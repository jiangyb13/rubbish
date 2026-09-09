"""
批量视频描述生成脚本。

整体流程：
1. 从 PKL 或 JSONL 读取视频相对路径；
2. 从 OBS/S3 将视频复制到本机 /cache；
3. 截取视频前 5 秒并均匀采样 32 帧；
4. 使用 Qwen3-VL 生成短描述；
5. 将 video_fn 与生成的 prompt 逐行写入远端 JSONL。

当前模型输入由 Qwen3-VL 的 AutoProcessor 和 qwen_vl_utils 负责构造。
"""

import argparse
import torch
from torch.utils.data import Dataset, DataLoader
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    # import transformer_npu
    # from npu.repetition_penality_logits_processor import *
except Exception as e:
    print("import torch_npu failed. maybe gpu env.")
import json
import os
import math
from tqdm import tqdm
import cv2
# from decord import VideoReader, cpu

from PIL import Image

import time
import json
import moxing as mox
import numpy as np
import pickle

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info





class VideoDataset(Dataset):
    """负责解析视频路径，并把每个远端视频下载到本地缓存。"""

    def __init__(self, video_paths, s3_data_root, s3_save_root):
        self.s3_data_root = s3_data_root
        self.s3_save_root = s3_save_root
        self.paths = video_paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # 兼容两种输入：
        # 1. PKL/JSONL 中直接保存路径字符串；
        # 2. JSONL 中保存包含 video_fn 字段的字典。
        if isinstance(self.paths[index], str):
            path = os.path.join(self.s3_data_root, self.paths[index])
            s3_save_path = os.path.join(self.s3_save_root, self.paths[index])
            # local_path = f"/cache/{index}.mp4" /cache/0.01.mp4    /cache/00999_2/f83e523297e8eb973d8c4373b8128333/011.mp4
            # 保留原相对目录结构，避免不同子目录下的同名视频互相覆盖。
            local_path = f"/cache/{self.paths[index]}"

            try:
                mox.file.copy(path, local_path)	
                return {"local_path": local_path, "s3_path": path, "s3_save_path": s3_save_path}
            except:
                print(f"copy file {path} fail, change index")
                return {"local_path": "", "s3_path": path, "s3_save_path": s3_save_path}
        else:
            # 字典输入会额外返回 video_fn，输出时可原样保留该相对路径。
            path = os.path.join(self.s3_data_root, self.paths[index]['video_fn'])
            s3_save_path = os.path.join(self.s3_save_root, self.paths[index]['video_fn'])
            # local_path = f"/cache/{index}.mp4" /cache/0.01.mp4    /cache/00999_2/f83e523297e8eb973d8c4373b8128333/011.mp4
            local_path = f"/cache/{self.paths[index]['video_fn']}"

            try:
                mox.file.copy(path, local_path)	
                return {"local_path": local_path, "s3_path": path, "s3_save_path": s3_save_path, "video_fn": self.paths[index]['video_fn']}
            except:
                print(f"copy file {path} fail, change index")
                return {"local_path": "", "s3_path": path, "s3_save_path": s3_save_path, "video_fn": ""}


def split_list(lst, n):
    """将列表近似均分为 n 份；当前主流程未调用。"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    """返回均分后的第 k 份；当前主流程未调用。"""
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    """解析数据路径、模型配置、采帧参数及推理参数。"""
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--video_path", help="Path to the video files.", default="")
    #parser.add_argument("--output_dir", help="Directory to save the model results JSON (single save).", required=False)
    parser.add_argument("--start", type=int, default=0, help="start index of inference")
    parser.add_argument("--end", type=int, default=-1, help="end index of inference. -1 means infering through the whole data")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="llava_onevision_qwen")
    parser.add_argument("--vision-tower", help=str, default="google/siglip-so400m-patch14-384")
    parser.add_argument("--num_samples_per_conv", "-nspc", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--mm_resampler_type", type=str, default=None)
    parser.add_argument("--mm_spatial_pool_stride", type=int, default=4)
    parser.add_argument("--mm_spatial_pool_out_channels", type=int, default=1024)
    parser.add_argument("--mm_spatial_pool_mode", type=str, default="bilinear")
    parser.add_argument("--mm_patch_merge_type", type=str, default="spatial_unpad")
    parser.add_argument("--image_aspect_ratio", type=str, default=None)
    # 计划送入视觉模型的采样帧数；后文目前固定覆盖为 32。
    parser.add_argument("--for_get_frames_num", type=int, default=32)
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--load_8bit", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--generation_config", type=str, default=None)
    parser.add_argument("--add_image_token", action="store_true", default=False)
    parser.add_argument("--no_do_sample", action="store_true", default=False)
    parser.add_argument("--prompt", type=str, default="Describe the video in details.")
    # pkl_path 和 jsonl_path 二选一；jsonl_path 非空时优先使用 JSONL。
    parser.add_argument('--pkl_path', type=str, default='')
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", default="pred_res_")
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", default="s3://bucket-6824-huanan/code/l60037260/video_50_res/")
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    # parser.add_argument('--s3_data_root', type=str,default='s3://bucket-6824-huanan/data/video/DIT_RES_l00503064/20240430/')
    parser.add_argument('--s3_data_root', type=str,default='s3://bucket-6824-huanan/data/video/DIT_RES_l00503064/')
    parser.add_argument('--s3_save_path', type=str,default='s3://bucket-6824-huanan/code/l60037260/video_50_res/')
    parser.add_argument("--s3_llava_path", type=str, default='s3://bucket-6824-huanan/code/l60037260/oneversion_1120/llava_onevision_infer/llava/models/Model_VideoCaption_v1.3/')
    parser.add_argument("--s3_clip_path", type=str, default='s3://bucket-6824-huanan/code/l60037260/oneversion_1120/llava_onevision_infer/llava/models/siglip-so400m-patch14-384/')
    parser.add_argument("--s3_base_path", type=str, default='s3://bucket-6824-huanan/code/l60037260/oneversion_1120/llava_onevision_infer/llava/models/llava-onevision-qwen2-7b-ov/')
    parser.add_argument("--s3_config_path", type=str, default='s3://bucket-6824-huanan/code/l60037260/oneversion_1120/llava_onevision_infer/configs/generation_configs/llava_onevision_npu/generation_config.json')
    parser.add_argument('--jsonl_path', type=str, default='')

    return parser.parse_args()

def get_all_videos_from_dir(video_path):
    """递归收集目录中的 MP4；当前主流程未调用。"""
    mp4_files = []
    count = 0
    for root, dirs, files in os.walk(video_path):
        for file in files:
            if file.endswith(".mp4"):
                count += 1
                mp4_files.append(os.path.join(root, file))
    return mp4_files, count

def load_video(video_path, max_frames_num):
    """从视频前 5 秒内均匀抽取 max_frames_num 帧，并返回 RGB 数组。

    例如 24 FPS 视频会先限定到前 120 个原始帧，再从中等间隔取 32 帧。
    这里的抽帧只用于生成 caption，不是 VAE target 使用的逐帧采样。
    """
    cv2_vr = cv2.VideoCapture(video_path)
    # OpenCV 给出的原始视频总帧数。
    duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = int(cv2_vr.get(cv2.CAP_PROP_FPS))
    # caption 只描述前 5 秒；长于 5 秒的后续内容不会进入模型。
    if duration > fps*5:
        duration = int(fps*5)
        print(f"new total_frames:{duration}")

    # 在有效时间区间首尾之间等间隔选帧，而不是连续读取前 32 帧。
    frame_id_list = np.linspace(0, (duration - 1), max_frames_num, dtype=int).tolist()
    print(frame_id_list)
    frames = []
    for frame_idx in frame_id_list:
        cv2_vr.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cv2_vr.read()
        if not ret:
            raise ValueError(f'video error at {video_path}')
        # OpenCV 默认 BGR；视觉预处理器需要 RGB。
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(str(frame_idx).zfill(4) + ".jpg", frame)
        frames.append(frame)
    cv2_vr.release()
    return np.array(frames)





# ========================= 模型初始化 =========================
args = parse_args()

args.for_get_frames_num = 32
args.num_samples_per_conv = 1
args.mm_spatial_pool_stride = 4
args.overwrite = True
args.add_image_token = True
args.no_do_sample = True

model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.model_path,
    dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(args.model_path)

def write_json(file_path, data):
    """将列表写为 JSONL；当前主流程使用 mox.file.append 直接写远端。"""
    f = open(file_path, "w", encoding='utf-8')
    for item in tqdm(data):
        f.writelines(json.dumps(item, ensure_ascii=False)+"\n")
    f.close()

def run_inference():
    """执行数据读取、逐视频推理、结果写入和本地缓存清理。"""

    pid = os.getpid()
    output_name = args.output_name+str(pid)+"_"+str(args.start_index)+"_"+str(args.end_index)+"_3s_short_caption"
    
    #print("当前进程pid：",pid)
    # 每个进程使用包含 PID 和数据切片范围的独立输出文件，降低并发写冲突。
    answers_file = os.path.join(args.output_dir, f"{output_name}.jsonl")
    mox.file.File(answers_file, "w")
    sample_set = {}
    # JSONL 优先；未提供时把远端 PKL 下载到固定本地路径。
    if args.jsonl_path != '':
        input_file = args.jsonl_path
    else:
        os.makedirs("/cache/wl_48", exist_ok=True)
        mox.file.copy(args.pkl_path, "/cache/wl_48/human240W_FI_2_intersection_set_new.pkl")
        input_file = "/cache/wl_48/human240W_FI_2_intersection_set_new.pkl"

    # 将两类输入统一整理为 paths：
    # - 路径字符串列表；
    # - 含 video_fn 字段的字典列表。
    if input_file.endswith('jsonl'):
        paths = []
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)

                # 如果 jsonl 每行就是一个路径字符串
                if isinstance(data, str):
                    paths.append(data)

                # 如果 jsonl 每行是 dict，根据你的字段改这里
                else:
                    # 例如：
                    # paths.append(data["video_fn"])
                    # paths.append(data["shot_video_path"])
                    paths.append(data)
    else:
        with open(input_file, "rb") as infile:
            paths = pickle.load(infile)
    # 多任务并行时可用起止下标切分数据。
    paths = paths[args.start_index: args.end_index]
    data_set = VideoDataset(paths, args.s3_data_root, args.s3_save_path)
    loader = DataLoader(data_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False,
                        drop_last=False)

    # DataLoader 批量完成下载；模型推理仍在 batch 内逐条执行。
    for batch in tqdm(loader,desc="当前进度"): 
        for i in range(len(batch["local_path"])):
            local_path = batch["local_path"][i]
            s3_path = batch["s3_path"][i]
            s3_save_path = batch["s3_save_path"][i]
            if local_path == "":
                print("Ignore the currently processed video and proceed to the next step.")
                continue


            try:
                question = args.prompt
                # 获取前 5 秒的 32 个均匀采样 RGB 帧。
                video_frames = load_video(local_path, max_frames_num=args.for_get_frames_num)

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": [Image.fromarray(frame) for frame in video_frames],
                            },
                            {
                                "type": "text",
                                "text": question,
                            },
                        ],
                    }
                ]

                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages,
                    image_patch_size=16,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
                if video_inputs is not None:
                    video_inputs, video_metadatas = zip(*video_inputs)
                    video_inputs = list(video_inputs)
                    video_metadatas = list(video_metadatas)
                else:
                    video_metadatas = None

                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    video_metadata=video_metadatas,
                    padding=True,
                    return_tensors="pt",
                    do_resize=False,
                    **video_kwargs,
                ).to(model.device)


                with torch.inference_mode():
                    start_time = time.time()

                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )

                    end_time = time.time()
                    print(f"Time taken for inference: {end_time - start_time} seconds")

                    generated_ids_trimmed = [
                        output_ids[len(input_ids):]
                        for input_ids, output_ids
                        in zip(inputs.input_ids, generated_ids)
                    ]
                    outputs = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0].strip()
                    
                    # 字典输入优先保留原 video_fn；字符串输入则从 S3 路径截取末三级。
                    if 'video_fn' not in batch:
                        try:
                            sample_set["video_fn"] = "/".join(s3_path.split("/")[-3:])
                        except:
                            sample_set["video_fn"] = s3_path
                    else:
                        sample_set["video_fn"] = batch["video_fn"][i]
                    sample_set["prompt"] = outputs
                    # 每成功处理一个视频便立即追加一行，避免进程中断丢失全部结果。
                    mox.file.append(answers_file, json.dumps(sample_set)+"\n")
                    try:
                        os.remove(local_path)
                        print(f"Deleted file: {local_path}")
                    except Exception as e:
                        print(f"Error deleting file {local_path}: {e}")
            except Exception as e:
                print(f"Failed to caption {local_path}: {e}")
                # 推理失败时记录空描述，但当前写出语句被注释，因此失败样本
                # 实际不会出现在输出 JSONL 中。
                try:
                    sample_set["video_fn"] = "/".join(s3_path.split("/")[-3:])
                except:
                    sample_set["video_fn"] = s3_path
                sample_set["prompt"] = ""
                # mox.file.append(answers_file, json.dumps(sample_set)+"\n")
                try:
                    os.remove(local_path)
                    print(f"Deleted file: {local_path}")
                except Exception as cleanup_error:
                    print(f"Error deleting file {local_path}: {cleanup_error}")


if __name__ == "__main__":
    run_inference()

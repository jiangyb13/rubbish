

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

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image

import time
import json
import moxing as mox
import numpy as np
import pickle






class VideoDataset(Dataset):
	def __init__(self, video_paths, s3_data_root, s3_save_root):
		self.s3_data_root = s3_data_root
		self.s3_save_root = s3_save_root
		self.paths = video_paths
	
	def __len__(self):
		return len(self.paths)
	
	def __getitem__(self, index):

		path = os.path.join(self.s3_data_root, self.paths[index])
		s3_save_path = os.path.join(self.s3_save_root, self.paths[index])
		# local_path = f"/cache/{index}.mp4" /cache/0.01.mp4    /cache/00999_2/f83e523297e8eb973d8c4373b8128333/011.mp4
		local_path = f"/cache/{self.paths[index]}"

		try:
			mox.file.copy(path, local_path)	
			return {"local_path": local_path, "s3_path": path, "s3_save_path": s3_save_path}
		except:
			print(f"copy file {path} fail, change index")
			return {"local_path": "", "s3_path": path, "s3_save_path": s3_save_path}


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--video_path", help="Path to the video files.", default="")
    #parser.add_argument("--output_dir", help="Directory to save the model results JSON (single save).", required=False)
    parser.add_argument("--start", type=int, default=0, help="start index of inference")
    parser.add_argument("--end", type=int, default=-1, help="end index of inference. -1 means infering through the whole data")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
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
    parser.add_argument("--for_get_frames_num", type=int, default=32)
    parser.add_argument("--overwrite", type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument("--load_8bit", type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument("--generation_config", type=str, default=None)
    parser.add_argument("--add_image_token", action="store_true", default=False)
    parser.add_argument("--no_do_sample", action="store_true", default=False)
    parser.add_argument("--prompt", type=str, default="Describe the video in details.")
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

    return parser.parse_args()

def get_all_videos_from_dir(video_path):
    mp4_files = []
    count = 0
    for root, dirs, files in os.walk(video_path):
        for file in files:
            if file.endswith(".mp4"):
                count += 1
                mp4_files.append(os.path.join(root, file))
    return mp4_files, count

# Function to extract frames from video
def load_video(video_path, max_frames_num):
    cv2_vr = cv2.VideoCapture(video_path)
    duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = int(cv2_vr.get(cv2.CAP_PROP_FPS))
    #print(f"old total_frames:{duration}")
    if duration > fps*5:
        duration = int(fps*5)
        print(f"new total_frames:{duration}")

    frame_id_list = np.linspace(0, (duration - 1), max_frames_num, dtype=int).tolist()
    print(frame_id_list)
    frames = []
    for frame_idx in frame_id_list:
        cv2_vr.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cv2_vr.read()
        if not ret:
            raise ValueError(f'video error at {video_path}')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cv2.imwrite(str(frame_idx).zfill(4) + ".jpg", frame)
        frames.append(frame)
    cv2_vr.release()
    return np.array(frames)





""" 模型初始化 """
args = parse_args()


args.for_get_frames_num = 32
args.num_samples_per_conv = 1
args.mm_spatial_pool_stride = 4
args.overwrite = True
args.vision_tower = "/home/ma-user/modelarts/user-job-dir/llava_onevision_infer/llava/models/siglip-so400m-patch14-384"
args.add_image_token = True
args.no_do_sample = True
# args.prompt = "Please provide a detailed description of the video, focusing on the main subjects, their actions and the background scenes. Please give your most confident answer and do not answer with uncertain content."
args.prompt = "Please provide a detailed description of the video, focusing on the main subjects, their actions and the background scenes. Please give your most confident answer and do not answer with uncertain content. The number of words to output must be less than 25 words."

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(args.model_path)

def write_json(file_path, data):
    f = open(file_path, "w", encoding='utf-8')
    for item in tqdm(data):
        f.writelines(json.dumps(item, ensure_ascii=False)+"\n")
    f.close()

def run_inference():
    """
    Run inference
    """

    pid = os.getpid()
    output_name = args.output_name+str(pid)+"_"+str(args.start_index)+"_"+str(args.end_index)+"_3s_short_caption"
    
    #print("当前进程pid：",pid)
    answers_file = os.path.join(args.output_dir, f"{output_name}.jsonl")
    mox.file.File(answers_file, "w")
    sample_set = {}
    os.makedirs("/cache/wl_48", exist_ok=True)
    mox.file.copy(args.pkl_path, "/cache/wl_48/human240W_FI_2_intersection_set_new.pkl")
    input_file = "/cache/wl_48/human240W_FI_2_intersection_set_new.pkl"

    # 读取 pkl 文件并加载列表
    with open(input_file, "rb") as infile:
        paths = pickle.load(infile)
    paths = paths[args.start_index: args.end_index]
    data_set = VideoDataset(paths, args.s3_data_root, args.s3_save_path)
    loader = DataLoader(data_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False,
                        drop_last=False)

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
                video_frames = load_video(local_path, max_frames_num=args.for_get_frames_num)
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "video", "video": [Image.fromarray(frame) for frame in video_frames]},
                        {"type": "text", "text": question},
                    ],
                }]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)

                if args.generation_config is not None and os.path.exists(args.generation_config):
                    from transformers.generation.configuration_utils import GenerationConfig
                    with open(args.generation_config, "r") as f:
                        generation_config = json.load(f)
                    generation_config = GenerationConfig.from_dict(generation_config)
                else:
                    generation_config = None


                with torch.inference_mode():
                    start_time = time.time()
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        temperature=args.temperature,
                        do_sample=False,
                        generation_config=generation_config,
                    )
                    end_time = time.time()
                    print(f"Time taken for inference: {end_time - start_time} seconds")
                    output_ids = output_ids[:, inputs.input_ids.shape[1]:]
                    outputs = processor.batch_decode(
                        output_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]
                    #print(f"Question: {prompt}\n")
                    #print(f"Response: {outputs}\n")
                    outputs = outputs.strip()
                    try:
                        sample_set["video_fn"] = "/".join(s3_path.split("/")[-3:])
                    except:
                        sample_set["video_fn"] = s3_path
                    sample_set["prompt"] = outputs
                    mox.file.append(answers_file, json.dumps(sample_set)+"\n")
                    try:
                        os.remove(local_path)
                        print(f"Deleted file: {local_path}")
                    except:
                        print(f"Error deleting file {local_path}: {e}")
            except:
                try:
                    sample_set["video_fn"] = "/".join(s3_path.split("/")[-3:])
                except:
                    sample_set["video_fn"] = s3_path
                sample_set["prompt"] = ""
                mox.file.append(answers_file, json.dumps(sample_set)+"\n")
                try:
                    os.remove(local_path)
                    print(f"Deleted file: {local_path}")
                except:
                    print(f"Error deleting file {local_path}: {e}")


if __name__ == "__main__":
    run_inference()

import os
import sys

os.environ["MOX_SILENT_MODE"] = "1"
os.environ["MOX_FILE_LARGE_FILE_METHOD"] = "1"  # for moxing download acceleration
os.environ["PYTORCH_NPU_ALLOC_CONF"]="expandable_segments:True"

the_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(the_dir)

import torch
import torch_npu
torch_npu.npu.set_compile_mode(jit_compile=False)
torch_npu.npu.config.allow_internal_format = False
torch.npu.conv.allow_hf32 = False
torch.npu.matmul.allow_hf32 = False

from torch_npu.contrib import transfer_to_npu

try:
    import moxing as mox
except:
    print("no moxing")

import gc
import argparse
import json
import functools
import datetime
import time
import random

import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, \
    ShardedStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import BackwardPrefetch, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from torch.profiler import ProfilerActivity, tensorboard_trace_handler
from torch.profiler import profile, schedule
from torch.autograd.profiler import record_function

from torchvision.io import write_video
from torchvision.utils import save_image

from mimogpt.models.dit.parallel_states import initialize_distributed, get_context_parallel_group
from mimogpt.models import build_backbone
from mimogpt.engine.utils.profile_npu import Nothing
from mimogpt.utils.load_optimizations import skip_torch_weight_init, convert_to_dir_dist
from mimogpt.utils.log_utils import print_t, rank0_print
from mimogpt.functional import to_torch_dtype

from inference_mmdit import save_sample, merge_args, parse_configs, load_prompts
from inference_mmdit import warp_fsdp, get_profiling_fn

from mmengine.runner import set_random_seed
from mmengine.config import Config
from transformers.models.t5.modeling_t5 import T5Block
from PIL import Image
import numpy as np
from torchvision import transforms as T
import cv2

from mimogpt.utils.vid_utils import pad_and_resize, face_preprocess  # add by wzz
try:
    from mimogpt.datasets.feat_codec import decode_feat_tensor, decode_feat_tensor_mox
except:
    encode_feat_tensor, decode_feat_tensor = None, None

TARGET_SIZE_720 = {
    1.7778: [1280, 720],
    1.0000: [960, 960],
    0.5625: [720, 1280],
    0.6667: [768, 1152],
    1.5000: [1152, 768],
    0.7500: [832, 1088],
    1.3333: [1088, 832],
    2.0000: [1344, 672],
    0.5000: [672, 1344],
}

TARGET_SIZE_480 = {
    1.0000: [576, 576],
    1.5: [720, 480],
    0.6667: [480, 720],
    1.7778: [768, 432],
    0.5625: [432, 768],
    1.3333: [640, 480],
    0.75: [480, 640],
    0.5: [384, 768],
    2.0: [768, 384]
}

TARGET_SIZE_360 = {
    0.5625: [360, 720],
}


def get_resize_crop_size(input_size, cfg_input_size=480, enable_multi_resolution=0):
    input_ratio = input_size[0] / input_size[1]
    if cfg_input_size == 480:
        if enable_multi_resolution:
            new_y = int(16 * (np.sqrt(480 * 720 / input_ratio) // 16))
            new_x = int(16 * ((480 * 720 / new_y) // 16))
            crop_size = [new_x, new_y]
            closest_ratio = new_x / new_y
        else:
            closest_ratio = min(TARGET_SIZE_480.keys(), key=lambda ratio: abs(float(ratio) - input_ratio))
            crop_size = TARGET_SIZE_480[closest_ratio]
    elif cfg_input_size == 720:
        if enable_multi_resolution:
            new_y = int(16 * (np.sqrt(720 * 1280 / input_ratio) // 16))
            new_x = int(16 * ((720 * 1280 / new_y) // 16))
            crop_size = [new_x, new_y]
            closest_ratio = new_x / new_y
        else:
            closest_ratio = min(TARGET_SIZE_720.keys(), key=lambda ratio: abs(float(ratio) - input_ratio))
            crop_size = TARGET_SIZE_720[closest_ratio]
    else:
        print("cfg_input_size must select in [480, 720]")
        raise KeyError
    ratio = max(crop_size[0] / input_size[0], crop_size[1] / input_size[1])
    resize_size = (int(input_size[0] * ratio), int(input_size[1] * ratio))
    return resize_size, crop_size, closest_ratio


class NormalizeToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, reshape=True):
        self.reshape = reshape

    def __call__(self, image):
        image = np.array(image).astype(np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)
        if self.reshape:
            image = np.reshape(image, (image.shape[0], image.shape[1], -1))
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)


def preprocess(video, cfg_input_size=480, enable_multi_resolution=0):
    resize_size, crop_size, closest_ratio = get_resize_crop_size(video.shape[-3:-1], cfg_input_size,
                                                                 enable_multi_resolution)

    transform = T.Compose([
        T.Resize((resize_size[0], resize_size[1])),
        T.CenterCrop(crop_size),
        NormalizeToTensor()
    ])
    transform_1 = T.Compose([
        NormalizeToTensor()
    ])
    if video.shape[-3] == crop_size[0] and video.shape[-2] == crop_size[1]:
        video = torch.stack([transform_1(Image.fromarray(frame)) for frame in video], dim=1)
    else:
        video = torch.stack([transform(Image.fromarray(frame)) for frame in video], dim=1)
    return video, closest_ratio


def parse_args():
    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument("--config", help="model config file path")

    parser.add_argument("--seed", default=42, type=int, help="generation seed")
    parser.add_argument("--gpus", default=2, type=int)
    parser.add_argument("--save_double", default=False, type=bool)
    parser.add_argument('--backend', type=str, default="nccl")
    parser.add_argument("--ckpt-path", type=str, help="path to model ckpt; will overwrite cfg.ckpt_path if specified")
    parser.add_argument("--batch-size", default=None, type=int, help="batch size")
    parser.add_argument("--root", default=None, type=str, help="data dir")
    parser.add_argument("--val_root", default=None, type=str, help="val_data dir")
    parser.add_argument("--val-data-path", default=None, type=str, help="val path to data csv")

    # ======================================================
    # Inference
    # ======================================================
    # prompt
    parser.add_argument("--prompt-path", default=None, type=str, help="path to prompt txt file")
    parser.add_argument("--input-dir", default=None, type=str, help="path to prompt txt file")
    parser.add_argument("--input-finish-dir", default=None, type=str, help="path to prompt finish txt file")
    parser.add_argument("--save-dir", default=None, type=str, help="path to save generated samples")
    parser.add_argument("--save-latent", default=0, type=int, help="if save latent pt")
    # variation test
    parser.add_argument("--variation", default=False, type=bool, help="whether test multiratio or multilength")
    # hyperparameters
    parser.add_argument("--num-sampling-steps", default=None, type=int, help="sampling steps")
    parser.add_argument("--cfg-scale", default=None, type=float, help="balance between cond & uncond")

	# I2V
    parser.add_argument("--ref-image", type=str, choices=["true", "false"], default="true", help="use ref image")  # add by wzz
    parser.add_argument("--first-image", type=str, choices=["true", "false"], default="true", help="use first image")  # add by wzz
    parser.add_argument("--face_dir", default='face_img', type=str, help="face dir of image/video condition")   # add by wzz
    parser.add_argument("--face_aug_dir", default='aug_face', type=str, help="augmented face dir of image/video condition")   # add by wzz
    parser.add_argument("--test_jsonl", default='aug_face', type=str, help="")   # add by wzz

    parser.add_argument("--img_dir", default=None, type=str)
    parser.add_argument("--cond_name", default='name_list.txt', type=str, help="name list of image/video condition")
    parser.add_argument("--enable_multi_resolution", default=1, type=int,
                        help="Using any resolution or multiple buckets, 1 for any resolution, 0 for multiple buckets")
    parser.add_argument("--chunk_start", default=0, type=int,
                        help="if using mulit nodes infer, using chunk_start to tell the save name index")

    # ci
    parser.add_argument("--efficiency", default=None, type=str, help="time for each inference")
    # profiling
    parser.add_argument("--profile", default=0, type=int, help="profile enable")
    parser.add_argument("--skip_first", default=1, type=int, help="profile skip_first")
    parser.add_argument("--wait", default=0, type=int, help="profile wait")
    parser.add_argument("--warmup", default=0, type=int, help="profile warmup")
    parser.add_argument("--active", default=3, type=int, help="profile active")
    parser.add_argument("--repeat", default=1, type=int, help="profile repeat")
    parser.add_argument("--profile_save_path", default="profiling", type=str, help="profile enable")

    # debug
    parser.add_argument("--deterministic", default=0, type=int, help="deterministic")

    args, _ = parser.parse_known_args()
    return args

def main():
    gc.disable()
    gc.set_threshold(70, 10, 1000)
    # ======================================================
    # 1. cfg and init distributed env
    # ======================================================
    cfg = parse_configs(parse_args)

    # 兼容历史配置文件
    if "inference_algo" not in cfg.keys():
        cfg["inference_algo"] = {}

    for k, v in cfg.items():
        if isinstance(v, dict):
            rank0_print("  {}: ".format(k), newline=False)
            for _k, _v in v.items():
                rank0_print("      {} : {}".format(_k, _v), newline=False)
        else:
            rank0_print("  {}: {}".format(k, v), newline=False)

    # 确定性计算, 开启后, seed一致/模型切分一致(CP等)/环境一致，则推理结果可做到二进制一致
    if cfg.deterministic:
        torch.use_deterministic_algorithms(True)
        os.environ["HCCL_DETERMINISTIC"] = "True"
        os.environ["CLOSE_MATMUL_K_SHIFT"] = "1"

    # init distributed
    if "RANK" in os.environ:
        cfg.rank = int(os.environ['RANK'])
        cfg.gpus = int(os.environ['WORLD_SIZE'])
    torch.distributed.init_process_group(
        backend=cfg.backend,
        init_method=None,
        rank=cfg.rank,
        world_size=cfg.gpus,
        timeout=datetime.timedelta(hours=2.)
    )
    torch.cuda.set_device(cfg.rank % 8)

    cp_size = dist.get_world_size()
    initialize_distributed(cp_size)
    cfg.distributed.context_parallelism_size = cp_size
    rank0_print("WARNING:当前推理仅支持CP=world_size")

    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    device = "npu"
    dtype = to_torch_dtype(cfg.dtype)

    data = []
    with open(cfg.test_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"[WARN] 第 {line_num} 行解析失败: {e}")
    # if '.txt' in cfg.prompt_path:
    #     full_prompts = load_prompts(cfg.prompt_path)
    #     cond_name_list = load_prompts(cfg.cond_name)
    # else:
    #     import pandas as pd
    #     df = pd.read_excel(cfg.prompt_path)
    #     full_prompts = list(df['prompt'])
    #     cond_name_list = list(df['video_fn'])
    #     cond_name_list = [os.path.join(cfg.img_dir, i) for i in cond_name_list]

    # rank0_print("prompts:{}".format(full_prompts))

    if cfg.seed == 0:
        dist.barrier()
        ts_tensor = torch.zeros(dist.get_world_size()).cuda()
        ts_tensor[dist.get_rank()] = int(time.time())
        dist.all_reduce(ts_tensor, op=dist.ReduceOp.SUM)

        start_ts = int(ts_tensor[0].item() % 10000)
        print(f"[rank {dist.get_rank()}] 统一启动时间戳 = {start_ts}")

        set_random_seed(seed=start_ts)
    else:
        set_random_seed(seed=cfg.seed)

    # ======================================================
    # 3. build model & load weights
    # ======================================================
    # 3.1 build model
    # 3.1.1 vae
    input_size = (cfg.num_frames, *cfg.image_size)
    with skip_torch_weight_init():
        vae = build_backbone(cfg, cfg.vae.get("backbone", "vae_causal_16ch_dist"))
    vae = vae.to(device, dtype).eval()
    with skip_torch_weight_init():
        vae_en = build_backbone(cfg, cfg.vae_en.get("backbone", "motionvae_16ch"))
    vae_en = vae_en.to(device, dtype).eval()
    latent_size = vae.get_latent_size(input_size)
    rank0_print("latent size is: {}".format(latent_size))

    if cfg.vae.get('use_framescale', False):
        frame_scale = torch.tensor(cfg.vae.frame_scale, dtype=torch.float32)[:latent_size[0]].cuda()
        frame_bias = torch.tensor(cfg.vae.frame_bias, dtype=torch.float32)[:latent_size[0]].cuda()
    else:
        vae_scale = cfg.vae.get('scale', 1.0)
        vae_bias = cfg.vae.get('bias', 0.0)

    mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
    rank0_print("after build vae, memory_allocated: {:.1f}GB".format(mem_after))

    # 3.1.2 text_encoder
    with skip_torch_weight_init(), torch.device("cuda"):
        text_encoder = build_backbone(cfg, "t5_online")  # T5 must be fp32

    if cfg.inference_algo.get('t5_weights_bf16', False):
        text_encoder.t5.model = text_encoder.t5.model.to(dtype=torch.bfloat16)

    text_encoder.t5.model = warp_fsdp(text_encoder.t5.model, T5Block, dtype=text_encoder.dtype)

    mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
    rank0_print("after build t5, memory_allocated: {:.1f}GB".format(mem_after))

    # 3.1.3 dit
    with skip_torch_weight_init(), torch.device('meta'):
        model = build_backbone(cfg)
    model = model.to(dtype).eval()

    mmdit_pretrained = cfg.model.get('from_pretrained', None)
    if mmdit_pretrained is not None:
        assert os.path.isfile(mmdit_pretrained), f"Could not find DiT checkpoint at {mmdit_pretrained}"

        dist_ckpt_dir = convert_to_dir_dist(mmdit_pretrained)
        if os.path.exists(dist_ckpt_dir):
            cache_dist_ckpt = False
        else:
            model.to_empty(device="cpu")
            print(f"load model from {mmdit_pretrained}")
            from pathlib import Path
            model_path = Path(mmdit_pretrained)
            ext = model_path.suffix.lower()
            if ext == ".safetensors":
                from safetensors.torch import load_file as load_safetensors
                state_dict = load_safetensors(model_path)
            else:
                state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
            if "module" in state_dict:
                state_dict = state_dict["module"]
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            cache_dist_ckpt = True

    # 兼容alltoall_overlap推理，model对应的block自动区分
    model = warp_fsdp(model, model.blocks[0].__class__, dtype=dtype)  # use fsdp zero3

    if mmdit_pretrained is not None:
        if cache_dist_ckpt:
            dist_ckpt_dir = convert_to_dir_dist(mmdit_pretrained)
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                                    state_dict_config=ShardedStateDictConfig(offload_to_cpu=True)):
                state_dict = {"model": model.state_dict(), }
                dist_cp.save_state_dict(
                    state_dict=state_dict,
                    storage_writer=dist_cp.FileSystemWriter(dist_ckpt_dir),
                )
            os.system("chmod -R 777 {}".format(dist_ckpt_dir))
            rank0_print("save distributed checkpoint to {}".format(dist_ckpt_dir))
        else:
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                                    state_dict_config=ShardedStateDictConfig(offload_to_cpu=True)):
                state_dict = {"model": model.state_dict(), }
                dist_cp.load_state_dict(
                    state_dict=state_dict,
                    storage_reader=dist_cp.FileSystemReader(dist_ckpt_dir),
                )
                model.load_state_dict(state_dict["model"])
            rank0_print("load distributed checkpoint from {}".format(dist_ckpt_dir))

    if cfg.inference_algo.get("enable_dit_inference_offload", False):
        model.enable_dit_inference_offload()

    text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance

    mem_after = torch.cuda.memory_allocated() / (1024 ** 3)
    rank0_print("after build dit, memory_allocated: {:.1f}GB".format(mem_after))

    # 3.2 build scheduler
    scheduler = build_backbone(cfg, cfg['scheduler']['backbone'])

    # ======================================================
    # 4. inference
    # ======================================================
    save_dir = cfg.save_dir.replace('s3://', '/cache/') if 's3://' in cfg.save_dir else cfg.save_dir
    pt_savedir = os.path.join(save_dir, 'latents')
    os.makedirs(save_dir, exist_ok=True)
    # os.makedirs(pt_savedir, exist_ok=True)
    time_list = []

    cfg_input_size = min(cfg.image_size[0], cfg.image_size[1])

    profile_fn = get_profiling_fn(cfg)
    with profile_fn as prof:
        for (idx, item) in zip(range(len(data)), data):
            torch.cuda.synchronize()
            t_start = time.time()
            user_cond_name = os.path.join(cfg.img_dir, item['first_frame'])
            user_prompt = item['prompt']
            rank0_print("prompt: {}".format(user_prompt))

            # 4.1 prepare inputs
            prompts = [user_prompt, ]
            neg_prompts = scheduler.negative_prompt
            n = len(prompts)
            # print(f"system_prompt: {scheduler.system_prompt}")
            if scheduler.system_prompt != None:
                system_prompt = n * scheduler.system_prompt
                prompts = list(map(lambda x: x[0] + x[1], zip(prompts, system_prompt)))
                rank0_print("after add system_prompt: {}".format(user_prompt))

            video = np.expand_dims(np.array(Image.open(user_cond_name.strip()).convert('RGB')), 0)
            video = preprocess(video, cfg_input_size, cfg.enable_multi_resolution)[0].to(dtype)
            video_sub = video.unsqueeze(0)
            cond_input = vae_en.encode(video_sub.to(device, dtype))[0].unsqueeze(0)

            given_len = 1
            cond = torch.zeros([1, vae.out_channels, latent_size[0], cond_input.shape[-2], cond_input.shape[-1]]).cuda()
            cond[:, :, :given_len] = (cond_input[:, :, :given_len] - frame_bias[None, None, :given_len, None,
                                                                     None]) * frame_scale[None, None, :given_len, None,
                                                                              None]
            mask = torch.zeros(1, 8, latent_size[0], cond_input.shape[3], cond_input.shape[4]).cuda()
            mask[:, :, :given_len] = 1

            #############################################################################################
            # add by wzz
            if cfg.first_image == "false":
                print("==============drop first frame!!!")
                cond[:, :, :given_len] = 0
                mask[:, :, :given_len] = 0

            # uncond = cond.clone()   # add by wzz
            # uncond_mask = mask.clone()  # add by wzz

            if cfg.ref_image == "true":
                face_img_list = []
                print("==============use face ref!!!")
                print(item['in_cross_pair_face_fn'])
                for face_path in item['in_cross_pair_face_fn']:
                    face_path = os.path.join(cfg.face_aug_dir, face_path)
                    print(face_path)
                    if os.path.exists(face_path):
                        face_img = Image.open(face_path).convert("RGB")
                        face_img_list.append(face_img)
                
                # face_img_path = os.path.join(cfg.face_dir, os.path.basename(user_cond_name))
                # if os.path.exists(face_img_path):
                #     face_img = Image.open(face_img_path).convert('RGB')
                #     face_img_list = [face_img]
                # name, ext = os.path.basename(user_cond_name).split('.')
                # ext_candidates = [ext, "jpg", "jpeg", "png"]
                # for i in range(4):   # 支持最多4个参考
                #     if len(face_img_list) >= 4:
                #         break
                #     for e in ext_candidates:
                #         img_path = os.path.join(cfg.face_aug_dir, f"{name}_{i}.{e}")
                #         if os.path.exists(img_path):
                #             face_img = Image.open(img_path).convert("RGB")
                #             face_img_list.append(face_img)
                #             break  # 找到一个就够了，不再尝试其他ext
                print(f'Use {len(face_img_list)} ref imgs')
                
                pad_resize_face_img_list = []
                for face_img in face_img_list:
                    face_img = pad_and_resize([face_img], cond_input.shape[3]*8, cond_input.shape[4]*8)
                    pad_resize_face_img_list.append(face_img)
                    # face_img.save(f'img_{random.random()}.png')

                # face_img = pad_and_resize(face_img_list, cond_input.shape[3]*8, cond_input.shape[4]*8)

                # if dist.get_rank() % cp_size == 0:
                #     os.makedirs(os.path.join(save_dir, 'cond'), exist_ok=True)
                #     save_img_path = os.path.join(save_dir, 'cond', f"sample_{idx:02d}.jpg")
                #     face_img.save(save_img_path)

                video = []
                for face_img in pad_resize_face_img_list:
                    video.append(np.array(face_img))
                video = face_preprocess(video).unsqueeze(0)

                null_video = torch.zeros_like(video)

                # ref_img = np.expand_dims(np.array(face_img), 0)
                # ref_img = face_preprocess(ref_img).unsqueeze(0)
                videos = torch.split(video, 1, dim=2)
                null_videos = torch.split(null_video, 1, dim=2)
                ref_vae_list = []
                null_ref_vae_list = []
                for video in videos:
                    ref_vae = vae_en.encode(video.cuda().to(torch.bfloat16))[0].unsqueeze(0)
                    ref_vae_list.append(ref_vae)
                for video in null_videos:
                    null_ref_vae = vae_en.encode(video.cuda().to(torch.bfloat16))[0].unsqueeze(0)
                    null_ref_vae_list.append(null_ref_vae)
                ref_vae = torch.cat(ref_vae_list, dim=2)
                null_ref_vae = torch.cat(null_ref_vae_list, dim=2)
                # ref_vae = vae_en.encode(ref_img.cuda().to(dtype))[0].unsqueeze(0)

                ref_vae = ((ref_vae - frame_bias[None, None, :ref_vae.shape[2], None, None])
                           * frame_scale[None, None, :ref_vae.shape[2], None, None])
                null_ref_vae = ((null_ref_vae - frame_bias[None, None, :null_ref_vae.shape[2], None, None])
                           * frame_scale[None, None, :null_ref_vae.shape[2], None, None])

                ref_mask = torch.ones(ref_vae.shape[0], 8, ref_vae.shape[2], ref_vae.shape[3], ref_vae.shape[4]).cuda().to(dtype)
                ref_x = torch.cat([ref_vae, ref_vae, ref_mask], dim=1)
                null_ref_x = torch.cat([null_ref_vae, null_ref_vae, ref_mask], dim=1)
                # cond = torch.cat([cond, ref_vae], 2)
                # mask = torch.cat([mask, ref_mask], 2)

                # print("==============use zero_like ref!!! use zero image==============")
                # uncond_ref_img = torch.zeros_like(ref_img)

                # uncond_ref_vae = vae_en.encode(uncond_ref_img.cuda().to(dtype))[0].unsqueeze(0)

                # uncond_ref_vae = ((uncond_ref_vae - frame_bias[None, None, :uncond_ref_vae.shape[2], None, None])
                #            * frame_scale[None, None, :uncond_ref_vae.shape[2], None, None])

                # uncond_ref_mask = torch.ones(uncond_ref_vae.shape[0], 8, uncond_ref_vae.shape[2], uncond_ref_vae.shape[3], uncond_ref_vae.shape[4]).cuda().to(dtype)

                #############################################################################################
                # print("==============use zero_like ref!!! use zero vae==============")
                # uncond_ref_vae = torch.zeros_like(ref_vae)
                # uncond_ref_mask = torch.zeros(uncond_ref_vae.shape[0], 8, uncond_ref_vae.shape[2], uncond_ref_vae.shape[3], uncond_ref_vae.shape[4]).cuda().to(dtype)


                # uncond = torch.cat([uncond_ref_vae, uncond], 2)
                # uncond_mask = torch.cat([uncond_ref_mask, uncond_mask], 2)
                # uncond = torch.cat([uncond, uncond_ref_vae], 2)
                # uncond_mask = torch.cat([uncond_mask, uncond_ref_mask], 2)

            #############################################################################################

            cond = torch.cat([cond, mask], 1)
            # uncond = torch.cat([uncond, uncond_mask], 1)  # add by wzz

            z = torch.randn((n, vae.out_channels, cond.shape[-3], cond.shape[-2], cond.shape[-1]), device=device)
            # z = torch.randn((n, vae.out_channels, latent_size[0], cond.shape[-2], cond.shape[-1]), device=device)
            cfg_scale = cfg.scheduler.get('cfg_scale', 8.5)
            if cfg_scale != 1.0:
                z = torch.cat([z, z], 0)

            # 4.2 t5 text_encoder inference
            model_args = text_encoder.encode(prompts)
            # if t5_fn.startswith('s3://'):
            #     t5_feature_data = decode_feat_tensor_mox(t5_fn)
            # else:
            #     t5_feature_data = decode_feat_tensor(t5_fn)

            # model_args = {}
            # if len(t5_feature_data.shape) == 2:
            #     t5_feature_data = t5_feature_data.unsqueeze(0)
            
            # model_max_length = 400
            # t5_len = t5_feature_data.shape[1]
            # t5_feature_data_pad = torch.zeros((1, model_max_length, 4096), dtype=torch.bfloat16)
            # t5_feature_data_pad[:, :t5_len, :] = t5_feature_data[:, :model_max_length, :]
            # t5_mask = torch.zeros((model_max_length), dtype=torch.bool)
            # t5_mask[:t5_len] = 1
            # model_args["y"] = t5_feature_data_pad.unsqueeze(0)
            model_args_neg = text_encoder.encode(neg_prompts)
            # model_args["y"] = model_args["y"].to(model_args_neg["y"].device)
            # model_args["mask"] = t5_mask.unsqueeze(0).to(model_args_neg["mask"].device)
            model_args["x_mask"] = None
            if cfg.ref_image == "true":
                # model_args['ref_x'] = torch.cat([ref_x, torch.zeros_like(ref_x)], 0)
                # model_args['ref_x'] = torch.cat([ref_x, null_ref_x], 0)
                model_args['ref_x'] = torch.cat([ref_x, ref_x], 0)
                model_args['ref_timestep'] = torch.ones(2).to(ref_x.device).type_as(ref_x)
            if cfg_scale != 1.0:
                # print(model_args_neg["y"].shape)
                model_args["y"] = torch.cat([model_args["y"], model_args_neg["y"]], 0)
                model_args["y_mask"] = torch.cat([model_args["mask"], model_args_neg["mask"]], 0)
                model_args["cond"] = torch.cat([cond] * 2)
            else:
                model_args["y_mask"] = model_args["mask"]
                model_args["cond"] = cond

            actual_model = model._fsdp_wrapped_module if isinstance(model, FSDP) else model
            actual_model.enable_attn_map_extraction()   # ← 开关打开，列表清空 
            # 4.3 dit inference
            samples = scheduler.sample_pure(model, z, model_args)

            actual_model.disable_attn_map_extraction()

            # barrier 前只做极快的 tensor 保存，避免 rank0 视频编码过慢导致 HCCL 超时
            if dist.get_rank() % cp_size == 0:
                attn_save_dir = os.path.join(save_dir, 'attn_maps', f"sample{idx + cfg.chunk_start:02d}")
                os.makedirs(attn_save_dir, exist_ok=True)
                torch.save(
                    [m.cpu() for m in actual_model.all_timestep_attn_maps],
                    os.path.join(attn_save_dir, "tensors.pt")
                )
                rank0_print(f"Saved attn tensors for sample {idx + cfg.chunk_start:02d}")

            dist.barrier()

            # if cfg.ref_image == "true":
            #     # samples = samples[:, :, 1:, :, :] #todo
            #     samples = samples[:, :, :-1, :, :]  #todo

            # if dist.get_rank() == 0 and cfg.save_latent:
            #     torch.save(samples.cpu(), os.path.join(pt_savedir, f"sample_{idx}_{user_prompt[0:100]}.pt"))

            # 4.4 vae decoder inference
            # T C H W
            if cfg.vae.get('use_framescale', False):
                samples = samples / frame_scale[None, None, :, None, None] + frame_bias[None, None, :, None, None]
            else:
                samples = samples / vae_scale + vae_bias

            samples = vae.decode(samples.to(dtype))

            torch.cuda.synchronize()
            t_end = time.time()
            rank0_print("inference idx={}, inference time={:.1f}s".format(idx, t_end - t_start))
            time_list.append(float(t_end - t_start))

            # 4.5 save videos
            if dist.get_rank() % cp_size == 0:
                save_path = os.path.join(save_dir, f"sample_{idx + cfg.chunk_start:02d}")
                save_sample(samples[0], fps=cfg.fps, save_path=save_path, cfg=cfg)

            dist.barrier()

            prof.step()
            time.sleep(0.1)

            if cfg.profile and idx == (cfg.skip_first + (cfg.wait + cfg.warmup + cfg.active) * cfg.repeat):
                if dist.get_rank() % 8 == 0 and 's3://' in cfg.save_dir:
                    src = "mmdit_profiling_worker{}.zip".format(int(dist.get_rank() // 8))
                    dst = os.path.join(cfg.save_dir, 'profiling', src)
                    os.system("zip -r {} {}".format(src, cfg.profile_save_path))
                    try:
                        import moxing as mox
                        mox.file.copy(src, dst)
                        rank0_print("Succeed to Mox the {} to {}".format(src, dst))
                    except Exception as e:
                        rank0_print("ERROR: {}".format(e))

        # 所有分布式op结束后，rank0 把保存的tensor渲染成视频（此时不再有HCCL通信）
        if dist.get_rank() % cp_size == 0:
            from attn_map_utils import extract_and_visualize_attn_map
            import glob as _glob
            attn_root = os.path.join(save_dir, 'attn_maps')
            tensor_files = sorted(_glob.glob(os.path.join(attn_root, '*', 'tensors.pt')))
            for tensor_file in tensor_files:
                sample_dir = os.path.dirname(tensor_file)   # .../attn_maps/sample00
                sample_tag = os.path.basename(sample_dir)   # sample00
                rank0_print(f"Rendering attn map videos for {sample_tag} ...")
                saved_maps = torch.load(tensor_file, map_location='cpu')
                for t_idx, attn_map in enumerate(saved_maps):
                    extract_and_visualize_attn_map(
                        avg_attn_map=attn_map,
                        ref_lat_size=(80, 45),    # 竖屏；横屏改为 (45, 80)
                        ref_img_size=(1280, 720),  # 竖屏；横屏改为 (720, 1280)
                        save_dir=sample_dir,
                        save_name=f"t{t_idx:03d}.mp4",
                    )
                os.remove(tensor_file)

        if dist.get_rank() == 0 and cfg.efficiency is not None:
            with open(cfg.efficiency, 'w') as f:
                for num in time_list:
                    f.write(f"{num}\n")

if __name__ == "__main__":
    main()

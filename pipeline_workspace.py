import os
import argparse
import sys
import math

sys.path.append(".")
file_dir = os.path.abspath(__file__)
work_dir = file_dir.rsplit("/", 1)[0] + "/../../"
print(f'work_dir: {work_dir}')
sys.path.append(work_dir)
import torch
import torch.distributed as dist
import datetime
import collections
import random
import numpy as np
try:
    import moxing as mox
except:
    print("no moxing")

import argparse
import datetime
import collections
import random
import numpy as np
from torchvision import transforms as T
from PIL import Image
import time
import json
from einops import rearrange
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose
import torch.utils.data as data
import torch.distributed as dist
from tqdm import tqdm
import glob
import cv2
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from mimogpt.engine.utils import set_seed
import numbers

from mimogpt.models.modules.ldm.encoder.distributed_vae import DistributedVAE
from mimogpt.models.modules.ldm.encoder.motion_vae import MotionVAE
from mimogpt.utils.txt_utils import read_from_yaml, merge_args

try:
    from feat_codec import encode_feat_tensor, decode_feat_tensor
except:
    encode_feat_tensor = None
import subprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='outputs/')
    parser.add_argument('--backend', type=str, default="hccl")
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--bucket_path', type=str, default='')
    parser.add_argument('--fps', type=int, default=8)
    # need to config
    parser.add_argument('--config_path', type=str, default='./configs/vae/vae-1.2/encode.yml')
    parser.add_argument('--model_name', type=str, default='vae-1.2')
    parser.add_argument('--using_tiling', type=str, default="0")
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--width', type=int, default=480)
    parser.add_argument('--max_frame', type=int, default=121)
    parser.add_argument('--task', type=str, default="")
    parser.add_argument('--image_size', type=str, default="480p")
    parser.add_argument('--dataset_name', type=str, default="one_video")
    
    parser.add_argument('--json_path', type=str, default='/data/DIT_RES_l00503064/text/train.jsonl')
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--ckpt_path', type=str, default='/home/ma-user/work/ckpt/meitiyuan_sgm_from4wstep_30wdata_w_gan_relax_noresize01_edge05_epoch=000001-v7.ckpt')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--global_rank', type=int, default=0)
    parser.add_argument('--root_path', type=str, default='/data/DIT_RES_l00503064/data/')
    parser.add_argument('--pre_type', type=str, default="tensor")
    parser.add_argument('--is_i2v', type=int, default=1)
    parser.add_argument('--is_i2v_mid', type=int, default=1)
    parser.add_argument('--is_i2v_end', type=int, default=1)
    parser.add_argument('--is_t2v', type=int, default=1)
    parser.add_argument('--is_v2v', type=int, default=1)
    parser.add_argument('--video_spec', type=str, default="")
    # video limit
    parser.add_argument('--max_time', type=int, default=16)
    parser.add_argument('--max_frames_limit', type=int, default=121)
    
    args, unknown = parser.parse_known_args()
    
    random.seed(args.global_seed)
    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 你原有的，保持不变
TARGET_SIZE_480p = {
    1.0000: [512, 512],
    1.5: [720, 480],
    0.6667: [480, 720],
}

TARGET_SIZE_720p = {
    # 1.0000: [960, 960],
    # 0.5625: [720, 1280],
    1.7778: [1072, 720],
}

# 新增的（全部尺寸为 8 的倍数，比例与 480p 一致）
TARGET_SIZE_640p = {
    1.0000: [640, 640],
    1.5: [960, 640],
    0.6667: [640, 960],
}

TARGET_SIZE_560p = {
    1.0000: [576, 576],
    1.5: [840, 560],
    0.6667: [560, 840],
}

TARGET_SIZE_400p = {
    # 1.0000: [448, 448],
    # 1.7778: [712, 400],
    0.5625: [592, 400],
}
TARGET_SIZE_320p = {
    1.0000: [320, 320],
    1.5: [480, 320],
    0.6667: [320, 480],
}

TARGET_SIZE_240p = {
    1.0000: [256, 256],
    1.5: [360, 240],
    0.6667: [240, 360],
}


# TARGET_SIZE_720p = {
#     1.7778: [1280,720],
#     1.0000: [960,960],
#     0.5625: [720,1280],
#     0.6667: [768,1152],
#     1.5000: [1152,768],
#     0.7500: [832,1088],
#     1.3333: [1088,832],
# }

TARGET_SIZE_1080p = {
    1.0000: [1440, 1440],
    0.5625: [1080, 1920],
    1.7778: [1920, 1080],
}

TARGET_SIZE={}

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

def get_resize_crop_size(input_size):
    input_ratio = input_size[0] / input_size[1]
    closest_ratio = min(TARGET_SIZE.keys(), key=lambda ratio: abs(float(ratio) - input_ratio))
    crop_size = TARGET_SIZE[closest_ratio]

    ratio = max(crop_size[0] / input_size[0], crop_size[1] / input_size[1])
    # ratio = math.ceil(ratio * 10000) / 10000    # avoid resize_size < crop_size
    # print(ratio)
    resize_size = int(np.ceil(input_size[0] * ratio)),int(np.ceil(input_size[1] * ratio))
    # print(resize_size)
    # resize_size = (int(input_size[0] * ratio), int(input_size[1] * ratio)) 
    return resize_size, crop_size

def preprocess(video, pre_type="tensor"):
    assert pre_type in ["numpy", "tensor"], 'must use tensor of numpy'
    if pre_type == "tensor":
        # video: THWC, {0, ..., 255}
        #video = rearrange(torch.tensor(video), "t h w c -> t c h w")
        
        resize_size, crop_size = get_resize_crop_size(video.shape[-3:-1])

        transform = T.Compose([
            T.Resize((resize_size[0], resize_size[1])),
            T.CenterCrop(crop_size),
            NormalizeToTensor()
        ])
        video = torch.stack([transform(Image.fromarray(frame)) for frame in video], dim=1)
        return video
    
    elif pre_type == "numpy":
        np_inter = cv2.INTER_NEAREST

        scaled = np.array([
            cv2.resize(img, (resize_size[1], resize_size[0]), interpolation=np_inter) for img in video
        ])

        t, h, w, c = scaled.shape
        w_start = (w - crop_size[1]) // 2
        h_start = (h - crop_size[0]) // 2
        video = scaled[:, h_start:h_start + crop_size[0], w_start:w_start + crop_size[1], :]
        video = video.permute(0,3,1,2)  # t h w c -> t c h w
    # import pdb; pdb.set_trace()
    video /= 255.0
    video -= 0.5
    video *= 2
    return video


class RawVideoExtractorCV2:
    def __init__(self, max_frames=121, fps=24, ffmpeg_dir="", frame_extractor_online="ffmpeg-subprocess"):
        self.max_frames = max_frames  # 12
        self.fps = fps
        self.end_time = max_frames//fps+1
        self.frame_extractor_online = frame_extractor_online
        self.frame_extract_record = set()

        self.ffmpeg_dir = ffmpeg_dir
        self.ffmpeg = os.path.join(self.ffmpeg_dir, "ffmpeg")
        self.ffprobe = os.path.join(self.ffmpeg_dir, "ffprobe")

        # self.ffmpeg = "ffmpeg"
        # self.ffprobe = "ffprobe"

    def get_vid_info(self, video_file, attribute, ffprobe):
        video_file = video_file.replace('(','\(').replace(')','\)').replace(' ','\ ')
        cmd_str = f"{ffprobe} -v error -select_streams v:0 -show_entries stream={attribute} \
                    -of default=noprint_wrappers=1:nokey=1 {video_file} -loglevel quiet"
        out_bytes = subprocess.check_output(cmd_str, stderr=subprocess.STDOUT, shell=True)
        out_text  = out_bytes.decode('utf-8')
        
        if attribute == "duration":
            return float(out_text)
        return int(out_text)

    def video_to_tensor_ffp_sup(self, video_file):
        vid_name = os.path.splitext(os.path.basename(video_file))[0]
        vid_time = self.get_vid_info(video_file, "duration", self.ffprobe)
        vid_width = self.get_vid_info(video_file, "width", self.ffprobe)
        vid_height = self.get_vid_info(video_file, "height", self.ffprobe)
        
        start=0
        end = min(vid_time, self.end_time)

        resize_size, crop_size = get_resize_crop_size([vid_height, vid_width])
        # print(resize_size, crop_size)
        # -vf crop=w={crop_size[1]}:h={crop_size[0]}
        video_file = video_file.replace('(','\(').replace(')','\)').replace(' ','\ ')
        cmd_str = f"{self.ffmpeg} -ss {start} -to {end} -i {video_file} -vf fps={self.fps},scale={resize_size[1]}:{resize_size[0]}:flags=bilinear,crop=w={crop_size[1]}:h={crop_size[0]} -vsync 0 -vcodec \
                    rawvideo -pix_fmt rgb24 -f image2pipe -"
        print(cmd_str)
        pipe = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = pipe.communicate()

        bytes_per_img = crop_size[1] * crop_size[0] * 3
        if len(stdout) % bytes_per_img != 0:
            print("Error, get wrong video frames. vid name: {vid_name}, width: {crop_size[1]}, height: {crop_size[0]}")
        img_num = len(stdout) // bytes_per_img
        # print(img_num)
        img_list = []
        idx = 0
        print(img_num, self.max_frames)
        img_num = img_num-(img_num-1)%8
        while idx < min(img_num, self.max_frames):
            start_byte = (idx % img_num) * bytes_per_img
            end_byte = ((idx % img_num) + 1) * bytes_per_img
            img = Image.frombytes('RGB', (crop_size[1], crop_size[0]), stdout[start_byte:end_byte])
            img = self.preprocess(img)
            img_list.append(img)
            idx += 1

        video_data = torch.tensor(np.stack(img_list))
        return video_data

    def preprocess(self, img):  # [t, h, w, c]
        transform_training = T.Compose([
            NormalizeToTensor()
        ])
        return transform_training(img)

    def video_to_tensor_ffp_py(self, video_file, start, end, max_frames=121, end_time=6):
        import ffmpeg as ffp
        probe = ffp.probe(video_file)
        vid_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

        vid_width = int(vid_stream['width'])
        vid_height = int(vid_stream['height'])
        vid_time = float(vid_stream['duration'])
        ori_fps = eval(vid_stream["avg_frame_rate"])

        if start is None:
            start = 0
        if end is None:
            end = vid_time
        end = min(end, end_time)
        fps = f"{max_frames}/{end - start}"

        resize_size, crop_size = get_resize_crop_size([vid_height, vid_width])

        tic = time.time()
        out, _ = (
            ffp
            .input(video_file, ss=start, to=end, loglevel='quiet')
            .filter('fps', 24)
            .filter('scale', width='{}'.format(resize_size[1]), height='{}'.format(resize_size[0]))
            .filter('crop', w=crop_size[1], h=crop_size[0])
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True)
        )
        video = (
            np
            .frombuffer(out, np.uint8)
            .reshape([-1, crop_size[1], crop_size[0], 3])
        )
        toc = time.time()
        print("resize:", toc-tic)

        img_list = []
        tic = time.time()
        video_data = torch.stack([self.preprocess(Image.fromarray(frame)) for frame in video[:121]], dim=1)

        toc = time.time()
        print("process:", toc-tic)
        return video_data

    def get_video_data(self, video_path):
        if self.frame_extractor_online == "ffmpeg-subprocess":
            vid_tensor = self.video_to_tensor_ffp_sup(video_path)
        elif self.frame_extractor_online == "ffmpeg-python":
            vid_tensor = self.video_to_tensor_ffp_py(video_path, None, None,
                                                          max_frames=self.max_frames)
        return vid_tensor

class VideoDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-1, 1] """
    exts = ['mp4', 'MP4', 'avi']

    def __init__(self, json_path='/data/DIT_RES_l00503064/text/train.jsonl', fps=16, max_time=32, root_path=f'/data/DIT_RES_l00503064/data', save_path=f'', pre_type="tensor", world_size=1, global_rank=1):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.root_path = root_path
        with open(json_path, 'r', encoding="utf-8") as f:
            self.raw_paths = f.readlines()

        print(f'start dataloader: load json {json_path}')
        self.paths = []
        for p in tqdm(self.raw_paths):
            try:
                sub_dir, video_full_name = json.loads(p)["video_fn"].rsplit("/", 1)
                # video_full_name = json.loads(p)["video_fn"]
            except:
                print('Error!!! Path depth must larger than 1!!!', p)
                break
            sub_dir = os.path.join(save_path, sub_dir)
            # sub_dir = save_path
            save_pt_name = video_full_name.rsplit(".", 1)[0] + ".pt"
            save_pt_path = os.path.join(sub_dir, save_pt_name)            
            self.paths.append(p)
            # if not os.path.exists(save_pt_path):
            #     self.paths.append(p)
            
        total_len = len(self.paths)
        len_per_npu = total_len / world_size
        self.paths = self.paths[round(global_rank*len_per_npu) : round((global_rank + 1)*len_per_npu)]

        print(f'end dataloader json_len {len(self.raw_paths)} new_len {len(self.paths)}')
        self.fps = fps
        self.max_time = max_time
        self.save_path = save_path
        self.pre_type = pre_type
        self.save_json_path = save_path + ".jsonl"
        self.output_dict = {}
        self.video_reader = RawVideoExtractorCV2(max_frames=args.max_frames_limit, fps=self.fps, ffmpeg_dir=f"{work_dir}/package/ffmpeg-7.0.2-arm64-static")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        start = time.perf_counter()
        video_dict = json.loads(self.paths[idx])
        sub_path = video_dict['video_fn']
        sub_dir, video_full_name = sub_path.rsplit("/", 1)
        sub_dir = os.path.join(self.save_path, sub_dir)
        # os.makedirs(sub_dir, exist_ok=True)

        save_pt_name = video_full_name.rsplit(".", 1)[0] + ".pt"
        save_path = os.path.join(sub_dir, save_pt_name)

        file_path = os.path.join(self.root_path, sub_path)
        # if os.path.exists(save_path) or os.path.exists(save_path.replace('.pt','.bin')) or os.path.exists(save_path.replace('.pt','.json')) or not mox.file.exists(file_path):
        if not mox.file.exists(file_path):
            print('Save Path', save_path, os.path.exists(save_path))
            print('File Path', file_path, mox.file.exists(file_path))
            return "None"

        if dist.get_rank()==0: #not os.path.exists('/cache/'+file_path.split('/')[-1]):
            mox.file.copy(file_path, '/cache/'+file_path.split('/')[-1])
            with open('/cache/'+file_path.split('/')[-1][:-4]+'.txt', 'w') as fw:
                fw.write('1')
        else:
            while True:
                if os.path.exists('/cache/'+file_path.split('/')[-1][:-4]+'.txt'):
                    break
                time.sleep(0.1)
        try:
            # target_video = self.video_reader.get_video_data(file_path).permute(1, 0, 2, 3)
            target_video = self.video_reader.get_video_data('/cache/' + file_path.split('/')[-1]).permute(1, 0, 2, 3)
            # print(f"target_video: {target_video}")
            print(f"target_video_shape: {target_video.shape[1]}")
            assert target_video.shape[1]>=73
        except Exception as e:
            print(e)
            return "None"
        # print(f"=====get_video_data: {time.perf_counter() - start:.3f} s")
        return target_video, save_path, json.dumps(video_dict, ensure_ascii=False) #, resolution=512



def prepare_data(args):
        # Setup data:    
    dataset = VideoDataset(args.json_path, args.fps, args.max_time, args.root_path, args.save_path, args.pre_type, args.gpus, args.global_rank)
    print('=====>len dataset: ', len(dataset))
        

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        #sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    return dataloader

if __name__ == "__main__":
    args = parse_args()

    print(args)
    if "RANK" in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])

    print("rank:{}".format(args.rank))
    print("world_size:{}".format(args.world_size))

    torch.distributed.init_process_group(
        backend=args.backend,
        init_method=None,
        rank=args.rank,
        world_size=args.world_size,
        timeout=datetime.timedelta(hours=2.)
    )
    torch.cuda.set_device(args.rank % 8)

    seed = os.environ.get("seed", None)
    # if seed is None:
    #     seed = 123
    # else:
    #     seed = int(seed)
    # set_seed(seed)
    
    #global TARGET_SIZE
    TARGET_SIZE_dict = {
        '480p': TARGET_SIZE_480p, '720p': TARGET_SIZE_720p, '400p': TARGET_SIZE_400p,
        '640p': TARGET_SIZE_640p, '560p': TARGET_SIZE_560p, '320p': TARGET_SIZE_320p,
        '240p': TARGET_SIZE_240p,
        }

    TARGET_SIZE = TARGET_SIZE_dict[args.image_size]

    dataloader = prepare_data(args)

    distributed_vae = DistributedVAE(args, dataloader=dataloader)

    #===add by wz

    config = read_from_yaml(args.config_path)
    ddconfig = config["model_config"]["params"]["ddconfig"]
    vae_config = dict(
        ddconfig=ddconfig,
        embed_dim=16,
        inflation=False,
        spynet_pretrained="./spynet_20210409-c6c1bd09.pth",
        ckpt_path=args.ckpt_path)
    image_vae = MotionVAE(**vae_config).cuda().to(torch.bfloat16)
    #===add by wz over

    
    os.makedirs(args.save_path, exist_ok=True)
    video_latent = os.path.join(args.save_path, f"video_latent_{args.video_spec}")
    print(f"===video_latent:{video_latent}")
    os.makedirs(os.path.join(args.save_path, f"video_latent_{args.video_spec}"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, f"first_frame_latent_{args.video_spec}"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, f"json_{args.video_spec}"), exist_ok=True)

    pbar = tqdm(total=len(dataloader), desc=f'NPU {args.global_rank}')
    for b in dataloader:
        cur = time.time()
        sync = torch.zeros([1]).cuda()
        if b[0] == "None":
            pbar.update(1)
            sync += 1
            dist.all_reduce(sync)
            continue
        else:
            dist.all_reduce(sync)
            if torch.sum(sync) > 0:
                continue
        save_pt_path = b[1]
        print('rank',dist.get_rank(), args.global_rank, save_pt_path)
        output_dict = json.loads(b[2][0])
        # print(output_dict)
        x = b[0].cuda().to(torch.bfloat16)
        #print(x.shape, x.device, x.dtype)
        
        with torch.no_grad():
            T = 16
            C = 32
            H = 90
            W = 160
            if args.is_i2v:
                moments_frame = image_vae.encode(x[:,:,:1])
                z_frame = moments_frame.detach().cpu()
                z_frame = rearrange(z_frame, 'b c t h w -> b t c h w')
                H = z_frame.shape[3]
                W = z_frame.shape[4]
            if args.is_v2v:
                start = time.perf_counter()
                moments_continue = distributed_vae.encode(x[:,:,:25])[0]
                z_continue = moments_continue.detach().cpu()
                z_continue = rearrange(z_continue, 'b c t h w -> b t c h w')
                print(f"=====v2v: {time.perf_counter() - start:.3f} s")
                H = z_continue.shape[3]
                W = z_continue.shape[4]
            if args.is_i2v_mid:
                start = time.perf_counter()
                mid_index = x.shape[2]//2
                moments_frame1 = image_vae.encode(x[:,:,mid_index:mid_index+1])
                z_frame1 = moments_frame1.detach().cpu()
                z_frame1 = rearrange(z_frame1, 'b c t h w -> b t c h w')
                print(f"=====i2v_mid: {time.perf_counter() - start:.3f} s")
                H = z_frame1.shape[3]
                W = z_frame1.shape[4]
            if args.is_i2v_end:
                start = time.perf_counter()
                moments_frame2 = image_vae.encode(x[:,:,-1:])
                z_frame2 = moments_frame2.detach().cpu()
                z_frame2 = rearrange(z_frame2, 'b c t h w -> b t c h w')
                print(f"=====i2v_end: {time.perf_counter() - start:.3f} s")
                H = z_frame2.shape[3]
                W = z_frame2.shape[4]

            if args.is_t2v:
                moments = distributed_vae.encode(x)[0]
                z = moments.detach().cpu()
                z = rearrange(z, 'b c t h w -> b t c h w')
                T = z.shape[1]
                C = z.shape[2]
                H = z.shape[3]
                W = z.shape[4]

            output_dict["4_vae_feature_shape"] = (T, C, H, W)
            output_dict["4_vae_feature_length"] = T

        save_json_path = save_pt_path[0].rsplit(".", 1)[0] + ".json"
        

        if dist.get_rank() == 0:
            if args.is_t2v:
                save_video_pt_path = save_pt_path[0].replace(args.save_path.rstrip("/"), os.path.join(args.save_path, f"{args.video_spec}_video_latent"))
                os.makedirs(os.path.dirname(save_video_pt_path), exist_ok=True)
                if encode_feat_tensor is not None:
                    print(f'x.shape: {x.shape}, z.shape: {z.shape}, save_bin_path: {save_video_pt_path}')
                    save_video_pt_path = save_video_pt_path.replace('.pt','.bin')
                    encode_feat_tensor(z[0].to(torch.bfloat16), save_video_pt_path)
                else:
                    print(f'x.shape: {x.shape}, z.shape: {z.shape}, save_pt_path: {save_video_pt_path}')
                    torch.save(z[0].to(torch.bfloat16), save_video_pt_path)
                output_dict["vae_fn"] = save_video_pt_path.replace(args.save_path.rstrip("/") + "/", "")
            if args.is_i2v:
                save_frame_pt_path = save_pt_path[0].replace(args.save_path.rstrip("/"), os.path.join(args.save_path, f"{args.video_spec}_first_frame"))
                os.makedirs(os.path.dirname(save_frame_pt_path), exist_ok=True)
                if encode_feat_tensor is not None:
                    print(f'z_frame.shape: {z_frame.shape}, save_bin_path: {save_frame_pt_path}')
                    save_frame_pt_path = save_frame_pt_path.replace('.pt','.bin')
                    encode_feat_tensor(z_frame[0].to(torch.bfloat16), save_frame_pt_path)
                else:
                    print(f'z_frame.shape: {z_frame.shape}, save_pt_path: {save_frame_pt_path}')
                    torch.save(z_frame[0].to(torch.bfloat16), save_frame_pt_path)
                output_dict["image_fn"] = save_frame_pt_path.replace(args.save_path.rstrip("/") + "/", "")

            if args.is_i2v_mid:
                save_frame1_pt_path = save_pt_path[0].replace(args.save_path.rstrip("/"), os.path.join(args.save_path, f"{args.video_spec}_mid_frame"))
                os.makedirs(os.path.dirname(save_frame1_pt_path), exist_ok=True)
                if encode_feat_tensor is not None:
                    print(f'z_frame1.shape: {z_frame1.shape}, save_bin_path: {save_frame1_pt_path}')
                    save_frame1_pt_path = save_frame1_pt_path.replace('.pt','.bin')
                    encode_feat_tensor(z_frame1[0].to(torch.bfloat16), save_frame1_pt_path)
                else:
                    print(f'z_frame1.shape: {z_frame1.shape}, save_pt_path: {save_frame1_pt_path}')
                    torch.save(z_frame1[0].to(torch.bfloat16), save_frame1_pt_path)
                output_dict["image1_fn"] = save_frame1_pt_path.replace(args.save_path.rstrip("/") + "/", "")

            if args.is_i2v_end:
                save_frame2_pt_path = save_pt_path[0].replace(args.save_path.rstrip("/"), os.path.join(args.save_path, f"{args.video_spec}_end_frame"))
                os.makedirs(os.path.dirname(save_frame2_pt_path), exist_ok=True)
                if encode_feat_tensor is not None:
                    print(f'z_frame2.shape: {z_frame2.shape}, save_bin_path: {save_frame2_pt_path}')
                    save_frame2_pt_path = save_frame2_pt_path.replace('.pt','.bin')
                    encode_feat_tensor(z_frame2[0].to(torch.bfloat16), save_frame2_pt_path)
                else:
                    print(f'z_frame2.shape: {z_frame2.shape}, save_pt_path: {save_frame2_pt_path}')
                    torch.save(z_frame2[0].to(torch.bfloat16), save_frame2_pt_path)
                output_dict["image2_fn"] = save_frame2_pt_path.replace(args.save_path.rstrip("/") + "/", "")
            if args.is_v2v:
                save_continue_pt_path = save_pt_path[0].replace(args.save_path.rstrip("/"), os.path.join(args.save_path, f"{args.video_spec}_videocontinue"))
                os.makedirs(os.path.dirname(save_continue_pt_path), exist_ok=True)
                if encode_feat_tensor is not None:
                    print(f'z_continue.shape: {z_continue.shape}, save_bin_path: {save_continue_pt_path}')
                    save_continue_pt_path = save_continue_pt_path.replace('.pt','.bin')
                    encode_feat_tensor(z_continue[0].to(torch.bfloat16), save_continue_pt_path)
                else:
                    print(f'z_continue.shape: {z_continue.shape}, save_pt_path: {save_continue_pt_path}')
                    torch.save(z_continue[0].to(torch.bfloat16), save_continue_pt_path)
                output_dict["subvideo_fn"] = save_continue_pt_path.replace(args.save_path.rstrip("/") + "/", "")

            save_json_path = save_json_path.replace(args.save_path.rstrip("/"), os.path.join(args.save_path, f"{args.video_spec}_jsonl"))
            os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
            with open(save_json_path, "w") as f:
                f.write(json.dumps(output_dict, ensure_ascii=False))

            if 's3://' in args.bucket_path:
                if args.is_i2v:
                    mox.file.copy(save_frame_pt_path, args.bucket_path+save_frame_pt_path)
                    os.remove(save_frame_pt_path)
                if args.is_i2v_mid:
                    mox.file.copy(save_frame1_pt_path, args.bucket_path+save_frame1_pt_path)
                    os.remove(save_frame1_pt_path)
                if args.is_i2v_end:
                    mox.file.copy(save_frame2_pt_path, args.bucket_path+save_frame2_pt_path)
                    os.remove(save_frame2_pt_path)
                if args.is_v2v:
                    mox.file.copy(save_continue_pt_path, args.bucket_path+save_continue_pt_path)
                    os.remove(save_continue_pt_path)
                if args.is_t2v:
                    mox.file.copy(save_video_pt_path, args.bucket_path+save_video_pt_path)
                    os.remove(save_video_pt_path)
                mox.file.copy(save_json_path, args.bucket_path+save_json_path)
                os.remove(save_json_path)
        pbar.update(1)

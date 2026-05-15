"""extract_ref_attn_map.py
==========================
Runs a single denoising step of the I2V-ID model and extracts the
cross-attention maps between video tokens and reference-image tokens.

Usage (8-NPU example matching local_test.sh)::

    torchrun --master_port 26599 --nproc_per_node 8 \
        scripts/extract_ref_attn_map.py \
        --config  configs/mimo/id_inference/inference.py \
        --ckpt-path  /cache/I2V_ID_model.pth \
        --test_jsonl  test.jsonl \
        --img_dir  /path/to/images \
        --face_aug_dir  /path/to/face_aug \
        --save-dir  /path/to/outputs \
        --attn-save-dir  /path/to/attn_maps \
        --attn-layers  0 4 8 12 16 20 23   # block indices to collect (default: all 24)
        --attn-step   25                   # which denoising step to capture (default: 25)

Output
------
For each sample a directory ``<attn-save-dir>/<sample_id>/`` is created with:
  * ``attn_layer<L>_step<S>.npy``   – raw attention map, float32
      shape: [n_head, T_video, T_ref]  (batch-dim=1 is squeezed out)
  * ``attn_layer<L>_step<S>_mean_head.png``   – head-averaged heatmap on ref image
  * ``ref_image.png``   – the reference image used for conditioning
"""
import os
import sys
import argparse
import datetime
import json
import time

os.environ["MOX_SILENT_MODE"] = "1"
os.environ["MOX_FILE_LARGE_FILE_METHOD"] = "1"
os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:True"

the_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(the_dir)

import torch
import numpy as np

try:
    import torch_npu
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch_npu.npu.config.allow_internal_format = False
    torch.npu.conv.allow_hf32 = False
    torch.npu.matmul.allow_hf32 = False
    from torch_npu.contrib import transfer_to_npu
    DEVICE = "npu"
except ImportError:
    DEVICE = "cuda"

import torch.distributed as dist
from PIL import Image
from torchvision import transforms as T
from torchvision.io import write_video

from mimogpt.models import build_backbone, AttnMapCollector
from mimogpt.models.dit.parallel_states import initialize_distributed
from mimogpt.utils.load_optimizations import skip_torch_weight_init, convert_to_dir_dist
from mimogpt.utils.log_utils import rank0_print
from mimogpt.functional import to_torch_dtype

# Import shared helpers from the main inference script
from inference_mmdit import parse_configs, warp_fsdp, load_prompts
from inference_mmdit_i2v_ip_last_no_cfg_jsonl_prompt import (
    parse_args, preprocess, NormalizeToTensor,
)

from mmengine.runner import set_random_seed
from mmengine.config import Config
from transformers.models.t5.modeling_t5 import T5Block

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType, ShardedStateDictConfig
import torch.distributed.checkpoint as dist_cp
from pathlib import Path


# ---------------------------------------------------------------------------
# Helper: visualise attn map
# ---------------------------------------------------------------------------
def save_attn_heatmap(attn_map_hw, ref_img_pil, save_path):
    """Overlay a 2-D attention map on the reference image and save.

    Args:
        attn_map_hw: np.ndarray [H_ref_tokens, W_ref_tokens]  (already 2-D)
        ref_img_pil: PIL.Image – original reference image (will be resized)
        save_path:   str – output .png path
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import cv2

        ref_w, ref_h = ref_img_pil.size
        heatmap = attn_map_hw.astype(np.float32)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap_resized = cv2.resize(heatmap, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(ref_img_pil)
        axes[0].set_title("Reference Image")
        axes[0].axis("off")

        axes[1].imshow(heatmap_resized, cmap="hot")
        axes[1].set_title("Attention Map (resized)")
        axes[1].axis("off")

        axes[2].imshow(ref_img_pil)
        axes[2].imshow(heatmap_resized, cmap="jet", alpha=0.5)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Could not save heatmap (matplotlib/cv2 missing?): {e}")


# ---------------------------------------------------------------------------
# Scheduler wrapper that calls the model with AttnMapCollector active
# ---------------------------------------------------------------------------
class AttnCapturingScheduler:
    """Wraps a rectified-flow scheduler to intercept one specific step."""

    def __init__(self, scheduler, capture_step: int, attn_layers: list, output_dir: str,
                 sample_id: str, fn: int, lh: int, lw: int, fnr: int):
        self.scheduler = scheduler
        self.capture_step = capture_step
        self.attn_layers = attn_layers
        self.output_dir = output_dir
        self.sample_id = sample_id
        self.fn = fn    # video latent frames
        self.lh = lh   # latent height / patch_size
        self.lw = lw   # latent width  / patch_size
        self.fnr = fnr  # ref latent frames

    # Forward all attributes to the wrapped scheduler
    def __getattr__(self, name):
        return getattr(self.scheduler, name)

    def _post_capture(self, attn_store: dict, step_idx: int, ref_img_pil: Image.Image):
        """Save raw numpy maps and optional heatmaps after capture."""
        save_root = os.path.join(self.output_dir, self.sample_id)
        os.makedirs(save_root, exist_ok=True)

        # Save ref image for reference
        ref_save = os.path.join(save_root, "ref_image.png")
        if not os.path.exists(ref_save):
            ref_img_pil.save(ref_save)

        patch = 2  # MMDiT spatial patch_size
        H_tok = self.lh // patch   # tokens per row in ref image latent
        W_tok = self.lw // patch   # tokens per col in ref image latent
        T_video = self.fn * (self.lh // patch) * (self.lw // patch)

        for layer_key, attn_map in attn_store.items():
            layer_idx = int(layer_key.split("_")[1])
            if self.attn_layers and layer_idx not in self.attn_layers:
                continue

            # attn_map: [B, n_head, T_video, T_ref]
            attn_np = attn_map.squeeze(0).float().numpy()  # [n_head, T_video, T_ref]

            # Save raw
            npy_path = os.path.join(save_root, f"attn_{layer_key}_step{step_idx}.npy")
            np.save(npy_path, attn_np)
            rank0_print(f"Saved attention map: {npy_path}  shape={attn_np.shape}")

            # Head-averaged heatmap: average over heads and video frames → [T_ref]
            # Reshape to get spatial interpretation of ref tokens
            mean_over_heads = attn_np.mean(axis=0)  # [T_video, T_ref]
            # Average over all video tokens to get global ref attention
            ref_attn_flat = mean_over_heads.mean(axis=0)  # [T_ref]
            T_ref = ref_attn_flat.shape[0]
            # T_ref = fnr * H_tok * W_tok for a single-frame ref image
            tokens_per_frame = H_tok * W_tok
            n_ref_frames = T_ref // tokens_per_frame if tokens_per_frame > 0 else 1
            if T_ref % tokens_per_frame == 0 and n_ref_frames >= 1:
                ref_attn_2d = ref_attn_flat[:tokens_per_frame].reshape(H_tok, W_tok)
            else:
                side = int(T_ref ** 0.5)
                ref_attn_2d = ref_attn_flat[:side * side].reshape(side, side)

            png_path = os.path.join(save_root, f"attn_{layer_key}_step{step_idx}_mean_head.png")
            save_attn_heatmap(ref_attn_2d, ref_img_pil, png_path)

            # Per-video-frame attention map (first frame → ref image)
            # mean_over_heads[0 : H_tok*W_tok, :] = first video frame
            first_frame_attn = mean_over_heads[:tokens_per_frame, :]  # [T_vid_frame, T_ref]
            # average over first video-frame tokens → [T_ref]
            first_frame_ref = first_frame_attn.mean(axis=0)
            if T_ref % tokens_per_frame == 0:
                first_frame_ref_2d = first_frame_ref[:tokens_per_frame].reshape(H_tok, W_tok)
            else:
                side = int(T_ref ** 0.5)
                first_frame_ref_2d = first_frame_ref[:side * side].reshape(side, side)

            png_path2 = os.path.join(save_root, f"attn_{layer_key}_step{step_idx}_frame0.png")
            save_attn_heatmap(first_frame_ref_2d, ref_img_pil, png_path2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ---- Parse args -------------------------------------------------------
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--attn-save-dir", type=str, default="./attn_maps",
                             help="Root directory to save attention maps")
    base_parser.add_argument("--attn-layers", nargs="*", type=int, default=None,
                             help="Which block indices to collect (default: all). E.g. --attn-layers 0 6 12 18 23")
    base_parser.add_argument("--attn-step", type=int, default=25,
                             help="Which denoising step (0-indexed) to capture attention maps at")
    extra_args, remaining = base_parser.parse_known_args()
    # Restore argv for downstream parse_args
    sys.argv = [sys.argv[0]] + remaining

    cfg = parse_configs(parse_args)

    attn_save_dir = extra_args.attn_save_dir
    attn_layers = extra_args.attn_layers  # None means all layers
    attn_step = extra_args.attn_step

    if "inference_algo" not in cfg.keys():
        cfg["inference_algo"] = {}

    # ---- Init distributed -------------------------------------------------
    if "RANK" in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.gpus = int(os.environ["WORLD_SIZE"])

    torch.distributed.init_process_group(
        backend=cfg.backend,
        init_method=None,
        rank=cfg.rank,
        world_size=cfg.gpus,
        timeout=datetime.timedelta(hours=2.0),
    )
    torch.cuda.set_device(cfg.rank % 8)

    cp_size = dist.get_world_size()
    initialize_distributed(cp_size)
    cfg.distributed.context_parallelism_size = cp_size

    # ---- Build models -----------------------------------------------------
    torch.set_grad_enabled(False)
    dtype = to_torch_dtype(cfg.dtype)

    input_size = (cfg.num_frames, *cfg.image_size)
    with skip_torch_weight_init():
        vae = build_backbone(cfg, cfg.vae.get("backbone", "motionvae_16ch_dist"))
    vae = vae.to(DEVICE, dtype).eval()
    with skip_torch_weight_init():
        vae_en = build_backbone(cfg, cfg.vae_en.get("backbone", "motionvae_16ch"))
    vae_en = vae_en.to(DEVICE, dtype).eval()
    latent_size = vae.get_latent_size(input_size)
    rank0_print(f"latent size: {latent_size}")

    frame_scale = torch.tensor(cfg.vae.frame_scale, dtype=torch.float32)[:latent_size[0]].to(DEVICE)
    frame_bias  = torch.tensor(cfg.vae.frame_bias,  dtype=torch.float32)[:latent_size[0]].to(DEVICE)

    with skip_torch_weight_init(), torch.device("cuda"):
        text_encoder = build_backbone(cfg, "t5_online")
    text_encoder.t5.model = warp_fsdp(text_encoder.t5.model, T5Block, dtype=text_encoder.dtype)

    with skip_torch_weight_init(), torch.device("meta"):
        model = build_backbone(cfg)
    model = model.to(dtype).eval()

    mmdit_pretrained = cfg.model.get("from_pretrained", None)
    if mmdit_pretrained and os.path.isfile(mmdit_pretrained):
        dist_ckpt_dir = convert_to_dir_dist(mmdit_pretrained)
        if not os.path.exists(dist_ckpt_dir):
            model.to_empty(device="cpu")
            model_path = Path(mmdit_pretrained)
            if model_path.suffix.lower() == ".safetensors":
                from safetensors.torch import load_file
                state_dict = load_file(model_path)
            else:
                state_dict = torch.load(model_path, map_location="cpu")
            if "module" in state_dict:
                state_dict = state_dict["module"]
            model.load_state_dict(state_dict, strict=False)

    model = warp_fsdp(model, model.blocks[0].__class__, dtype=dtype)

    if mmdit_pretrained and os.path.isfile(mmdit_pretrained):
        dist_ckpt_dir = convert_to_dir_dist(mmdit_pretrained)
        if os.path.exists(dist_ckpt_dir):
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                                      state_dict_config=ShardedStateDictConfig(offload_to_cpu=True)):
                state_dict = {"model": model.state_dict()}
                dist_cp.load_state_dict(state_dict=state_dict,
                                        storage_reader=dist_cp.FileSystemReader(dist_ckpt_dir))
                model.load_state_dict(state_dict["model"])

    text_encoder.y_embedder = model.y_embedder
    scheduler = build_backbone(cfg, cfg["scheduler"]["backbone"])

    set_random_seed(seed=cfg.seed)

    # ---- Load test data ---------------------------------------------------
    data = []
    with open(cfg.test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    cfg_input_size = min(cfg.image_size[0], cfg.image_size[1])
    os.makedirs(attn_save_dir, exist_ok=True)

    # ---- Inference loop ---------------------------------------------------
    for idx, item in enumerate(data):
        rank0_print(f"\n=== Sample {idx} ===")
        sample_id = f"sample_{idx:04d}"

        user_cond_name = os.path.join(cfg.img_dir, item["first_frame"])
        user_prompt = item["prompt"]

        # Encode reference image
        ref_img_pil = Image.open(user_cond_name.strip()).convert("RGB")
        video_np = np.expand_dims(np.array(ref_img_pil), 0)
        video_t, _ = preprocess(video_np, cfg_input_size, cfg.enable_multi_resolution)
        video_sub = video_t.unsqueeze(0).to(DEVICE, dtype)
        cond_input = vae_en.encode(video_sub)[0].unsqueeze(0)

        # Encode text
        prompts = [user_prompt]
        model_args = text_encoder.encode(prompts)
        y       = model_args["y"].to(DEVICE, dtype)
        y_mask  = model_args["mask"].to(DEVICE)

        # Build noisy latent
        fn, lh, lw = latent_size
        z = torch.randn(1, 16, fn, lh, lw, device=DEVICE, dtype=dtype)

        # Prepare ref_x (reference image in latent space)
        # cond_input shape: [1, 1, C_lat, fnr, H_lat, W_lat]  → squeeze as needed
        ref_x = cond_input  # already latent-encoded

        # ------------------------------------------------------------------
        # Run denoising steps; capture attn map at step `attn_step`
        # ------------------------------------------------------------------
        timesteps = scheduler.get_timesteps(cfg.num_sampling_steps if hasattr(cfg, "num_sampling_steps") else 50)
        x_cur = z.clone()

        for step_i, t in enumerate(timesteps):
            t_batch = t.expand(x_cur.shape[0]).to(DEVICE, dtype)
            ref_t_batch = torch.zeros_like(t_batch)  # reference image timestep = 0

            model_kwargs = dict(
                y=y,
                y_mask=y_mask,
                ref_x=ref_x,
                ref_timestep=ref_t_batch,
                hh=lh // 2,   # base_size_h after patch_size=2
                ww=lw // 2,   # base_size_w after patch_size=2
            )

            if step_i == attn_step:
                rank0_print(f"  Capturing attention maps at denoising step {step_i} ...")
                # Only rank-0 captures to avoid redundant files with CP
                if dist.get_rank() == 0:
                    with AttnMapCollector() as attn_store:
                        v_pred = model(x_cur, t_batch, **model_kwargs)
                    # Build helper to save
                    capt = AttnCapturingScheduler(
                        scheduler=scheduler,
                        capture_step=step_i,
                        attn_layers=attn_layers if attn_layers else list(range(model.depth if hasattr(model, "depth") else 24)),
                        output_dir=attn_save_dir,
                        sample_id=sample_id,
                        fn=fn,
                        lh=lh,
                        lw=lw,
                        fnr=ref_x.shape[3] if ref_x is not None else 1,
                    )
                    capt._post_capture(attn_store, step_i, ref_img_pil)
                else:
                    v_pred = model(x_cur, t_batch, **model_kwargs)
            else:
                v_pred = model(x_cur, t_batch, **model_kwargs)

            # Euler step
            dt = timesteps[step_i + 1] - t if step_i + 1 < len(timesteps) else -t
            x_cur = x_cur + dt.to(dtype) * v_pred

        rank0_print(f"  Finished sample {idx}. Attention maps saved to: {attn_save_dir}/{sample_id}/")

    rank0_print("\nAll samples processed. Attention maps are in:", attn_save_dir)


if __name__ == "__main__":
    main()

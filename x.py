import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os

def extract_and_visualize_attn_map(avg_attn_map, ref_lat_size, ref_img_size, save_dir, save_name):
    """
    把一个 denoising step 的跨层平均 attn map 保存为灰度视频。
    自动根据 T_ref 推断 ref 图片数量，每张 ref 图保存一个独立视频。
    视频共 T_lat=16 帧，每帧是对应 ref 图片上的注意力灰度图（越亮注意力越高）。

    Args:
        avg_attn_map: [B, n_heads=20, T_lat=16, T_ref]
                      T_ref = n_ref × H_ref_patch × W_ref_patch
        ref_lat_size: 单张 ref 图片的 patch 尺寸 (H_ref_patch, W_ref_patch)
                      竖屏=(80, 45)，横屏=(45, 80)
        ref_img_size: 单张 ref 图片的像素尺寸 (H_ref, W_ref)
                      竖屏=(1280, 720)，横屏=(720, 1280)
        save_dir:     保存目录
        save_name:    基础文件名，如 "attn_sample00_t049.mp4"
                      多张 ref 时自动变为 "attn_sample00_t049_ref0.mp4" 等
    """
    H_ref_patch, W_ref_patch = ref_lat_size
    H_ref, W_ref = ref_img_size
    T_lat = avg_attn_map.shape[2]
    T_ref = avg_attn_map.shape[3]

    # 从 T_ref 推断 ref 图片数量
    patches_per_ref = H_ref_patch * W_ref_patch
    n_ref = T_ref // patches_per_ref
    print(f"[attn_shape] avg_attn_map.shape={avg_attn_map.shape}, n_ref={n_ref}", flush=True)

    # 1. 对 heads 求平均，取 cond 那份（batch index 0）
    avg_map = avg_attn_map.mean(dim=1)   # [B, T_lat, T_ref]
    avg_map = avg_map[0].float().cpu()   # [T_lat, T_ref]

    # 2. reshape T_ref → (n_ref, H_ref_patch, W_ref_patch)
    avg_map = avg_map.reshape(T_lat, n_ref, H_ref_patch, W_ref_patch)  # [T_lat, n_ref, H_p, W_p]

    os.makedirs(save_dir, exist_ok=True)
    base, ext = os.path.splitext(save_name)

    # 3. 先上采样所有 ref，再用全局 min/max 统一归一化，保证不同 ref 视频亮度可比
    ref_maps_up = []
    for ref_idx in range(n_ref):
        ref_map = avg_map[:, ref_idx, :, :]  # [T_lat, H_ref_patch, W_ref_patch]
        ref_map_up = F.interpolate(
            ref_map.unsqueeze(0).unsqueeze(0),   # [1, 1, T_lat, H_ref_patch, W_ref_patch]
            size=(T_lat, H_ref, W_ref),
            mode='trilinear',
            align_corners=False,
        ).squeeze()  # [T_lat, H_ref, W_ref]
        ref_maps_up.append(ref_map_up)

    global_min = min(m.min() for m in ref_maps_up)
    global_max = max(m.max() for m in ref_maps_up)

    for ref_idx in range(n_ref):
        ref_map_up = ref_maps_up[ref_idx]

        # 4. 用全局 min/max 归一化到 [0, 255]，不同 ref 视频亮度可比
        ref_map_np = ((ref_map_up - global_min) / (global_max - global_min + 1e-8) * 255).numpy().astype(np.uint8)

        # 5. 保存灰度视频
        fname = f"{base}_ref{ref_idx}{ext}" if n_ref > 1 else save_name
        save_path = os.path.join(save_dir, fname)
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 8, (W_ref, H_ref))
        for t in range(T_lat):
            gray = ref_map_np[t]
            writer.write(cv2.merge([gray, gray, gray]))
        writer.release()
        print(f"Saved attn map video to {save_path}")

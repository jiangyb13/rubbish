"""
把 attn map 视频和生成视频的帧拼在一起，方便查看每一帧的 ref 注意力情况。

用法：
  # 单个 attn 视频 + 生成视频
  python test_viz.py --attn path/to/t049.mp4 --gen path/to/sample_00.mp4 --output out.mp4

  # 文件夹（所有 t*.mp4）+ 生成视频，每个 attn 视频都会生成一个对应的输出
  python test_viz.py --attn path/to/attn_maps/sample00/ --gen path/to/sample_00.mp4 --output path/to/output_dir/

拼接方式（横向）：
  [ 生成视频帧 | ref0 attn | ref1 attn | ref2 attn ]

attn 视频帧数 (T_lat=16) 通常少于生成视频帧数，
脚本会按比例把生成视频帧映射到 attn 帧上（最近邻）。
"""

import cv2
import numpy as np
import os
import argparse
from pathlib import Path


def read_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def combine_videos(attn_video_path, gen_video_path, output_path):
    attn_frames, _ = read_frames(attn_video_path)
    gen_frames, _ = read_frames(gen_video_path)

    if not attn_frames:
        print(f"[WARN] 无法读取 attn 视频: {attn_video_path}")
        return
    if not gen_frames:
        print(f"[WARN] 无法读取生成视频: {gen_video_path}")
        return

    n_attn = len(attn_frames)
    n_gen = len(gen_frames)
    print(f"  attn 帧数={n_attn}, 生成视频帧数={n_gen}")

    # attn 帧的高度作为统一高度
    target_h = attn_frames[0].shape[0]
    # 生成视频帧宽度（单帧）
    gen_w_orig = gen_frames[0].shape[1]
    gen_h_orig = gen_frames[0].shape[0]
    # 等比缩放生成视频帧到 target_h
    gen_w_scaled = int(gen_w_orig * target_h / gen_h_orig)

    combined_frames = []
    for i, attn_frame in enumerate(attn_frames):
        # 按比例找对应的生成帧（最近邻）
        gen_idx = round(i * (n_gen - 1) / max(n_attn - 1, 1))
        gen_frame = gen_frames[gen_idx]
        gen_resized = cv2.resize(gen_frame, (gen_w_scaled, target_h))

        # 在生成帧上标注帧号
        cv2.putText(gen_resized, f"frame {gen_idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # 在 attn 帧上标注 attn 帧号
        attn_labeled = attn_frame.copy()
        cv2.putText(attn_labeled, f"attn t={i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        combined = np.concatenate([gen_resized, attn_labeled], axis=1)
        combined_frames.append(combined)

    out_h, out_w = combined_frames[0].shape[:2]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 8, (out_w, out_h))
    for frame in combined_frames:
        writer.write(frame)
    writer.release()
    print(f"  已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn', type=str, required=True,
                        help='attn map 视频路径（单个 .mp4）或文件夹（含多个 t*.mp4）')
    parser.add_argument('--gen', type=str, required=True,
                        help='生成视频路径（.mp4）')
    parser.add_argument('--output', type=str, default='viz_output',
                        help='输出路径（单文件时为 .mp4，文件夹时为目录）')
    args = parser.parse_args()

    attn_path = Path(args.attn)
    gen_path = Path(args.gen)

    if attn_path.is_file():
        # 单个 attn 视频
        out = args.output if args.output.endswith('.mp4') else args.output + '.mp4'
        print(f"处理: {attn_path.name}")
        combine_videos(attn_path, gen_path, out)
    elif attn_path.is_dir():
        # 文件夹：处理所有 .mp4
        videos = sorted(attn_path.glob('*.mp4'))
        if not videos:
            print(f"[ERROR] 文件夹中没有 .mp4 文件: {attn_path}")
            return
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for v in videos:
            print(f"处理: {v.name}")
            combine_videos(v, gen_path, out_dir / v.name)
    else:
        print(f"[ERROR] --attn 路径不存在: {attn_path}")


if __name__ == '__main__':
    main()

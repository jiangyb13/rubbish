import argparse
import json
import math
import os
import pickle
# import tempfile
import time
import traceback
from pathlib import PurePosixPath

import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import moxing as mox
import json

import matplotlib.pyplot as plt

def plot_resolution_distribution(
    dict_img,
    top_k=50,
    save_path="resolution_distribution.png"
):
    # 按 value 从大到小排序
    sorted_items = sorted(
        dict_img.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # 只保留 Top K
    top_items = sorted_items[:top_k]

    resolutions = [item[0] for item in top_items]
    counts = [item[1] for item in top_items]

    # 根据实际数量调整画布宽度
    fig_width = max(12, len(top_items) * 0.35)

    plt.figure(figsize=(fig_width, 8))

    plt.bar(
        range(len(resolutions)),
        counts
    )

    plt.xticks(
        range(len(resolutions)),
        resolutions,
        rotation=60,
        ha="right"
    )

    plt.xlabel("Resolution")
    plt.ylabel("Count")
    plt.title(
        f"Top {min(top_k, len(dict_img))} Frame Resolutions "
        f"(Total: {len(dict_img)} resolutions)"
    )

    plt.tight_layout()

    # 保存
    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()



# 在导入 transformers 前启用原有的 NPU 迁移逻辑。
try:
    import torch_npu
except ImportError:
    torch_npu = None

if torch_npu is not None:
    from torch_npu.contrib import transfer_to_npu

import moxing as mox


import cv2
import numpy as np


def get_nonzero_bbox(image_path):
    """
    读取图片，找到所有非0像素的最小外接矩形。

    返回：
        min_x, min_y, max_x, max_y, width, height
    """

    # 读取图片
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # 判断哪些位置是非0像素
    if img.ndim == 2:
        # 灰度图
        mask = img != 0
    else:
        mask = np.any(img != 0, axis=2)

    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise ValueError("图片中没有非0像素")

    # 最小/最大坐标
    min_x = xs.min()
    max_x = xs.max()
    min_y = ys.min()
    max_y = ys.max()

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    return {
        "min_x": int(min_x),
        "min_y": int(min_y),
        "max_x": int(max_x),
        "max_y": int(max_y),
        "width": int(width),
        "height": int(height),
    }



with open("/data/huanan/code/jwx1416454/ID_cross_train_data_0914_Final/final_qwen3.8_caption.jsonl", "r", encoding="utf-8") as stream:
    rows = [
        json.loads(line)
        for line in stream
        if line.strip()
    ]


cnt = 0
missing_cnt = 0
dict_img = {}

log_path = "log.jsonl"

# 清空旧 log
with open(log_path, "w", encoding="utf-8") as f:
    pass

for row in tqdm(rows):
    filepath_list = row["in_cross_pair_face_fn"]

    for path in filepath_list:
        filepath = (
            "s3://bucket-4931-huanan/data/l00966586/ID_cross_train_data_v1/face_imgs/"
            + path
        )
        cnt += 1
        try:
            mox.file.copy(filepath, "./pic.png")

            # img_message = get_nonzero_bbox("./pic.png")
            img = cv2.imread("./pic.png", cv2.IMREAD_UNCHANGED)
            
            # dict_img[(img_message["width"], img_message['height'])] = dict_img.get((img_message["width"], img_message['height']), 0) + 1
            dict_img[img.shape] = dict_img.get(img.shape, 0) + 1

            print(f"pic: {filepath}, (width, height): {img.shape}")

        except Exception as e:
            missing_cnt += 1
            print(f"Failed: {filepath}: {e}")

        # 每 100 个视频记录一次中间结果
        if cnt % 100 == 0:
            # tuple 不能作为 JSON object 的 key，
            # 所以保存时转换成字符串
            resolution_count = {
                str(k): v
                for k, v in sorted(
                    dict_img.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            }

            log_record = {
                "processed": cnt,
                "missing": missing_cnt,
                "resolution_count": resolution_count,
            }

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(log_record, ensure_ascii=False) + "\n"
                )

# 最后再写一次最终结果
resolution_count = {
    str(k): v
    for k, v in sorted(
        dict_img.items(),
        key=lambda x: x[1],
        reverse=True
    )
}

with open(log_path, "a", encoding="utf-8") as f:
    f.write(
        json.dumps(
            {
                "processed": cnt,
                "missing": missing_cnt,
                "resolution_count": resolution_count,
                "final": True,
            },
            ensure_ascii=False
        )
        + "\n"
    )


plot_resolution_distribution(dict_img)
        # breakpoint()

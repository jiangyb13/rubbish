"""
根据 face_white 图片路径，从 identity_matching 的 output.jsonl 中查询对应的人脸角度。

支持两种图片路径格式：
  1. person_clusters 目录下：{shot_key}_id{obj_id}_frame{frame_idx:04d}.jpg
  2. Stage 2 输出目录下：{shot_basename}/id_{obj_id}/face/cropped_face/face_pic_white/{frame_idx}.jpg

用法：
  python query_face_angle.py --jsonl /path/to/output.jsonl --image /path/to/face.jpg
  python query_face_angle.py --jsonl /path/to/output.jsonl --image /path/to/face.jpg --batch  # 批量模式（读取stdin或glob）
"""

import argparse
import json
import os
import re
import sys
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 解析图片路径，提取 (shot_key, obj_id, frame_idx)
# ---------------------------------------------------------------------------

def parse_person_cluster_filename(filename: str) -> Optional[Tuple[str, str, int]]:
    """
    解析 person_clusters/face_white/ 下的文件名。
    格式：{shot_key}_id{obj_id}_frame{frame_idx:04d}.ext
    示例：shot_0001_id2_frame0010.jpg → ('shot_0001', '2', 10)
    """
    stem = os.path.splitext(filename)[0]
    # 从右往左找 _frame{digits}
    m = re.search(r'^(.+)_id(\w+)_frame(\d+)$', stem)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    return None


def parse_stage2_path(image_path: str) -> Optional[Tuple[str, str, int]]:
    """
    解析 Stage 2 输出目录下的路径。
    格式：.../id_{obj_id}/face/*/face_pic_white/{frame_idx}.ext
           或 .../id_{obj_id}/full/full_pic_white/{frame_idx}.ext
    shot_key 取 id_{obj_id} 的父目录名（去扩展名）。
    """
    parts = image_path.replace("\\", "/").split("/")
    stem = os.path.splitext(parts[-1])[0]
    if not stem.isdigit():
        return None
    frame_idx = int(stem)

    # 找 id_xxx 层
    for i, part in enumerate(parts):
        m = re.match(r'^id_(\w+)$', part)
        if m:
            obj_id = m.group(1)
            # shot_key = id_xxx 的父目录名（不含扩展名）
            if i > 0:
                shot_key = os.path.splitext(parts[i - 1])[0]
                return shot_key, obj_id, frame_idx
    return None


def parse_image_path(image_path: str) -> Optional[Tuple[str, str, int]]:
    """尝试所有解析策略，返回 (shot_key, obj_id, frame_idx) 或 None。"""
    filename = os.path.basename(image_path)

    # 策略 1：person_clusters 文件名格式
    result = parse_person_cluster_filename(filename)
    if result:
        return result

    # 策略 2：Stage 2 目录结构
    result = parse_stage2_path(image_path)
    if result:
        return result

    return None


# ---------------------------------------------------------------------------
# 加载 output.jsonl，建立索引
# ---------------------------------------------------------------------------

def build_index(jsonl_path: str) -> dict:
    """
    读取 output.jsonl，建立索引：
      index[(shot_key, obj_id)][frame_idx] = pose
    """
    index = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            # 获取 shot_key
            shot_path = record.get("source_shot_path", "")
            shot_key = os.path.splitext(os.path.basename(shot_path))[0]

            obj_id = str(record.get("obj_id", ""))

            frontal_scores = record.get("identity_matching_per_frame_frontal_scores") or []
            frame_map = {}
            for entry in frontal_scores:
                fidx = entry.get("frame_idx")
                if fidx is None:
                    continue
                frame_map[int(fidx)] = entry.get("pose")

            key = (shot_key, obj_id)
            if key not in index:
                index[key] = {}
            index[key].update(frame_map)

    return index


# ---------------------------------------------------------------------------
# 查询单张图片的角度
# ---------------------------------------------------------------------------

def query_angle(image_path: str, index: dict) -> dict:
    """
    查询单张图片的角度，返回结果字典。
    """
    result = {
        "image_path": image_path,
        "shot_key": None,
        "obj_id": None,
        "frame_idx": None,
        "pose": None,
        "error": None,
    }

    parsed = parse_image_path(image_path)
    if parsed is None:
        result["error"] = "无法从路径解析 shot_key / obj_id / frame_idx，请检查路径格式"
        return result

    shot_key, obj_id, frame_idx = parsed
    result["shot_key"] = shot_key
    result["obj_id"] = obj_id
    result["frame_idx"] = frame_idx

    key = (shot_key, obj_id)
    if key not in index:
        # 尝试模糊匹配 shot_key（有时路径中带有额外前缀）
        candidates = [k for k in index if k[1] == obj_id and shot_key in k[0]]
        if len(candidates) == 1:
            key = candidates[0]
            result["shot_key"] = key[0]
        else:
            result["error"] = f"在 jsonl 中找不到 shot_key='{shot_key}', obj_id='{obj_id}'"
            return result

    frame_map = index[key]
    if frame_idx not in frame_map:
        result["error"] = (
            f"找到了 shot_key='{result['shot_key']}', obj_id='{obj_id}'，"
            f"但 frame_idx={frame_idx} 不在 per_frame_frontal_scores 中"
            f"（可能该帧未通过质量过滤，或 DECA 推理失败）"
        )
        return result

    result["pose"] = frame_map[frame_idx]
    return result


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="查询 face_white 图片对应的人脸角度")
    parser.add_argument("--jsonl", required=True, help="identity_matching 输出的 output.jsonl 路径")
    parser.add_argument("--image", help="单张图片路径")
    parser.add_argument("--image_list", help="图片路径列表文件（每行一个路径）")
    parser.add_argument("--output", help="批量查询结果输出路径（json），不指定则打印到 stdout")
    args = parser.parse_args()

    if not args.image and not args.image_list:
        parser.error("请指定 --image 或 --image_list")

    print(f"加载索引：{args.jsonl} ...", file=sys.stderr)
    index = build_index(args.jsonl)
    print(f"索引加载完成，共 {len(index)} 个 (shot_key, obj_id) 对", file=sys.stderr)

    # 收集待查图片
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    if args.image_list:
        with open(args.image_list, "r", encoding="utf-8") as f:
            image_paths.extend(line.strip() for line in f if line.strip())

    # 查询
    results = [query_angle(p, index) for p in image_paths]

    # 输出
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已写入：{args.output}", file=sys.stderr)
    else:
        for r in results:
            print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

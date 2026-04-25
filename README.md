"""
Double Check Face Orientation using VLM (e.g., Qwen3-VL-8B-Instruct)

功能：对已经分好类的 front, left, right 文件夹中的图片进行二次校验，判断分类是否准确。
支持多 GPU 并行：通过 --phase / --total 参数分片处理。

用法示例:
  python face_orientation_checker.py \
      --base_dir path/to/dataset \
      --output_jsonl outputs/double_check_result_0.jsonl \
      --model_path Qwen/Qwen3-VL-8B-Instruct \
      --device cuda:0 \
      --phase 0 --total 4
"""

import argparse
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ─────────────────────────── Prompt ───────────────────────────

def get_robust_verification_prompt(expected_orientation):
    return f"""You are a strict computer vision data annotator specializing in Head Pose Estimation. 
Your task is to double-check the yaw angle (horizontal orientation) of the face in the provided image.

This image has been pre-filtered. Its EXPECTED orientation label is: "{expected_orientation}".

### Class Definitions (Based on Camera Perspective & Facial Landmarks)
- "front": The face is looking directly (or almost directly) at the camera. Both eyes, both cheeks, and the contours of both sides of the face are relatively equally visible. The tip of the nose is roughly centered.
- "left": The person's face is turned towards the LEFT side of the image frame. The right side of their face (right cheek/ear) is more visible to the camera, while their left eye or left cheek is partially or fully occluded by the bridge of the nose.
- "right": The person's face is turned towards the RIGHT side of the image frame. The left side of their face (left cheek/ear) is more visible to the camera, while their right eye or right cheek is partially or fully occluded by the bridge of the nose.

### Evaluation Rules
1. Reference Frame: Always judge "left" and "right" based on the image frame's left and right, NOT the subject's left and right hands.
2. Ignore Pitch/Roll: Ignore if the person is looking up, looking down, or tilting their head to the shoulder. Focus ONLY on the left/right rotation (Yaw).
3. Borderline Cases: If the face is only very slightly turned (e.g., < 15 degrees) but still maintains binocular eye contact with the camera, classify it as "front".

Please evaluate the image and output STRICTLY in the following JSON format without any markdown code blocks or additional text:
{{
  "is_correct": true or false,
  "actual_orientation": "front" | "left" | "right",
  "confidence": "high" | "medium" | "low",
  "reason": "Explain briefly using visible landmarks, e.g., 'The nose tip points left and the left ear is occluded, indicating a left turn.'"
}}"""


# ─────────────────────────── 工具函数 ───────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser(description="Double check face orientation using VLM")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="包含 front, left, right 子文件夹的根目录")
    parser.add_argument("--output_jsonl", type=str, required=True,
                        help="输出结果的 JSONL 文件路径")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="VLM 模型路径")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="推理设备")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="生成的最大 token 数量")
    parser.add_argument("--phase", type=int, default=0,
                        help="Worker 编号 (从 0 开始)")
    parser.add_argument("--total", type=int, default=1,
                        help="Worker 总数，用于分片并行")
    return parser.parse_args()


def get_image_files(base_dir):
    """递归获取所有图片文件，并记录其所属文件夹（预期标签）。"""
    files = []
    sub_dirs = ['front', 'left', 'right']
    
    for sd in sub_dirs:
        dir_path = os.path.join(base_dir, sd)
        if not os.path.exists(dir_path):
            print(f"[Warning] 找不到目录: {dir_path}")
            continue
            
        for fname in os.listdir(dir_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                files.append({
                    "path": os.path.join(dir_path, fname),
                    "expected_label": sd
                })
    return files


def parse_vlm_response(response_text, expected_label):
    """从 VLM 输出中提取 JSON 并进行容错解析。"""
    try:
        # 提取花括号包裹的 JSON 字符串
        json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
    except Exception as e:
        pass
    
    # 极简备用解析逻辑（如果 JSON 结构破损严重）
    response_lower = response_text.lower()
    is_correct = "true" in response_lower.split("is_correct")[1].split(",")[0] if "is_correct" in response_lower else False
    
    actual_ori = "unknown"
    for ori in ["front", "left", "right"]:
        if f'"actual_orientation": "{ori}"' in response_lower or f'"actual_orientation":"{ori}"' in response_lower:
            actual_ori = ori
            break
            
    return {
        "is_correct": is_correct,
        "actual_orientation": actual_ori,
        "confidence": "low",
        "reason": f"[fallback parse] {response_text[:100]}"
    }


# ─────────────────────────── 主流程 ───────────────────────────

def main():
    args = parse_arguments()

    # 创建输出目录
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)

    # 获取所有图片并进行分片
    all_files = get_image_files(args.base_dir)
    print(f"共加载 {len(all_files)} 张图片记录")

    if args.total > 1:
        worker_files = [f for idx, f in enumerate(all_files) if idx % args.total == args.phase]
        print(f"[Worker {args.phase}/{args.total}] 分配到 {len(worker_files)} 张图片")
    else:
        worker_files = all_files

    if not worker_files:
        print("没有需要处理的图片，退出。")
        return

    # 加载模型
    print(f"正在加载模型 {args.model_path} 到 {args.device} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map=args.device,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    print("模型加载完成！")

    # 逐条处理并写入 JSONL
    file_out = open(args.output_jsonl, "a", encoding="utf-8")
    correct_count = 0
    incorrect_count = 0

    try:
        for item in tqdm(worker_files, desc=f"Worker {args.phase}"):
            img_path = item["path"]
            expected = item["expected_label"]
            
            prompt_text = get_robust_verification_prompt(expected)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            try:
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                
            except Exception as exc:
                print(f"  [ERROR] 推理失败 {img_path}: {exc}")
                continue

            # 解析结果
            vlm_result = parse_vlm_response(output_text, expected)
            
            # 双重保险：修正 vlm 可能的 is_correct 矛盾
            # 如果 VLM 认为的 actual_orientation 恰好等于预期标签，那 is_correct 强制设为 True
            if vlm_result.get("actual_orientation") == expected:
                vlm_result["is_correct"] = True
            
            if vlm_result.get("is_correct", False):
                correct_count += 1
            else:
                incorrect_count += 1

            # 组装最终结果
            output_record = {
                "file_path": img_path,
                "expected_label": expected,
                "vlm_verification": vlm_result,
                "raw_response": output_text
            }

            file_out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            file_out.flush()

    finally:
        file_out.close()

    print(f"\n[Worker {args.phase}] 处理完成！")
    print(f"  校验一致 (正确): {correct_count}")
    print(f"  校验不一致 (错误): {incorrect_count}")
    print(f"  输出保存至: {args.output_jsonl}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ─────────────────────────── Prompt ───────────────────────────

FACE_CHECK_PROMPT = """\
You are an expert in facial orientation analysis. Your task is to analyze the face in the image and classify its orientation relative to the camera.

Categories:
- "front": The person is looking directly or nearly directly at the camera. The face is centered.
- "left": The person is facing towards the left side of the frame (profile or semi-profile).
- "right": The person is facing towards the right side of the frame (profile or semi-profile).

Please respond strictly in the following JSON format:
```json
{
  "orientation": "front" | "left" | "right",
  "reason": "Brief explanation of your judgment"
}
```"""

# ─────────────────────────── 工具函数 ───────────────────────────

def parse_arguments():
    parser = argparse.ArgumentParser(description="Double check face orientation using Qwen3-VL")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing front/left/right subfolders")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output JSONL results")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--phase", type=int, default=0)
    parser.add_argument("--total", type=int, default=1)
    return parser.parse_args()

def get_image_files(base_dir):
    """递归获取所有图片文件，并记录其所属文件夹（Ground Truth）。"""
    files = []
    # 假设结构: base_dir/front/xxx.jpg, base_dir/left/xxx.jpg 等
    sub_dirs = ['front', 'left', 'right']
    for sd in sub_dirs:
        dir_path = os.path.join(base_dir, sd)
        if not os.path.exists(dir_path):
            continue
        for fname in os.listdir(dir_path):
            if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                files.append({
                    "path": os.path.join(dir_path, fname),
                    "folder_label": sd
                })
    return files

# ─────────────────────────── 主流程 ───────────────────────────

def main():
    args = parse_arguments()
    
    all_files = get_image_files(args.base_dir)
    # 分片逻辑
    worker_files = [f for idx, f in enumerate(all_files) if idx % args.total == args.phase]
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype="auto", device_map=args.device)
    processor = AutoProcessor.from_pretrained(args.model_path)

    with open(args.output_jsonl, "a", encoding="utf-8") as f_out:
        for item in tqdm(worker_files, desc=f"Worker {args.phase}"):
            img_path = item["path"]
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {"type": "text", "text": FACE_CHECK_PROMPT},
                    ],
                }
            ]

            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=100)
                
            output_text = processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            
            # 解析 JSON (建议增加错误容忍)
            try:
                import re
                json_str = re.search(r"\{.*\}", output_text, re.DOTALL).group()
                vlm_res = json.loads(json_str)
                orientation = vlm_res.get("orientation")
            except:
                orientation = "unknown"
                vlm_res = {"reason": "parse error"}

            # Double Check 逻辑：检查 VLM 预测与文件夹标签是否一致
            is_consistent = (orientation == item["folder_label"])
            
            result_entry = {
                "file_path": img_path,
                "folder_label": item["folder_label"],
                "vlm_prediction": orientation,
                "is_consistent": is_consistent,
                "reason": vlm_res.get("reason", "")
            }
            
            f_out.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()

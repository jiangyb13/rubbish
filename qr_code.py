import os
import sys
import torch
import pandas as pd
import json
import argparse
from PIL import Image
from tqdm import tqdm
from modeling.longclip_b.model import longclip

# --- 1. 初始化 CLIP 模型 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在加载 LongCLIP 模型至 {device}...")
model, preprocess = longclip.load("/home/ma-user/work/wx1468559/Bagel-Reca/pretrained_models/LongCLIP-B/longclip-B.pt", device=device)

def calc_clip_score(image_path, prompt):
    """计算相似度分数"""
    try:
        if not os.path.exists(image_path):
            return -1.0
        image = Image.open(image_path).convert("RGB")
        text_inputs = longclip.tokenize([prompt]).to(device)
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_embeds = model.encode_image(image_tensor)
            text_embeds = model.encode_text(text_inputs)

        image_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return (image_norm @ text_norm.T).item()
    except Exception as e:
        print(f"  [错误] 无法计算分数 {image_path}: {e}")
        return -1.0

# --- 2. 参数设置 ---
parser = argparse.ArgumentParser(description="根据 CLIP 分数择优并合并 A 和 B 的并集。")
parser.add_argument("--file_a", type=str, default="merged_best_results_union4.jsonl")
parser.add_argument("--file_b", type=str, default="/home/ma-user/work/wx1468559/geneval/Bagel/outputs_bagel_set1seed_dual_h.jsonl")
parser.add_argument("--output_file", type=str, default="merged_best_results_union4.jsonl", help="输出文件路径")
args = parser.parse_args()

# --- 3. 数据加载与预处理 ---
def load_and_prepare(filepath):
    df = pd.read_json(filepath, orient="records", lines=True)
    # 将 metadata 转为 string 方便后续作为 merge/join 的 Key 进行对比
    df['meta_key'] = df['metadata'].apply(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, dict) else x)
    # 提取文件名作为 ID
    df['sample_id'] = df['filename'].apply(os.path.basename)
    return df

print("正在读取 A 和 B 的结果文件...")
df_a = load_and_prepare(args.file_a)
df_b = load_and_prepare(args.file_b)

# --- 4. 求并集并处理冲突 ---
# 使用 meta_key 和 sample_id 求并集
all_keys = pd.concat([df_a[['meta_key', 'sample_id']], df_b[['meta_key', 'sample_id']]]).drop_duplicates()

final_results = []
stats = {"a_only": 0, "b_only": 0, "a_better": 0, "b_better": 0}

for _, key_row in tqdm(all_keys.iterrows(), total=len(all_keys), desc="并集择优处理"):
    m_key = key_row['meta_key']
    s_id = key_row['sample_id']
    
    # 查找该 key 在 A 和 B 中是否存在
    match_a = df_a[(df_a['meta_key'] == m_key) & (df_a['sample_id'] == s_id)]
    match_b = df_b[(df_b['meta_key'] == m_key) & (df_b['sample_id'] == s_id)]
    
    in_a = not match_a.empty
    in_b = not match_b.empty
    
    if in_a and not in_b:
        # 只在 A 中有
        selected_row = match_a.iloc[0].to_dict()
        stats["a_only"] += 1
    elif in_b and not in_a:
        # 只在 B 中有
        selected_row = match_b.iloc[0].to_dict()
        stats["b_only"] += 1
    else:
        # A 和 B 都有，择优
        row_a = match_a.iloc[0]
        row_b = match_b.iloc[0]
        
        # 获取 prompt
        prompt = json.loads(m_key)['prompt']
        
        score_a = calc_clip_score(row_a['filename'], prompt)
        score_b = calc_clip_score(row_b['filename'], prompt)
        
        if score_a >= score_b:
            selected_row = row_a.to_dict()
            stats["a_better"] += 1
        else:
            selected_row = row_b.to_dict()
            stats["b_better"] += 1

    # 清理临时字段并添加
    selected_row.pop('meta_key', None)
    selected_row.pop('sample_id', None)
    final_results.append(selected_row)

# --- 5. 保存结果 ---
with open(args.output_file, 'w', encoding='utf-8') as f:
    for entry in final_results:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print("\n" + "="*50)
print(f"并集处理完成！")
print(f"总样本数: {len(final_results)}")
print(f"- 仅存在于 A: {stats['a_only']}")
print(f"- 仅存在于 B: {stats['b_only']}")
print(f"- A/B 共存且 A 优: {stats['a_better']}")
print(f"- A/B 共存且 B 优: {stats['b_better']}")
print(f"输出文件: {args.output_file}")
print("="*50)

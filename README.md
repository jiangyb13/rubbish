import os
import json
from pathlib import Path

def convert_to_target_relative(data, target_folder_name="data"):
    """
    终极截断逻辑：
    不管是 "/mnt/data/uuu" (绝对)
    还是 "../../mnt/data/uuu" (被搞坏的相对)
    甚至是 "data/uuu" (已经是正确的)
    只要切分后包含 target_folder_name，全部统一修正为 "data/..."
    """
    if isinstance(data, dict):
        return {k: convert_to_target_relative(v, target_folder_name) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_target_relative(item, target_folder_name) for item in data]
    elif isinstance(data, str):
        # 1. 统一斜杠，防止系统差异
        normalized_str = data.replace('\\', '/')
        
        # 2. 只要字符串里有 '/'，就说明它可能是一个路径
        if '/' in normalized_str:
            # 例如 "../../mnt/data/uuu" 会被拆成 ['..', '..', 'mnt', 'data', 'uuu']
            parts = normalized_str.split('/')
            
            # 3. 如果我们找到了目标锚点文件夹（例如 'data'）
            if target_folder_name in parts:
                # 找到它的索引位置
                idx = parts.index(target_folder_name)
                # 直接从该位置重新拼接，前面的 '..' 和 'mnt' 全都丢弃！
                return "/".join(parts[idx:])
                
        # 如果不符合条件（比如普通文本），原样返回
        return data
    else:
        return data

def process_file(file_path, target_folder_name):
    # 根据后缀判断是 json 还是 jsonl
    is_jsonl = file_path.suffix.lower() == '.jsonl'
    
    if is_jsonl:
        new_lines = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    new_data = convert_to_target_relative(data, target_folder_name)
                    new_lines.append(json.dumps(new_data, ensure_ascii=False))
                except json.JSONDecodeError:
                    new_lines.append(line.strip())
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines) + '\n')
            
    else: # json
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return
        new_data = convert_to_target_relative(data, target_folder_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=4)

def main(dataset_root, target_folder_name):
    base_dir = Path(dataset_root).resolve()
    print(f"🚀 开始扫描: {base_dir}")
    print(f"✂️  截断规则: 将绝对路径截断为以 '{target_folder_name}/' 开头\n")

    count = 0
    for file_path in base_dir.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in ['.json', '.jsonl']:
            process_file(file_path, target_folder_name)
            print(f"✅ 已处理: {file_path}")
            count += 1

    print(f"\n🎉 搞定！共修改了 {count} 个文件。")

if __name__ == "__main__":
    # 1. 你要处理的文件夹所在路径
    FOLDER_TO_PROCESS = "./my_dataset" 
    
    # 2. 你想保留的起点文件夹名（比如你想留下 data/uuu，这里就填 data）
    STARTING_FOLDER = "data" 
    
    main(FOLDER_TO_PROCESS, STARTING_FOLDER)

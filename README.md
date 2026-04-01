import os
import json
from pathlib import Path

def convert_to_target_relative(data, target_folder_name="data"):
    """
    不管前面是 /mnt/ 还是 /root/autodl-tmp/，
    只要在路径里碰到 target_folder_name (比如 'data')，
    就把前面的全砍掉，只保留从 'data' 开始的部分。
    """
    if isinstance(data, dict):
        return {k: convert_to_target_relative(v, target_folder_name) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_target_relative(item, target_folder_name) for item in data]
    elif isinstance(data, str):
        # 1. 统一斜杠，防止系统差异
        normalized_str = data.replace('\\', '/')
        
        # 2. 判断是不是绝对路径（以 '/' 开头或包含 ':/'）
        if normalized_str.startswith('/') or ':/' in normalized_str:
            # 拆分路径，比如 "/mnt/data/uuu" 会变成 ['', 'mnt', 'data', 'uuu']
            parts = normalized_str.split('/')
            
            # 3. 核心：寻找目标文件夹名并截断
            if target_folder_name in parts:
                idx = parts.index(target_folder_name)
                # 重新拼接成 "data/uuu"
                return "/".join(parts[idx:])
                
        # 如果不是绝对路径，或者没找到目标文件夹，就原样返回
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

import os
import json
from pathlib import Path

def is_absolute_path(val_str):
    """
    判断一个字符串是否是绝对路径。
    注意：在 Linux 系统中，任何以 '/' 开头的字符串都会被判定为绝对路径。
    如果你的 json 包含以 '/' 开头的普通非路径文本，建议在这里增加额外判断（比如后缀名检查）。
    """
    if not isinstance(val_str, str):
        return False
    # 排除空字符串或纯空格
    if not val_str.strip():
        return False
    return os.path.isabs(val_str)

def convert_paths_in_data(data, reference_dir):
    """
    递归遍历字典或列表，将绝对路径转换为相对路径。
    """
    if isinstance(data, dict):
        return {k: convert_paths_in_data(v, reference_dir) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_paths_in_data(item, reference_dir) for item in data]
    elif isinstance(data, str):
        if is_absolute_path(data):
            try:
                # 转为相对于 reference_dir 的相对路径
                return os.path.relpath(data, reference_dir)
            except ValueError:
                # 在 Windows 下，如果跨盘符（例如 C: 到 D:），relpath 会报错
                # 这种情况下直接返回原路径
                return data
        return data
    else:
        # 其他类型（int, float, bool, None 等）保持不变
        return data

def process_json_file(file_path, reference_dir):
    """处理单个 .json 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"⚠️ 跳过: 无法解析 JSON 格式 - {file_path}")
            return

    new_data = convert_paths_in_data(data, reference_dir)

    # 将修改后的内容写回文件，覆盖原文件
    with open(file_path, 'w', encoding='utf-8') as f:
        # ensure_ascii=False 保证中文字符正常显示
        json.dump(new_data, f, ensure_ascii=False, indent=4)

def process_jsonl_file(file_path, reference_dir):
    """处理单个 .jsonl 文件"""
    new_lines = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                new_data = convert_paths_in_data(data, reference_dir)
                new_lines.append(json.dumps(new_data, ensure_ascii=False))
            except json.JSONDecodeError:
                print(f"⚠️ 警告: 无法解析 {file_path} 的第 {line_num} 行，已保持原样。")
                new_lines.append(line)

    # 将修改后的内容写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line + '\n')

def main(target_folder, relative_to_folder=None):
    """
    主函数
    :param target_folder: 需要遍历的根目录
    :param relative_to_folder: 计算相对路径的基准目录。如果为 None，则默认使用 target_folder 作为基准。
    """
    target_path = Path(target_folder).resolve()
    
    if relative_to_folder is None:
        reference_dir = target_path
    else:
        reference_dir = Path(relative_to_folder).resolve()

    print(f"🚀 开始遍历目录: {target_path}")
    print(f"📍 相对路径基准: {reference_dir}\n")

    # rglob('*') 递归遍历该文件夹下的所有文件
    for file_path in target_path.rglob('*'):
        if file_path.is_file():
            if file_path.suffix.lower() == '.json':
                print(f"处理 JSON: {file_path}")
                process_json_file(file_path, reference_dir)
            elif file_path.suffix.lower() == '.jsonl':
                print(f"处理 JSONL: {file_path}")
                process_jsonl_file(file_path, reference_dir)
                
    print("\n✅ 处理完成！")

if __name__ == "__main__":
    # 替换为你实际想要遍历的文件夹路径
    FOLDER_TO_PROCESS = "./my_dataset_folder" 
    
    # 运行代码
    main(FOLDER_TO_PROCESS)

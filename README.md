from pathlib import Path

def get_dirname_up_to_target(path_str, target_folder="data"):
    # 1. 把字符串转换成 Path 对象
    p = Path(path_str)
    
    # 2. p.parts 会把路径拆成元组
    # 例如 "/home/flask/data/images/1.jpg" 变成 ('/', 'home', 'flask', 'data', 'images', '1.jpg')
    parts = p.parts
    
    if target_folder in parts:
        # 3. 找到目标文件夹的索引
        idx = parts.index(target_folder)
        
        # 4. 切片：取从开头到目标文件夹（包含目标文件夹，所以要 idx + 1）的所有部分
        # 然后用 Path(*...) 重新组装成路径，并转回字符串
        return str(Path(*parts[:idx + 1]))
    else:
        # 如果路径里根本没有 data，就返回原路或报错，看你的需求
        return path_str

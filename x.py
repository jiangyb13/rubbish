import os
import shutil

def batch_rename_and_move(source_dir, target_dir, prefix="video"):
    """
    读取源文件夹下的所有 mp4 文件，修改名字后保存到指定的目标文件夹。
    
    :param source_dir: 存放原始 mp4 文件的文件夹路径
    :param target_dir: 移动后存放文件的目标文件夹路径
    :param prefix: 新文件名的前缀，默认是 'video'
    """
    # 确保目标文件夹存在，如果不存在则自动创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建目标文件夹: {target_dir}")

    # 获取源文件夹下的所有文件
    files = os.listdir(source_dir)
    
    # 过滤出所有的 .mp4 文件
    mp4_files = [f for f in files if f.lower().endswith('.mp4')]
    
    if not mp4_files:
        print("没有在源文件夹中找到 .mp4 文件。")
        return

    print(f"共找到 {len(mp4_files)} 个 MP4 文件，开始处理...")

    # 计数器，用于给文件编号
    count = 1

    for filename in mp4_files:
        # 构建旧文件的完整路径
        old_file_path = os.path.join(source_dir, filename)
        
        # ----------------【这里可以自定义你的命名规则】----------------
        # 举例：默认改成 "video_1.mp4", "video_2.mp4" 这种格式
        # zfill(3) 意思是编号保持3位数，比如 001, 002... 方便排序
        new_filename = f"{prefix}_{str(count).zfill(3)}.mp4"
        # -----------------------------------------------------------
        
        # 构建新文件的完整路径
        new_file_path = os.path.join(target_dir, new_filename)
        
        try:
            # 使用 shutil.copy 复制并重命名（安全，原文件不动）
            # 如果想直接“剪切”过去，可以把下面这行换成 shutil.move(old_file_path, new_file_path)
            shutil.copy(old_file_path, new_file_path)
            print(f"成功: {filename} -> {new_filename}")
            count += 1
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

    print("\n所有文件处理完毕！")

# ==================== 使用前配置 ====================
if __name__ == "__main__":
    # 1. 你的原始视频文件夹路径（请替换成你自己的实际路径）
    # 注意：Windows 路径中的 \ 最好改成 / 或者在字符串前加 r
    source_folder = r"C:\Users\YourUsername\Desktop\OldVideos"
    
    # 2. 你想要保存到的目标文件夹路径
    target_folder = r"C:\Users\YourUsername\Desktop\NewVideos"
    
    # 3. 新名字的前缀
    file_prefix = "我的视频"

    # 执行函数
    batch_rename_and_move(source_folder, target_folder, file_prefix)

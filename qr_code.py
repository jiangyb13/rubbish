import base64
import os
import math

def batch_image_to_base64_and_split(input_folder_path, num_parts):
    """
    将指定文件夹中的每张图片转换为Base64编码，并将其分割成指定数量的部分。

    :param input_folder_path: 包含图片的文件夹路径。
    :param num_parts: 要将Base64字符串分割成的部分数量 (N)。
    """
    if not os.path.isdir(input_folder_path):
        print(f"错误：找不到文件夹 '{input_folder_path}' 或它不是一个有效的文件夹。")
        return

    if not isinstance(num_parts, int) or num_parts <= 0:
        print(f"错误：分割数量 (num_parts) 必须是一个大于0的整数。")
        return

    # 定义支持的图片文件类型
    supported_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(input_folder_path):
        if file_name.lower().endswith(supported_extensions):
            image_path = os.path.join(input_folder_path, file_name)
            # 从文件名（不含扩展名）创建输出文件夹的路径
            base_name = os.path.splitext(file_name)[0]
            output_folder = os.path.join(input_folder_path, base_name)

            print(f"--- 正在处理图片: {file_name} ---")

            try:
                # 1. 创建与图片同名的文件夹
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                    print(f"创建了文件夹: {output_folder}")

                # 2. 读取图片并进行 Base64 编码
                with open(image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                
                total_length = len(encoded_string)
                # 计算每部分的基本长度（使用向上取整，让前面的部分更长一点）
                part_length = math.ceil(total_length / num_parts)

                # 3. 循环分割 Base64 字符串并写入文件
                for i in range(num_parts):
                    start_index = i * part_length
                    end_index = start_index + part_length
                    
                    # 获取当前部分的数据
                    part_data = encoded_string[start_index:end_index]
                    
                    # 如果当前部分数据为空（在总长度小于分割数时可能发生），则跳过
                    if not part_data:
                        continue

                    # 定义输出文件名，例如 part_1.txt, part_2.txt ...
                    output_file_name = f"part_{i + 1}.txt"
                    output_file_path = os.path.join(output_folder, output_file_name)

                    with open(output_file_path, "w", encoding='utf-8') as f:
                        f.write(part_data)
                
                print(f"'{file_name}' 已成功转换为 Base64 并分割成 {num_parts} 份，保存在 '{output_folder}'。")

            except Exception as e:
                print(f"处理图片 '{file_name}' 时发生错误: {e}")

# --- 如何使用 ---
if __name__ == "__main__":
    # 1. 请在这里指定你的图片文件夹路径
    # 例如: "C:/Users/YourUser/Pictures/my_photos" 或 "/Users/user/Desktop/images"
    folder_path = "./" 

    # 2. 请在这里指定你想要分割成的份数 (N)
    number_of_splits = 20  # 例如，这里设置为5份

    # 运行主函数
    batch_image_to_base64_and_split(folder_path, number_of_splits)
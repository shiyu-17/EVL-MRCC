import os

def rename_images(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否以 "002_" 开头
        if filename.startswith("003_"):
            # 构造新的文件名，将 "002_" 替换为 "003_"
            new_filename = "002_" + filename[4:]  # 从第5个字符开始保留原始内容
            # 获取完整的文件路径
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)
            # 重命名文件
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")

# 指定目标文件夹路径
folder_path = "/Users/shiyu/mycode/data/002/002_frames"  # 替换为你的文件夹路径
rename_images(folder_path)
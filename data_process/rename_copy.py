#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rename_and_copy.py

功能：
    - 读取源文件夹中所有形如 frame-xxxxxx.color.png 的文件
    - 去掉 ".color"，将 "frame-" 改为 "001_"
    - 将重命名后的文件复制到目标文件夹

用法：
    修改下面的 src_dir 和 dst_dir 为你自己的路径，然后：
    $ python3 rename_and_copy.py
"""

import os
import shutil

def process_images(src_dir: str, dst_dir: str):
    """
    遍历 src_dir，处理并复制符合条件的图片到 dst_dir。
    
    参数：
        src_dir: 源文件夹路径（7scenes 数据集中 seq1 的路径）
        dst_dir: 目标文件夹路径（如果不存在会自动创建）
    """
    # 如果目标文件夹不存在，则创建
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        print(f"已创建目标文件夹：{dst_dir}")

    # 遍历源文件夹
    for fname in os.listdir(src_dir):
        # 判断文件名是否以 frame- 开头并以 .color.png 结尾
        if  fname.endswith(".png"):
            src_path = os.path.join(src_dir, fname)

            # 加上前缀 '003_'
            new_name = "003_" + fname
            dst_path = os.path.join(dst_dir, new_name)

            # 复制文件
            shutil.copy2(src_path, dst_path)
            print(f"复制并重命名：{fname} → {new_name}")

if __name__ == "__main__":
    # TODO：根据实际情况修改下面两个路径
    src_dir = "/Users/shiyu/Downloads/rgbd_dataset_freiburg1_desk/rgbd_dataset_freiburg1_desk/rgb"       # 7scenes 数据集中 seq1 文件夹的路径
    dst_dir = "/Users/shiyu/mycode/data/003/003_frames"         # 重命名后图片要保存到的文件夹

    print("开始处理图片……")
    process_images(src_dir, dst_dir)
    print("处理完成！")

# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image

# 你的 mask 文件夹路径（训练集+验证集）
mask_dirs = [
    "C:/Users/LingZiheng/PycharmProjects/PythonProject/dataset/train/mask",
    "C:/Users/LingZiheng/PycharmProjects/PythonProject/dataset/val/mask"
]


def convert_gray_to_white(mask_path):
    """将 mask 中所有非黑像素（>0）转为全白（255）"""
    # 读取 mask
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)

    # 非黑像素（>0）转为 255（全白），背景保持 0（黑色）
    mask_np[mask_np > 0] = 255

    # 保存修改后的 mask（覆盖原文件，或修改路径保存新文件）
    modified_mask = Image.fromarray(mask_np.astype(np.uint8))
    modified_mask.save(mask_path)  # 覆盖原文件（建议先备份）
    # modified_mask.save(f"{os.path.splitext(mask_path)[0]}_white.png")  # 保存新文件，不覆盖原文件


if __name__ == "__main__":
    for dir_path in mask_dirs:
        print(f"\n正在处理文件夹：{dir_path}")
        mask_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        for idx, file_name in enumerate(mask_files, 1):
            file_path = os.path.join(dir_path, file_name)
            convert_gray_to_white(file_path)
            print(f"✅ 处理完成 {idx}/{len(mask_files)}：{file_name}")

    print(f"\n🎉 所有 mask 处理完成！浅灰裂缝已转为全白（像素值=255）")
# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image

mask_dir = "C:/Users/LingZiheng/PycharmProjects/PythonProject/dataset/train/mask"  # 你的mask目录

print("开始检查mask图像素值（正常：裂缝=255，背景=0）")
print("=" * 60)

for filename in os.listdir(mask_dir)[:10]:  # 检查前5张
    if filename.endswith(".png"):
        img_path = os.path.join(mask_dir, filename)
        img = Image.open(img_path).convert("L")  # 转为单通道
        img_np = np.array(img)

        # 打印像素统计
        max_pix = img_np.max()
        min_pix = img_np.min()
        non_zero_ratio = (img_np > 0).sum() / (img_np.shape[0] * img_np.shape[1])  # 非零像素占比（裂缝占比）

        print(f"文件：{filename}")
        print(f"  最大像素值：{max_pix}（正常应为255）")
        print(f"  最小像素值：{min_pix}（正常应为0）")
        print(f"  非零像素占比：{non_zero_ratio:.4f}（正常应>0且<1）")
        print("-" * 40)

print("检查完成！若最大像素值≠255或非零占比=0/1，说明mask标签异常")
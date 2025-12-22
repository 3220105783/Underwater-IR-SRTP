# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image

def fix_mask_background(mask_dir, target_background=0):
    """
    批量修复mask图背景：将像素值=1的背景转为0，保留裂缝区域（255）
    :param mask_dir: mask图文件夹路径
    :param target_background: 目标背景像素值（默认0）
    """
    if not os.path.exists(mask_dir):
        print(f"❌ 错误：文件夹不存在 → {mask_dir}")
        return

    fixed_count = 0
    # 遍历所有mask图
    for filename in os.listdir(mask_dir):
        if filename.endswith(".png"):  # 只处理png格式的mask图
            mask_path = os.path.join(mask_dir, filename)
            # 打开mask图（单通道灰度图）
            mask = Image.open(mask_path).convert("L")
            mask_np = np.array(mask)

            # 打印修复前的像素统计（验证问题）
            min_pix = mask_np.min()
            max_pix = mask_np.max()
            print(f"📊 修复前 - {filename}：最小像素={min_pix}，最大像素={max_pix}")

            # 核心修复：将像素值=1的区域转为0（背景），像素值=255的区域保留（裂缝）
            # 先将所有像素>1的转为255（确保裂缝是纯白色），像素≤1的转为0（纯黑色背景）
            mask_np[mask_np > 1] = 255  # 裂缝区域
            mask_np[mask_np <= 1] = target_background  # 背景区域

            # 转回Image格式并保存（覆盖原文件，建议先备份）
            fixed_mask = Image.fromarray(mask_np.astype(np.uint8))
            fixed_mask.save(mask_path)

            print(f"✅ 修复完成 - {filename}：最小像素={mask_np.min()}，最大像素={mask_np.max()}")
            fixed_count += 1
            print("-"*40)

    # 总结
    print("\n" + "="*50)
    print(f"🎉 批量修复完成！共处理 {fixed_count} 张mask图")
    print(f"🔧 修复内容：将背景像素1转为0，裂缝保留255")
    print(f"📁 处理文件夹：{mask_dir}")
    print("="*50)

if __name__ == "__main__":
    # 替换为你的mask文件夹路径（训练集和验证集都要处理）
    train_mask_dir = "C:/Users/LingZiheng/PycharmProjects/PythonProject/dataset/train/mask"
    val_mask_dir = "C:/Users/LingZiheng/PycharmProjects/PythonProject/dataset/val/mask"

    # 先处理训练集mask
    print("开始处理训练集mask图...")
    fix_mask_background(train_mask_dir)

    # 再处理验证集mask
    print("\n开始处理验证集mask图...")
    fix_mask_background(val_mask_dir)
# -*- coding: utf-8 -*-
import os
import argparse


def batch_add_suffix(folder_path, target_suffix, file_extensions=None, skip_suffix=None):
    """
    批量给文件添加后缀（在原文件名和扩展名之间插入）
    :param folder_path: 目标文件夹路径（必填）
    :param target_suffix: 要添加的后缀（如 "_img"、"_mask"，必填）
    :param file_extensions: 要处理的文件扩展名列表（默认：所有文件）
    :param skip_suffix: 已包含该后缀的文件跳过（避免重复添加，默认：None）
    """
    # 校验文件夹路径是否存在
    if not os.path.exists(folder_path):
        print(f"❌ 错误：文件夹路径不存在 → {folder_path}")
        return

    # 默认处理所有文件，若指定扩展名则过滤
    if file_extensions is None:
        file_extensions = []
    else:
        # 统一转为小写，避免大小写问题（如 .JPG → .jpg）
        file_extensions = [ext.lower() for ext in file_extensions]

    # 遍历文件夹下所有文件（不递归子文件夹）
    file_count = 0  # 统计处理的文件数
    for filename in os.listdir(folder_path):
        # 跳过子文件夹，只处理文件
        file_path = os.path.join(folder_path, filename)
        if os.path.isdir(file_path):
            continue

        # 分离文件名和扩展名（如 "776.rf.xxx.jpg" → ("776.rf.xxx", ".jpg")）
        file_name_without_ext, file_ext = os.path.splitext(filename)
        file_ext_lower = file_ext.lower()

        # 1. 按扩展名过滤（只处理指定类型的文件）
        if file_extensions and file_ext_lower not in file_extensions:
            continue

        # 2. 跳过已包含目标后缀的文件（避免重复添加，如 "776.rf.xxx_img.jpg" 不再处理）
        if skip_suffix and skip_suffix in file_name_without_ext:
            print(f"⚠️  跳过已含后缀的文件 → {filename}")
            continue

        # 3. 构造新文件名（原文件名 + 目标后缀 + 原扩展名）
        new_filename = f"{file_name_without_ext}{target_suffix}{file_ext}"
        new_file_path = os.path.join(folder_path, new_filename)

        # 4. 重命名文件（处理文件名重复的极端情况）
        if os.path.exists(new_file_path):
            print(f"❌ 跳过：新文件名已存在 → {new_filename}")
            continue

        # 执行重命名
        os.rename(file_path, new_file_path)
        print(f"✅ 已处理 → 原文件：{filename} → 新文件：{new_filename}")
        file_count += 1

    # 处理完成总结
    print("\n" + "=" * 50)
    if file_count > 0:
        print(f"🎉 批量添加后缀完成！共处理 {file_count} 个文件")
        print(f"📁 处理文件夹：{folder_path}")
        print(f"🔧 添加的后缀：{target_suffix}")
        if file_extensions:
            print(f"📌 处理的文件类型：{file_extensions}")
    else:
        print(f"ℹ️  未找到符合条件的文件，无需处理")
    print("=" * 50)


if __name__ == "__main__":
    # 解析命令行参数（方便直接运行时配置）
    parser = argparse.ArgumentParser(description="批量给文件添加后缀（在文件名和扩展名之间）")
    parser.add_argument("--folder", required=True, help="目标文件夹路径（如：C:/new_data/images）")
    parser.add_argument("--suffix", required=True, help="要添加的后缀（如：_img、_mask）")
    parser.add_argument("--ext", nargs="+", default=[], help="要处理的文件扩展名（如：.jpg .png，默认所有文件）")
    parser.add_argument("--skip", default=None, help="已包含该后缀则跳过（如：_img，避免重复添加）")

    args = parser.parse_args()

    # 调用函数执行批量添加后缀
    batch_add_suffix(
        folder_path=args.folder,
        target_suffix=args.suffix,
        file_extensions=args.ext,
        skip_suffix=args.skip
    )
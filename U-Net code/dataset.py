# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# -------------------------- 核心：统一尺寸为 512x512，适配所有输入图片 --------------------------
TARGET_SIZE = (512, 512)  # 统一目标尺寸

# -------------------------- 训练集 transforms（含数据增强，尺寸统一为512x512）--------------------------
train_transform_img = transforms.Compose([
    transforms.Resize(TARGET_SIZE),  # 强制 resize 为 512x512（无论原图尺寸）
    #transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转（数据增强）
    #transforms.RandomRotation(degrees=15),  # 随机旋转±15度（适应不同角度裂缝）
    #transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.9, 1.1)),  # 轻微缩放裁剪（增强鲁棒性）
    #transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # 高斯模糊（降低光照差异影响）
    #transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # 颜色抖动（适配风格差异）
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 标准化（提升训练稳定性）
                         std=[0.229, 0.224, 0.225])
])

train_transform_mask = transforms.Compose([
    transforms.Resize(TARGET_SIZE, interpolation=Image.NEAREST),  # mask 用最近邻插值（保持像素纯净）
    #transforms.RandomHorizontalFlip(p=0.5),  # 与原图同步翻转
    #transforms.RandomRotation(degrees=15),  # 与原图同步旋转
    #transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.9, 1.1), interpolation=Image.NEAREST),  # 与原图同步裁剪
    transforms.Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.float32)))
])

# -------------------------- 验证集 transforms（仅统一尺寸，无数据增强）--------------------------
val_transform_img = transforms.Compose([
    transforms.Resize(TARGET_SIZE),  # 验证集统一为 512x512
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform_mask = transforms.Compose([
    transforms.Resize(TARGET_SIZE, interpolation=Image.NEAREST),
    transforms.Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.float32)))
])

# 给 predict.py 预留别名
val_transform = val_transform_img


# -------------------------- 数据集类（统一尺寸+适配命名规则+无mask筛选）--------------------------
class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, is_train=True, oversample=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        self.oversample = oversample

        # 1. 加载所有文件（支持 jpg/jpeg/png/bmp 格式）
        img_suffixes = ('.jpg', '.jpeg', '.png', '.bmp')
        mask_suffixes = ('.jpg', '.jpeg', '.png', '.bmp')
        self.img_filenames = [f for f in os.listdir(img_dir) if f.lower().endswith(img_suffixes)]
        self.mask_filenames = [f for f in os.listdir(mask_dir) if f.lower().endswith(mask_suffixes)]

        # 2. 文件名匹配（核心：适配 xxx.jpg ↔ xxx_mask.jpg）
        self.common_filenames = []
        # 建立 mask 映射：key=xxx（前缀），value=mask原始文件名
        mask_name_map = {}
        for mask_name in self.mask_filenames:
            if "_mask" in mask_name.lower():  # 仅匹配含 "_mask" 的 mask 文件
                mask_prefix = mask_name.lower().split("_mask")[0]  # 提取前缀（如 "001_mask.jpg" → "001"）
                mask_name_map[mask_prefix] = mask_name

        # 匹配原图和 mask（原图前缀 == mask 前缀）
        for img_name in self.img_filenames:
            img_prefix = os.path.splitext(img_name.lower())[0]  # 原图前缀（如 "001.jpg" → "001"）
            if img_prefix in mask_name_map:
                # 保存（原图前缀，mask原始文件名），确保后续能正确加载
                self.common_filenames.append((img_prefix, mask_name_map[img_prefix]))

        # 3. 过采样（可选：训练集开启，提升样本量）
        if self.is_train and self.oversample:
            self.common_filenames = self._oversample_crack_samples()

        # 打印加载信息
        print(f"🔍 数据集加载详情：")
        print(f"   - 文件夹：{os.path.basename(img_dir)}")
        print(f"   - 原图总数：{len(self.img_filenames)}")
        print(f"   - mask总数：{len(self.mask_filenames)}")
        print(
            f"   - 匹配样本数：{len(self.common_filenames)}（{'含过采样' if self.is_train and self.oversample else '无过采样'}）")

        # 无匹配样本时报错（提示命名规则）
        if len(self.common_filenames) == 0:
            raise ValueError(
                "❌ 无任何匹配的样本！请检查：\n"
                "1. 原图命名：xxx.jpg（支持 jpg/png 等）\n"
                "2. mask命名：xxx_mask.jpg（需与原图前缀一致）\n"
                "3. 示例：原图 001.jpg → mask 001_mask.jpg"
            )

    def _oversample_crack_samples(self):
        """过采样：所有匹配样本重复2次（适配小数据集）"""
        oversampled = []
        for item in self.common_filenames:
            oversampled.extend([item] * 2)  # 重复2次，可改为3次（根据需求调整）
        return oversampled

    def __len__(self):
        """返回样本总数"""
        return len(self.common_filenames)

    def __getitem__(self, idx):
        """加载单样本（自动统一为 512x512）"""
        # 获取当前样本的（原图前缀，mask原始文件名）
        img_prefix, mask_name = self.common_filenames[idx]

        # 1. 加载原图（自动适配所有支持的后缀）
        img_path = None
        for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            temp_path = os.path.join(self.img_dir, f"{img_prefix}{ext}")
            if os.path.exists(temp_path):
                img_path = temp_path
                break
        if not img_path:
            raise FileNotFoundError(f"❌ 未找到原图：{img_prefix}（支持后缀：jpg/jpeg/png/bmp）")

        # 2. 加载 mask（直接用匹配到的原始文件名）
        mask_path = os.path.join(self.mask_dir, mask_name)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"❌ 未找到 mask 文件：{mask_name}")

        # 3. 读取图片
        image = Image.open(img_path).convert("RGB")  # 原图转为 RGB 通道
        mask = Image.open(mask_path).convert("L")  # mask 转为单通道灰度图

        # 4. 应用 transforms（训练集含增强，验证集仅 resize）
        if self.is_train:
            image = train_transform_img(image)
            mask = train_transform_mask(mask)
        else:
            image = val_transform_img(image)
            mask = val_transform_mask(mask)

        # 5. mask 二值化（非黑像素>0 → 1，背景→0，添加通道维度适配模型）
        mask = (mask > 0).float().unsqueeze(0)

        return image, mask
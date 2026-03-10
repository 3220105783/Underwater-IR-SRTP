# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from PIL import Image

# 统一尺寸（与原代码一致）
TARGET_SIZE = (768, 768)

# ImageNet均值/标准差（TF1.x标准化）
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def preprocess_img(img, is_train=True):
    """图片预处理（TF1.x版）"""
    # 转为float32并归一化到[0,1]
    img = tf.cast(img, tf.float32) / 255.0

    # 调整尺寸
    img = tf.image.resize_images(img, TARGET_SIZE, method=tf.image.ResizeMethod.BILINEAR)

    # 标准化（ImageNet）
    img = (img - MEAN) / STD

    # 训练集数据增强（可选，与原代码注释对应）
    if is_train:
        # img = tf.image.random_flip_left_right(img)  # 水平翻转
        # img = tf.image.random_brightness(img, max_delta=0.3)  # 亮度抖动
        # img = tf.image.random_contrast(img, lower=0.7, upper=1.3)  # 对比度抖动
        pass

    return img


def preprocess_mask(mask):
    """Mask预处理（TF1.x版）"""
    # 调整尺寸（最近邻插值）
    mask = tf.image.resize_images(mask, TARGET_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # 二值化（>0 → 1，背景→0）
    mask = tf.cast(mask > 0, tf.float32)

    # 添加通道维度 (H,W) → (H,W,1)
    mask = tf.expand_dims(mask, axis=-1)
    return mask


def load_image_mask_pair(img_path, mask_path, is_train=True):
    """加载单对图片+mask（TF1.x版）"""
    # 读取图片
    img_raw = tf.read_file(img_path)
    img = tf.image.decode_image(img_raw, channels=3)
    img.set_shape([None, None, 3])
    img = preprocess_img(img, is_train)

    # 读取mask
    mask_raw = tf.read_file(mask_path)
    mask = tf.image.decode_image(mask_raw, channels=1)
    mask.set_shape([None, None, 1])
    mask = preprocess_mask(mask)

    return img, mask


def build_dataset(img_dir, mask_dir, batch_size=1, is_train=True, shuffle=True):
    """构建TF1.x Dataset（替代PyTorch DataLoader）"""
    # 匹配文件名（xxx.jpg ↔ xxx_mask.png）
    img_suffixes = ('.jpg', '.jpeg', '.png', '.bmp')
    mask_map = {}

    # 构建mask映射表
    for mask_name in os.listdir(mask_dir):
        if '_mask' in mask_name.lower() and mask_name.lower().endswith(img_suffixes):
            mask_prefix = mask_name.lower().split('_mask')[0]
            mask_map[mask_prefix] = os.path.join(mask_dir, mask_name)

    # 收集匹配的图片路径
    img_paths = []
    mask_paths = []
    for img_name in os.listdir(img_dir):
        if img_name.lower().endswith(img_suffixes):
            img_prefix = os.path.splitext(img_name.lower())[0]
            if img_prefix in mask_map:
                img_paths.append(os.path.join(img_dir, img_name))
                mask_paths.append(mask_map[img_prefix])

    if len(img_paths) == 0:
        raise ValueError("No matching samples found! Please check the naming convention:  image xxx.jpg → mask xxx_mask.png")

    # 构建TF Dataset
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))

    # 加载+预处理
    dataset = dataset.map(
        lambda x, y: load_image_mask_pair(x, y, is_train),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # 训练集配置
    if is_train:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(img_paths))
        dataset = dataset.repeat()  # 无限重复（训练时控制步数）
        dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
        dataset = dataset.batch(batch_size, drop_remainder=False)

    # 预取数据（提升性能）
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # 创建迭代器
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    return iterator, next_batch, len(img_paths)
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

# 数据增强配置
AUG_CONFIG = {
    'random_flip_left_right': True,  # 水平翻转
    'random_flip_up_down': True,  # 垂直翻转
    'random_brightness': False,  # 亮度调整
    'brightness_max_delta': 0.2,  # 亮度调整范围
    'random_contrast': False,  # 对比度调整
    'contrast_lower': 0.8,  # 对比度下限
    'contrast_upper': 1.2,  # 对比度上限
    'random_hue': False,  # 色调调整（仅RGB）
    'hue_max_delta': 0.1,  # 色调调整范围
    'random_saturation': False,  # 饱和度调整
    'saturation_lower': 0.8,  # 饱和度下限
    'saturation_upper': 1.2,  # 饱和度上限
    'random_rotation': True,  # 随机旋转
    'rotation_angle': 15,  # 最大旋转角度（±15°）
    'random_zoom': True,  # 随机缩放
    'zoom_lower': 0.5,  # 缩放下限
    'zoom_upper': 1.5,  # 缩放上限
    'random_crop': True,  # 随机裁剪后resize
    'noise_aug': True,  # 高斯噪声
    'noise_std': 0.01  # 噪声标准差
}


def get_random_aug_params(is_train):
    """生成共享的随机增强参数（确保img和mask同步）"""
    if not is_train:
        return {}

    aug_params = {}
    # 1. 水平翻转参数
    if AUG_CONFIG['random_flip_left_right']:
        aug_params['flip_left_right'] = tf.random_uniform([], 0, 1) > 0.5
    # 2. 垂直翻转参数
    if AUG_CONFIG['random_flip_up_down']:
        aug_params['flip_up_down'] = tf.random_uniform([], 0, 1) > 0.5
    # 3. 旋转角度参数
    if AUG_CONFIG['random_rotation']:
        aug_params['rotation_angle'] = tf.random_uniform([], -AUG_CONFIG['rotation_angle'],
                                                         AUG_CONFIG['rotation_angle'])
    # 4. 缩放因子参数
    if AUG_CONFIG['random_zoom']:
        aug_params['zoom_scale'] = tf.random_uniform([], AUG_CONFIG['zoom_lower'], AUG_CONFIG['zoom_upper'])
    # 5. 亮度调整参数（仅img）
    if AUG_CONFIG['random_brightness']:
        aug_params['brightness_delta'] = tf.random_uniform([], -AUG_CONFIG['brightness_max_delta'],
                                                           AUG_CONFIG['brightness_max_delta'])
    # 6. 对比度调整参数（仅img）
    if AUG_CONFIG['random_contrast']:
        aug_params['contrast_factor'] = tf.random_uniform([], AUG_CONFIG['contrast_lower'],
                                                          AUG_CONFIG['contrast_upper'])
    # 7. 色调调整参数（仅img）
    if AUG_CONFIG['random_hue']:
        aug_params['hue_delta'] = tf.random_uniform([], -AUG_CONFIG['hue_max_delta'], AUG_CONFIG['hue_max_delta'])
    # 8. 饱和度调整参数（仅img）
    if AUG_CONFIG['random_saturation']:
        aug_params['saturation_factor'] = tf.random_uniform([], AUG_CONFIG['saturation_lower'],
                                                            AUG_CONFIG['saturation_upper'])

    return aug_params


def apply_rotation(image, angle, is_mask=False):
    """应用旋转（共享角度参数）"""
    interpolation = 'NEAREST' if is_mask else 'BILINEAR'
    image = tf.contrib.image.rotate(image, angle * np.pi / 180, interpolation=interpolation)
    return image


def apply_zoom(image, target_size, scale, is_mask=False):
    """应用缩放（共享缩放因子参数）"""
    # 计算缩放后的尺寸
    h = tf.cast(target_size[0] * scale, tf.int32)
    w = tf.cast(target_size[1] * scale, tf.int32)
    # 缩放
    resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR if is_mask else tf.image.ResizeMethod.BILINEAR
    image = tf.image.resize_images(image, [h, w], method=resize_method)
    # 裁剪或填充回原尺寸
    if scale > 1.0:
        image = tf.image.random_crop(image, [target_size[0], target_size[1], tf.shape(image)[-1]])
    else:
        pad_h = (target_size[0] - h) // 2
        pad_w = (target_size[1] - w) // 2
        image = tf.pad(image, [[pad_h, target_size[0] - h - pad_h],
                               [pad_w, target_size[1] - w - pad_w], [0, 0]])
    return image


def preprocess_img(img, aug_params, is_train=True):
    """图片预处理（使用共享增强参数 TF1.x）"""
    # 转为float32并归一化到[0,1]
    img = tf.cast(img, tf.float32) / 255.0

    # 训练集数据增强（使用共享参数）
    if is_train:
        # 1. 水平翻转（共享参数）
        if AUG_CONFIG['random_flip_left_right'] and aug_params.get('flip_left_right', False):
            img = tf.image.flip_left_right(img)

        # 2. 垂直翻转（共享参数）
        if AUG_CONFIG['random_flip_up_down'] and aug_params.get('flip_up_down', False):
            img = tf.image.flip_up_down(img)

        # 3. 随机亮度调整（共享参数）
        if AUG_CONFIG['random_brightness']:
            img = tf.image.adjust_brightness(img, delta=aug_params['brightness_delta'])

        # 4. 随机对比度调整（共享参数）
        if AUG_CONFIG['random_contrast']:
            img = tf.image.adjust_contrast(img, contrast_factor=aug_params['contrast_factor'])

        # 5. 随机色调调整（仅RGB图像有效，共享参数）
        if AUG_CONFIG['random_hue']:
            img = tf.image.adjust_hue(img, delta=aug_params['hue_delta'])

        # 6. 随机饱和度调整（共享参数）
        if AUG_CONFIG['random_saturation']:
            img = tf.image.adjust_saturation(img, saturation_factor=aug_params['saturation_factor'])

        # 7. 随机旋转（共享角度参数）
        if AUG_CONFIG['random_rotation']:
            img = apply_rotation(img, aug_params['rotation_angle'], is_mask=False)

        # 8. 随机缩放（共享缩放因子参数）
        if AUG_CONFIG['random_zoom']:
            img = apply_zoom(img, TARGET_SIZE, aug_params['zoom_scale'], is_mask=False)

        # 9. 添加高斯噪声
        if AUG_CONFIG['noise_aug']:
            noise = tf.random_normal(shape=tf.shape(img), mean=0.0, stddev=AUG_CONFIG['noise_std'], dtype=tf.float32)
            img = img + noise

        # 裁剪到有效范围（防止增强后超出[0,1]）
        img = tf.clip_by_value(img, 0.0, 1.0)

    # 调整尺寸（最终resize到目标尺寸）
    img = tf.image.resize_images(img, TARGET_SIZE, method=tf.image.ResizeMethod.BILINEAR)

    # 标准化（ImageNet）
    img = (img - MEAN) / STD

    return img


def preprocess_mask(mask, aug_params, is_train=True):
    """Mask预处理（使用共享增强参数 TF1.x版）"""
    # 转为float32
    mask = tf.cast(mask, tf.float32)

    # 训练集mask同步增强（与图片增强保持完全一致）
    if is_train:
        # 1. 水平翻转（共享参数）
        if AUG_CONFIG['random_flip_left_right'] and aug_params.get('flip_left_right', False):
            mask = tf.image.flip_left_right(mask)

        # 2. 垂直翻转（共享参数）
        if AUG_CONFIG['random_flip_up_down'] and aug_params.get('flip_up_down', False):
            mask = tf.image.flip_up_down(mask)

        # 3. 随机旋转（共享角度参数）
        if AUG_CONFIG['random_rotation']:
            mask = apply_rotation(mask, aug_params['rotation_angle'], is_mask=True)

        # 4. 随机缩放（共享缩放因子参数）
        if AUG_CONFIG['random_zoom']:
            mask = apply_zoom(mask, TARGET_SIZE, aug_params['zoom_scale'], is_mask=True)

    # 调整尺寸（最终resize到目标尺寸）
    mask = tf.image.resize_images(mask, TARGET_SIZE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # 二值化（>0 → 1，背景→0）
    mask = tf.cast(mask > 0, tf.float32)

    # 添加通道维度 (H,W) → (H,W,1)
    mask = tf.expand_dims(mask, axis=-1)
    return mask


def load_image_mask_pair(img_path, mask_path, is_train=True):
    """加载单对图片+mask（同步增强 TF1.x版）"""
    # 读取图片
    img_raw = tf.read_file(img_path)
    img = tf.image.decode_image(img_raw, channels=3)
    img.set_shape([None, None, 3])

    # 读取mask
    mask_raw = tf.read_file(mask_path)
    mask = tf.image.decode_image(mask_raw, channels=1)
    mask.set_shape([None, None, 1])

    # 生成共享的增强参数（核心：确保img和mask用同一组随机参数）
    aug_params = get_random_aug_params(is_train)

    # 预处理（传入共享参数）
    img = preprocess_img(img, aug_params, is_train)
    mask = preprocess_mask(mask, aug_params, is_train)

    return img, mask


def build_dataset(img_dir, mask_dir, batch_size=1, is_train=True, shuffle=True):
    """构建TF1.x Dataset（替代PyTorch DataLoader）"""
    # 匹配文件名（xxx.png ↔ xxx_mask.png）
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
        raise ValueError(
            "No matching samples found! Please check the naming convention:  image xxx.png → mask xxx_mask.png")

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
# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from config import AUG_CONFIG, INPUT_SIZE

# 统一尺寸
TARGET_SIZE = INPUT_SIZE
TARGET_H, TARGET_W = TARGET_SIZE

# ImageNet均值/标准差
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_random_aug_params(is_train):
    """生成共享的随机增强参数"""
    if not is_train:
        return {'selected_method': None}

    aug_params = {}
    # 筛选启用的候选方法
    enabled_methods = []
    for method in AUG_CONFIG['all_aug_methods']:
        if AUG_CONFIG.get(f'enable_{method}', False):
            enabled_methods.append(method)

    # 随机选1种方法
    if enabled_methods:
        aug_params['selected_method'] = tf.random_shuffle(enabled_methods)[0]
    else:
        aug_params['selected_method'] = None

    # 生成对应参数
    selected_method = aug_params['selected_method']
    if selected_method == 'flip_left_right':
        aug_params['flip_left_right'] = tf.random_uniform([], 0, 1) > 0.5
    elif selected_method == 'flip_up_down':
        aug_params['flip_up_down'] = tf.random_uniform([], 0, 1) > 0.5
    elif selected_method == 'rotation':
        aug_params['rotation_angle'] = tf.random_uniform([], -AUG_CONFIG['rotation_angle'],
                                                         AUG_CONFIG['rotation_angle'])
    elif selected_method == 'zoom':
        aug_params['zoom_scale'] = tf.random_uniform([], AUG_CONFIG['zoom_lower'], AUG_CONFIG['zoom_upper'])
        aug_params['crop_offset_h'] = tf.random_uniform([], 0, 1)
        aug_params['crop_offset_w'] = tf.random_uniform([], 0, 1)
    elif selected_method == 'brightness':
        aug_params['brightness_delta'] = tf.random_uniform([], -AUG_CONFIG['brightness_max_delta'],
                                                           AUG_CONFIG['brightness_max_delta'])
    elif selected_method == 'contrast':
        aug_params['contrast_factor'] = tf.random_uniform([], AUG_CONFIG['contrast_lower'],
                                                          AUG_CONFIG['contrast_upper'])
    elif selected_method == 'hue':
        aug_params['hue_delta'] = tf.random_uniform([], -AUG_CONFIG['hue_max_delta'], AUG_CONFIG['hue_max_delta'])
    elif selected_method == 'saturation':
        aug_params['saturation_factor'] = tf.random_uniform([], AUG_CONFIG['saturation_lower'],
                                                            AUG_CONFIG['saturation_upper'])

    return aug_params


def apply_rotation(image, angle, is_mask=False):
    """应用旋转"""
    interpolation = 'NEAREST' if is_mask else 'BILINEAR'
    image = tf.contrib.image.rotate(image, angle * np.pi / 180, interpolation=interpolation)
    image = tf.image.resize_with_crop_or_pad(image, TARGET_H, TARGET_W)
    # 显式设置shape
    image.set_shape([TARGET_H, TARGET_W, tf.shape(image)[-1]])
    return image


def apply_zoom(image, scale, crop_offset_h, crop_offset_w, is_mask=False):
    """应用缩放"""
    scaled_h = tf.cast(tf.round(TARGET_H * scale), tf.int32)
    scaled_w = tf.cast(tf.round(TARGET_W * scale), tf.int32)

    resize_method = tf.image.ResizeMethod.NEAREST_NEIGHBOR if is_mask else tf.image.ResizeMethod.BILINEAR
    image = tf.image.resize_images(image, [scaled_h, scaled_w], method=resize_method)

    if scale > 1.0:
        max_offset_h = tf.maximum(scaled_h - TARGET_H, 0)
        max_offset_w = tf.maximum(scaled_w - TARGET_W, 0)
        offset_h = tf.cast(crop_offset_h * tf.cast(max_offset_h, tf.float32), tf.int32)
        offset_w = tf.cast(crop_offset_w * tf.cast(max_offset_w, tf.float32), tf.int32)
        image = image[offset_h:offset_h + TARGET_H, offset_w:offset_w + TARGET_W, :]
    else:
        image = tf.image.resize_with_crop_or_pad(image, TARGET_H, TARGET_W)

    image.set_shape([TARGET_H, TARGET_W, tf.shape(image)[-1]])
    return image


def preprocess_img(img, aug_params, is_train=True):
    """图片预处理"""
    # 确保img有基础shape（高宽暂时为None，通道固定为3）
    img.set_shape([None, None, 3])
    img = tf.cast(img, tf.float32) / 255.0
    selected_method = aug_params.get('selected_method')

    if is_train and selected_method is not None:
        if selected_method == 'flip_left_right':
            img = tf.image.flip_left_right(img)
        elif selected_method == 'flip_up_down':
            img = tf.image.flip_up_down(img)
        elif selected_method == 'rotation':
            img = apply_rotation(img, aug_params['rotation_angle'], is_mask=False)
        elif selected_method == 'zoom':
            img = apply_zoom(img, aug_params['zoom_scale'], aug_params['crop_offset_h'],
                             aug_params['crop_offset_w'], is_mask=False)
        elif selected_method == 'brightness':
            img = tf.image.adjust_brightness(img, delta=aug_params['brightness_delta'])
        elif selected_method == 'contrast':
            img = tf.image.adjust_contrast(img, contrast_factor=aug_params['contrast_factor'])
        elif selected_method == 'hue':
            img = tf.image.adjust_hue(img, delta=aug_params['hue_delta'])
        elif selected_method == 'saturation':
            img = tf.image.adjust_saturation(img, saturation_factor=aug_params['saturation_factor'])
        elif selected_method == 'noise':
            noise = tf.random_normal(shape=tf.shape(img), mean=0.0,
                                     stddev=AUG_CONFIG['noise_std'], dtype=tf.float32)
            img = img + noise

        img = tf.clip_by_value(img, 0.0, 1.0)

    # 最终预处理（显式指定resize方法，确保shape）
    img = tf.image.resize_images(img, [TARGET_H, TARGET_W], method=tf.image.ResizeMethod.BILINEAR)
    img = (img - MEAN) / STD
    img.set_shape([TARGET_H, TARGET_W, 3])
    return img


def preprocess_mask(mask, aug_params, is_train=True):
    """Mask预处理（补全逻辑）"""
    # 确保mask有基础shape（高宽暂时为None，通道固定为1）
    mask.set_shape([None, None, 1])
    # 基础预处理：归一化到[0,1]
    mask = tf.cast(mask, tf.float32) / 255.0
    selected_method = aug_params.get('selected_method')

    # 仅执行同步增强方法（img/mask同步）
    if is_train and selected_method is not None:
        if selected_method == 'flip_left_right':
            mask = tf.image.flip_left_right(mask)
        elif selected_method == 'flip_up_down':
            mask = tf.image.flip_up_down(mask)
        elif selected_method == 'rotation':
            mask = apply_rotation(mask, aug_params['rotation_angle'], is_mask=True)
        elif selected_method == 'zoom':
            mask = apply_zoom(mask, aug_params['zoom_scale'], aug_params['crop_offset_h'],
                              aug_params['crop_offset_w'], is_mask=True)

    # 最终resize + 确保维度正确
    mask = tf.image.resize_images(mask, [TARGET_H, TARGET_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.expand_dims(mask, axis=-1)  # 增加通道维度
    mask.set_shape([TARGET_H, TARGET_W, 1])
    return mask


def load_img(img_path, mask_path, target_size=(768, 768), is_train=True):
    """修复扩展名提取逻辑 + 兼容TF1.x + 维度强制保证"""
    # ========== 核心修复：提取文件扩展名 ==========
    split_result = tf.string_split([img_path], '.')  # 返回SparseTensor
    img_ext = tf.sparse.to_dense(split_result)[0][-1]  # 转为密集张量后取最后一个元素

    # 读取图片并保证维度
    img = tf.read_file(img_path)
    if img_ext == b'jpg' or img_ext == b'jpeg':
        img = tf.image.decode_jpeg(img, channels=3)
    elif img_ext == b'png':
        img = tf.image.decode_png(img, channels=3)
    # 强制保证img是3维 (H, W, 3)，避免维度异常
    img = tf.ensure_shape(img, [None, None, 3])
    # 调整尺寸
    img = tf.image.resize_images(img, target_size, method=tf.image.ResizeMethod.BILINEAR)

    # 读取mask并保证维度
    mask = tf.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    # 强制保证mask是3维 (H, W, 1)
    mask = tf.ensure_shape(mask, [None, None, 1])
    # 调整尺寸（最近邻插值）
    mask = tf.image.resize_images(mask, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.where(mask > 0.5, 1.0, 0.0)  # 二值化

    return img, mask


def build_dataset(img_dir, mask_dir, batch_size, is_train=True):
    """构建TF1.x数据集迭代器（完整逻辑）"""
    # 获取文件列表
    img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
    mask_paths = [os.path.join(mask_dir, f.replace('.jpg', '_mask.png').replace('.png', '_mask.png'))
                  for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

    # 构建数据集
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))

    # 训练集打乱 + 重复
    if is_train:
        dataset = dataset.shuffle(buffer_size=len(img_paths))
        dataset = dataset.repeat()

    # 修复：通过lambda传递target_size和is_train参数给load_img
    dataset = dataset.map(lambda img_path, mask_path: load_img(
        img_path, mask_path, target_size=TARGET_SIZE, is_train=is_train
    ), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # 生成增强参数
    dataset = dataset.map(lambda img, mask: (img, mask, get_random_aug_params(is_train)),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # 预处理
    dataset = dataset.map(lambda img, mask, aug_params:
                          (preprocess_img(img, aug_params, is_train),
                           preprocess_mask(mask, aug_params, is_train)),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # 批处理 + 预取
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # 构建迭代器
    iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    next_batch = iterator.get_next()
    init_op = iterator.make_initializer(dataset)

    return init_op, next_batch, len(img_paths)
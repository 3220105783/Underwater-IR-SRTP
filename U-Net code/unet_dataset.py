# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from PIL import Image

# 统一尺寸
TARGET_SIZE = (768, 768)
TARGET_H, TARGET_W = TARGET_SIZE

# ImageNet均值/标准差
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 数据增强配置
# 核心：定义9种候选方法 + 每种方法的开关/参数
AUG_CONFIG = {
    # 9种候选增强方法（按需求调整顺序，确保总数为9）
    'all_aug_methods': [
        'flip_left_right',  # 1. 水平翻转（同步）
        'flip_up_down',  # 2. 垂直翻转（同步）
        'rotation',  # 3. 随机旋转（同步）
        'zoom',  # 4. 随机缩放（同步）
        'brightness',  # 5. 亮度调整（仅img）
        'contrast',  # 6. 对比度调整（仅img）
        'hue',  # 7. 色调调整（仅img）
        'saturation',  # 8. 饱和度调整（仅img）
        'noise'  # 9. 高斯噪声（仅img）
    ],
    # 方法开关（控制是否纳入候选池）
    'enable_flip_left_right': True,
    'enable_flip_up_down': True,
    'enable_rotation': True,
    'enable_zoom': False,
    'enable_brightness': False,
    'enable_contrast': False,
    'enable_hue': False,
    'enable_saturation': False,
    'enable_noise': False,
    # 方法参数
    'rotation_angle': 15,  # 旋转最大角度
    'zoom_lower': 0.5,  # 缩放下限
    'zoom_upper': 1.5,  # 缩放上限
    'brightness_max_delta': 0.2,  # 亮度调整范围
    'contrast_lower': 0.8,  # 对比度下限
    'contrast_upper': 1.2,  # 对比度上限
    'hue_max_delta': 0.1,  # 色调调整范围
    'saturation_lower': 0.8,  # 饱和度下限
    'saturation_upper': 1.2,  # 饱和度上限
    'noise_std': 0.01  # 噪声标准差
}


def get_random_aug_params(is_train):
    """
    生成共享的随机增强参数（核心：严格选1种方法，仅生成该方法的参数）
    返回：
        aug_params: dict，包含selected_method（选中的方法） + 该方法的随机参数
    """
    if not is_train:
        return {'selected_method': None}

    aug_params = {}
    # 第一步：筛选启用的候选方法（从9种中挑出开启的）
    enabled_methods = []
    for method in AUG_CONFIG['all_aug_methods']:
        if AUG_CONFIG.get(f'enable_{method}', False):
            enabled_methods.append(method)

    # 第二步：严格随机选1种方法（共享！确保img/mask选同一个）
    if enabled_methods:
        # 随机打乱后取第一个，确保均匀选择
        aug_params['selected_method'] = tf.random_shuffle(enabled_methods)[0]
    else:
        aug_params['selected_method'] = None

    # 第三步：仅为选中的方法生成对应参数（避免冗余，提升效率）
    selected_method = aug_params['selected_method']

    # 同步方法参数（img/mask共用）
    if selected_method == 'flip_left_right':
        aug_params['flip_left_right'] = tf.random_uniform([], 0, 1) > 0.5

    elif selected_method == 'flip_up_down':
        aug_params['flip_up_down'] = tf.random_uniform([], 0, 1) > 0.5

    elif selected_method == 'rotation':
        aug_params['rotation_angle'] = tf.random_uniform([], -AUG_CONFIG['rotation_angle'], AUG_CONFIG['rotation_angle'])

    elif selected_method == 'zoom':
        aug_params['zoom_scale'] = tf.random_uniform([], AUG_CONFIG['zoom_lower'], AUG_CONFIG['zoom_upper'])
        aug_params['crop_offset_h'] = tf.random_uniform([], 0, 1)
        aug_params['crop_offset_w'] = tf.random_uniform([], 0, 1)

    # 仅img方法参数（mask不执行）
    elif selected_method == 'brightness':
        aug_params['brightness_delta'] = tf.random_uniform([], -AUG_CONFIG['brightness_max_delta'], AUG_CONFIG['brightness_max_delta'])

    elif selected_method == 'contrast':
        aug_params['contrast_factor'] = tf.random_uniform([], AUG_CONFIG['contrast_lower'], AUG_CONFIG['contrast_upper'])

    elif selected_method == 'hue':
        aug_params['hue_delta'] = tf.random_uniform([], -AUG_CONFIG['hue_max_delta'], AUG_CONFIG['hue_max_delta'])

    elif selected_method == 'saturation':
        aug_params['saturation_factor'] = tf.random_uniform([], AUG_CONFIG['saturation_lower'], AUG_CONFIG['saturation_upper'])

    elif selected_method == 'noise':
        # 噪声参数无需提前生成，执行时直接用配置值即可
        pass

    return aug_params


def apply_rotation(image, angle, is_mask=False):
    """应用旋转（同步方法）"""
    interpolation = 'NEAREST' if is_mask else 'BILINEAR'
    image = tf.contrib.image.rotate(image, angle * np.pi / 180, interpolation=interpolation)
    image = tf.image.resize_with_crop_or_pad(image, TARGET_H, TARGET_W)
    return image


def apply_zoom(image, scale, crop_offset_h, crop_offset_w, is_mask=False):
    """应用缩放（同步方法）"""
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
    """
    图片预处理（严格执行选中的1种方法）
    逻辑：仅执行selected_method对应的变换，其余所有方法都不执行
    """
    # 基础预处理：归一化到[0,1]
    img = tf.cast(img, tf.float32) / 255.0
    selected_method = aug_params.get('selected_method')

    # 训练集：仅执行选中的1种增强方法
    if is_train and selected_method is not None:
        # 1. 水平翻转（同步方法）
        if selected_method == 'flip_left_right':
            img = tf.image.flip_left_right(img)
        # 2. 垂直翻转（同步方法）
        elif selected_method == 'flip_up_down':
            img = tf.image.flip_up_down(img)
        # 3. 随机旋转（同步方法）
        elif selected_method == 'rotation':
            img = apply_rotation(img, aug_params['rotation_angle'], is_mask=False)
        # 4. 随机缩放（同步方法）
        elif selected_method == 'zoom':
            img = apply_zoom(img,
                             aug_params['zoom_scale'],
                             aug_params['crop_offset_h'],
                             aug_params['crop_offset_w'],
                             is_mask=False)
        # 5. 亮度调整（仅img方法）
        elif selected_method == 'brightness':
            img = tf.image.adjust_brightness(img, delta=aug_params['brightness_delta'])
        # 6. 对比度调整（仅img方法）
        elif selected_method == 'contrast':
            img = tf.image.adjust_contrast(img, contrast_factor=aug_params['contrast_factor'])
        # 7. 色调调整（仅img方法）
        elif selected_method == 'hue':
            img = tf.image.adjust_hue(img, delta=aug_params['hue_delta'])
        # 8. 饱和度调整（仅img方法）
        elif selected_method == 'saturation':
            img = tf.image.adjust_saturation(img, saturation_factor=aug_params['saturation_factor'])
        # 9. 高斯噪声（仅img方法）
        elif selected_method == 'noise':
            noise = tf.random_normal(shape=tf.shape(img),
                                     mean=0.0,
                                     stddev=AUG_CONFIG['noise_std'],
                                     dtype=tf.float32)
            img = img + noise

        # 边界约束：确保像素值在[0,1]范围内
        img = tf.clip_by_value(img, 0.0, 1.0)

    # 最终兜底：强制resize到768x768 + 标准化
    img = tf.image.resize_images(img, [TARGET_H, TARGET_W], method=tf.image.ResizeMethod.BILINEAR)
    img = (img - MEAN) / STD
    img.set_shape([TARGET_H, TARGET_W, 3])
    return img


def preprocess_mask(mask, aug_params, is_train=True):
    """
    Mask预处理（严格同步img的选中方法：仅执行同步方法，仅img方法不执行）
    逻辑：
        - 若选中的是同步方法（1-4）：执行对应变换
        - 若选中的是仅img方法（5-9）：不做任何变换
    """
    # 基础预处理：转为float32
    mask = tf.cast(mask, tf.float32)
    selected_method = aug_params.get('selected_method')

    # 训练集：仅执行同步类增强方法（和img严格同步）
    if is_train and selected_method is not None:
        # 1. 水平翻转（同步）
        if selected_method == 'flip_left_right':
            mask = tf.image.flip_left_right(mask)
        # 2. 垂直翻转（同步）
        elif selected_method == 'flip_up_down':
            mask = tf.image.flip_up_down(mask)
        # 3. 随机旋转（同步）
        elif selected_method == 'rotation':
            mask = apply_rotation(mask, aug_params['rotation_angle'], is_mask=True)
        # 4. 随机缩放（同步）
        elif selected_method == 'zoom':
            mask = apply_zoom(mask,
                              aug_params['zoom_scale'],
                              aug_params['crop_offset_h'],
                              aug_params['crop_offset_w'],
                              is_mask=True)
        # 选中仅img方法（5-9）：不执行任何变换，直接跳过

    # 最终兜底：强制resize到768x768 + 二值化 + 增加通道维度
    mask = tf.image.resize_images(mask, [TARGET_H, TARGET_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.cast(mask > 0, tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    mask.set_shape([TARGET_H, TARGET_W, 1])
    return mask


def load_image_mask_pair(img_path, mask_path, is_train=True):
    """加载单对img/mask（核心：共享aug_params，确保选同一种方法）"""
    # 读取img
    img_raw = tf.read_file(img_path)
    img = tf.image.decode_image(img_raw, channels=3)
    img.set_shape([None, None, 3])

    # 读取mask
    mask_raw = tf.read_file(mask_path)
    mask = tf.image.decode_image(mask_raw, channels=1)
    mask.set_shape([None, None, 1])

    # 生成共享增强参数（仅选1种方法，仅生成该方法参数）
    aug_params = get_random_aug_params(is_train)

    # 预处理（严格同步）
    img = preprocess_img(img, aug_params, is_train)
    mask = preprocess_mask(mask, aug_params, is_train)

    return img, mask


def build_dataset(img_dir, mask_dir, batch_size=1, is_train=True, shuffle=True):
    """构建TF1.x Dataset"""
    img_suffixes = ('.jpg', '.jpeg', '.png', '.bmp')
    mask_map = {}

    # 构建mask映射表（xxx.png ↔ xxx_mask.png）
    for mask_name in os.listdir(mask_dir):
        if '_mask' in mask_name.lower() and mask_name.lower().endswith(img_suffixes):
            mask_prefix = mask_name.lower().split('_mask')[0]
            mask_map[mask_prefix] = os.path.join(mask_dir, mask_name)

    # 收集匹配的img/mask路径
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
            "No matching img/mask found! Please check the naming convention: img should be like xxx.png, mask should be like xxx_mask.png.")

    # 构建Dataset
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    dataset = dataset.map(
        lambda x, y: load_image_mask_pair(x, y, is_train),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    # 训练集配置
    if is_train:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(img_paths))
        dataset = dataset.repeat()  # 无限重复
        dataset = dataset.batch(batch_size, drop_remainder=True)
    else:
        dataset = dataset.batch(batch_size, drop_remainder=False)

    # 预取数据提升性能
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    iterator = dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    return iterator, next_batch, len(img_paths)
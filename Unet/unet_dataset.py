import os
import numpy as np
import tensorflow as tf
from config import Config


def data_generator(img_dir, mask_dir, batch_size, img_size, augment=True, seed=42):
    """数据生成器：支持数据增强，图像和掩码同步变换"""
    # 图像数据增强参数
    img_data_gen_args = dict(
        rotation_range=Config.ROTATION_RANGE if augment else 0,
        width_shift_range=Config.WIDTH_SHIFT_RANGE if augment else 0,
        height_shift_range=Config.HEIGHT_SHIFT_RANGE if augment else 0,
        shear_range=Config.SHEAR_RANGE if augment else 0,
        zoom_range=Config.ZOOM_RANGE if augment else 0,
        horizontal_flip=Config.HORIZONTAL_FLIP if augment else False,
        fill_mode=Config.FILL_MODE,
        rescale=1. / 255
    )

    # 掩码数据增强参数（与图像一致，仅二值化处理不同）
    mask_data_gen_args = dict(
        rotation_range=Config.ROTATION_RANGE if augment else 0,
        width_shift_range=Config.WIDTH_SHIFT_RANGE if augment else 0,
        height_shift_range=Config.HEIGHT_SHIFT_RANGE if augment else 0,
        shear_range=Config.SHEAR_RANGE if augment else 0,
        zoom_range=Config.ZOOM_RANGE if augment else 0,
        horizontal_flip=Config.HORIZONTAL_FLIP if augment else False,
        fill_mode=Config.FILL_MODE,
        rescale=1. / 255,
        preprocessing_function=lambda x: np.where(x > 0.5, 1, 0)  # 二值化掩码
    )

    # 创建图像和掩码生成器
    img_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**img_data_gen_args)
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**mask_data_gen_args)

    img_generator = img_datagen.flow_from_directory(
        directory=os.path.dirname(img_dir),
        classes=[os.path.basename(img_dir)],
        class_mode=None,
        color_mode='grayscale' if Config.IN_CHANNELS == 1 else 'rgb',
        target_size=img_size,
        batch_size=batch_size,
        seed=seed
    )

    mask_generator = mask_datagen.flow_from_directory(
        directory=os.path.dirname(mask_dir),
        classes=[os.path.basename(mask_dir)],
        class_mode=None,
        color_mode='grayscale',
        target_size=img_size,
        batch_size=batch_size,
        seed=seed
    )

    # 同步生成图像和掩码
    while True:
        img = next(img_generator)
        mask = next(mask_generator)
        yield (img, mask)
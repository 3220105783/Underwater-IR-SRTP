# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import config

# 设置随机种子
np.random.seed(42)
tf.set_random_seed(42)


def load_image_mask_pairs(img_dir, mask_dir):
    """
    加载图像和掩码对路径（适配xxx.png和xxx_mask.png命名规则）
    """
    img_paths = []
    mask_paths = []

    # 获取所有图像文件
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    for img_file in img_files:
        # 构建图像路径
        img_path = os.path.join(img_dir, img_file)

        # 构建掩码路径（xxx.png -> xxx_mask.png）
        img_name, img_ext = os.path.splitext(img_file)
        mask_file = f"{img_name}_mask{img_ext}"
        mask_path = os.path.join(mask_dir, mask_file)

        if os.path.exists(mask_path):
            img_paths.append(img_path)
            mask_paths.append(mask_path)
        else:
            print(f"警告：未找到{img_file}对应的掩码文件{mask_file}")

    return img_paths, mask_paths


def augment_image_mask(image, mask):
    """
    同步增强图像和掩码（numpy实现）
    核心修改：
    1. 对每对image/mask随机选择一种增强方式（而非多种叠加）
    2. 确保选中的增强方式使用完全相同的参数同步作用于图像和掩码
    3. 修复cv2.copyMakeBorder的参数类型警告
    """
    # 定义可用的增强方式列表
    augmentation_methods = []
    if config.RANDOM_FLIP_HORIZONTAL:
        augmentation_methods.append('flip_horizontal')
    if config.RANDOM_FLIP_VERTICAL:
        augmentation_methods.append('flip_vertical')
    if config.RANDOM_ROTATION:
        augmentation_methods.append('rotate')
    if config.RANDOM_ZOOM:
        augmentation_methods.append('zoom')

    # 如果没有启用任何增强方式，直接返回
    if not augmentation_methods or not config.DATA_AUGMENTATION:
        return image, mask

    # 随机选择一种增强方式
    selected_method = np.random.choice(augmentation_methods)

    # 根据选中的方式进行增强（参数完全同步）
    if selected_method == 'flip_horizontal':
        # 水平翻转
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    elif selected_method == 'flip_vertical':
        # 垂直翻转
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    elif selected_method == 'rotate':
        # 随机旋转（相同角度）
        angle = np.random.uniform(-config.ROTATION_RANGE, config.ROTATION_RANGE)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        # 使用INTER_LINEAR插值图像，INTER_NEAREST插值掩码（保持掩码像素值）
        image = cv2.warpAffine(image, m,  (w, h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, m, (w, h), flags=cv2.INTER_NEAREST)

    elif selected_method == 'zoom':
        # 随机缩放（相同缩放因子）
        zoom_factor = np.random.uniform(config.ZOOM_RANGE[0], config.ZOOM_RANGE[1])
        h, w = image.shape[:2]
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)

        # 缩放（图像用线性插值，掩码用最近邻插值）
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # 裁剪/填充回原尺寸（修复参数类型警告）
        if zoom_factor > 1:
            # 裁剪到原尺寸
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            # 防止越界
            start_h = max(0, start_h)
            start_w = max(0, start_w)
            end_h = start_h + h
            end_w = start_w + w
            # 处理缩放后尺寸仍小于原尺寸的边界情况
            if end_h > new_h:
                end_h = new_h
                start_h = end_h - h
            if end_w > new_w:
                end_w = new_w
                start_w = end_w - w

            image = image[start_h:end_h, start_w:end_w]
            mask = mask[start_h:end_h, start_w:end_w]
        else:
            # 填充到原尺寸（修复参数类型警告：value需为序列类型）
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            # 计算上下左右填充量（确保总尺寸正确）
            top = pad_h
            bottom = h - new_h - pad_h
            left = pad_w
            right = w - new_w - pad_w

            # 修复警告：value参数改为序列类型（与图像通道数匹配）
            if len(image.shape) == 3:
                # 彩色图像：填充黑色（[0,0,0]）
                image_border_value = [0.0, 0.0, 0.0]
            else:
                # 灰度图像：填充黑色（[0.0]）
                image_border_value = [0.0]
            # 掩码填充0（[0.0]）
            mask_border_value = [0.0]

            # 执行填充
            image = cv2.copyMakeBorder(
                image,
                top, bottom, left, right,
                cv2.BORDER_CONSTANT,
                value=image_border_value  # 序列类型，修复警告
            )
            mask = cv2.copyMakeBorder(
                mask,
                top, bottom, left, right,
                cv2.BORDER_CONSTANT,
                value=mask_border_value  # 序列类型，修复警告
            )

    # 确保输出尺寸正确（防御性编程）
    assert image.shape[:2] == (config.IMAGE_HEIGHT, config.IMAGE_WIDTH), \
        f"Incorrect dimensions after image enhancement: {image.shape[:2]}, expected{(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)}"
    assert mask.shape[:2] == (config.IMAGE_HEIGHT, config.IMAGE_WIDTH), \
        f"Incorrect dimensions after mask enhancement: {mask.shape[:2]}, expected{(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)}"

    return image, mask


def preprocess_image_mask(img_path, mask_path, augment=False):
    """
    预处理单张图像和掩码（TF1.x兼容）
    """
    try:
        # 加载图像（保持原始通道数）
        img = Image.open(img_path).resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
        image = np.array(img, dtype=np.float32) / 255.0
        # 确保图像是3通道（防止灰度图）
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.repeat(image, 3, axis=-1)

        # 加载掩码（灰度图）
        mask = Image.open(mask_path).resize((config.IMAGE_WIDTH, config.IMAGE_HEIGHT)).convert('L')
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = np.where(mask > 0.5, 1.0, 0.0)
        mask = np.expand_dims(mask, axis=-1)  # [H, W, 1]

        # 数据增强（仅训练时启用）
        if augment and config.DATA_AUGMENTATION:
            image, mask = augment_image_mask(image, mask)

        return image, mask
    except Exception as e:
        print(f"Error occurred while preprocessing image {img_path}: {e}")
        raise


def create_dataset_generator(img_paths, mask_paths, batch_size=config.BATCH_SIZE, augment=False):
    """
    创建TF1.x数据集生成器（基于队列）
    """

    def generator():
        while True:
            # 打乱数据
            indices = np.arange(len(img_paths))
            np.random.shuffle(indices)

            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_images = []
                batch_masks = []

                for idx in batch_indices:
                    img_path = img_paths[idx]
                    mask_path = mask_paths[idx]

                    # 预处理
                    image, mask = preprocess_image_mask(img_path, mask_path, augment)

                    batch_images.append(image)
                    batch_masks.append(mask)

                yield np.array(batch_images), np.array(batch_masks)

    # 创建TF数据集
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            (None, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
            (None, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1)
        )
    )

    # 创建迭代器
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    return next_batch


def get_train_val_datasets():
    """
    获取训练和验证数据集
    """
    # 加载路径
    train_img_paths, train_mask_paths = load_image_mask_pairs(config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR)
    val_img_paths, val_mask_paths = load_image_mask_pairs(config.VAL_IMG_DIR, config.VAL_MASK_DIR)

    # 检查数据量
    if len(train_img_paths) == 0:
        raise ValueError("The training set is empty! Please check the paths for TRAIN_IMG_DIR and TRAIN_MASK_DIR, as well as the file naming convention (xxx.png corresponds to xxx_mask.png).")
    if len(val_img_paths) == 0:
        raise ValueError("The validation set is empty! Please check the VAL_IMG_DIR and VAL_MASK_DIR paths, as well as the file naming convention (xxx.png corresponds to xxx_mask.png).")

    # 创建生成器
    train_batch = create_dataset_generator(train_img_paths, train_mask_paths, augment=True)
    val_batch = create_dataset_generator(val_img_paths, val_mask_paths, augment=False)

    # 计算步数
    train_steps = len(train_img_paths) // config.BATCH_SIZE
    val_steps = len(val_img_paths) // config.BATCH_SIZE

    # 确保步数至少为1
    train_steps = max(1, train_steps)
    val_steps = max(1, val_steps)

    return train_batch, val_batch, train_steps, val_steps, len(train_img_paths), len(val_img_paths)


if __name__ == "__main__":
    # 测试数据集加载
    tf.reset_default_graph()
    with tf.Session(config=config.TF_CONFIG) as sess:
        try:
            train_batch, val_batch, train_steps, val_steps, train_len, val_len = get_train_val_datasets()

            print(f"Number of training set samples: {train_len}, steps: {train_steps}")
            print(f"Number of validation set samples: {val_len}, steps: {val_steps}")

            # 测试生成器
            images, masks = sess.run(train_batch)
            print(f"Image shape: {images.shape}, mask shape: {masks.shape}")
            print(f"Image value range: {np.min(images):.4f} ~ {np.max(images):.4f}")
            print(f"Mask value range: {np.min(masks):.4f} ~ {np.max(masks):.4f}")

            # 测试增强函数
            test_image = np.random.rand(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3).astype(np.float32)
            test_mask = np.random.rand(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1).astype(np.float32)
            aug_image, aug_mask = augment_image_mask(test_image, test_mask)
            print(f"Augmented image shape: {aug_image.shape}, mask shape: {aug_mask.shape}")
            print("Data augmentation function test passed!")

        except Exception as e:
            print(f"Test failed: {e}")
# -*- coding: utf-8 -*-
import os

# ====================== 路径配置 ======================
# 数据集路径
TRAIN_IMG_DIR = "/dataset/train/img"
TRAIN_MASK_DIR = "/dataset/train/mask"
VAL_IMG_DIR = "/dataset/val/img"
VAL_MASK_DIR = "/dataset/val/mask"
# 测试集路径（预测用）
TEST_IMG_DIR = "/dataset/test/img"
TEST_MASK_DIR = "/dataset/test/mask"

# 模型保存路径
MODEL_SAVE_PATH = "/model/best_model.ckpt"
LATEST_MODEL_DIR = "/model/latest"
EXIST_MODEL_DIR = "/model/exist"
EXIST_MODEL_NAME = "exist.ckpt"
EXIST_MODEL_PATH = os.path.join(EXIST_MODEL_DIR, EXIST_MODEL_NAME)

# 结果保存路径
TRAIN_RESULT_DIR = "/results/train"
EVAL_RESULT_DIR = "/results/evaluation"
PRED_RESULT_DIR = "/results/predict"

# 日志路径
LOG_DIR = "/tensorboard_logs"

# ====================== 训练配置 ======================
# 硬件配置
CUDA_VISIBLE_DEVICES = "0"

# 训练参数
BATCH_SIZE = 1
EPOCHS = 512
LEARNING_RATE = 5e-5
PATIENCE = 30  # 早停耐心值
MAX_LATEST_MODELS = 3  # 保留最新模型数
LOAD_EXIST_MODEL = False  # 是否加载已有模型

# ====================== 模型配置 ======================
INPUT_SIZE = (768, 768)
N_CHANNELS = 3
N_CLASSES = 1

# ====================== 数据增强配置 ======================
AUG_CONFIG = {
    'all_aug_methods': [
        'flip_left_right', 'flip_up_down', 'rotation', 'zoom',
        'brightness', 'contrast', 'hue', 'saturation', 'noise'
    ],
    'enable_flip_left_right': True,
    'enable_flip_up_down': True,
    'enable_rotation': True,
    'enable_zoom': False,
    'enable_brightness': False,
    'enable_contrast': False,
    'enable_hue': False,
    'enable_saturation': False,
    'enable_noise': False,
    'rotation_angle': 15,
    'zoom_lower': 0.5,
    'zoom_upper': 1.5,
    'brightness_max_delta': 0.2,
    'contrast_lower': 0.8,
    'contrast_upper': 1.2,
    'hue_max_delta': 0.1,
    'saturation_lower': 0.8,
    'saturation_upper': 1.2,
    'noise_std': 0.01
}

# ====================== 评估配置 ======================
THRESHOLD = 0.5
SMOOTH = 1e-6
FOCAL_GAMMA = 2
FOCAL_ALPHA = 50.0
FOCAL_WEIGHT = 0.4
DICE_WEIGHT = 0.6

# ====================== 预测配置 ======================
# 归一化均值/标准差（ImageNet）
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
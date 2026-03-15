# -*- coding: utf-8 -*-
import os
import cv2
import tensorflow as tf

# ===================== 基础配置 =====================
# 环境设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第0块GPU

# 图像配置
IMAGE_HEIGHT = 768
IMAGE_WIDTH = 768
IMAGE_CHANNELS = 3
NUM_CLASSES = 1  # 二分类：裂缝/非裂缝
THRESHOLD = 0.5  # 预测二值化阈值

# 文件路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "dataset/train/img")
TRAIN_MASK_DIR = os.path.join(BASE_DIR, "dataset/train/mask")
VAL_IMG_DIR = os.path.join(BASE_DIR, "dataset/val/img")
VAL_MASK_DIR = os.path.join(BASE_DIR, "dataset/val/mask")
TEST_IMG_DIR = os.path.join(BASE_DIR, "dataset/test/img")
TEST_MASK_DIR = os.path.join(BASE_DIR, "dataset/test/mask")

# 模型保存路径
BEST_MODEL_DIR = os.path.join(BASE_DIR, "model/best_model")
LATEST_MODEL_DIR = os.path.join(BASE_DIR, "model/latest_model")
TENSORBOARD_LOG_DIR = os.path.join(BASE_DIR, "tensorboard_logs")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TEST_OUTPUT_DIR = os.path.join(RESULTS_DIR, "test_output")

# 创建目录
for dir_path in [BEST_MODEL_DIR, LATEST_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR, TEST_OUTPUT_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# ===================== 训练配置 =====================
# 训练参数
BATCH_SIZE = 2
EPOCHS = 200
INITIAL_LEARNING_RATE = 2e-4
#DECAY_STEPS = 1000
#DECAY_RATE = 0.9
MIN_LEARNING_RATE = 1e-6  # 新增：学习率下限
CLIP_NORM = 1.0
# 余弦退火学习率配置（新增）
COSINE_T_MAX = EPOCHS  # 余弦退火周期（以epoch为单位，完整周期的epoch数）
COSINE_ETA_MIN = MIN_LEARNING_RATE  # 余弦退火最小学习率（复用已有下限）

# 早停机制
EARLY_STOPPING_PATIENCE = 25
EARLY_STOPPING_MIN_DELTA = 1e-4
BEST_VAL_LOSS = float('inf')
STOPPING_COUNTER = 0

# 模型保存
SAVE_BEST_ONLY = True
SAVE_CHECKPOINT_EVERY_N_EPOCHS = 1
KEEP_LATEST_MODELS = 3  # 保留最新的5个模型

# ===================== 数据增强配置 =====================
DATA_AUGMENTATION = True
RANDOM_FLIP_HORIZONTAL = True
RANDOM_FLIP_VERTICAL = True
RANDOM_ROTATION = True
ROTATION_RANGE = 15  # 旋转角度范围
RANDOM_ZOOM = True
ZOOM_RANGE = (0.8, 1.2)  # 缩放范围

# ===================== 损失函数配置 =====================
# 损失函数权重
BCE_WEIGHT = 0.2
DICE_WEIGHT = 0.5
FOCAL_WEIGHT = 0.2
IOU_WEIGHT = 0.1

# Focal Loss参数
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# ===================== 测试配置 =====================
# 测试模式
PREDICT_MODE = True  # 生成预测结果
EVALUATE_MODE = True  # 评估模型性能

# 预测结果可视化
PREDICTION_ALPHA = 0.5  # 遮罩透明度
PREDICTION_COLOR = (255, 0, 0)  # 红色裂缝遮罩
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0
FONT_COLOR = (255, 0, 0)
FONT_THICKNESS = 2

# 评估指标保存
SAVE_EVALUATION_RESULTS = True
EVALUATION_SAVE_PATH = os.path.join(RESULTS_DIR, "evaluation_results.txt")

# TF1.x 配置
TF_CONFIG = tf.ConfigProto()
TF_CONFIG.gpu_options.allow_growth = True  # 动态分配GPU内存
TF_CONFIG.allow_soft_placement = True
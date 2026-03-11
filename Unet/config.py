import os


class Config:
    """配置类：控制所有训练和模型参数"""
    # 数据路径
    DATA_DIR = 'dataset'
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train', 'img')
    TRAIN_MASK_DIR = os.path.join(DATA_DIR, 'train', 'mask')
    VAL_IMG_DIR = os.path.join(DATA_DIR, 'val', 'img')
    VAL_MASK_DIR = os.path.join(DATA_DIR, 'val', 'mask')
    TEST_IMG_DIR = os.path.join(DATA_DIR, 'test', 'img')
    TEST_MASK_DIR = os.path.join(DATA_DIR, 'test', 'mask')

    # 模型路径
    MODEL_DIR = 'model'
    BEST_MODEL_DIR = os.path.join(MODEL_DIR, 'best')
    LATEST_MODEL_DIR = os.path.join(MODEL_DIR, 'latest')
    EXIST_MODEL_DIR = os.path.join(MODEL_DIR, 'exist')
    BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, 'best_model.ckpt')

    # 输出路径
    RESULTS_DIR = 'results'
    TENSORBOARD_DIR = 'tensorboard_logs'
    TEST_OUTPUT_DIR = 'test_output'

    # 训练参数
    IMG_SIZE = (768, 768)
    IN_CHANNELS = 1
    OUT_CHANNELS = 1
    BATCH_SIZE = 8
    EPOCHS = 200
    LEARNING_RATE = 1e-4
    PATIENCE = 20  # 早停机制耐心值

    # 数据增强参数
    ROTATION_RANGE = 15
    WIDTH_SHIFT_RANGE = 0.1
    HEIGHT_SHIFT_RANGE = 0.1
    SHEAR_RANGE = 0.1
    ZOOM_RANGE = 0.1
    HORIZONTAL_FLIP = True
    FILL_MODE = 'nearest'
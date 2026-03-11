import os
import cv2
import numpy as np
import tensorflow as tf
from config import Config
from utils import create_dirs, set_gpu_config
from unet_model import unet_model

# 禁用 TensorFlow 2.x 行为
tf.disable_v2_behavior()


def test():
    """测试模块：加载模型处理test/img中的图片，保存结果到test_output"""
    create_dirs()
    set_gpu_config()

    # 加载模型
    model = unet_model(input_size=(*Config.IMG_SIZE, Config.IN_CHANNELS), out_channels=Config.OUT_CHANNELS)
    if not os.path.exists(Config.BEST_MODEL_PATH + '.index'):
        print(f"Best model not found at {Config.BEST_MODEL_PATH}")
        return
    model.load_weights(Config.BEST_MODEL_PATH)
    print(f"Loaded best model from {Config.BEST_MODEL_PATH}")

    # 处理测试图片
    test_files = [f for f in os.listdir(Config.TEST_IMG_DIR) if os.path.isfile(os.path.join(Config.TEST_IMG_DIR, f))]
    for img_file in test_files:
        img_path = os.path.join(Config.TEST_IMG_DIR, img_file)

        # 读取图片
        if Config.IN_CHANNELS == 1:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        orig_size = img.shape[:2]
        # 预处理
        img_resized = cv2.resize(img, Config.IMG_SIZE)
        img_norm = img_resized / 255.0
        img_input = np.expand_dims(img_norm, axis=0)
        if Config.IN_CHANNELS == 1:
            img_input = np.expand_dims(img_input, axis=-1)

        # 预测
        pred = model.predict(img_input)
        pred_mask = (pred[0, :, :, 0] > 0.5).astype(np.uint8) * 255
        # 恢复原始尺寸
        pred_mask = cv2.resize(pred_mask, (orig_size[1], orig_size[0]), interpolation=cv2.INTER_NEAREST)

        # 保存结果
        output_path = os.path.join(Config.TEST_OUTPUT_DIR, f"pred_{img_file}")
        cv2.imwrite(output_path, pred_mask)
        print(f"Saved prediction to {output_path}")


if __name__ == '__main__':
    test()
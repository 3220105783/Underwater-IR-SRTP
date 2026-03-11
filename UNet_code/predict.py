# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from unet_model import UNet
from config import (CUDA_VISIBLE_DEVICES, MODEL_SAVE_PATH, TEST_IMG_DIR, TEST_MASK_DIR,
                    PRED_RESULT_DIR, INPUT_SIZE, MEAN, STD, THRESHOLD)

# -------------------------- 配置参数 --------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

TARGET_SIZE = INPUT_SIZE

# -------------------------- 预处理 --------------------------
def preprocess_img(img_path):
    """单张图片预处理"""
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    # 调整尺寸
    img_resized = Image.fromarray(img_np).resize(TARGET_SIZE, Image.BILINEAR)  # type: ignore
    img_resized = np.array(img_resized, dtype=np.float32) / 255.0
    # 标准化
    img_resized = (img_resized - MEAN) / STD
    # 添加batch维度
    img_input = np.expand_dims(img_resized, axis=0)
    return img_np, img_input

# -------------------------- 预测主流程 --------------------------
def predict():
    # 构建图
    tf.reset_default_graph()

    # 构建模型
    inputs = tf.placeholder(tf.float32, [1, *TARGET_SIZE, 3], name='inputs')
    unet = UNet()
    logits = unet.build_model(inputs, is_training=False)
    pred_mask = tf.nn.sigmoid(logits) > THRESHOLD
    pred_mask = tf.cast(pred_mask, tf.float32)

    # 加载模型
    saver = tf.train.Saver()

    # 启动Session
    with tf.Session(config=config) as sess:
        saver.restore(sess, MODEL_SAVE_PATH)
        print(f"Model loaded: {MODEL_SAVE_PATH}")

        # 确保保存目录存在
        os.makedirs(PRED_RESULT_DIR, exist_ok=True)

        # 遍历测试图片
        test_img_names = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith((".jpg", ".png"))]
        for img_name in test_img_names:
            img_path = os.path.join(TEST_IMG_DIR, img_name)

            # 预处理
            img_np, img_input = preprocess_img(img_path)

            # 预测
            pred_mask_val = sess.run(pred_mask, feed_dict={inputs: img_input})
            pred_mask_val = np.squeeze(pred_mask_val)

            # 读取真实mask
            mask_name = img_name.replace(".jpg", "_mask.png").replace(".png", "_mask.png")
            mask_path = os.path.join(TEST_MASK_DIR, mask_name)
            if os.path.exists(mask_path):
                true_mask = np.array(Image.open(mask_path).convert("L")) / 255.0
                true_mask = np.array(Image.fromarray(true_mask).resize(TARGET_SIZE, Image.NEAREST))  # type: ignore
            else:
                true_mask = np.zeros(TARGET_SIZE)
                print(f"True mask not found: {mask_path}")

            # 可视化
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(img_np)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(true_mask, cmap="gray")
            plt.title("True Mask")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(pred_mask_val, cmap="gray")
            plt.title("Predicted Mask")
            plt.axis("off")

            # 保存结果
            save_name = img_name.replace(".jpg", "_pred.png").replace(".png", "_pred.png")
            save_path = os.path.join(PRED_RESULT_DIR, save_name)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Prediction saved: {save_path}")

if __name__ == "__main__":
    predict()
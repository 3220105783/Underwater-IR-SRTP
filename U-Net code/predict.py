# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from unet_model import UNet

# 配置参数
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

MODEL_PATH = "/model/best_model.ckpt"  # 最优模型覆盖保存路径（DEFAULT: "/model/best_model.ckpt"）
TEST_IMG_DIR = "/dataset/val/img"  #（DEFAULT: "/dataset/val/img"）
TEST_MASK_DIR = "/dataset/val/mask"  #（DEFAULT: "/dataset/val/mask"）
TARGET_SIZE = (768, 768)

#训练结果保存路径
TRAIN_RESULT_DIR = "/results/train"  # 训练相关结果（DEFAULT: "/results/train"）
EVAL_RESULT_DIR = "/results/evaluation"    # 评估相关结果（DEFAULT: "/results/evaluation"）
PRED_RESULT_DIR = "/results/predict" # 预测相关结果（DEFAULT: "/results/predict"）


# ImageNet均值/标准差
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def preprocess_img(img_path):
    """单张图片预处理"""
    # 读取图片
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    # 调整尺寸
    img_resized = Image.fromarray(img_np).resize(TARGET_SIZE, Image.BILINEAR)
    img_resized = np.array(img_resized, dtype=np.float32) / 255.0

    # 标准化
    img_resized = (img_resized - MEAN) / STD

    # 添加batch维度
    img_input = np.expand_dims(img_resized, axis=0)
    return img_np, img_input


def predict():
    # 1. 构建图
    tf.reset_default_graph()

    # 构建模型
    inputs = tf.placeholder(tf.float32, [1, 768, 768, 3], name='inputs')
    unet = UNet(n_channels=3, n_classes=1)
    # 关键修改：评估时is_training=False（推理模式）
    logits = unet.build_model(inputs, is_training=False)
    pred_mask = tf.nn.sigmoid(logits) > 0.5
    pred_mask = tf.cast(pred_mask, tf.float32)

    # 加载模型
    saver = tf.train.Saver()

    # 2. 启动Session预测
    with tf.Session(config=config) as sess:
        saver.restore(sess, MODEL_PATH)
        print(f"Model loaded successfully: {MODEL_PATH}")

        # 遍历测试图片
        test_img_names = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith((".jpg", ".png"))]
        for img_name in test_img_names:
            img_path = os.path.join(TEST_IMG_DIR, img_name)

            # 预处理
            img_np, img_input = preprocess_img(img_path)

            # 预测
            pred_mask_val = sess.run(pred_mask, feed_dict={inputs: img_input})
            pred_mask_val = np.squeeze(pred_mask_val)  # 去除batch/通道维度

            # 读取真实mask
            mask_name = img_name.replace(".jpg", "_mask.png").replace(".png", "_mask.png")
            mask_path = os.path.join(TEST_MASK_DIR, mask_name)
            true_mask = np.array(Image.open(mask_path).convert("L")) / 255.0
            true_mask = np.array(Image.fromarray(true_mask).resize(TARGET_SIZE, Image.NEAREST))

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
            save_path = os.path.join(PRED_RESULT_DIR, f"result_{img_name}")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Prediction results saved: {save_path}")

    print(f"All images prediction completed! Results saved in: {PRED_RESULT_DIR}")


if __name__ == "__main__":
    predict()
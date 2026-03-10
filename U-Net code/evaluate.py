# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from unet_dataset import build_dataset
from unet_model import UNet
from unet_utils import calculate_iou, calculate_metrics, calculate_dsc  # 新增导入calculate_dsc

# 配置参数
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

MODEL_PATH = "/model/best_crack_model.ckpt"
VAL_IMG_DIR = "/dataset/val/img"
VAL_MASK_DIR = "/dataset/val/mask"
BATCH_SIZE = 2


def evaluate():
    # 1. 构建图
    tf.reset_default_graph()

    # 加载验证集
    val_iter, val_batch, val_size = build_dataset(
        VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, is_train=False
    )

    # 构建模型
    inputs = tf.placeholder(tf.float32, [None, 768, 768, 3], name='inputs')
    masks = tf.placeholder(tf.float32, [None, 768, 768, 1], name='masks')

    unet = UNet(n_channels=3, n_classes=1)
    logits = unet.build_model(inputs)

    # 评估指标（新增dsc计算）
    iou = calculate_iou(logits, masks)
    metrics = calculate_metrics(logits, masks)
    dsc = calculate_dsc(logits, masks)  # 新增DSC计算节点

    # 加载模型
    saver = tf.train.Saver()

    # 2. 启动Session评估
    with tf.Session(config=config) as sess:
        # 加载权重
        saver.restore(sess, MODEL_PATH)
        print(f"Model loaded successfully: {MODEL_PATH}")

        # 初始化迭代器
        sess.run(val_iter.initializer)

        # 累计指标（新增total_dsc）
        total_iou = 0.0
        total_acc = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_dsc = 0.0  # 新增DSC累计变量
        steps = 0

        # 批量评估
        while True:
            try:
                img_batch, mask_batch = sess.run(val_batch)
                # 新增dsc_val的计算
                iou_val, metrics_val, dsc_val = sess.run(
                    [iou, metrics, dsc],
                    feed_dict={inputs: img_batch, masks: mask_batch}
                )

                total_iou += iou_val
                total_acc += metrics_val['accuracy']
                total_precision += metrics_val['precision']
                total_recall += metrics_val['recall']
                total_f1 += metrics_val['f1']
                total_dsc += dsc_val  # 累计DSC值
                steps += 1

                print(f"Batch {steps} evaluation completed")
            except tf.errors.OutOfRangeError:
                break

        # 计算平均指标（新增avg_dsc）
        avg_iou = total_iou / steps
        avg_acc = total_acc / steps
        avg_precision = total_precision / steps
        avg_recall = total_recall / steps
        avg_f1 = total_f1 / steps
        avg_dsc = total_dsc / steps  # 计算平均DSC

        # 整理结果（新增DSC指标）
        final_metrics = {
            "IoU（交并比）": round(avg_iou, 4),
            "DSC（Dice相似系数）": round(avg_dsc, 4),  # 新增DSC输出
            "Precision（精确率）": round(avg_precision, 4),
            "Recall（召回率）": round(avg_recall, 4),
            "F1-Score": round(avg_f1, 4),
            "Accuracy（准确率）": round(avg_acc, 4)
        }

        # 打印结果
        print("\n" + "=" * 60)
        print("Final Quantitative Evaluation Results")
        print("=" * 60)
        for name, value in final_metrics.items():
            print(f"{name}: {value}")
        print("=" * 60)

        # 可视化（新增DSC的柱状图）
        plt.rcParams['font.sans-serif'] = ['SimHei']
        metric_names = list(final_metrics.keys())
        metric_values = list(final_metrics.values())

        plt.figure(figsize=(12, 6))  # 调整画布宽度以适配新增的指标
        bars = plt.bar(metric_names, metric_values,
                       color=['#2ecc71', '#1abc9c', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'])

        # 添加数值标签
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{value}', ha='center', va='bottom', fontsize=12)

        plt.ylim(0, 1.1)
        plt.title("裂缝分割模型量化评估结果", fontsize=14, fontweight='bold')
        plt.ylabel("指标值", fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig("evaluation_results_tf1.png", dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    evaluate()
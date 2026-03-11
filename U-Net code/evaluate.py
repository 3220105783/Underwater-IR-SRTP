# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.metrics import hausdorff_distance
from unet_dataset import build_dataset
from unet_model import UNet
from unet_utils import calculate_iou, calculate_metrics, calculate_dsc

# 配置参数
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

MODEL_PATH = "/model/best_model.ckpt"
VAL_IMG_DIR = "/dataset/val/img"
VAL_MASK_DIR = "/dataset/val/mask"
BATCH_SIZE = 2


def calculate_asd(pred_mask, gt_mask):
    """
    计算平均表面距离（ASD）
    :param pred_mask: 预测掩码 (batch, H, W, 1)，二值化（0/1）
    :param gt_mask: 真实掩码 (batch, H, W, 1)，二值化（0/1）
    :return: 批次平均ASD
    """
    asd_list = []
    for i in range(pred_mask.shape[0]):
        pred = np.squeeze(pred_mask[i])
        gt = np.squeeze(gt_mask[i])

        # 处理全0的情况（无裂缝）
        if np.sum(pred) == 0 and np.sum(gt) == 0:
            asd_list.append(0.0)
            continue
        if np.sum(pred) == 0 or np.sum(gt) == 0:
            asd_list.append(np.inf)
            continue

        # 计算距离变换
        pred_dist = distance_transform_edt(1 - pred)
        gt_dist = distance_transform_edt(1 - gt)

        # 提取表面点（距离>0的点）
        pred_surface = pred_dist[gt == 1]
        gt_surface = gt_dist[pred == 1]

        # 计算平均表面距离
        asd = (np.mean(pred_surface) + np.mean(gt_surface)) / 2.0
        asd_list.append(asd)
    return np.mean(asd_list)


def calculate_hd95(pred_mask, gt_mask):
    """
    计算95%豪斯多夫距离（HD95）
    :param pred_mask: 预测掩码 (batch, H, W, 1)，二值化（0/1）
    :param gt_mask: 真实掩码 (batch, H, W, 1)，二值化（0/1）
    :return: 批次平均HD95
    """
    hd95_list = []
    for i in range(pred_mask.shape[0]):
        pred = np.squeeze(pred_mask[i])
        gt = np.squeeze(gt_mask[i])

        # 处理全0的情况（无裂缝）
        if np.sum(pred) == 0 and np.sum(gt) == 0:
            hd95_list.append(0.0)
            continue
        if np.sum(pred) == 0 or np.sum(gt) == 0:
            hd95_list.append(np.inf)
            continue

        # 计算豪斯多夫距离（skimage的hausdorff_distance返回最大距离）
        hd = hausdorff_distance(pred, gt)

        # 计算HD95（基于距离排序取95%分位数）
        # 步骤1：获取所有预测和真实掩码的坐标
        pred_coords = np.argwhere(pred == 1)
        gt_coords = np.argwhere(gt == 1)

        # 步骤2：计算所有点对的距离
        all_distances = []
        for p in pred_coords:
            dists = np.sqrt(np.sum((gt_coords - p) ** 2, axis=1))
            all_distances.append(np.min(dists))
        for g in gt_coords:
            dists = np.sqrt(np.sum((pred_coords - g) ** 2, axis=1))
            all_distances.append(np.min(dists))

        # 步骤3：取95%分位数
        hd95 = np.percentile(all_distances, 95) if all_distances else 0.0
        hd95_list.append(hd95)
    return np.mean(hd95_list)


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
    # 生成二值化预测掩码（用于ASD/HD95计算）
    pred_masks = tf.cast(tf.sigmoid(logits) > 0.5, tf.float32)

    # 评估指标
    iou = calculate_iou(logits, masks)
    metrics = calculate_metrics(logits, masks)
    dsc = calculate_dsc(logits, masks)

    # 加载模型
    saver = tf.train.Saver()

    # 2. 启动Session评估
    with tf.Session(config=config) as sess:
        # 加载权重
        saver.restore(sess, MODEL_PATH)
        print(f"Model loaded successfully: {MODEL_PATH}")

        # 初始化迭代器
        sess.run(val_iter.initializer)

        # 累计指标
        total_iou = 0.0
        total_acc = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_dsc = 0.0
        total_asd = 0.0  # 新增ASD累计
        total_hd95 = 0.0  # 新增HD95累计
        steps = 0

        # 批量评估
        while True:
            try:
                img_batch, mask_batch = sess.run(val_batch)
                # 运行计算（新增pred_masks）
                iou_val, metrics_val, dsc_val, pred_masks_val = sess.run(
                    [iou, metrics, dsc, pred_masks],
                    feed_dict={inputs: img_batch, masks: mask_batch}
                )

                # 计算ASD和HD95
                asd_val = calculate_asd(pred_masks_val, mask_batch)
                hd95_val = calculate_hd95(pred_masks_val, mask_batch)

                # 累计指标
                total_iou += iou_val
                total_acc += metrics_val['accuracy']
                total_precision += metrics_val['precision']
                total_recall += metrics_val['recall']
                total_f1 += metrics_val['f1']
                total_dsc += dsc_val
                total_asd += asd_val
                total_hd95 += hd95_val
                steps += 1

                print(f"Batch {steps} evaluation completed | ASD: {asd_val:.4f} | HD95: {hd95_val:.4f}")
            except tf.errors.OutOfRangeError:
                break

        # 计算平均指标
        avg_iou = total_iou / steps
        avg_acc = total_acc / steps
        avg_precision = total_precision / steps
        avg_recall = total_recall / steps
        avg_f1 = total_f1 / steps
        avg_dsc = total_dsc / steps
        avg_asd = total_asd / steps  # 平均ASD
        avg_hd95 = total_hd95 / steps  # 平均HD95

        # 整理结果（新增ASD/HD95）
        final_metrics = {
            "IoU（交并比）": round(avg_iou, 4),
            "DSC（Dice相似系数）": round(avg_dsc, 4),
            "Precision（精确率）": round(avg_precision, 4),
            "Recall（召回率）": round(avg_recall, 4),
            "F1-Score": round(avg_f1, 4),
            "Accuracy（准确率）": round(avg_acc, 4),
            "ASD（平均表面距离）": round(avg_asd, 4),  # 新增ASD
            "HD95（平均95%豪斯多夫距离）": round(avg_hd95, 4)  # 新增HD95
        }

        # 打印结果
        print("\n" + "=" * 60)
        print("Final Quantitative Evaluation Results")
        print("=" * 60)
        for name, value in final_metrics.items():
            print(f"{name}: {value}")
        print("=" * 60)

        # 可视化（适配新增指标）
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        metric_names = list(final_metrics.keys())
        metric_values = list(final_metrics.values())

        plt.figure(figsize=(15, 6))  # 加宽画布适配更多指标
        bars = plt.bar(metric_names, metric_values,
                       color=['#2ecc71', '#1abc9c', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#8e44ad', '#16a085'])

        # 添加数值标签
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            # 处理无穷大的情况（显示"inf"）
            if np.isinf(value):
                label = "inf"
            else:
                label = f'{value}'
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     label, ha='center', va='bottom', fontsize=10)

        plt.ylim(0, max([v for v in metric_values if not np.isinf(v)]) * 1.1)  # 适配非无穷大值
        plt.title("裂缝分割模型量化评估结果", fontsize=14, fontweight='bold')
        plt.ylabel("指标值", fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=15)  # 旋转x轴标签防止重叠
        plt.tight_layout()
        plt.savefig("evaluation_results.png", dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    evaluate()
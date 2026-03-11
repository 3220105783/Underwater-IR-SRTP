# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from skimage.metrics import hausdorff_distance
# 注意：需确保以下自定义模块存在
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

# 训练结果保存路径
TRAIN_RESULT_DIR = "/results/train"
EVAL_RESULT_DIR = "/results/evaluation"
PRED_RESULT_DIR = "/results/predict"


def calculate_asd(pred_mask, gt_mask):
    """
    计算平均表面距离（ASD）
    修复点：将inf转为np.nan，避免后续均值计算失效
    :param pred_mask: 预测掩码 (batch, H, W, 1)，二值化（0/1）
    :param gt_mask: 真实掩码 (batch, H, W, 1)，二值化（0/1）
    :return: 批次ASD列表（含np.nan，无inf）
    """
    asd_list = []
    for i in range(pred_mask.shape[0]):
        pred = np.squeeze(pred_mask[i])
        gt = np.squeeze(gt_mask[i])

        # 处理全0的情况（无裂缝）
        if np.sum(pred) == 0 and np.sum(gt) == 0:
            asd_list.append(0.0)
            continue
        # 其一为空时，赋值np.nan而非inf
        if np.sum(pred) == 0 or np.sum(gt) == 0:
            asd_list.append(np.nan)
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
    return asd_list  # 改为返回列表，便于后续按样本计算均值


def calculate_hd95(pred_mask, gt_mask):
    """
    计算95%豪斯多夫距离（HD95）
    修复点1：正确计算所有点对的距离
    修复点2：将inf转为np.nan，避免后续均值计算失效
    :param pred_mask: 预测掩码 (batch, H, W, 1)，二值化（0/1）
    :param gt_mask: 真实掩码 (batch, H, W, 1)，二值化（0/1）
    :return: 批次HD95列表（含np.nan，无inf）
    """
    hd95_list = []
    for i in range(pred_mask.shape[0]):
        pred = np.squeeze(pred_mask[i])
        gt = np.squeeze(gt_mask[i])

        # 处理全0的情况（无裂缝）
        if np.sum(pred) == 0 and np.sum(gt) == 0:
            hd95_list.append(0.0)
            continue
        # 其一为空时，赋值np.nan而非inf
        if np.sum(pred) == 0 or np.sum(gt) == 0:
            hd95_list.append(np.nan)
            continue

        # 步骤1：获取所有预测和真实掩码的坐标
        pred_coords = np.argwhere(pred == 1)
        gt_coords = np.argwhere(gt == 1)

        # 步骤2：计算所有点对的距离（修复核心：不再取最小距离）
        all_distances = []
        for p in pred_coords:
            # 计算当前预测点到所有真实点的距离
            dists = np.sqrt(np.sum((gt_coords - p) ** 2, axis=1))
            all_distances.extend(dists)  # 用extend添加所有距离，而非append最小距离
        for g in gt_coords:
            # 计算当前真实点到所有预测点的距离
            dists = np.sqrt(np.sum((pred_coords - g) ** 2, axis=1))
            all_distances.extend(dists)  # 用extend添加所有距离，而非append最小距离

        # 步骤3：取95%分位数（确保非空）
        hd95 = np.percentile(all_distances, 95) if len(all_distances) > 0 else 0.0
        hd95_list.append(hd95)
    return hd95_list  # 改为返回列表，便于后续按样本计算均值


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
    # 关键修改：评估时is_training=False（推理模式）
    logits = unet.build_model(inputs, is_training=False)
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

        # 累计指标（改为列表存储每个样本的指标，而非批量累加）
        all_iou = []
        all_acc = []
        all_precision = []
        all_recall = []
        all_f1 = []
        all_dsc = []
        all_asd = []
        all_hd95 = []

        # 批量评估
        while True:
            try:
                img_batch, mask_batch = sess.run(val_batch)
                # 运行计算
                iou_val, metrics_val, dsc_val, pred_masks_val = sess.run(
                    [iou, metrics, dsc, pred_masks],
                    feed_dict={inputs: img_batch, masks: mask_batch}
                )

                # 计算ASD和HD95（返回当前batch的样本级列表）
                asd_val_list = calculate_asd(pred_masks_val, mask_batch)
                hd95_val_list = calculate_hd95(pred_masks_val, mask_batch)

                # 收集样本级指标（适配batch内多个样本）
                batch_size = len(asd_val_list)
                all_iou.extend([iou_val] * batch_size)  # IoU批量值复制到样本级
                all_acc.extend([metrics_val['accuracy']] * batch_size)
                all_precision.extend([metrics_val['precision']] * batch_size)
                all_recall.extend([metrics_val['recall']] * batch_size)
                all_f1.extend([metrics_val['f1']] * batch_size)
                all_dsc.extend([dsc_val] * batch_size)
                all_asd.extend(asd_val_list)
                all_hd95.extend(hd95_val_list)

                # 打印batch级日志
                current_step = len(all_asd) // BATCH_SIZE
                batch_asd = np.nanmean(asd_val_list)  # 忽略nan计算当前batch均值
                batch_hd95 = np.nanmean(hd95_val_list)
                print(f"Batch {current_step} evaluation completed | ASD: {batch_asd:.4f} | HD95: {batch_hd95:.4f}")

            except tf.errors.OutOfRangeError:
                break

        # 计算平均指标（使用np.nanmean忽略nan值）
        avg_iou = np.mean(all_iou) if all_iou else 0.0
        avg_acc = np.mean(all_acc) if all_acc else 0.0
        avg_precision = np.mean(all_precision) if all_precision else 0.0
        avg_recall = np.mean(all_recall) if all_recall else 0.0
        avg_f1 = np.mean(all_f1) if all_f1 else 0.0
        avg_dsc = np.mean(all_dsc) if all_dsc else 0.0
        avg_asd = np.nanmean(all_asd) if all_asd else 0.0  # 忽略nan计算均值
        avg_hd95 = np.nanmean(all_hd95) if all_hd95 else 0.0  # 忽略nan计算均值

        # 整理结果
        final_metrics = {
            "IoU（交并比）": round(avg_iou, 4),
            "DSC（Dice相似系数）": round(avg_dsc, 4),
            "Precision（精确率）": round(avg_precision, 4),
            "Recall（召回率）": round(avg_recall, 4),
            "F1-Score": round(avg_f1, 4),
            "Accuracy（准确率）": round(avg_acc, 4),
            "ASD（平均表面距离）": round(avg_asd, 4) if not np.isnan(avg_asd) else np.nan,
            "HD95（95%豪斯多夫距离）": round(avg_hd95, 4) if not np.isnan(avg_hd95) else np.nan
        }

        # 打印结果
        print("\n" + "=" * 60)
        print("Final Quantitative Evaluation Results")
        print("=" * 60)
        for name, value in final_metrics.items():
            # 处理nan值的显示
            if np.isnan(value):
                print(f"{name}: NaN (部分样本无有效掩码)")
            else:
                print(f"{name}: {value}")
        print("=" * 60)

        # 可视化（适配nan/inf值）
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        metric_names = list(final_metrics.keys())
        metric_values = list(final_metrics.values())

        plt.figure(figsize=(15, 6))
        # 过滤nan值，避免绘图报错
        plot_names = []
        plot_values = []
        plot_colors = ['#2ecc71', '#1abc9c', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#8e44ad', '#16a085']
        for name, value, color in zip(metric_names, metric_values, plot_colors):
            if not np.isnan(value) and not np.isinf(value):
                plot_names.append(name)
                plot_values.append(value)
            else:
                plot_names.append(name)
                plot_values.append(0.0)  # NaN值显示为0，并标注

        bars = plt.bar(plot_names, plot_values, color=plot_colors[:len(plot_names)])

        # 添加数值标签（处理NaN/0值）
        for bar, name, value in zip(bars, plot_names, metric_values):
            height = bar.get_height()
            if np.isnan(value):
                label = "NaN"
            elif np.isinf(value):
                label = "inf"
            else:
                label = f'{value:.4f}'
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     label, ha='center', va='bottom', fontsize=10)

        # 适配y轴范围（排除NaN/inf值）
        valid_values = [v for v in metric_values if not np.isnan(v) and not np.isinf(v)]
        if valid_values:
            plt.ylim(0, max(valid_values) * 1.1)
        plt.title("裂缝分割模型量化评估结果", fontsize=14, fontweight='bold')
        plt.ylabel("指标值", fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=15)
        plt.tight_layout()

        # 确保保存目录存在
        os.makedirs(EVAL_RESULT_DIR, exist_ok=True)
        eval_result_path = os.path.join(EVAL_RESULT_DIR, "evaluation_results.png")
        plt.savefig(eval_result_path, dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    evaluate()
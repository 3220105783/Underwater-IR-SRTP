# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
from unet_dataset import build_dataset
from unet_model import UNet
from unet_utils import (calculate_iou_per_sample,
                        calculate_metrics_per_sample,
                        calculate_dsc_per_sample)
from config import (CUDA_VISIBLE_DEVICES, BATCH_SIZE, MODEL_SAVE_PATH,
                    VAL_IMG_DIR, VAL_MASK_DIR, EVAL_RESULT_DIR, THRESHOLD, SMOOTH)

# -------------------------- 配置参数 --------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# -------------------------- 评估指标 --------------------------
def calculate_asd(pred_mask, gt_mask):
    """计算平均表面距离（ASD）"""
    asd_list = []
    for i in range(pred_mask.shape[0]):
        pred = np.squeeze(pred_mask[i])
        gt = np.squeeze(gt_mask[i])

        if np.sum(pred) == 0 and np.sum(gt) == 0:
            asd_list.append(0.0)
            continue
        if np.sum(pred) == 0 or np.sum(gt) == 0:
            asd_list.append(np.nan)
            continue

        pred_dist = distance_transform_edt(1 - pred)
        gt_dist = distance_transform_edt(1 - gt)

        pred_surface = pred_dist[gt == 1]
        gt_surface = gt_dist[pred == 1]

        asd = (np.mean(pred_surface) + np.mean(gt_surface)) / 2.0
        asd_list.append(asd)
    return asd_list

def calculate_hd95(pred_mask, gt_mask):
    """修正HD95计算逻辑（基于directed_hausdorff）"""
    hd95_list = []
    for i in range(pred_mask.shape[0]):
        pred = np.squeeze(pred_mask[i])
        gt = np.squeeze(gt_mask[i])

        if np.sum(pred) == 0 and np.sum(gt) == 0:
            hd95_list.append(0.0)
            continue
        if np.sum(pred) == 0 or np.sum(gt) == 0:
            hd95_list.append(np.nan)
            continue

        # 正确HD95计算：双向有向豪斯多夫距离的95%分位数
        hd1 = directed_hausdorff(pred, gt)[0]
        hd2 = directed_hausdorff(gt, pred)[0]
        hd95 = np.percentile([hd1, hd2], 95)
        hd95_list.append(hd95)
    return hd95_list

# -------------------------- 评估主流程 --------------------------
def evaluate():
    # 构建图
    tf.reset_default_graph()

    # 加载验证集
    val_iter, val_batch, val_size = build_dataset(
        VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, is_train=False
    )

    # 构建模型
    inputs = tf.placeholder(tf.float32, [None, 768, 768, 3], name='inputs')
    masks = tf.placeholder(tf.float32, [None, 768, 768, 1], name='masks')

    unet = UNet()
    logits = unet.build_model(inputs, is_training=False)
    pred_masks = tf.cast(tf.sigmoid(logits) > THRESHOLD, tf.float32)

    # 评估指标（样本级）
    iou_per_sample = calculate_iou_per_sample(logits, masks, THRESHOLD, SMOOTH)
    metrics_per_sample = calculate_metrics_per_sample(logits, masks, THRESHOLD, SMOOTH)
    dsc_per_sample = calculate_dsc_per_sample(logits, masks, SMOOTH)

    # 加载模型
    saver = tf.train.Saver()

    # 启动Session
    with tf.Session(config=config) as sess:
        saver.restore(sess, MODEL_SAVE_PATH)
        print(f"Model loaded: {MODEL_SAVE_PATH}")

        # 初始化迭代器
        sess.run(val_iter.initializer)

        # 累计样本级指标
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
                # 运行样本级指标
                iou_val_list, metrics_val_dict, dsc_val_list, pred_masks_val = sess.run(
                    [iou_per_sample, metrics_per_sample, dsc_per_sample, pred_masks],
                    feed_dict={inputs: img_batch, masks: mask_batch}
                )

                # 计算ASD/HD95
                asd_val_list = calculate_asd(pred_masks_val, mask_batch)
                hd95_val_list = calculate_hd95(pred_masks_val, mask_batch)

                # 收集样本级指标
                all_iou.extend(iou_val_list)
                all_acc.extend(metrics_val_dict['accuracy'])
                all_precision.extend(metrics_val_dict['precision'])
                all_recall.extend(metrics_val_dict['recall'])
                all_f1.extend(metrics_val_dict['f1'])
                all_dsc.extend(dsc_val_list)
                all_asd.extend(asd_val_list)
                all_hd95.extend(hd95_val_list)

                # 打印batch日志
                current_step = len(all_asd) // BATCH_SIZE
                batch_asd = np.nanmean(asd_val_list)
                batch_hd95 = np.nanmean(hd95_val_list)
                print(f"Batch {current_step} | ASD: {batch_asd:.4f} | HD95: {batch_hd95:.4f}")

            except tf.errors.OutOfRangeError:
                break

        # 计算平均指标
        avg_iou = np.mean(all_iou) if all_iou else 0.0
        avg_acc = np.mean(all_acc) if all_acc else 0.0
        avg_precision = np.mean(all_precision) if all_precision else 0.0
        avg_recall = np.mean(all_recall) if all_recall else 0.0
        avg_f1 = np.mean(all_f1) if all_f1 else 0.0
        avg_dsc = np.mean(all_dsc) if all_dsc else 0.0
        avg_asd = np.nanmean(all_asd) if all_asd else 0.0
        avg_hd95 = np.nanmean(all_hd95) if all_hd95 else 0.0

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
            if np.isnan(value):
                print(f"{name}: NaN (部分样本无有效掩码)")
            else:
                print(f"{name}: {value}")

        # 保存结果到文件
        os.makedirs(EVAL_RESULT_DIR, exist_ok=True)
        eval_path = os.path.join(EVAL_RESULT_DIR, "evaluation_results.txt")
        with open(eval_path, 'w', encoding='utf-8') as f:
            f.write("Evaluation Results\n")
            f.write("=" * 60 + "\n")
            for name, value in final_metrics.items():
                if np.isnan(value):
                    f.write(f"{name}: NaN\n")
                else:
                    f.write(f"{name}: {value}\n")
        print(f"\nResults saved to: {eval_path}")

if __name__ == "__main__":
    evaluate()
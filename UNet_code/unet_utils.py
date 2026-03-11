# -*- coding: utf-8 -*-
import tensorflow as tf


def focal_loss(pred_logits, target, gamma=2, alpha=50.0):
    """TF1.x版Focal Loss（解决样本不平衡）"""
    # BCE损失（logits输入）
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=pred_logits)

    # 聚焦因子
    pt = tf.exp(-bce)
    focal = (1 - pt) ** gamma * bce

    # 类别权重（裂缝权重alpha，背景权重1）
    alpha_t = tf.where(tf.equal(target, 1.0), alpha, 1.0)
    focal = alpha_t * focal

    return tf.reduce_mean(focal)


def dice_loss(pred_logits, target, smooth=1e-6):
    """TF1.x版Dice Loss（小目标优化）"""
    pred = tf.nn.sigmoid(pred_logits)

    # 展平（统一target类型转换）
    pred_flat = tf.reshape(pred, [-1])
    target_flat = tf.reshape(tf.cast(target, tf.float32), [-1])  # 新增类型转换

    # Dice系数
    intersection = tf.reduce_sum(pred_flat * target_flat)
    union = tf.reduce_sum(pred_flat) + tf.reduce_sum(target_flat)
    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice


def focal_dice_loss(pred_logits, target, focal_weight=0.4, dice_weight=0.6):
    """TF1.x版组合损失（Focal+Dice）"""
    focal = focal_loss(pred_logits, target)
    dice = dice_loss(pred_logits, target)
    return focal_weight * focal + dice_weight * dice


# 样本级IoU计算（返回每个样本的IoU）
def calculate_iou_per_sample(pred_logits, target, threshold=0.5, smooth=1e-6):
    pred = tf.nn.sigmoid(pred_logits) > threshold
    pred = tf.cast(pred, tf.float32)
    target = tf.cast(target, tf.float32)

    # 保留batch维度计算（shape: [batch,]）
    intersection = tf.reduce_sum(tf.reshape(pred * target, [tf.shape(pred)[0], -1]), axis=1)
    union = tf.reduce_sum(tf.reshape(pred, [tf.shape(pred)[0], -1]), axis=1) + \
            tf.reduce_sum(tf.reshape(target, [tf.shape(target)[0], -1]), axis=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

# 样本级Metrics计算（返回每个样本的指标字典）
def calculate_metrics_per_sample(pred_logits, target, threshold=0.5, smooth=1e-6):
    pred = tf.nn.sigmoid(pred_logits) > threshold
    pred = tf.cast(pred, tf.float32)
    target = tf.cast(target, tf.float32)

    # 展平（保留batch维度）
    pred_flat = tf.reshape(pred, [tf.shape(pred)[0], -1])
    target_flat = tf.reshape(target, [tf.shape(target)[0], -1])

    # 样本级混淆矩阵
    tp = tf.reduce_sum(tf.cast((pred_flat == 1) & (target_flat == 1), tf.float32), axis=1)
    tn = tf.reduce_sum(tf.cast((pred_flat == 0) & (target_flat == 0), tf.float32), axis=1)
    fp = tf.reduce_sum(tf.cast((pred_flat == 1) & (target_flat == 0), tf.float32), axis=1)
    fn = tf.reduce_sum(tf.cast((pred_flat == 0) & (target_flat == 1), tf.float32), axis=1)

    # 样本级指标计算
    accuracy = (tp + tn) / (tp + tn + fp + fn + smooth)
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)  # 修复原代码recall计算错误（原代码误用fp）
    f1 = 2 * precision * recall / (precision + recall + smooth)

    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'f1': f1
    }

# 样本级DSC计算（返回每个样本的DSC）
def calculate_dsc_per_sample(pred_logits, target, smooth=1e-6):
    pred = tf.nn.sigmoid(pred_logits)
    pred_flat = tf.reshape(pred, [tf.shape(pred)[0], -1])
    target_flat = tf.reshape(tf.cast(target, tf.float32), [tf.shape(target)[0], -1])

    intersection = tf.reduce_sum(pred_flat * target_flat, axis=1)
    dsc = (2. * intersection + smooth) / (tf.reduce_sum(pred_flat, axis=1) + tf.reduce_sum(target_flat, axis=1) + smooth)
    return dsc
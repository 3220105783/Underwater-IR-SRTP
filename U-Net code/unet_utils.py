# -*- coding: utf-8 -*-
import tensorflow as tf

def calculate_iou(pred_logits, target, threshold=0.5, smooth=1e-6):
    """TF1.x版IoU计算（匹配原逻辑）"""
    # Sigmoid+二值化
    pred = tf.nn.sigmoid(pred_logits) > threshold
    pred = tf.cast(pred, tf.float32)
    target = tf.cast(target, tf.float32)

    # 展平
    pred_flat = tf.reshape(pred, [-1])
    target_flat = tf.reshape(target, [-1])

    # 计算交并集
    intersection = tf.reduce_sum(pred_flat * target_flat)
    union = tf.reduce_sum(pred_flat) + tf.reduce_sum(target_flat) - intersection

    # IoU（平滑项避免除零）
    iou = (intersection + smooth) / (union + smooth)
    return iou


def focal_loss(pred_logits, target, gamma=2, alpha=50.0, smooth=1e-6):
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

    # 展平
    pred_flat = tf.reshape(pred, [-1])
    target_flat = tf.reshape(target, [-1])

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


def calculate_metrics(pred_logits, target):
    """计算Precision/Recall/F1/Accuracy（TF1.x版）"""
    pred = tf.nn.sigmoid(pred_logits) > 0.5
    pred = tf.cast(pred, tf.float32)
    target = tf.cast(target, tf.float32)

    # 展平
    pred_flat = tf.reshape(pred, [-1])
    target_flat = tf.reshape(target, [-1])

    # 混淆矩阵
    TP = tf.reduce_sum((pred_flat == 1) & (target_flat == 1))
    TN = tf.reduce_sum((pred_flat == 0) & (target_flat == 0))
    FP = tf.reduce_sum((pred_flat == 1) & (target_flat == 0))
    FN = tf.reduce_sum((pred_flat == 0) & (target_flat == 1))

    # 指标计算（添加平滑项避免除零）
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'f1': f1
    }


# 新增DSC计算函数
def calculate_dsc(pred_logits, target, smooth=1e-6):
    """计算DSC（Dice Similarity Coefficient）"""
    pred = tf.nn.sigmoid(pred_logits)
    pred_flat = tf.reshape(pred, [-1])
    target_flat = tf.reshape(tf.cast(target, tf.float32), [-1])

    intersection = tf.reduce_sum(pred_flat * target_flat)
    dsc = (2. * intersection + smooth) / (tf.reduce_sum(pred_flat) + tf.reduce_sum(target_flat) + smooth)
    return dsc
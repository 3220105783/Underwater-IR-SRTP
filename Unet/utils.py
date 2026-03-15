# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import config

# 设置matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 评估指标函数（TF1.x） =====================
def iou_score(y_true, y_pred, name='iou_score'):
    """
    TF1.x 计算IoU
    """
    with tf.variable_scope(name):
        y_pred = tf.round(y_pred)  # 二值化
        intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=[1, 2, 3])
        union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
        iou = (intersection + tf.keras.backend.epsilon()) / (union + tf.keras.backend.epsilon())
        return tf.reduce_mean(iou)


def dice_coefficient(y_true, y_pred, name='dice_coefficient'):
    """
    TF1.x 计算Dice系数
    """
    with tf.variable_scope(name):
        y_pred = tf.round(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        dice = (2. * intersection + tf.keras.backend.epsilon()) / (
                tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred,
                                                                      axis=[1, 2, 3]) + tf.keras.backend.epsilon()
        )
        return tf.reduce_mean(dice)


def precision(y_true, y_pred, name='precision'):
    """
    TF1.x 计算精确率
    """
    with tf.variable_scope(name):
        y_pred = tf.round(y_pred)
        true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
        predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        return precision


def recall(y_true, y_pred, name='recall'):
    """
    TF1.x 计算召回率
    """
    with tf.variable_scope(name):
        y_pred = tf.round(y_pred)
        true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
        possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
        return recall


def f1_score(y_true, y_pred, name='f1_score'):
    """
    TF1.x 计算F1分数
    """
    with tf.variable_scope(name):
        prec = precision(y_true, y_pred)
        rec = recall(y_true, y_pred)
        f1 = 2 * ((prec * rec) / (prec + rec + tf.keras.backend.epsilon()))
        return f1


# ===================== 损失函数（TF1.x） =====================
def focal_loss(y_true, y_pred, alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA, name='focal_loss'):
    """
    TF1.x Focal Loss
    """
    with tf.variable_scope(name):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Binary focal loss
        p_t = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)

        # 关键修复：将alpha转换为与y_true形状相同的张量
        alpha_tensor = tf.ones_like(y_true) * alpha  # 生成和y_true同形状的alpha张量
        alpha_t = tf.where(tf.equal(y_true, 1), alpha_tensor, 1 - alpha_tensor)

        focal_loss = -alpha_t * tf.pow((1 - p_t), gamma) * tf.log(p_t)

        return tf.reduce_mean(focal_loss)


def bce_dice_focal_iou_loss(y_true, y_pred, name='combined_loss'):
    """
    TF1.x 组合损失函数
    """
    with tf.variable_scope(name):
        # 修复BCE损失：适配sigmoid输出（y_pred∈[0,1]）
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        bce = tf.reduce_mean(
            - (y_true * tf.log(y_pred) + (1 - y_true) * tf.log(1 - y_pred))
        )

        # Dice损失
        dice = 1 - dice_coefficient(y_true, y_pred)

        # Focal损失
        focal = focal_loss(y_true, y_pred)

        # IoU损失
        iou = 1 - iou_score(y_true, y_pred)

        # 加权组合
        total_loss = (
                config.BCE_WEIGHT * bce +
                config.DICE_WEIGHT * dice +
                config.FOCAL_WEIGHT * focal +
                config.IOU_WEIGHT * iou
        )

        return total_loss


# ===================== 可视化函数 =====================
def plot_training_history(history, save_path):
    """
    绘制训练历史曲线
    """
    metrics = ['loss', 'iou_score', 'accuracy', 'precision', 'recall', 'f1_score']
    val_metrics = ['val_' + m for m in metrics]

    plt.figure(figsize=(15, 10))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i + 1)
        if metric in history:
            plt.plot(history[metric], label=f'Train {metric}')
        if val_metrics[i] in history:
            plt.plot(history[val_metrics[i]], label=f'Val {metric}')
        plt.title(f'{metric} over epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def visualize_prediction(image, mask, pred_mask, save_path):
    """
    可视化预测结果
    """
    # 转换为RGB图像
    image = (image * 255).astype(np.uint8)
    mask = (mask * 255).astype(np.uint8)
    pred_mask = (pred_mask * 255).astype(np.uint8)

    # 创建叠加图像
    overlay = image.copy()
    pred_mask_rgb = np.zeros_like(image)
    pred_mask_rgb[:, :, 0] = pred_mask[:, :, 0]  # 红色通道
    overlay = cv2.addWeighted(overlay, 1 - config.PREDICTION_ALPHA,
                              pred_mask_rgb, config.PREDICTION_ALPHA, 0)

    # 检测裂缝区域并标注
    contours, _ = cv2.findContours(pred_mask.squeeze(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # 过滤小区域
            x, y, w, h = cv2.boundingRect(contour)
            cv2.putText(overlay, 'crack', (x, y - 10), config.FONT,
                        config.FONT_SCALE, config.FONT_COLOR, config.FONT_THICKNESS)

    # 创建拼接图像
    mask_3ch = cv2.cvtColor(mask.squeeze(), cv2.COLOR_GRAY2RGB)
    pred_mask_3ch = cv2.cvtColor(pred_mask.squeeze(), cv2.COLOR_GRAY2RGB)

    combined = np.hstack((image, mask_3ch, pred_mask_3ch, overlay))

    # 保存图像
    cv2.imwrite(save_path, combined)


# ===================== 模型保存与加载（TF1.x） =====================
def save_model_checkpoint(sess, saver, epoch, val_loss, is_best):
    """
    TF1.x 保存模型检查点
    """
    # 保存最佳模型
    if is_best:
        best_model_base = os.path.join(config.BEST_MODEL_DIR, 'best_model')
        # 定义best_model对应的所有后缀文件
        best_model_exts = ['.meta', '.index', '.data-00000-of-00001']

        # 第一步：删除旧的best_model相关文件（避免残留）
        for ext in best_model_exts:
            old_file = best_model_base + ext
            if os.path.exists(old_file):
                os.remove(old_file)
                print(f"Deleted old best model file: {old_file}")

        # 第二步：保存新的最佳模型
        saver.save(sess, best_model_base)
        print(f"Saved new best model to: {best_model_base}")

    # 保存最新模型
    model_filename = f'model_epoch_{epoch}_val_loss_{val_loss:.4f}'
    model_path = os.path.join(config.LATEST_MODEL_DIR, model_filename)
    saver.save(sess, model_path)

    # 清理旧模型
    model_files = sorted([f for f in os.listdir(config.LATEST_MODEL_DIR) if f.endswith('.meta')],
                         key=lambda x: os.path.getctime(os.path.join(config.LATEST_MODEL_DIR, x)))
    if len(model_files) > config.KEEP_LATEST_MODELS:
        for old_meta in model_files[:-config.KEEP_LATEST_MODELS]:
            # 删除相关文件
            base_name = old_meta[:-5]  # 去掉.meta后缀
            for ext in ['.meta', '.index', '.data-00000-of-00001']:
                file_path = os.path.join(config.LATEST_MODEL_DIR, base_name + ext)
                if os.path.exists(file_path):
                    os.remove(file_path)
            print(f"Deleting old model file: {base_name}")


def load_best_model(sess, model_path=None):
    """
    TF1.x 加载最佳模型
    """
    if model_path is None:
        model_path = os.path.join(config.BEST_MODEL_DIR, 'best_model')

    if not os.path.exists(model_path + '.meta'):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    # 创建Saver
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print(f"Successfully loaded model: {model_path}")

    return saver


# ===================== 评估结果保存 =====================
def save_evaluation_results(results, save_path):
    """
    保存评估结果到文件
    """
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("U-Net Model Evaluation Results (Concrete Crack Detection)\n")
        f.write("=" * 50 + "\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")

    print(f"Evaluation results have been saved to: {save_path}")
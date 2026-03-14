# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import config
from unet_model import unet_model
from unet_dataset import load_image_mask_pairs, preprocess_image_mask
from utils import (iou_score, dice_coefficient, precision, recall, f1_score,
                   load_best_model, visualize_prediction, save_evaluation_results)


def predict_single_image(sess, inputs, logits, is_training_pl, img_path):
    """
    TF1.x 预测单张图像
    """
    # 预处理图像
    image, _ = preprocess_image_mask(img_path, None, augment=False)
    image_batch = np.expand_dims(image, axis=0)

    # 预测
    pred_mask = sess.run(
        logits,
        feed_dict={
            inputs: image_batch,
            is_training_pl: False
        }
    )[0]
    pred_mask = np.where(pred_mask > config.THRESHOLD, 1.0, 0.0)

    return image, pred_mask


def evaluate_model(sess, inputs, logits, is_training_pl, test_img_paths, test_mask_paths):
    """
    TF1.x 评估模型性能
    """
    all_iou = []
    all_dice = []
    all_precision = []
    all_recall = []
    all_f1 = []

    # 创建评估指标计算图
    labels_pl = tf.placeholder(tf.float32, [None, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1], name='labels_pl')
    iou_op = iou_score(labels_pl, logits)
    dice_op = dice_coefficient(labels_pl, logits)
    prec_op = precision(labels_pl, logits)
    rec_op = recall(labels_pl, logits)
    f1_op = f1_score(labels_pl, logits)

    for img_path, mask_path in zip(test_img_paths, test_mask_paths):
        # 加载数据
        image, true_mask = preprocess_image_mask(img_path, mask_path)
        image_batch = np.expand_dims(image, axis=0)
        mask_batch = np.expand_dims(true_mask, axis=0)

        # 计算指标
        iou_val, dice_val, prec_val, rec_val, f1_val = sess.run(
            [iou_op, dice_op, prec_op, rec_op, f1_op],
            feed_dict={
                inputs: image_batch,
                labels_pl: mask_batch,
                is_training_pl: False
            }
        )

        all_iou.append(iou_val)
        all_dice.append(dice_val)
        all_precision.append(prec_val)
        all_recall.append(rec_val)
        all_f1.append(f1_val)

    # 计算平均指标
    results = {
        'mean_iou': np.mean(all_iou),
        'mean_dice': np.mean(all_dice),
        'mean_precision': np.mean(all_precision),
        'mean_recall': np.mean(all_recall),
        'mean_f1': np.mean(all_f1),
        'std_iou': np.std(all_iou),
        'std_dice': np.std(all_dice)
    }

    return results


def test_unet():
    """
    TF1.x 测试U-Net模型
    """
    # 重置图
    tf.reset_default_graph()

    # 1. 创建模型图
    inputs = tf.placeholder(
        tf.float32,
        [None, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS],
        name='inputs'
    )
    is_training_pl = tf.placeholder(tf.bool, name='is_training')
    logits = unet_model(inputs, is_training=is_training_pl)

    # 2. 创建Session
    sess = tf.Session(config=config.TF_CONFIG)

    # 3. 加载最佳模型
    print("Loading the best model...")
    try:
        load_best_model(sess)
    except FileNotFoundError as e:
        print(f"Model loading failed: {e}")
        sess.close()
        return

    # 4. 加载测试数据集
    print("Loading test dataset...")
    test_img_paths, test_mask_paths = load_image_mask_pairs(config.TEST_IMG_DIR, config.TEST_MASK_DIR)
    print(f"Number of test set samples: {len(test_img_paths)}")

    if len(test_img_paths) == 0:
        print("Test set is empty!")
        sess.close()
        return

    # 5. 预测模式
    if config.PREDICT_MODE:
        print("Starting prediction and generating results...")
        for idx, (img_path, mask_path) in enumerate(zip(test_img_paths, test_mask_paths)):
            # 预测
            image, pred_mask = predict_single_image(sess, inputs, logits, is_training_pl, img_path)

            # 加载真实掩码
            _, true_mask = preprocess_image_mask(img_path, mask_path)

            # 保存可视化结果
            img_name = os.path.basename(img_path)
            save_path = os.path.join(config.TEST_OUTPUT_DIR, f'pred_{img_name}')
            visualize_prediction(image, true_mask, pred_mask, save_path)

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(test_img_paths)} images")

        print(f"Prediction results saved to: {config.TEST_OUTPUT_DIR}")

    # 6. 评估模式
    if config.EVALUATE_MODE:
        print("Starting model performance evaluation...")
        results = evaluate_model(sess, inputs, logits, is_training_pl, test_img_paths, test_mask_paths)

        # 打印评估结果
        print("\nModel evaluation results:")
        print("=" * 50)
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")

        # 保存评估结果
        if config.SAVE_EVALUATION_RESULTS:
            save_evaluation_results(results, config.EVALUATION_SAVE_PATH)

    # 7. 关闭Session
    sess.close()
    print("\nTest completed!")


if __name__ == "__main__":
    # 设置GPU内存增长
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # 运行测试
    test_unet()
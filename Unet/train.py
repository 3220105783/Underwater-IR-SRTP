# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import config
from unet_model import unet_model
from unet_dataset import get_train_val_datasets
from utils import (iou_score, dice_coefficient, precision, recall, f1_score,
                   bce_dice_focal_iou_loss, plot_training_history, save_model_checkpoint)


def lr_scheduler(global_step):
    """
    TF1.x 学习率调度器（修复类型不匹配问题）
    """
    # 1. 将所有参数转换为TensorFlow张量（浮点类型）
    global_step_float = tf.cast(global_step, tf.float32)
    decay_steps_float = tf.cast(config.DECAY_STEPS, tf.float32)
    initial_lr = tf.cast(config.INITIAL_LEARNING_RATE, tf.float32)
    decay_rate = tf.cast(config.DECAY_RATE, tf.float32)

    # 2. 使用TF操作替代Python原生运算（避免类型错误）
    decay_exponent = tf.floor_div(global_step_float, decay_steps_float)  # 整数除法（浮点结果）
    lr = initial_lr * tf.pow(decay_rate, decay_exponent)

    # 3. 确保学习率不低于最小值（可选，增强鲁棒性）
    if hasattr(config, 'MIN_LEARNING_RATE'):
        min_lr = tf.cast(config.MIN_LEARNING_RATE, tf.float32)
        lr = tf.maximum(lr, min_lr)

    return lr


def train_unet():
    """
    TF1.x 训练U-Net模型
    """
    # 重置图
    tf.reset_default_graph()

    # 1. 准备数据集
    print("Loading dataset...")
    train_batch, val_batch, train_steps, val_steps, train_len, val_len = get_train_val_datasets()
    print(f"Training set: {train_len}sample(s), {train_steps}step(s)/epoch")
    print(f"Validation set: {val_len}sample(s), {val_steps}step(s)/epoch")

    # 2. 创建占位符
    inputs = tf.placeholder(
        tf.float32,
        [None, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS],
        name='inputs'
    )
    labels = tf.placeholder(
        tf.float32,
        [None, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 1],
        name='labels'
    )
    is_training = tf.placeholder(tf.bool, name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # 3. 构建模型
    print("Building U-Net model...")
    logits = unet_model(inputs, is_training=is_training)

    # 4. 定义损失和优化器
    print("Define loss and optimizer...")
    # 损失函数
    loss = bce_dice_focal_iou_loss(labels, logits)
    # L2正则化损失
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = loss + tf.reduce_sum(reg_losses)

    # 学习率
    lr = lr_scheduler(global_step)
    tf.summary.scalar('learning_rate', lr)

    # 优化器（带BN更新）
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # 梯度裁剪
        grads_and_vars = optimizer.compute_gradients(total_loss)
        clipped_grads = [(tf.clip_by_norm(grad, config.CLIP_NORM), var)
                         for grad, var in grads_and_vars if grad is not None]
        train_op = optimizer.apply_gradients(clipped_grads, global_step=global_step)

    # 5. 定义评估指标
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(logits), labels), tf.float32), name='accuracy')
    iou = iou_score(labels, logits)
    dice = dice_coefficient(labels, logits)
    prec = precision(labels, logits)
    rec = recall(labels, logits)
    f1 = f1_score(labels, logits)

    # 6. TensorBoard汇总
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.scalar('iou_score', iou)
    tf.summary.scalar('precision', prec)
    tf.summary.scalar('recall', rec)
    tf.summary.scalar('f1_score', f1)
    tf.summary.scalar('dice_coefficient', dice)

    # 合并所有汇总
    summary_op = tf.summary.merge_all()

    # 7. 创建Session
    sess = tf.Session(config=config.TF_CONFIG)
    saver = tf.train.Saver(max_to_keep=config.KEEP_LATEST_MODELS)

    # 初始化变量
    sess.run(tf.global_variables_initializer())

    # 初始化TensorBoard
    train_writer = tf.summary.FileWriter(os.path.join(config.TENSORBOARD_LOG_DIR, 'train'), sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join(config.TENSORBOARD_LOG_DIR, 'val'))

    # 8. 训练循环
    print("Start training...")
    best_val_loss = config.BEST_VAL_LOSS
    stopping_counter = config.STOPPING_COUNTER
    training_history = {
        'loss': [], 'accuracy': [], 'iou_score': [], 'precision': [], 'recall': [], 'f1_score': [],
        'val_loss': [], 'val_accuracy': [], 'val_iou_score': [], 'val_precision': [], 'val_recall': [],
        'val_f1_score': []
    }

    for epoch in range(config.EPOCHS):
        # 训练阶段
        train_losses = []
        train_accuracies = []
        train_ious = []

        for step in range(train_steps):
            # 获取批次数据
            batch_images, batch_masks = sess.run(train_batch)

            # 训练步骤
            _, loss_val, acc_val, iou_val, summary = sess.run(
                [train_op, total_loss, accuracy, iou, summary_op],
                feed_dict={
                    inputs: batch_images,
                    labels: batch_masks,
                    is_training: True
                }
            )

            train_losses.append(loss_val)
            train_accuracies.append(acc_val)
            train_ious.append(iou_val)

            # 写入TensorBoard
            if step % 10 == 0:
                train_writer.add_summary(summary, epoch * train_steps + step)

        # 验证阶段
        val_losses = []
        val_accuracies = []
        val_ious = []

        for step in range(val_steps):
            batch_images, batch_masks = sess.run(val_batch)

            loss_val, acc_val, iou_val, summary = sess.run(
                [total_loss, accuracy, iou, summary_op],
                feed_dict={
                    inputs: batch_images,
                    labels: batch_masks,
                    is_training: False
                }
            )

            val_losses.append(loss_val)
            val_accuracies.append(acc_val)
            val_ious.append(iou_val)

            if step % 10 == 0:
                val_writer.add_summary(summary, epoch * val_steps + step)

        # 计算平均指标
        avg_train_loss = np.mean(train_losses)
        avg_train_acc = np.mean(train_accuracies)
        avg_train_iou = np.mean(train_ious)

        avg_val_loss = np.mean(val_losses)
        avg_val_acc = np.mean(val_accuracies)
        avg_val_iou = np.mean(val_ious)

        # 保存历史
        training_history['loss'].append(avg_train_loss)
        training_history['accuracy'].append(avg_train_acc)
        training_history['iou_score'].append(avg_train_iou)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_accuracy'].append(avg_val_acc)
        training_history['val_iou_score'].append(avg_val_iou)

        # 打印日志
        print(f"Epoch {epoch + 1}/{config.EPOCHS}:")
        print(f"  Training loss: {avg_train_loss:.4f}, Training accuracy: {avg_train_acc:.4f}, Training IoU: {avg_train_iou:.4f}")
        print(f"  Validation loss: {avg_val_loss:.4f}, Validation accuracy: {avg_val_acc:.4f}, Validation IoU: {avg_val_iou:.4f}")

        # 早停机制
        if avg_val_loss < best_val_loss - config.EARLY_STOPPING_MIN_DELTA:
            best_val_loss = avg_val_loss
            stopping_counter = 0
            # 保存最佳模型
            save_model_checkpoint(sess, saver, epoch + 1, avg_val_loss, is_best=True)
        else:
            stopping_counter += 1
            print(f"  Early Stop Counter: {stopping_counter}/{config.EARLY_STOPPING_PATIENCE}")

        # 定期保存模型
        if (epoch + 1) % config.SAVE_CHECKPOINT_EVERY_N_EPOCHS == 0:
            save_model_checkpoint(sess, saver, epoch + 1, avg_val_loss)

        # 检查早停
        if stopping_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered! Best validation loss: {best_val_loss:.4f}")
            break

    # 9. 训练结束
    train_writer.close()
    val_writer.close()

    # 绘制训练历史
    history_plot_path = os.path.join(config.RESULTS_DIR, 'training_history.png')
    plot_training_history(training_history, history_plot_path)
    print(f"The training history graph has been saved to：{history_plot_path}")

    # 保存最终模型
    save_model_checkpoint(sess, saver, config.EPOCHS, best_val_loss)

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    sess.close()
    return training_history


if __name__ == "__main__":
    # 设置GPU内存增长
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    # 开始训练
    training_history = train_unet()
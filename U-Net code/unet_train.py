# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from unet_dataset import build_dataset
from unet_model import UNet
from unet_utils import focal_dice_loss, calculate_iou

# -------------------------- 配置参数 --------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 动态显存分配

# 训练参数
BATCH_SIZE = 1
EPOCHS = 512
LEARNING_RATE = 5e-5
PATIENCE = 30  # 早停耐心值
MODEL_SAVE_PATH = "/model/best_crack_model.ckpt"

# 数据集路径
TRAIN_IMG_DIR = "/dataset/train/img"
TRAIN_MASK_DIR = "/dataset/train/mask"
VAL_IMG_DIR = "/dataset/val/img"
VAL_MASK_DIR = "/dataset/val/mask"


# -------------------------- 构建训练图 --------------------------
def build_train_graph():
    # 1. 加载数据集
    train_iter, train_batch, train_size = build_dataset(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR, BATCH_SIZE, is_train=True
    )
    val_iter, val_batch, val_size = build_dataset(
        VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, is_train=False
    )

    # 2. 构建模型
    inputs = tf.placeholder(tf.float32, [None, 512, 512, 3], name='inputs')
    masks = tf.placeholder(tf.float32, [None, 512, 512, 1], name='masks')

    unet = UNet(n_channels=3, n_classes=1)
    logits = unet.build_model(inputs)

    # 3. 损失函数+优化器
    loss = focal_dice_loss(logits, masks)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(loss)

    # 4. 评估指标
    iou = calculate_iou(logits, masks)

    # 5. 模型保存
    saver = tf.train.Saver(max_to_keep=1)

    return {
        'train_iter': train_iter, 'train_batch': train_batch, 'train_size': train_size,
        'val_iter': val_iter, 'val_batch': val_batch, 'val_size': val_size,
        'inputs': inputs, 'masks': masks,
        'train_op': train_op, 'loss': loss, 'iou': iou,
        'saver': saver
    }


# -------------------------- 训练主流程 --------------------------
def train():
    # 构建图
    graph = build_train_graph()

    # 训练记录
    train_losses = []
    val_losses = []
    val_ious = []
    best_iou = 0.0
    patience_counter = 0

    # 启动Session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # 训练循环
        steps_per_epoch = graph['train_size'] // BATCH_SIZE
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            print("-" * 40)

            # 初始化训练迭代器
            sess.run(graph['train_iter'].initializer)

            # 训练一轮
            train_loss = 0.0
            for step in range(steps_per_epoch):
                img_batch, mask_batch = sess.run(graph['train_batch'])
                _, loss_val = sess.run(
                    [graph['train_op'], graph['loss']],
                    feed_dict={graph['inputs']: img_batch, graph['masks']: mask_batch}
                )
                train_loss += loss_val
                if step % 10 == 0:
                    print(f"Step {step}/{steps_per_epoch} - Loss: {loss_val:.4f}")

            avg_train_loss = train_loss / steps_per_epoch
            train_losses.append(avg_train_loss)

            # 验证一轮
            sess.run(graph['val_iter'].initializer)
            val_loss = 0.0
            val_iou = 0.0
            val_steps = 0

            while True:
                try:
                    img_batch, mask_batch = sess.run(graph['val_batch'])
                    loss_val, iou_val = sess.run(
                        [graph['loss'], graph['iou']],
                        feed_dict={graph['inputs']: img_batch, graph['masks']: mask_batch}
                    )
                    val_loss += loss_val
                    val_iou += iou_val
                    val_steps += 1
                except tf.errors.OutOfRangeError:
                    break

            avg_val_loss = val_loss / val_steps
            avg_val_iou = val_iou / val_steps
            val_losses.append(avg_val_loss)
            val_ious.append(avg_val_iou)

            # 打印本轮结果
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}")

            # 早停+保存最佳模型
            if avg_val_iou > best_iou:
                best_iou = avg_val_iou
                graph['saver'].save(sess, MODEL_SAVE_PATH)
                print(f"Saving best model (IoU: {best_iou:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Early stopping counter: {patience_counter}/{PATIENCE}")
                if patience_counter >= PATIENCE:
                    print("IoU has not improved for multiple consecutive epochs, stopping training early")
                    break

        # 训练完成
        print(f"\nTraining completed! Best Val IoU: {best_iou:.4f}")

        # 绘制训练曲线
        plt.rcParams['font.sans-serif'] = ['SimHei']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 损失曲线
        ax1.plot(train_losses, label='训练损失', color='#e74c3c')
        ax1.plot(val_losses, label='验证损失', color='#3498db')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失值')
        ax1.set_title('训练/验证损失曲线')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # IoU曲线
        ax2.plot(val_ious, label='验证IoU', color='#2ecc71', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('IoU值')
        ax2.set_title('验证IoU曲线')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history_tf1.png', dpi=150)
        plt.show()


if __name__ == "__main__":
    train()
# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from unet_dataset import build_dataset
from unet_model import UNet
from unet_utils import (focal_dice_loss, calculate_iou_per_sample,
                        calculate_metrics_per_sample,
                        calculate_dsc_per_sample)
from config import (CUDA_VISIBLE_DEVICES, BATCH_SIZE, EPOCHS, LEARNING_RATE, PATIENCE,
                    MODEL_SAVE_PATH, LATEST_MODEL_DIR, MAX_LATEST_MODELS, LOG_DIR,
                    LOAD_EXIST_MODEL, EXIST_MODEL_PATH, TRAIN_IMG_DIR, TRAIN_MASK_DIR,
                    VAL_IMG_DIR, VAL_MASK_DIR, TRAIN_RESULT_DIR, FOCAL_WEIGHT, DICE_WEIGHT)

# -------------------------- 配置参数 --------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# -------------------------- 保存最新模型 --------------------------
def save_latest_model(sess, saver, epoch, val_iou):
    """保存最新模型，修复冗余目录判断"""
    model_name = f"model_epoch_{epoch}_iou_{val_iou:.4f}"
    save_path = os.path.join(LATEST_MODEL_DIR, model_name)
    saver.save(sess, save_path)
    print(f"Saved latest model to: {save_path}")

    # 筛选模型文件（仅保留.ckpt相关文件）
    model_files = []
    for item in os.listdir(LATEST_MODEL_DIR):
        item_path = os.path.join(LATEST_MODEL_DIR, item)
        if item.endswith('.ckpt') or item.endswith('.index') or item.endswith('.data-00000-of-00001') or item.endswith(
                '.meta'):
            # 提取基础名称
            base_name = item.split('.ckpt')[0]
            mtime = os.path.getmtime(item_path)
            model_files.append((base_name, mtime))

    # 去重 + 排序
    unique_models = list({name: mtime for name, mtime in model_files}.items())
    unique_models.sort(key=lambda x: x[1])

    # 超出数量则删除最旧模型
    while len(unique_models) > MAX_LATEST_MODELS:
        oldest_model = unique_models.pop(0)[0]
        for ext in ['.ckpt.data-00000-of-00001', '.ckpt.index', '.ckpt.meta']:
            file_path = os.path.join(LATEST_MODEL_DIR, f"{oldest_model}{ext}")
            if os.path.exists(file_path):
                os.remove(file_path)
        print(f"Removed oldest model: {oldest_model}")


# -------------------------- 构建训练图 --------------------------
def build_train_graph():
    # 加载数据集
    train_iter, train_batch, train_size = build_dataset(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR, BATCH_SIZE, is_train=True
    )
    val_iter, val_batch, val_size = build_dataset(
        VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, is_train=False
    )

    # 构建模型
    inputs = tf.placeholder(tf.float32, [None, 768, 768, 3], name='inputs')
    masks = tf.placeholder(tf.float32, [None, 768, 768, 1], name='masks')
    unet = UNet()
    logits_train = unet.build_model(inputs, is_training=True)
    logits_val = unet.build_model(inputs, is_training=False)

    # 损失函数
    loss_train = focal_dice_loss(logits_train, masks, focal_weight=FOCAL_WEIGHT, dice_weight=DICE_WEIGHT)
    loss_val = focal_dice_loss(logits_val, masks, focal_weight=FOCAL_WEIGHT, dice_weight=DICE_WEIGHT)

    # 优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(loss_train)

    # 评估指标（支持样本级）
    iou_per_sample = calculate_iou_per_sample(logits_val, masks)
    iou = tf.reduce_mean(iou_per_sample)
    metrics_per_sample = calculate_metrics_per_sample(logits_val, masks)
    metrics = {k: tf.reduce_mean(v) for k, v in metrics_per_sample.items()}
    dsc_per_sample = calculate_dsc_per_sample(logits_val, masks)
    dsc = tf.reduce_mean(dsc_per_sample)

    # TensorBoard摘要
    tf.summary.scalar('train/loss', loss_train)
    tf.summary.scalar('val/iou', iou)
    tf.summary.scalar('val/accuracy', metrics['accuracy'])
    tf.summary.scalar('val/precision', metrics['precision'])
    tf.summary.scalar('val/recall', metrics['recall'])
    tf.summary.scalar('val/f1', metrics['f1'])
    tf.summary.scalar('val/dsc', dsc)
    tf.summary.scalar('val/loss', loss_val)
    merged_summary = tf.summary.merge_all()

    # 模型保存
    saver = tf.train.Saver(max_to_keep=None)

    return {
        'train_iter': train_iter, 'train_batch': train_batch, 'train_size': train_size,
        'val_iter': val_iter, 'val_batch': val_batch, 'val_size': val_size,
        'inputs': inputs, 'masks': masks,
        'train_op': train_op, 'loss_train': loss_train, 'loss_val': loss_val,
        'iou': iou, 'iou_per_sample': iou_per_sample,
        'metrics': metrics, 'metrics_per_sample': metrics_per_sample,
        'dsc': dsc, 'dsc_per_sample': dsc_per_sample,
        'merged_summary': merged_summary, 'saver': saver
    }


# -------------------------- 训练主流程 --------------------------
def train():
    graph = build_train_graph()
    train_losses = []
    val_losses = []
    val_ious = []
    best_iou = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    with tf.Session(config=config) as sess:
        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

        # 加载已有模型
        if LOAD_EXIST_MODEL:
            required_exts = ['.data-00000-of-00001', '.index', '.meta']
            model_files_exist = all([os.path.exists(EXIST_MODEL_PATH + ext) for ext in required_exts])
            if model_files_exist:
                try:
                    graph['saver'].restore(sess, EXIST_MODEL_PATH)
                    print(f"Loaded existing model: {EXIST_MODEL_PATH}")
                    # 验证模型
                    test_iou = sess.run(graph['iou'], feed_dict={
                        graph['inputs']: np.zeros((1, 768, 768, 3)),
                        graph['masks']: np.zeros((1, 768, 768, 1))
                    })
                    print(f"Model validated, test IoU: {test_iou:.4f}")
                except Exception as e:
                    print(f"Load model failed: {str(e)[:100]}")
                    sess.run(tf.global_variables_initializer())
            else:
                print(f"Model files missing: {EXIST_MODEL_PATH}")
                sess.run(tf.global_variables_initializer())
        else:
            sess.run(tf.global_variables_initializer())
            print("Training from scratch")

        steps_per_epoch = graph['train_size'] // BATCH_SIZE
        global_step = 0
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            print("-" * 40)

            # 训练批次（修复迭代器初始化：每个epoch重新初始化）
            sess.run(graph['train_iter'].initializer)
            train_loss = 0.0
            for step in range(steps_per_epoch):
                global_step += 1
                img_batch, mask_batch = sess.run(graph['train_batch'])
                _, loss_val, summary = sess.run(
                    [graph['train_op'], graph['loss_train'], graph['merged_summary']],
                    feed_dict={graph['inputs']: img_batch, graph['masks']: mask_batch}
                )
                summary_writer.add_summary(summary, global_step)
                train_loss += loss_val
                if step % 10 == 0:
                    print(f"Step {step}/{steps_per_epoch} - Loss: {loss_val:.4f}")
            avg_train_loss = train_loss / steps_per_epoch
            train_losses.append(avg_train_loss)

            # 验证批次
            sess.run(graph['val_iter'].initializer)
            val_loss, val_iou = 0.0, 0.0
            val_precision, val_recall, val_f1, val_dsc, val_acc = 0.0, 0.0, 0.0, 0.0, 0.0
            val_steps = 0
            while True:
                try:
                    img_batch, mask_batch = sess.run(graph['val_batch'])
                    # 运行样本级指标
                    loss_val_i, iou_per_sample_val, metrics_per_sample_val, dsc_per_sample_val, summary = sess.run(
                        [graph['loss_val'], graph['iou_per_sample'], graph['metrics_per_sample'],
                         graph['dsc_per_sample'], graph['merged_summary']],
                        feed_dict={graph['inputs']: img_batch, graph['masks']: mask_batch}
                    )
                    summary_writer.add_summary(summary, global_step)

                    # 累加样本级指标均值
                    val_loss += loss_val_i
                    val_iou += np.mean(iou_per_sample_val)
                    val_precision += np.mean(metrics_per_sample_val['precision'])
                    val_recall += np.mean(metrics_per_sample_val['recall'])
                    val_f1 += np.mean(metrics_per_sample_val['f1'])
                    val_dsc += np.mean(dsc_per_sample_val)
                    val_acc += np.mean(metrics_per_sample_val['accuracy'])
                    val_steps += 1
                except tf.errors.OutOfRangeError:
                    break

            # 计算验证平均指标
            avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0
            avg_val_iou = val_iou / val_steps if val_steps > 0 else 0.0
            avg_val_precision = val_precision / val_steps if val_steps > 0 else 0.0
            avg_val_recall = val_recall / val_steps if val_steps > 0 else 0.0
            avg_val_f1 = val_f1 / val_steps if val_steps > 0 else 0.0
            avg_val_dsc = val_dsc / val_steps if val_steps > 0 else 0.0
            avg_val_acc = val_acc / val_steps if val_steps > 0 else 0.0
            val_losses.append(avg_val_loss)
            val_ious.append(avg_val_iou)

            # 打印本轮指标
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"Val IoU: {avg_val_iou:.4f} | Val DSC: {avg_val_dsc:.4f}")
            print(
                f"Val Precision: {avg_val_precision:.4f} | Val Recall: {avg_val_recall:.4f} | Val F1: {avg_val_f1:.4f}")
            print(f"Val Accuracy: {avg_val_acc:.4f}")

            # 保存最优模型（修复早停逻辑）
            improved = False
            # 条件：IoU提升 或 损失下降且IoU未显著下降（仅当best_iou>0时判断）
            if (avg_val_iou > best_iou) or (
                    avg_val_loss < best_val_loss and (best_iou == 0 or (best_iou - avg_val_iou) < 0.01)):
                if avg_val_iou > best_iou:
                    best_iou = avg_val_iou
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                graph['saver'].save(sess, MODEL_SAVE_PATH)
                print(f"Saved best model (IoU: {best_iou:.4f}, Loss: {best_val_loss:.4f}) to {MODEL_SAVE_PATH}")
                save_latest_model(sess, graph['saver'], epoch + 1, avg_val_iou)
                patience_counter = 0
                improved = True
            # 每5epoch强制保存
            if (epoch + 1) % 5 == 0 and not improved:
                save_latest_model(sess, graph['saver'], epoch + 1, avg_val_iou)
                print(f"Forced save latest model at epoch {epoch + 1}")

            # 早停逻辑（修复初始阶段误判）
            if not improved:
                patience_counter += 1
                print(f"Early stopping counter: {patience_counter}/{PATIENCE}")
                if patience_counter >= PATIENCE:
                    print("Early stopping triggered (no improvement)")
                    break
            else:
                patience_counter = 0

        # 训练完成
        summary_writer.close()
        print(f"\nTraining completed! Best Val IoU: {best_iou:.4f}")

        # 绘制训练曲线
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
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
        # 隐藏多余子图
        ax3.axis('off')
        ax4.axis('off')
        plt.tight_layout()
        # 确保目录存在
        os.makedirs(TRAIN_RESULT_DIR, exist_ok=True)
        train_history_path = os.path.join(TRAIN_RESULT_DIR, "training_history.png")
        plt.savefig(train_history_path, dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    train()
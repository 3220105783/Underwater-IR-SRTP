# -*- coding: utf-8 -*-
import os
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from unet_dataset import build_dataset
from unet_model import UNet
from unet_utils import focal_dice_loss, calculate_iou, calculate_metrics, calculate_dsc

# -------------------------- 配置参数 --------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# 训练核心参数
BATCH_SIZE = 1
EPOCHS = 512
LEARNING_RATE = 5e-5
PATIENCE = 30  # 早停耐心值
MODEL_SAVE_PATH = "/model/best_model.ckpt"  # 最优模型覆盖保存路径（DEFAULT: "/model/best_model.ckpt"）
LATEST_MODEL_DIR = "/model/latest"  # 最新3个模型保存目录（DEFAULT: "/model/latest"）
MAX_LATEST_MODELS = 3  # 保留最新的3个模型
LOG_DIR = "./tensorboard_logs"  # TensorBoard日志保存路径（DEFAULT: "./tensorboard_logs"）

# ====================== 基于已有模型训练配置（核心修改）======================
LOAD_EXIST_MODEL = False  # 开关：True=使用已有模型继续训练 | False=从头开始训练
EXIST_MODEL_DIR = "/model/exist"  # 已有模型存放目录：将之前的任意模型放入此目录即可（DEFAULT: "/model/exist"）
EXIST_MODEL_NAME = "exist.ckpt"  # 已有模型的文件名（保持和训练保存的一致即可，DEFAULT: "exist.ckpt"）
EXIST_MODEL_PATH = os.path.join(EXIST_MODEL_DIR, EXIST_MODEL_NAME)
# ============================================================================

# 数据集路径
TRAIN_IMG_DIR = "/dataset/train/img"  #（DEFAULT: "/dataset/train/img"）
TRAIN_MASK_DIR = "/dataset/train/mask"  #（DEFAULT: "/dataset/train/mask"）
VAL_IMG_DIR = "/dataset/val/img"  #（DEFAULT: "/dataset/val/img"）
VAL_MASK_DIR = "/dataset/val/mask"  #（DEFAULT: "/dataset/val/mask"）


#训练结果保存路径
TRAIN_RESULT_DIR = "/results/train"  # 训练相关结果（DEFAULT: "/results/train"）
EVAL_RESULT_DIR = "/results/evaluation"    # 评估相关结果（DEFAULT: "/results/evaluation"）
PRED_RESULT_DIR = "/results/predict" # 预测相关结果（DEFAULT: "/results/predict"）


# -------------------------- 保存最新模型 --------------------------
def save_latest_model(sess, saver, epoch, val_iou):
    """保存最新模型，并只保留最近的3个"""
    model_name = f"model_epoch_{epoch}_iou_{val_iou:.4f}"
    save_path = os.path.join(LATEST_MODEL_DIR, model_name)
    saver.save(sess, save_path)
    print(f"Saved latest model to: {save_path}")

    # 筛选并排序模型文件（按修改时间，最旧在前）
    model_files = []
    for item in os.listdir(LATEST_MODEL_DIR):
        item_path = os.path.join(LATEST_MODEL_DIR, item)
        if os.path.isdir(item_path) or item.endswith('.ckpt'):
            base_name = item[:-5] if item.endswith('.ckpt') else item
            mtime = os.path.getmtime(item_path)
            model_files.append((base_name, mtime))
    model_files.sort(key=lambda x: x[1])

    # 超出数量则删除最旧模型
    while len(model_files) > MAX_LATEST_MODELS:
        oldest_model = model_files.pop(0)[0]
        for ext in ['.ckpt.data-00000-of-00001', '.ckpt.index', '.ckpt.meta', 'checkpoint']:
            file_path = os.path.join(LATEST_MODEL_DIR, f"{oldest_model}{ext}")
            if os.path.exists(file_path):
                os.remove(file_path)
        dir_path = os.path.join(LATEST_MODEL_DIR, oldest_model)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
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
    unet = UNet(n_channels=3, n_classes=1)
    logits = unet.build_model(inputs)

    # 损失函数+优化器
    loss = focal_dice_loss(logits, masks)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_op = optimizer.minimize(loss)

    # 评估指标
    iou = calculate_iou(logits, masks)
    metrics = calculate_metrics(logits, masks)
    dsc = calculate_dsc(logits, masks)

    # TensorBoard摘要
    tf.summary.scalar('train/loss', loss)
    tf.summary.scalar('val/iou', iou)
    tf.summary.scalar('val/accuracy', metrics['accuracy'])
    tf.summary.scalar('val/precision', metrics['precision'])
    tf.summary.scalar('val/recall', metrics['recall'])
    tf.summary.scalar('val/f1', metrics['f1'])
    tf.summary.scalar('val/dsc', dsc)
    tf.summary.scalar('val/loss', loss)
    merged_summary = tf.summary.merge_all()

    # 模型保存（不限制数量，手动管理）
    saver = tf.train.Saver(max_to_keep=None)

    return {
        'train_iter': train_iter, 'train_batch': train_batch, 'train_size': train_size,
        'val_iter': val_iter, 'val_batch': val_batch, 'val_size': val_size,
        'inputs': inputs, 'masks': masks,
        'train_op': train_op, 'loss': loss, 'iou': iou, 'metrics': metrics, 'dsc': dsc,
        'merged_summary': merged_summary, 'saver': saver
    }


# -------------------------- 训练主流程 --------------------------
def train():
    graph = build_train_graph()
    # 训练记录容器
    train_losses = []
    val_losses = []
    val_ious = []
    best_iou = 0.0
    patience_counter = 0

    # 启动Session
    with tf.Session(config=config) as sess:
        summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

        # ====================== 已有模型加载逻辑 =======================
        if LOAD_EXIST_MODEL:
            # 检查模型文件是否存在，避免加载失败
            required_exts = ['.ckpt.data-00000-of-00001', '.ckpt.index', '.ckpt.meta']
            model_files_exist = all([os.path.exists(EXIST_MODEL_PATH + ext) for ext in required_exts])
            if model_files_exist:
                try:
                    graph['saver'].restore(sess, EXIST_MODEL_PATH)
                    print(f"Successfully loaded existing model: {EXIST_MODEL_PATH}")
                    # 简单验证模型可用性
                    test_iou = sess.run(graph['iou'], feed_dict={
                        graph['inputs']: np.zeros((1, 768, 768, 3)),  #（DEFAULT：768，768）
                        graph['masks']: np.zeros((1, 768, 768, 1))  #（DEFAULT：768，768）
                    })
                    print(f"The existing model has been validated, basic IoU value: {test_iou:.4f}")
                except Exception as e:
                    print(f"Failed to load existing model, error message: {str(e)[:100]}")
                    print("Train starting from initializing model parameters from scratch")
                    sess.run(tf.global_variables_initializer())
            else:
                print(f"The existing model files are missing (must include {required_exts}), path: {EXIST_MODEL_PATH}")
                print("Train starting from initializing model parameters from scratch")
                sess.run(tf.global_variables_initializer())
        else:
            # 不加载已有模型，从头训练
            sess.run(tf.global_variables_initializer())
            print("Did not load an existing model; training begins from scratch with model parameters initialized.")
        # =================================================================

        # 训练循环（原有逻辑完全不变）
        steps_per_epoch = graph['train_size'] // BATCH_SIZE
        global_step = 0
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch + 1}/{EPOCHS}")
            print("-" * 40)
            sess.run(graph['train_iter'].initializer)

            # 训练批次
            train_loss = 0.0
            for step in range(steps_per_epoch):
                global_step += 1
                img_batch, mask_batch = sess.run(graph['train_batch'])
                _, loss_val, summary = sess.run(
                    [graph['train_op'], graph['loss'], graph['merged_summary']],
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
            val_loss, val_iou, val_precision, val_recall, val_f1, val_dsc, val_acc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            val_steps = 0
            while True:
                try:
                    img_batch, mask_batch = sess.run(graph['val_batch'])
                    loss_val, iou_val, metrics_val, dsc_val, summary = sess.run(
                        [graph['loss'], graph['iou'], graph['metrics'], graph['dsc'], graph['merged_summary']],
                        feed_dict={graph['inputs']: img_batch, graph['masks']: mask_batch}
                    )
                    summary_writer.add_summary(summary, global_step)
                    val_loss += loss_val
                    val_iou += iou_val
                    val_precision += metrics_val['precision']
                    val_recall += metrics_val['recall']
                    val_f1 += metrics_val['f1']
                    val_dsc += dsc_val
                    val_acc += metrics_val['accuracy']
                    val_steps += 1
                except tf.errors.OutOfRangeError:
                    break

            # 计算验证平均指标
            avg_val_loss = val_loss / val_steps
            avg_val_iou = val_iou / val_steps
            avg_val_precision = val_precision / val_steps
            avg_val_recall = val_recall / val_steps
            avg_val_f1 = val_f1 / val_steps
            avg_val_dsc = val_dsc / val_steps
            avg_val_acc = val_acc / val_steps
            val_losses.append(avg_val_loss)
            val_ious.append(avg_val_iou)

            # 打印本轮指标
            print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            print(f"Val IoU: {avg_val_iou:.4f} | Val DSC: {avg_val_dsc:.4f}")
            print(f"Val Precision: {avg_val_precision:.4f} | Val Recall: {avg_val_recall:.4f} | Val F1: {avg_val_f1:.4f}")
            print(f"Val Accuracy: {avg_val_acc:.4f}")

            # 保存最优模型+最新模型
            if avg_val_iou > best_iou:
                best_iou = avg_val_iou
                graph['saver'].save(sess, MODEL_SAVE_PATH)
                print(f"Saving best model (IoU: {best_iou:.4f}) to {MODEL_SAVE_PATH}")
                save_latest_model(sess, graph['saver'], epoch + 1, avg_val_iou)
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Early stopping counter: {patience_counter}/{PATIENCE}")
                if patience_counter >= PATIENCE:
                    print("IoU has not improved for multiple consecutive epochs, stopping training early")
                    break

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
        # 隐藏多余子图（保持原有布局，不改动）
        ax3.axis('off')
        ax4.axis('off')
        plt.tight_layout()
        train_history_path = os.path.join(TRAIN_RESULT_DIR, "training_history.png")
        plt.savefig(train_history_path, dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    train()
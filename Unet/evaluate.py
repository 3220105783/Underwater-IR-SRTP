import os
import matplotlib.pyplot as plt
import tensorflow as tf
from config import Config
from utils import create_dirs, set_gpu_config, count_files
from unet_model import build_compiled_model
from unet_dataset import data_generator

# 禁用 TensorFlow 2.x 行为
tf.disable_v2_behavior()


def evaluate():
    """评估模块：加载最佳模型并在验证/测试集上评估，绘制示例结果"""
    create_dirs()
    set_gpu_config()

    # 加载模型
    model = build_compiled_model()

    if not os.path.exists(Config.BEST_MODEL_PATH + '.index'):
        print(f"Best model not found at {Config.BEST_MODEL_PATH}")
        return
    model.load_weights(Config.BEST_MODEL_PATH)
    print(f"Loaded best model from {Config.BEST_MODEL_PATH}")

    # 数据生成器
    val_gen = data_generator(Config.VAL_IMG_DIR, Config.VAL_MASK_DIR, Config.BATCH_SIZE, Config.IMG_SIZE, augment=False)
    test_gen = data_generator(Config.TEST_IMG_DIR, Config.TEST_MASK_DIR, Config.BATCH_SIZE, Config.IMG_SIZE,
                              augment=False)

    num_val = count_files(Config.VAL_IMG_DIR)
    num_test = count_files(Config.TEST_IMG_DIR)
    val_steps = num_val // Config.BATCH_SIZE
    test_steps = num_test // Config.BATCH_SIZE

    # 评估
    print("\nEvaluating on validation set:")
    val_loss, val_acc, val_iou = model.evaluate(val_gen, steps=val_steps)
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val IoU: {val_iou:.4f}")

    print("\nEvaluating on test set:")
    test_loss, test_acc, test_iou = model.evaluate(test_gen, steps=test_steps)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test IoU: {test_iou:.4f}")

    # 绘制示例结果
    plt.figure(figsize=(12, 16))
    for i in range(5):
        img, mask = next(val_gen)
        pred = model.predict(img)

        plt.subplot(5, 3, i * 3 + 1)
        plt.imshow(img[0, :, :, 0], cmap='gray')
        plt.title('Input Image')
        plt.axis('off')

        plt.subplot(5, 3, i * 3 + 2)
        plt.imshow(mask[0, :, :, 0], cmap='gray')
        plt.title('Ground Truth')
        plt.axis('off')

        plt.subplot(5, 3, i * 3 + 3)
        plt.imshow(pred[0, :, :, 0], cmap='gray')
        plt.title('Prediction')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, 'evaluation_examples.png'))
    plt.close()
    print("\nEvaluation examples saved to results/evaluation_examples.png")


if __name__ == '__main__':
    evaluate()
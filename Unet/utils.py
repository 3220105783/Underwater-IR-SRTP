import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from config import Config


def create_dirs():
    """创建所需的文件夹结构"""
    dirs = [
        Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR,
        Config.VAL_IMG_DIR, Config.VAL_MASK_DIR,
        Config.TEST_IMG_DIR, Config.TEST_MASK_DIR,
        Config.BEST_MODEL_DIR, Config.LATEST_MODEL_DIR, Config.EXIST_MODEL_DIR,
        Config.RESULTS_DIR, Config.TENSORBOARD_DIR, Config.TEST_OUTPUT_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def set_gpu_config():
    """配置 GPU 内存增长"""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)


def count_files(dir_path):
    """统计目录下的文件数量"""
    return len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])


def plot_training_history(history):
    """绘制训练过程的损失、准确率和IoU曲线"""
    plt.figure(figsize=(12, 4))

    # 损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # IoU曲线
    plt.subplot(1, 3, 3)
    plt.plot(history['mean_io_u'], label='Train IoU')
    plt.plot(history['val_mean_io_u'], label='Val IoU')
    plt.title('Mean IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, 'training_history.png'))
    plt.close()


class CleanLatestModels(Callback):
    """自定义Callback：清理旧的latest模型（保留最近3个）"""

    def on_epoch_end(self, epoch, logs=None):
        files = [f for f in os.listdir(Config.LATEST_MODEL_DIR) if
                 f.startswith('latest_model_') and f.endswith('.ckpt.index')]
        if len(files) > 3:
            files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
            for f in files[:-3]:
                base = f.replace('.index', '')
                for ext in ['.index', '.data-00000-of-00001', '.meta']:
                    os.remove(os.path.join(Config.LATEST_MODEL_DIR, base + ext))


class SaveHistory(Callback):
    """自定义Callback：保存训练历史"""

    def on_train_begin(self, logs=None):
        self.history = {k: [] for k in ['loss', 'accuracy', 'mean_io_u', 'val_loss', 'val_accuracy', 'val_mean_io_u']}

    def on_epoch_end(self, epoch, logs=None):
        for k in self.history.keys():
            self.history[k].append(logs[k])
        np.save(os.path.join(Config.RESULTS_DIR, 'training_history.npy'), self.history)
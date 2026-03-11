import os
import tensorflow as tf
from config import Config
from utils import create_dirs, set_gpu_config, count_files, plot_training_history, CleanLatestModels, SaveHistory
from unet_model import build_compiled_model
from unet_dataset import data_generator

# 禁用 TensorFlow 2.x 行为
tf.disable_v2_behavior()


def train():
    """训练模块：包含早停、TensorBoard、继续训练和模型保存"""
    create_dirs()
    set_gpu_config()

    # 构建并编译模型
    model = build_compiled_model()

    # 继续训练：加载exist文件夹中的模型
    initial_epoch = 0
    exist_models = [f for f in os.listdir(Config.EXIST_MODEL_DIR) if f.endswith('.ckpt.index')]
    if exist_models:
        exist_models.sort(key=lambda x: os.path.getmtime(os.path.join(Config.EXIST_MODEL_DIR, x)), reverse=True)
        latest_exist = os.path.join(Config.EXIST_MODEL_DIR, exist_models[0].replace('.index', ''))
        print(f"Loading existing model from {latest_exist}")
        model.load_weights(latest_exist)

    # 数据生成器
    train_gen = data_generator(Config.TRAIN_IMG_DIR, Config.TRAIN_MASK_DIR, Config.BATCH_SIZE, Config.IMG_SIZE,
                               augment=True)
    val_gen = data_generator(Config.VAL_IMG_DIR, Config.VAL_MASK_DIR, Config.BATCH_SIZE, Config.IMG_SIZE, augment=False)

    # 计算步数
    num_train = count_files(Config.TRAIN_IMG_DIR)
    num_val = count_files(Config.VAL_IMG_DIR)
    steps_per_epoch = num_train // Config.BATCH_SIZE
    validation_steps = num_val // Config.BATCH_SIZE

    # Callbacks列表
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=Config.PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir=Config.TENSORBOARD_DIR, write_graph=True, write_images=True),
        tf.keras.callbacks.ModelCheckpoint(Config.BEST_MODEL_PATH, monitor='val_loss', save_best_only=True,
                                           save_weights_only=True),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(Config.LATEST_MODEL_DIR, 'latest_model_{epoch:02d}.ckpt'),
                                           save_weights_only=True),
        CleanLatestModels(),
        SaveHistory()
    ]

    # 开始训练
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=Config.EPOCHS,
        initial_epoch=initial_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    # 绘制训练曲线
    plot_training_history(history.history)


if __name__ == '__main__':
    train()
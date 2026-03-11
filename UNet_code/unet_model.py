# -*- coding: utf-8 -*-
import tensorflow as tf
from config import INPUT_SIZE, N_CHANNELS, N_CLASSES

tf.disable_eager_execution()
tf.reset_default_graph()

class UNet:
    def __init__(self, n_channels=N_CHANNELS, n_classes=N_CLASSES, input_size=INPUT_SIZE):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_size = input_size

    @staticmethod  # 新增装饰器
    def double_conv(x, out_channels, is_training, name="double_conv"):
        """(Conv → BN → ReLU) × 2"""
        with tf.variable_scope(name):
            x = tf.layers.conv2d(
                x, out_channels, kernel_size=3, padding='same',
                use_bias=False, name='conv1'
            )
            x = tf.layers.batch_normalization(x, training=is_training, name='bn1')
            x = tf.nn.relu(x, name='relu1')

            x = tf.layers.conv2d(
                x, out_channels, kernel_size=3, padding='same',
                use_bias=False, name='conv2'
            )
            x = tf.layers.batch_normalization(x, training=is_training, name='bn2')
            x = tf.nn.relu(x, name='relu2')
        return x

    def down_sample(self, x, out_channels, is_training, name="down"):
        """下采样：MaxPool → DoubleConv"""
        with tf.variable_scope(name):
            x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, name='maxpool')
            x = self.double_conv(x, out_channels, is_training, name='double_conv')
        return x

    def up_sample(self, x1, x2, out_channels, is_training, name="up"):
        """上采样：转置卷积 → 拼接 → DoubleConv"""
        with tf.variable_scope(name):
            x1 = tf.layers.conv2d_transpose(
                x1, x1.get_shape()[-1] // 2, kernel_size=2, strides=2,
                padding='same', use_bias=False, name='transpose_conv'
            )

            # 处理尺寸不匹配
            x1_shape = tf.shape(x1)
            x2_shape = tf.shape(x2)
            diff_h = x2_shape[1] - x1_shape[1]
            diff_w = x2_shape[2] - x1_shape[2]
            x1 = tf.pad(x1, [[0, 0], [diff_h // 2, diff_h - diff_h // 2],
                             [diff_w // 2, diff_w - diff_w // 2], [0, 0]])

            x = tf.concat([x2, x1], axis=-1, name='concat')
            x = self.double_conv(x, out_channels, is_training, name='double_conv')
        return x

    def build_model(self, inputs, is_training):
        """构建完整U-Net模型（补全输出层）"""
        # 输入层
        x1 = self.double_conv(inputs, 64, is_training, name='inc')

        # 下采样
        x2 = self.down_sample(x1, 128, is_training, name='down1')
        x3 = self.down_sample(x2, 256, is_training, name='down2')
        x4 = self.down_sample(x3, 512, is_training, name='down3')
        x5 = self.down_sample(x4, 1024, is_training, name='down4')

        # 上采样
        x = self.up_sample(x5, x4, 512, is_training, name='up1')
        x = self.up_sample(x, x3, 256, is_training, name='up2')
        x = self.up_sample(x, x2, 128, is_training, name='up3')
        x = self.up_sample(x, x1, 64, is_training, name='up4')

        # 输出层（1x1卷积，无激活）
        output = tf.layers.conv2d(
            x, self.n_classes, kernel_size=1, padding='same',
            use_bias=True, name='output_conv'
        )
        return output
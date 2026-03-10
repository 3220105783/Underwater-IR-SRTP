# -*- coding: utf-8 -*-
import tensorflow as tf

tf.disable_eager_execution()  # TF1.x必须禁用eager模式
tf.reset_default_graph()


class UNet:
    def __init__(self, n_channels=3, n_classes=1, input_size=(512, 512)):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_size = input_size

    def double_conv(self, x, out_channels, name="double_conv"):
        """(Conv → BN → ReLU) × 2（TF1.x版）"""
        with tf.variable_scope(name):
            # 第一层卷积
            x = tf.layers.conv2d(
                x, out_channels, kernel_size=3, padding='same',
                use_bias=False, name='conv1'
            )
            x = tf.layers.batch_normalization(x, name='bn1')
            x = tf.nn.relu(x, name='relu1')

            # 第二层卷积
            x = tf.layers.conv2d(
                x, out_channels, kernel_size=3, padding='same',
                use_bias=False, name='conv2'
            )
            x = tf.layers.batch_normalization(x, name='bn2')
            x = tf.nn.relu(x, name='relu2')
        return x

    def down_sample(self, x, out_channels, name="down"):
        """下采样：MaxPool → DoubleConv"""
        with tf.variable_scope(name):
            x = tf.layers.max_pooling2d(x, pool_size=2, strides=2, name='maxpool')
            x = self.double_conv(x, out_channels, name='double_conv')
        return x

    def up_sample(self, x1, x2, out_channels, name="up"):
        """上采样：转置卷积 → 拼接 → DoubleConv"""
        with tf.variable_scope(name):
            # 转置卷积上采样（匹配PyTorch ConvTranspose2d）
            x1 = tf.layers.conv2d_transpose(
                x1, x1.get_shape()[-1] // 2, kernel_size=2, strides=2,
                padding='same', use_bias=False, name='transpose_conv'
            )

            # 处理尺寸不匹配（padding）
            x1_shape = tf.shape(x1)
            x2_shape = tf.shape(x2)
            diff_h = x2_shape[1] - x1_shape[1]
            diff_w = x2_shape[2] - x1_shape[2]
            x1 = tf.pad(x1, [[0, 0], [diff_h // 2, diff_h - diff_h // 2], [diff_w // 2, diff_w - diff_w // 2], [0, 0]])

            # 拼接通道维度（x2在前，x1在后）
            x = tf.concat([x2, x1], axis=-1, name='concat')
            x = self.double_conv(x, out_channels, name='double_conv')
        return x

    def build_model(self, inputs):
        """构建完整U-Net模型（返回logits）"""
        # 输入层
        x1 = self.double_conv(inputs, 64, name='inc')

        # 下采样
        x2 = self.down_sample(x1, 128, name='down1')
        x3 = self.down_sample(x2, 256, name='down2')
        x4 = self.down_sample(x3, 512, name='down3')
        x5 = self.down_sample(x4, 1024, name='down4')

        # 上采样
        x = self.up_sample(x5, x4, 512, name='up1')
        x = self.up_sample(x, x3, 256, name='up2')
        x = self.up_sample(x, x2, 128, name='up3')
        x = self.up_sample(x, x1, 64, name='up4')

        # 输出层（1x1卷积，无激活）
        logits = tf.layers.conv2d(
            x, self.n_classes, kernel_size=1, padding='same',
            name='out_conv'
        )
        return logits
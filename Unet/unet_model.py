# -*- coding: utf-8 -*-
import tensorflow as tf
import config


def conv_block(input_tensor, num_filters, kernel_size=3, batch_norm=True, dropout_rate=0.0, is_training=True,
               name='conv_block'):
    """
    TF1.x 卷积块：Conv2D -> BatchNorm -> ReLU (两层)
    """
    with tf.variable_scope(name):
        # 第一层卷积
        x = tf.layers.conv2d(
            inputs=input_tensor,
            filters=num_filters,
            kernel_size=kernel_size,
            padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            name='conv1'
        )

        if batch_norm:
            x = tf.layers.batch_normalization(
                inputs=x,
                training=is_training,
                name='bn1'
            )
        x = tf.nn.relu(x, name='relu1')

        # 第二层卷积
        x = tf.layers.conv2d(
            inputs=x,
            filters=num_filters,
            kernel_size=kernel_size,
            padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4),
            name='conv2'
        )

        if batch_norm:
            x = tf.layers.batch_normalization(
                inputs=x,
                training=is_training,
                name='bn2'
            )
        x = tf.nn.relu(x, name='relu2')

        # Dropout
        if dropout_rate > 0:
            x = tf.layers.dropout(
                inputs=x,
                rate=dropout_rate,
                training=is_training,
                name='dropout'
            )

        return x


def unet_model(inputs, is_training=True, num_classes=config.NUM_CLASSES, name='unet'):
    """
    构建U-Net模型（纯TF1.x实现）
    inputs: 输入张量 [batch_size, height, width, channels]
    """
    with tf.variable_scope(name):
        # 下采样（编码）
        c1 = conv_block(inputs, 64, dropout_rate=0.1, is_training=is_training, name='block1')
        p1 = tf.layers.max_pooling2d(c1, pool_size=(2, 2), strides=(2, 2), name='pool1')

        c2 = conv_block(p1, 128, dropout_rate=0.1, is_training=is_training, name='block2')
        p2 = tf.layers.max_pooling2d(c2, pool_size=(2, 2), strides=(2, 2), name='pool2')

        c3 = conv_block(p2, 256, dropout_rate=0.2, is_training=is_training, name='block3')
        p3 = tf.layers.max_pooling2d(c3, pool_size=(2, 2), strides=(2, 2), name='pool3')

        c4 = conv_block(p3, 512, dropout_rate=0.2, is_training=is_training, name='block4')
        p4 = tf.layers.max_pooling2d(c4, pool_size=(2, 2), strides=(2, 2), name='pool4')

        # 瓶颈层
        c5 = conv_block(p4, 1024, dropout_rate=0.3, is_training=is_training, name='block5')

        # 上采样（解码）
        u6 = tf.layers.conv2d_transpose(
            inputs=c5,
            filters=512,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='same',
            name='up6'
        )
        u6 = tf.concat([u6, c4], axis=-1, name='concat6')
        c6 = conv_block(u6, 512, dropout_rate=0.2, is_training=is_training, name='block6')

        u7 = tf.layers.conv2d_transpose(
            inputs=c6,
            filters=256,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='same',
            name='up7'
        )
        u7 = tf.concat([u7, c3], axis=-1, name='concat7')
        c7 = conv_block(u7, 256, dropout_rate=0.2, is_training=is_training, name='block7')

        u8 = tf.layers.conv2d_transpose(
            inputs=c7,
            filters=128,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='same',
            name='up8'
        )
        u8 = tf.concat([u8, c2], axis=-1, name='concat8')
        c8 = conv_block(u8, 128, dropout_rate=0.1, is_training=is_training, name='block8')

        u9 = tf.layers.conv2d_transpose(
            inputs=c8,
            filters=64,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='same',
            name='up9'
        )
        u9 = tf.concat([u9, c1], axis=-1, name='concat9')
        c9 = conv_block(u9, 64, dropout_rate=0.1, is_training=is_training, name='block9')

        # 输出层
        outputs = tf.layers.conv2d(
            inputs=c9,
            filters=num_classes,
            kernel_size=(1, 1),
            activation=tf.nn.sigmoid,
            name='output'
        )

        return outputs


if __name__ == "__main__":
    # 测试模型构建（TF1.x方式）
    tf.reset_default_graph()
    with tf.Session(config=config.TF_CONFIG) as sess:
        # 创建输入占位符
        inputs = tf.placeholder(
            tf.float32,
            [None, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS],
            name='inputs'
        )

        # 构建模型
        outputs = unet_model(inputs, is_training=True)

        # 初始化变量
        sess.run(tf.global_variables_initializer())

        # 打印模型信息
        print("The U-Net model has been built successfully!")
        print(f"Inputs Shape: {inputs.shape}")
        print(f"Outputs Shape: {outputs.shape}")

        # 打印可训练变量数量
        trainable_vars = tf.trainable_variables()
        print(f"Number of trainable variables: {len(trainable_vars)}")
        for var in trainable_vars[:5]:  # 打印前5个变量
            print(f"Variable Name: {var.name}, Shape: {var.shape}")
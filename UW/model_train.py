from utils import (
    imsave,
    prepare_data
)

import time
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import tensorflow as tf
import scipy.io as scio
from ops import *
import vgg
# 新增：导入颜色空间转换相关库
from skimage import color
from tensorflow.image import ssim as tf_ssim


class T_CNN(object):

    def __init__(self,
                 sess,
                 image_height=230,
                 image_width=310,
                 label_height=230,
                 label_width=310,
                 batch_size=2,
                 c_dim=3,
                 checkpoint_dir=None,
                 sample_dir=None,
                 tblog_dir=None  # 新增日志目录参数
                 ):

        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_height = image_height
        self.image_width = image_width
        self.label_height = label_height
        self.label_width = label_width
        self.batch_size = batch_size
        self.dropout_keep_prob = 0.5

        self.c_dim = c_dim
        self.df_dim = 64
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.tblog_dir = tblog_dir  # 初始化日志目录
        self.vgg_dir = '/root/autodl-tmp/UW/vgg_pretrained/imagenet-vgg-verydeep-19.mat'
        self.CONTENT_LAYER = 'relu5_4'
        # ===================== 新增代码开始（早停参数） =====================
        # 早停机制参数
        self.early_stop_patience = 40  # 耐心值：连续多少个epoch验证损失不下降则停止
        self.early_stop_min_delta = 5e-7  # 最小损失变化：小于该值认为无改进
        self.best_val_loss = float('inf')  # 最佳验证损失
        self.early_stop_counter = 0  # 早停计数器
        self.early_stop_flag = False  # 早停标志
        # ===================== 新增代码结束 =====================
        # 新增：高光区域参数
        self.highlight_threshold = 0.8  # 高光亮度阈值
        self.highlight_weight = 3.0  # 高光区域损失权重
        self.build_model()
        # 初始化 TensorBoard SummaryWriter
        self._init_summary_writer()

    def _init_summary_writer(self):
        """Initialize the TensorBoard log writer"""
        if self.tblog_dir is not None:
            # 按时间创建子目录，避免日志覆盖
            current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
            self.summary_writer = tf.summary.FileWriter(
                os.path.join(self.tblog_dir, current_time),
                self.sess.graph  # 写入计算图结构
            )
        else:
            self.summary_writer = None

    # 新增：RGB转LAB颜色空间（TensorFlow实现）
    def rgb_to_lab(self, rgb):
        """
        修复版：RGB转LAB（增加数值裁剪，防止NaN/Inf）
        """
        # 强制裁剪到0~1范围，避免溢出
        rgb = tf.clip_by_value(rgb, 1e-8, 1.0 - 1e-8)  # 避开0和1，防止log/幂运算出错

        # RGB gamma校正（增加防溢出）
        rgb_gamma = tf.pow(rgb, 2.2)
        rgb_gamma = tf.clip_by_value(rgb_gamma, 1e-8, 1.0 - 1e-8)

        # RGB转XYZ
        rgb_to_xyz = tf.constant([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=tf.float32)
        xyz = tf.tensordot(rgb_gamma, rgb_to_xyz, axes=[-1, -1])
        xyz = tf.clip_by_value(xyz, 1e-8, 1e6)  # 限制XYZ范围

        # XYZ转LAB（D65参考白）
        xyz_ref_white = tf.constant([0.95047, 1.0, 1.08883], dtype=tf.float32)
        xyz_scaled = xyz / xyz_ref_white
        xyz_scaled = tf.clip_by_value(xyz_scaled, 1e-8, 1e6)

        # 非线性变换（增加epsilon防除零）
        epsilon = 6 / 29

        def f(t):
            t = tf.clip_by_value(t, 1e-8, 1e6)
            return tf.where(
                t > epsilon ** 3,
                tf.pow(t, 1 / 3),
                (t / (3 * epsilon ** 2)) + (4 / 29)
            )

        xyz_f = f(xyz_scaled)
        L = 116 * xyz_f[..., 1] - 16
        a = 500 * (xyz_f[..., 0] - xyz_f[..., 1])
        b = 200 * (xyz_f[..., 1] - xyz_f[..., 2])

        # 裁剪LAB值到合理范围，防止极端值
        L = tf.clip_by_value(L, 0.0, 100.0)
        a = tf.clip_by_value(a, -128.0, 128.0)
        b = tf.clip_by_value(b, -128.0, 128.0)

        lab = tf.stack([L, a, b], axis=-1)
        return lab

    def build_model(self):
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim],
                                     name='images')
        self.images_wb = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim],
                                        name='images_wb')
        self.images_ce = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim],
                                        name='images_ce')
        self.images_gc = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim],
                                        name='images_gc')
        self.labels_image = tf.placeholder(tf.float32,
                                           [self.batch_size, self.image_height, self.image_width, self.c_dim],
                                           name='labels_image')

        self.images_test = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim],
                                          name='images_test')
        self.images_test_wb = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim],
                                             name='images_test_wb')
        self.images_test_ce = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim],
                                             name='images_test_ce')
        self.images_test_gc = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim],
                                             name='images_test_gc')

        self.labels_test = tf.placeholder(tf.float32, [1, self.label_height, self.label_width, self.c_dim],
                                          name='labels_test')

        self.pred_h1 = self.model()

        self.enhanced_texture_vgg1 = vgg.net(self.vgg_dir, vgg.preprocess(self.pred_h1 * 255))
        self.labels_texture_vgg = vgg.net(self.vgg_dir, vgg.preprocess(self.labels_image * 255))
        self.loss_texture1 = tf.reduce_mean(
            tf.square(self.enhanced_texture_vgg1[self.CONTENT_LAYER] - self.labels_texture_vgg[self.CONTENT_LAYER]))

        self.loss_h1 = tf.reduce_mean(tf.abs(self.labels_image - self.pred_h1))

        # ===================== 新增代码开始 =====================
        # 1. 高光区域加权MSE损失
        # 计算亮度（RGB转灰度）
        pred_luminance = tf.reduce_mean(self.pred_h1, axis=-1, keepdims=True)
        label_luminance = tf.reduce_mean(self.labels_image, axis=-1, keepdims=True)

        # 生成高光掩码（亮度>阈值的区域为1，否则为0）
        highlight_mask = tf.cast(tf.reduce_mean(self.labels_image, axis=-1, keepdims=True) > self.highlight_threshold, tf.float32)
        # 加权MSE损失
        mse_loss = tf.square(self.labels_image - self.pred_h1)
        weighted_mse_loss = tf.reduce_mean(mse_loss * (1.0 + highlight_mask * (self.highlight_weight - 1.0)))

        # 2. SSIM损失（最大化SSIM等价于最小化1-SSIM）
        self.ssim = tf.image.ssim(
            tf.clip_by_value(self.labels_image, 0.0, 1.0),
            tf.clip_by_value(self.pred_h1, 0.0, 1.0),
            max_val=1.0,
            filter_size=11,
            filter_sigma=1.5,
            k1=0.01,
            k2=0.03
        )
        self.ssim = tf.clip_by_value(self.ssim, 0.0, 1.0)  # 限制SSIM在0~1
        self.ssim_loss = 1 - tf.reduce_mean(self.ssim)

        # 3. LAB颜色空间损失
        pred_lab = self.rgb_to_lab(self.pred_h1)
        label_lab = self.rgb_to_lab(self.labels_image)
        self.lab_loss = tf.reduce_mean(tf.abs(pred_lab - label_lab))

        # 4. 总损失重构（平衡各损失项）
        # ---- AB高光加权损失 ----
        pred_lab = self.rgb_to_lab(self.pred_h1)
        label_lab = self.rgb_to_lab(self.labels_image)
        ab_diff = tf.abs(pred_lab[...,1:3] - label_lab[...,1:3])
        self.highlight_ab_loss = tf.reduce_mean(ab_diff * highlight_mask) * self.highlight_weight
        # ---- 蓝色非对称惩罚 ----
        b_diff = label_lab[...,2:3] - pred_lab[...,2:3]
        self.blue_penalty_loss = tf.reduce_mean(tf.nn.relu(b_diff))
        # ---- TV Loss (AB channels only, LAB space) ----
        # 只对 AB 通道做总变分，L 通道不动，压制色度噪声
        tv_a_h = tf.reduce_mean(tf.abs(pred_lab[...,1:2][:,1:,:,:] - pred_lab[...,1:2][:,:-1,:,:]))
        tv_a_w = tf.reduce_mean(tf.abs(pred_lab[...,1:2][:,:,1:,:] - pred_lab[...,1:2][:,:,:-1,:]))
        tv_b_h = tf.reduce_mean(tf.abs(pred_lab[...,2:3][:,1:,:,:] - pred_lab[...,2:3][:,:-1,:,:]))
        tv_b_w = tf.reduce_mean(tf.abs(pred_lab[...,2:3][:,:,1:,:] - pred_lab[...,2:3][:,:,:-1,:]))
        self.tv_loss = (tv_a_h + tv_a_w + tv_b_h + tv_b_w) * 1e-4
        # ---- 总损失 ----
        self.loss = (0.10 * self.loss_texture1 + 0.20 * self.loss_h1 + 0.15 * weighted_mse_loss + 0.15 * self.ssim_loss + 0.25 * self.lab_loss + 0.10 * self.highlight_ab_loss + 0.05 * self.blue_penalty_loss + self.tv_loss)

        # 5. 计算PSNR（峰值信噪比）
        self.mse = tf.reduce_mean(mse_loss)
        self.psnr = tf.image.psnr(self.labels_image, self.pred_h1, max_val=1.0)
        # ===================== 新增代码结束 =====================

        t_vars = tf.trainable_variables()

        """"Add: Define TensorBoard monitoring metrics"""
        # 原有标量监控
        tf.summary.scalar('total_loss', self.loss)
        tf.summary.scalar('loss_h1 (MAE)', self.loss_h1)
        tf.summary.scalar('loss_texture1 (VGG)', self.loss_texture1)

        # 新增监控指标
        tf.summary.scalar('weighted_mse_loss', weighted_mse_loss)
        tf.summary.scalar('ssim', tf.reduce_mean(self.ssim))
        tf.summary.scalar('ssim_loss', self.ssim_loss)
        tf.summary.scalar('lab_color_loss', self.lab_loss)
        tf.summary.scalar('psnr', tf.reduce_mean(self.psnr))
        tf.summary.scalar('mse', self.mse)

        """
        # 图片监控（输入/标签/预测结果）
        # 裁剪到前3张图片避免日志过大
        tf.summary.image('input_images', self.images[:3], max_outputs=3)
        tf.summary.image('label_images', self.labels_image[:3], max_outputs=3)
        tf.summary.image('predicted_images', self.pred_h1[:3], max_outputs=3)
        """

        def resize_images(images, height=64, width=64):
            return tf.image.resize(images, [height, width])

        tf.summary.image('input_images', resize_images(self.images[:1]), max_outputs=1)
        tf.summary.image('label_images', resize_images(self.labels_image[:1]), max_outputs=1)
        tf.summary.image('predicted_images', resize_images(self.pred_h1[:1]), max_outputs=1)

        # 合并所有 summary
        self.merged_summary = tf.summary.merge_all()

        self.saver = tf.train.Saver(max_to_keep=3)
        # ===================== 新增代码开始（最佳模型保存器） =====================
        # 额外保存最佳模型的saver
        self.best_saver = tf.train.Saver(max_to_keep=1)
        # ===================== 新增代码结束 =====================

    def train(self, config):
        if config.is_train:
            data_train_list = prepare_data(self.sess, dataset="/root/autodl-tmp/UW/input_train")
            data_wb_train_list = prepare_data(self.sess, dataset="/root/autodl-tmp/UW/input_wb_train")
            data_ce_train_list = prepare_data(self.sess, dataset="/root/autodl-tmp/UW/input_ce_train")
            data_gc_train_list = prepare_data(self.sess, dataset="/root/autodl-tmp/UW/input_gc_train")
            image_train_list = prepare_data(self.sess, dataset="/root/autodl-tmp/UW/gt_train")

            data_test_list = prepare_data(self.sess, dataset="/root/autodl-tmp/UW/input_test")
            data_wb_test_list = prepare_data(self.sess, dataset="/root/autodl-tmp/UW/input_wb_test")
            data_ce_test_list = prepare_data(self.sess, dataset="/root/autodl-tmp/UW/input_ce_test")
            data_gc_test_list = prepare_data(self.sess, dataset="/root/autodl-tmp/UW/input_gc_test")
            image_test_list = prepare_data(self.sess, dataset="/root/autodl-tmp/UW/gt_test")

            seed = 568
            np.random.seed(seed)
            np.random.shuffle(data_train_list)
            np.random.seed(seed)
            np.random.shuffle(data_wb_train_list)
            np.random.seed(seed)
            np.random.shuffle(data_ce_train_list)
            np.random.seed(seed)
            np.random.shuffle(data_gc_train_list)
            np.random.seed(seed)
            np.random.shuffle(image_train_list)

        else:
            data_test_list = prepare_data(self.sess, dataset="/root/autodl-tmp/UW/input_test")
            data_wb_test_list = prepare_data(self.sess, dataset="/root/autodl-tmp/UW/input_wb_test")
            data_ce_test_list = prepare_data(self.sess, dataset="/root/autodl-tmp/UW/input_ce_test")
            data_gc_test_list = prepare_data(self.sess, dataset="/root/autodl-tmp/UW/input_gc_test")
            image_test_list = prepare_data(self.sess, dataset="/root/autodl-tmp/UW/gt_test")

        sample_data_files = data_test_list[16:20]
        sample_wb_data_files = data_wb_test_list[16:20]
        sample_ce_data_files = data_ce_test_list[16:20]
        sample_gc_data_files = data_gc_test_list[16:20]
        sample_image_files = image_test_list[16:20]

        sample_data = [
            get_image(sample_data_file,
                      is_grayscale=self.is_grayscale) for sample_data_file in sample_data_files]
        sample_lable_image = [
            get_image(sample_image_file,
                      is_grayscale=self.is_grayscale) for sample_image_file in sample_image_files]

        sample_inputs_data = np.array(sample_data).astype(np.float32)
        sample_inputs_lable_image = np.array(sample_lable_image).astype(np.float32)

        self.train_op = tf.train.AdamOptimizer(config.learning_rate, 0.9).minimize(self.loss)
        tf.global_variables_initializer().run()

        counter = 0
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if config.is_train:
            print("Training...")
            loss = np.ones(config.epoch)
            # 新增：记录验证集的MSE/PSNR/SSIM
            val_mse = np.ones(config.epoch)
            val_psnr = np.ones(config.epoch)
            val_ssim = np.ones(config.epoch)

            for ep in range(config.epoch):
                # ===================== 新增代码开始（早停检查） =====================
                # 检查是否触发早停
                if self.early_stop_flag:
                    print(f"Early stopping triggered at epoch {ep + 1}! Best validation loss: {self.best_val_loss:.8f}")
                    break
                # ===================== 新增代码结束 =====================
                # Run by batch images

                batch_idxs = len(data_train_list) // config.batch_size
                for idx in range(0, batch_idxs):

                    batch_files = data_train_list[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_files_wb = data_wb_train_list[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_files_ce = data_ce_train_list[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_files_gc = data_gc_train_list[idx * config.batch_size:(idx + 1) * config.batch_size]
                    batch_image_files = image_train_list[idx * config.batch_size: (idx + 1) * config.batch_size]

                    batch_ = [
                        get_image(batch_file,
                                  is_grayscale=self.is_grayscale) for batch_file in batch_files]
                    batch_wb = [
                        get_image(batch_wb_file,
                                  is_grayscale=self.is_grayscale) for batch_wb_file in batch_files_wb]
                    batch_ce = [
                        get_image(batch_ce_file,
                                  is_grayscale=self.is_grayscale) for batch_ce_file in batch_files_ce]
                    batch_gc = [
                        get_image(batch_gc_file,
                                  is_grayscale=self.is_grayscale) for batch_gc_file in batch_files_gc]
                    batch_labels_image = [
                        get_image(batch_image_file,
                                  is_grayscale=self.is_grayscale) for batch_image_file in batch_image_files]

                    batch_input = np.array(batch_).astype(np.float32)
                    batch_wb_input = np.array(batch_wb).astype(np.float32)
                    batch_ce_input = np.array(batch_ce).astype(np.float32)
                    batch_gc_input = np.array(batch_gc).astype(np.float32)
                    batch_image_input = np.array(batch_labels_image).astype(np.float32)

                    counter += 1
                    # ========== 新增：运行训练+记录 Summary + 监控MSE/PSNR/SSIM ==========
                    _, err, summary, train_mse, train_psnr, train_ssim = self.sess.run(
                        [self.train_op, self.loss, self.merged_summary, self.mse, self.psnr, self.ssim],
                        feed_dict={
                            self.images: batch_input,
                            self.images_wb: batch_wb_input,
                            self.images_ce: batch_ce_input,
                            self.images_gc: batch_gc_input,
                            self.labels_image: batch_image_input
                        }
                    )

                    # 写入 TensorBoard 日志（每步/每100步）
                    if self.summary_writer is not None and counter % 10 == 0:
                        self.summary_writer.add_summary(summary, counter)
                    # print(batch_light)

                    if counter % 100 == 0:
                        # 新增：打印MSE/PSNR/SSIM
                        print(
                            "Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f], MSE: [%.8f], PSNR: [%.2f], SSIM: [%.4f]" \
                            % ((ep + 1), counter, time.time() - start_time, err, train_mse, np.mean(train_psnr),
                               np.mean(train_ssim)))

                    if idx == batch_idxs - 1:
                        batch_test_idxs = len(data_test_list) // config.batch_size
                        err_test = np.ones(batch_test_idxs)
                        test_mse_list = []
                        test_psnr_list = []
                        test_ssim_list = []

                        for idx_test in range(0, batch_test_idxs):

                            sample_data_files = data_train_list[
                                idx_test * config.batch_size:(idx_test + 1) * config.batch_size]
                            sample_wb_files = data_wb_train_list[
                                idx_test * config.batch_size: (idx_test + 1) * config.batch_size]
                            sample_ce_files = data_ce_train_list[
                                idx_test * config.batch_size: (idx_test + 1) * config.batch_size]
                            sample_gc_files = data_gc_train_list[
                                idx_test * config.batch_size: (idx_test + 1) * config.batch_size]
                            sample_image_files = image_train_list[
                                idx_test * config.batch_size: (idx_test + 1) * config.batch_size]

                            sample_data = [get_image(sample_data_file,
                                                     is_grayscale=self.is_grayscale) for sample_data_file in
                                           sample_data_files]
                            sample_wb_image = [get_image(sample_wb_file,
                                                         is_grayscale=self.is_grayscale) for sample_wb_file in
                                               sample_wb_files]
                            sample_ce_image = [get_image(sample_ce_file,
                                                         is_grayscale=self.is_grayscale) for sample_ce_file in
                                               sample_ce_files]
                            sample_gc_image = [get_image(sample_gc_file,
                                                         is_grayscale=self.is_grayscale) for sample_gc_file in
                                               sample_gc_files]

                            sample_lable_image = [get_image(sample_image_file,
                                                            is_grayscale=self.is_grayscale) for sample_image_file in
                                                  sample_image_files]

                            sample_inputs_data = np.array(sample_data).astype(np.float32)
                            sample_inputs_wb_image = np.array(sample_wb_image).astype(np.float32)
                            sample_inputs_ce_image = np.array(sample_ce_image).astype(np.float32)
                            sample_inputs_gc_image = np.array(sample_gc_image).astype(np.float32)
                            sample_inputs_lable_image = np.array(sample_lable_image).astype(np.float32)

                            # ========== 新增：测试损失+MSE/PSNR/SSIM也写入 TensorBoard ==========
                            err_test[idx_test], test_mse, test_psnr, test_ssim, test_summary = self.sess.run(
                                [self.loss, self.mse, self.psnr, self.ssim, self.merged_summary],
                                feed_dict={
                                    self.images: sample_inputs_data,
                                    self.images_wb: sample_inputs_wb_image,
                                    self.images_ce: sample_inputs_ce_image,
                                    self.images_gc: sample_inputs_gc_image,
                                    self.labels_image: sample_inputs_lable_image
                                }
                            )
                            test_mse_list.append(test_mse)
                            test_psnr_list.append(np.mean(test_psnr))
                            test_ssim_list.append(np.mean(test_ssim))

                            # 写入测试集日志（标记为test_loss）
                            if self.summary_writer is not None and counter % 10 == 0:
                                test_summary = tf.Summary(value=[
                                    tf.Summary.Value(tag='test_loss', simple_value=np.mean(err_test[idx_test])),
                                    tf.Summary.Value(tag='test_mse', simple_value=test_mse),
                                    tf.Summary.Value(tag='test_psnr', simple_value=np.mean(test_psnr)),
                                    tf.Summary.Value(tag='test_ssim', simple_value=np.mean(test_ssim))
                                ])
                                self.summary_writer.add_summary(test_summary, counter)

                        loss[ep] = np.mean(err_test)
                        val_mse[ep] = np.mean(test_mse_list)
                        val_psnr[ep] = np.mean(test_psnr_list)
                        val_ssim[ep] = np.mean(test_ssim_list)

                        # 新增：打印验证集MSE/PSNR/SSIM
                        print(
                            f"Epoch {ep + 1} validation - Loss: {loss[ep]:.8f}, MSE: {val_mse[ep]:.8f}, PSNR: {val_psnr[ep]:.2f}, SSIM: {val_ssim[ep]:.4f}")

                        # ===================== 新增代码开始（早停逻辑） =====================
                        # 更新早停计数器和最佳损失
                        current_val_loss = loss[ep]
                        if current_val_loss < self.best_val_loss - self.early_stop_min_delta:
                            self.best_val_loss = current_val_loss
                            self.early_stop_counter = 0
                            # 保存最佳模型
                            best_model_dir = os.path.join(self.checkpoint_dir, "best_model")
                            if not os.path.exists(best_model_dir):
                                os.makedirs(best_model_dir)
                            best_save_path = self.best_saver.save(
                                self.sess,
                                os.path.join(best_model_dir, "best_coarse.model"),
                                global_step=counter
                            )
                            print(
                                f"Best model saved to: {best_save_path} (Validation loss improved to {self.best_val_loss:.8f})")
                        else:
                            self.early_stop_counter += 1
                            print(f"Early stop counter: {self.early_stop_counter}/{self.early_stop_patience}")
                            if self.early_stop_counter >= self.early_stop_patience:
                                self.early_stop_flag = True
                        # ===================== 新增代码结束 =====================
                        self.save(config.checkpoint_dir, counter)

    def model(self):

        with tf.variable_scope("main_branch") as scope3:
            conb0 = tf.concat(axis=3, values=[self.images, self.images_wb, self.images_ce, self.images_gc])
            conv_wb1 = tf.nn.relu(conv2d(conb0, 16, 128, k_h=7, k_w=7, d_h=1, d_w=1, name="conv2wb_1"))
            conv_wb2 = tf.nn.relu(conv2d(conv_wb1, 128, 128, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2wb_2"))
            conv_wb3 = tf.nn.relu(conv2d(conv_wb2, 128, 128, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2wb_3"))
            conv_wb4 = tf.nn.relu(conv2d(conv_wb3, 128, 64, k_h=1, k_w=1, d_h=1, d_w=1, name="conv2wb_4"))
            conv_wb5 = tf.nn.relu(conv2d(conv_wb4, 64, 64, k_h=7, k_w=7, d_h=1, d_w=1, name="conv2wb_5"))
            conv_wb6 = tf.nn.relu(conv2d(conv_wb5, 64, 64, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2wb_6"))
            conv_wb7 = tf.nn.relu(conv2d(conv_wb6, 64, 64, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2wb_7"))

            conv_wb77 = tf.nn.sigmoid(conv2d(conv_wb7, 64, 3, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2wb_77"))

            conb00 = tf.concat(axis=3, values=[self.images, self.images_wb])
            conv_wb9 = tf.nn.relu(conv2d(conb00, 3, 32, k_h=7, k_w=7, d_h=1, d_w=1, name="conv2wb_9"))
            conv_wb10 = tf.nn.relu(conv2d(conv_wb9, 32, 32, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2wb_10"))
            wb1 = tf.nn.relu(conv2d(conv_wb10, 32, 3, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2wb_11"))

            conb11 = tf.concat(axis=3, values=[self.images, self.images_ce])
            conv_wb99 = tf.nn.relu(conv2d(conb11, 3, 32, k_h=7, k_w=7, d_h=1, d_w=1, name="conv2wb_99"))
            conv_wb100 = tf.nn.relu(conv2d(conv_wb99, 32, 32, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2wb_100"))
            ce1 = tf.nn.relu(conv2d(conv_wb100, 32, 3, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2wb_111"))

            conb111 = tf.concat(axis=3, values=[self.images, self.images_gc])
            conv_wb999 = tf.nn.relu(conv2d(conb111, 3, 32, k_h=7, k_w=7, d_h=1, d_w=1, name="conv2wb_999"))
            conv_wb1000 = tf.nn.relu(conv2d(conv_wb999, 32, 32, k_h=5, k_w=5, d_h=1, d_w=1, name="conv2wb_1000"))
            gc1 = tf.nn.relu(conv2d(conv_wb1000, 32, 3, k_h=3, k_w=3, d_h=1, d_w=1, name="conv2wb_1111"))

            weight_wb, weight_ce, weight_gc = tf.split(conv_wb77, 3, 3)

            # ========== SE 全局权重模块 ==========
            # 对三个子分支输出做 GAP（全局平均池化），提取全局统计特征
            # 通过 FC → ReLU → FC → Sigmoid 生成三个标量权重 α, β, γ
            with tf.variable_scope("se_global"):
                gap_wb = tf.reduce_mean(wb1, axis=[1, 2], keepdims=False)   # 白平衡分支全局特征
                gap_ce = tf.reduce_mean(ce1, axis=[1, 2], keepdims=False)   # 直方图均衡分支全局特征
                gap_gc = tf.reduce_mean(gc1, axis=[1, 2], keepdims=False)   # Gamma校正分支全局特征
                concat_gap = tf.concat([gap_wb, gap_ce, gap_gc], axis=1)    # 拼接全局特征 [B, C_wb+C_ce+C_gc]
                se_fc1 = tf.layers.dense(concat_gap, 9, activation=tf.nn.relu, name='se_fc1')    # 降维至中间层
                se_fc2 = tf.layers.dense(se_fc1, 3, activation=tf.nn.sigmoid, name='se_fc2')     # 生成3个全局标量权重
                alpha = tf.reshape(se_fc2[:, 0:1], [-1, 1, 1, 1])   # 白平衡全局权重
                beta  = tf.reshape(se_fc2[:, 1:2], [-1, 1, 1, 1])   # 直方图均衡全局权重
                gamma = tf.reshape(se_fc2[:, 2:3], [-1, 1, 1, 1])   # Gamma校正全局权重
            # ====================================

            # 两级融合：SE全局标量 × 空间权重 × 子分支输出
            output1 = tf.add(
                tf.add(
                    tf.multiply(tf.multiply(wb1, weight_wb), alpha),
                    tf.multiply(tf.multiply(ce1, weight_ce), beta)
                ),
                tf.multiply(tf.multiply(gc1, weight_gc), gamma)
            )

        return output1

    def save(self, checkpoint_dir, step):
        model_name = "coarse.model"
        model_dir = "%s_%s" % ("coarse", self.label_height)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        """
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
        """
        # 保存checkpoint并获取保存路径
        save_path = self.saver.save(
            self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step
        )
        # 修改2：保存后立即删除当前生成的.meta文件
        meta_file = f"{save_path}.meta"
        if os.path.exists(meta_file):
            os.remove(meta_file)
            print(f"Deleted .meta file: {meta_file}")

    def load(self, checkpoint_dir):
        import tensorflow as tf
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("coarse", self.label_height)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            try:
                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                print(" [*] Full model restored (including SE module)")
            except:
                print(" [!] Full restore failed, trying backbone-only restore...")
                import tensorflow as tf
                bv = [v for v in tf.trainable_variables() if "se_global" not in v.name]
                if bv:
                    tf.train.Saver(var_list=bv).restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                    print(" [*] Backbone restored, SE module randomly initialized")
            return True
        else:
            return False
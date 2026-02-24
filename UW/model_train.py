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
               tblog_dir = None  # 新增日志目录参数
               ):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_height = image_height
    self.image_width = image_width
    self.label_height = label_height
    self.label_width = label_width
    self.batch_size = batch_size
    self.dropout_keep_prob=0.5


    self.c_dim = c_dim
    self.df_dim = 64
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.tblog_dir = tblog_dir  # 初始化日志目录
    self.vgg_dir='/root/autodl-tmp/UW/vgg_pretrained/imagenet-vgg-verydeep-19.mat'
    self.CONTENT_LAYER = 'relu5_4'
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

  def build_model(self):
    self.images = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images')
    self.images_wb = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_wb')
    self.images_ce = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_ce')
    self.images_gc = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_gc')
    self.labels_image = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='labels_image')


    self.images_test = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test')
    self.images_test_wb = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test_wb')
    self.images_test_ce = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test_ce')
    self.images_test_gc = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test_gc')

    self.labels_test = tf.placeholder(tf.float32, [1,self.label_height,self.label_width, self.c_dim], name='labels_test')
    
    self.pred_h1= self.model()

    self.enhanced_texture_vgg1 = vgg.net(self.vgg_dir, vgg.preprocess(self.pred_h1 * 255))
    self.labels_texture_vgg = vgg.net(self.vgg_dir, vgg.preprocess(self.labels_image* 255))
    self.loss_texture1 =tf.reduce_mean(tf.square(self.enhanced_texture_vgg1[self.CONTENT_LAYER]-self.labels_texture_vgg[self.CONTENT_LAYER]))
    
    self.loss_h1= tf.reduce_mean(tf.abs(self.labels_image-self.pred_h1))
    self.loss = 0.05*self.loss_texture1+ self.loss_h1
    t_vars = tf.trainable_variables()

    """"Add: Define TensorBoard monitoring metrics"""
    # 标量监控（损失）
    tf.summary.scalar('total_loss', self.loss)
    tf.summary.scalar('loss_h1 (MAE)', self.loss_h1)
    tf.summary.scalar('loss_texture1 (VGG)', self.loss_texture1)

    # 图片监控（输入/标签/预测结果）
    # 裁剪到前3张图片避免日志过大
    tf.summary.image('input_images', self.images[:3], max_outputs=3)
    tf.summary.image('label_images', self.labels_image[:3], max_outputs=3)
    tf.summary.image('predicted_images', self.pred_h1[:3], max_outputs=3)

    # 合并所有 summary
    self.merged_summary = tf.summary.merge_all()

    self.saver = tf.train.Saver(max_to_keep=0)
    
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


    self.train_op = tf.train.AdamOptimizer(config.learning_rate,0.9).minimize(self.loss)
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

      for ep in range(config.epoch):
        # Run by batch images
        
        batch_idxs = len(data_train_list) // config.batch_size
        for idx in range(0, batch_idxs):

          batch_files       = data_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_files_wb       = data_wb_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_files_ce       = data_ce_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_files_gc       = data_gc_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_image_files = image_train_list[idx*config.batch_size : (idx+1)*config.batch_size]


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
          # ========== 新增：运行训练+记录 Summary ==========
          _, err, summary = self.sess.run(
              [self.train_op, self.loss, self.merged_summary],
              feed_dict={
                  self.images: batch_input,
                  self.images_wb: batch_wb_input,
                  self.images_ce: batch_ce_input,
                  self.images_gc: batch_gc_input,
                  self.labels_image: batch_image_input
              }
          )

          # 写入 TensorBoard 日志（每步/每100步）
          if self.summary_writer is not None:
            self.summary_writer.add_summary(summary, counter)
          # print(batch_light)

          if counter % 100 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err ))
            
          if idx  == batch_idxs-1: 
            batch_test_idxs = len(data_test_list) // config.batch_size
            err_test =  np.ones(batch_test_idxs)
            for idx_test in range(0,batch_test_idxs):

              sample_data_files = data_train_list[idx_test*config.batch_size:(idx_test+1)*config.batch_size]
              sample_wb_files = data_wb_train_list[idx_test*config.batch_size : (idx_test+1)*config.batch_size]
              sample_ce_files = data_ce_train_list[idx_test*config.batch_size : (idx_test+1)*config.batch_size]
              sample_gc_files = data_gc_train_list[idx_test*config.batch_size : (idx_test+1)*config.batch_size]
              sample_image_files = image_train_list[idx_test*config.batch_size : (idx_test+1)*config.batch_size]
             
              sample_data = [get_image(sample_data_file,
                            is_grayscale=self.is_grayscale) for sample_data_file in sample_data_files]
              sample_wb_image = [get_image(sample_wb_file,
                                    is_grayscale=self.is_grayscale) for sample_wb_file in sample_wb_files]
              sample_ce_image = [get_image(sample_ce_file,
                                    is_grayscale=self.is_grayscale) for sample_ce_file in sample_ce_files]
              sample_gc_image = [get_image(sample_gc_file,
                                    is_grayscale=self.is_grayscale) for sample_gc_file in sample_gc_files]

              sample_lable_image = [get_image(sample_image_file,
                                    is_grayscale=self.is_grayscale) for sample_image_file in sample_image_files]

              sample_inputs_data = np.array(sample_data).astype(np.float32)
              sample_inputs_wb_image = np.array(sample_wb_image).astype(np.float32)
              sample_inputs_ce_image = np.array(sample_ce_image).astype(np.float32)
              sample_inputs_gc_image = np.array(sample_gc_image).astype(np.float32)
              sample_inputs_lable_image = np.array(sample_lable_image).astype(np.float32)

              # ========== 新增：测试损失也写入 TensorBoard ==========
              err_test[idx_test], test_summary = self.sess.run(
                  [self.loss, self.merged_summary],
                  feed_dict={
                      self.images: sample_inputs_data,
                      self.images_wb: sample_inputs_wb_image,
                      self.images_ce: sample_inputs_ce_image,
                      self.images_gc: sample_inputs_gc_image,
                      self.labels_image: sample_inputs_lable_image
                  }
              )
              # 写入测试集日志（标记为test_loss）
              if self.summary_writer is not None:
                  test_summary = tf.Summary(value=[
                      tf.Summary.Value(tag='test_loss', simple_value=np.mean(err_test[idx_test]))
                  ])
                  self.summary_writer.add_summary(test_summary, counter)

            loss[ep]=np.mean(err_test)
            print(loss)
            self.save(config.checkpoint_dir, counter)


      
  def model(self):

    with tf.variable_scope("main_branch") as scope3: 

      conb0 = tf.concat(axis = 3, values = [self.images,self.images_wb,self.images_ce,self.images_gc]) 
      conv_wb1 = tf.nn.relu(conv2d(conb0, 16,128, k_h=7, k_w=7, d_h=1, d_w=1,name="conv2wb_1"))
      conv_wb2 = tf.nn.relu(conv2d(conv_wb1, 128,128, k_h=5, k_w=5, d_h=1, d_w=1,name="conv2wb_2"))
      conv_wb3 = tf.nn.relu(conv2d(conv_wb2, 128,128, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2wb_3"))
      conv_wb4 = tf.nn.relu(conv2d(conv_wb3, 128,64, k_h=1, k_w=1, d_h=1, d_w=1,name="conv2wb_4"))
      conv_wb5 = tf.nn.relu(conv2d(conv_wb4, 64,64, k_h=7, k_w=7, d_h=1, d_w=1,name="conv2wb_5"))
      conv_wb6 = tf.nn.relu(conv2d(conv_wb5, 64,64, k_h=5, k_w=5, d_h=1, d_w=1,name="conv2wb_6"))
      conv_wb7 = tf.nn.relu(conv2d(conv_wb6, 64,64, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2wb_7"))

      conv_wb77 =tf.nn.sigmoid(conv2d(conv_wb7, 64,3, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2wb_77"))

      conb00 = tf.concat(axis = 3, values = [self.images,self.images_wb]) 
      conv_wb9 = tf.nn.relu(conv2d(conb00, 3,32, k_h=7, k_w=7, d_h=1, d_w=1,name="conv2wb_9"))
      conv_wb10 = tf.nn.relu(conv2d(conv_wb9, 32,32, k_h=5, k_w=5, d_h=1, d_w=1,name="conv2wb_10"))
      wb1 =tf.nn.relu(conv2d(conv_wb10, 32,3, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2wb_11"))

      conb11 = tf.concat(axis = 3, values = [self.images,self.images_ce]) 
      conv_wb99 = tf.nn.relu(conv2d(conb11, 3,32, k_h=7, k_w=7, d_h=1, d_w=1,name="conv2wb_99"))
      conv_wb100 = tf.nn.relu(conv2d(conv_wb99, 32,32, k_h=5, k_w=5, d_h=1, d_w=1,name="conv2wb_100"))
      ce1 =tf.nn.relu(conv2d(conv_wb100, 32,3, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2wb_111"))

      conb111 = tf.concat(axis = 3, values = [self.images,self.images_gc]) 
      conv_wb999 = tf.nn.relu(conv2d(conb111, 3,32, k_h=7, k_w=7, d_h=1, d_w=1,name="conv2wb_999"))
      conv_wb1000 = tf.nn.relu(conv2d(conv_wb999, 32,32, k_h=5, k_w=5, d_h=1, d_w=1,name="conv2wb_1000"))
      gc1 =tf.nn.relu(conv2d(conv_wb1000, 32,3, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2wb_1111"))

      weight_wb,weight_ce,weight_gc=tf.split(conv_wb77,3,3)
      output1=tf.add(tf.add(tf.multiply(wb1,weight_wb),tf.multiply(ce1,weight_ce)),tf.multiply(gc1,weight_gc))


    return output1
 

  def save(self, checkpoint_dir, step):
    model_name = "coarse.model"
    model_dir = "%s_%s" % ("coarse", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("coarse", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False

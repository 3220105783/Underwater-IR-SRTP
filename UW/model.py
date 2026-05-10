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

class T_CNN(object):

  def __init__(self, 
               sess, 
               image_height=460,
               image_width=620,
               label_height=460, 
               label_width=620,
               batch_size=1,
               c_dim=3, 
               c_depth_dim=1,
               checkpoint_dir=None, 
               sample_dir=None,
               test_image_name = None,
               test_wb_name = None,
               test_ce_name = None,
               test_gc_name = None,
               id = None
               ):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_height = image_height
    self.image_width = image_width
    self.label_height = label_height
    self.label_width = label_width
    self.batch_size = batch_size
    self.dropout_keep_prob=0.5
    self.test_image_name = test_image_name
    self.test_wb_name = test_wb_name
    self.test_ce_name = test_ce_name
    self.test_gc_name = test_gc_name
    self.id = id
    self.c_dim = c_dim
    self.df_dim = 64
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    #self.c_depth_dim=c_depth_dim
    #
    # self.d_bn1 = batch_norm(name='d_bn1')
    # self.d_bn2 = batch_norm(name='d_bn2')
    # self.d_bn3 = batch_norm(name='d_bn3')
    self.build_model()

  def build_model(self):
    self.images = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images')
    self.images_wb = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_wb')
    self.images_ce = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_ce')
    self.images_gc = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_gc')
    self.pred_h = self.model()


    self.saver = tf.train.Saver()
     
  def train(self, config):


    # Stochastic gradient descent with the standard backpropagation,var_list=self.model_c_vars
    image_test =  get_image(self.test_image_name,is_grayscale=False)
    shape = image_test.shape
    expand_test = image_test[np.newaxis,:,:,:]
    expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    batch_test_image = np.append(expand_test,expand_zero,axis = 0)

    wb_test =  get_image(self.test_wb_name,is_grayscale=False)
    shape = wb_test.shape
    expand_test = wb_test[np.newaxis,:,:,:]
    expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    batch_test_wb = np.append(expand_test,expand_zero,axis = 0)

    ce_test =  get_image(self.test_wb_name,is_grayscale=False)
    shape = ce_test.shape
    expand_test = ce_test[np.newaxis,:,:,:]
    expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    batch_test_ce = np.append(expand_test,expand_zero,axis = 0)

    gc_test =  get_image(self.test_wb_name,is_grayscale=False)
    shape = gc_test.shape
    expand_test = gc_test[np.newaxis,:,:,:]
    expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    batch_test_gc = np.append(expand_test,expand_zero,axis = 0)


    tf.global_variables_initializer().run()
    
    
    counter = 0
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")
    start_time = time.time()   
    result_h  = self.sess.run(self.pred_h, feed_dict={self.images: batch_test_image,self.images_wb: batch_test_wb,self.images_ce: batch_test_ce,self.images_gc: batch_test_gc})
    all_time = time.time()
    final_time=all_time - start_time
    print(final_time)


    _,h ,w , c = result_h.shape
    for id in range(0,1):
        result_h0 = result_h[id,:,:,:].reshape(h , w , 3)
        result_h0 = result_h0.squeeze()
        image_path0 = os.path.join(os.getcwd(), config.sample_dir)
        image_path = os.path.join(image_path0, self.test_image_name)
        imsave_lable(result_h0, image_path)



  def model(self):


    with tf.variable_scope("main_branch") as scope:      

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

import tensorflow as tf
import os
import numpy
import random
import time
import numpy as np
from corpus import Corpus
from model import Model


class VBPR(Model):
  def __init__(self, session, corpus, sampler, k, k2, factor_reg, bias_reg):
    
    self.K=k
    self.K2=k2
    self.lam=factor_reg
    self.bias_reg=bias_reg
    
    self.sampler = sampler
    
    self.u, self.i, self.j, self.iv, self.jv, self.loss, self.auc, self.train_op = VBPR.vbpr(corpus.user_count, corpus.item_count, len(corpus.image_features[1]),  hidden_dim=k, hidden_img_dim=k2, l2_regulization =factor_reg, bias_regulization=bias_reg)
    
    Model.__init__(self, corpus, session)
    
    print "VBPR - K=%d, K2=%d, reg_lf=%.2f, reg_bias=%.2f"%(k, k2, factor_reg, bias_reg)
  
  @classmethod
  def vbpr(cls, user_count, item_count, image_feat_dim, hidden_dim=20, hidden_img_dim=128,
           learning_rate=0.001,
            l2_regulization=0.001,
            bias_regulization=0.001,
            embed_regulization = 0.001,
            image_regulization = 0.0,
            visual_bias_regulization=0.0):
      """
      user_count: total number of users
      item_count: total number of items
      hidden_dim: hidden feature size of MF
      hidden_img_dim: [4096, hidden_img_dim]
      """
            
      
      u = tf.placeholder(tf.int32, [None])
      i = tf.placeholder(tf.int32, [None])
      j = tf.placeholder(tf.int32, [None])
      iv = tf.placeholder(tf.float32, [None, image_feat_dim])
      jv = tf.placeholder(tf.float32, [None, image_feat_dim])
      
      user_emb_w = tf.get_variable("user_emb_w", [user_count+1, hidden_dim],
                                  initializer=tf.random_normal_initializer(0, 0.1))
      user_img_w = tf.get_variable("user_img_w", [user_count+1, hidden_img_dim],
                                  initializer=tf.random_normal_initializer(0, 0.1)) #theta_u
      item_emb_w = tf.get_variable("item_emb_w", [item_count+1, hidden_dim],
                                  initializer=tf.random_normal_initializer(0, 0.1))
      item_b = tf.get_variable("item_b", [item_count+1, 1],
                                  initializer=tf.constant_initializer(0.0))
      visual_bias = tf.get_variable("visual_bias", [1, image_feat_dim], initializer=tf.constant_initializer(0.0))
      
      img_emb_w = tf.get_variable("image_embedding_weights", [image_feat_dim, hidden_img_dim],
                                 initializer=tf.random_normal_initializer(0, 0.1))
    
    
      
      #lookup the latent factors by user and id
      u_emb = tf.nn.embedding_lookup(user_emb_w, u)
      u_img = tf.nn.embedding_lookup(user_img_w, u)
      
      i_emb = tf.nn.embedding_lookup(item_emb_w, i)
      i_b = tf.nn.embedding_lookup(item_b, i)
      j_emb = tf.nn.embedding_lookup(item_emb_w, j)
      j_b = tf.nn.embedding_lookup(item_b, j)
                               

    
      
      # MF predict: u_i > u_j
      theta_i = tf.matmul(iv, img_emb_w) # (f_i * E), eq. 3 1xK2 x 4096xK2 => 1xK2 #plot these on 2d scatter
      theta_j = tf.matmul(jv, img_emb_w) # (f_j * E), eq. 3
      xui = i_b + tf.reduce_sum(tf.multiply(u_emb, i_emb), 1, keep_dims=True) + tf.reduce_sum(tf.multiply(u_img, theta_i), 1, keep_dims=True) + tf.reduce_sum(tf.multiply(visual_bias, iv), 1, keep_dims=True)
      xuj = j_b + tf.reduce_sum(tf.multiply(u_emb, j_emb), 1, keep_dims=True) + tf.reduce_sum(tf.multiply(u_img, theta_j), 1, keep_dims=True) + tf.reduce_sum(tf.multiply(visual_bias, jv), 1, keep_dims=True)
      xuij = xui - xuj
      #
      
      # auc score is used in test/cv
      # reduce_mean is reasonable BECAUSE
      # all test (i, j) pairs of one user is in ONE batch
      auc = tf.reduce_mean(tf.to_float(xuij > 0))
      
      l2_norm = tf.add_n([
              l2_regulization * tf.reduce_sum(tf.multiply(u_emb, u_emb)),
              image_regulization * tf.reduce_sum(tf.multiply(u_img, u_img)),
              l2_regulization * tf.reduce_sum(tf.multiply(i_emb, i_emb)),
              l2_regulization * tf.reduce_sum(tf.multiply(j_emb, j_emb)),
              embed_regulization * tf.reduce_sum(tf.multiply(img_emb_w, img_emb_w)),
              bias_regulization * tf.reduce_sum(tf.multiply(i_b, i_b)),
              bias_regulization * tf.reduce_sum(tf.multiply(j_b, j_b)),
              visual_bias_regulization * tf.reduce_sum(tf.multiply(visual_bias,visual_bias))
          ])
      
      loss = l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(xuij)))
      train_op =  tf.train.AdamOptimizer().minimize(loss)
      
      return u, i, j, iv, jv, loss, auc, train_op
  
  def evaluate(self, eval_set, sample_size=1000 , cold_start=False):
    u, i, j, iv, jv, auc, loss, train_op = self.u, self.i, self.j, self.iv, self.jv, self.auc, self.loss, self.train_op
    
    auc_vals=[]
    loss_vals=[]
    for d, fi, fj in self.sampler.generate_user_eval_batch(self.corpus.user_items, eval_set, self.corpus.item_count, self.corpus.item_dist, self.corpus.image_features, sample_size=sample_size, cold_start=cold_start):
        _loss, _auc = self.session.run([loss, auc], feed_dict={u:d[:,0], i:d[:,1], j:d[:,2], iv:fi, jv:fj})
        loss_vals.append(_loss)
        auc_vals.append(_auc)
    auc = np.mean(auc_vals)
    loss = np.mean(loss_vals)
    
    return auc, loss
  
  def train(self, max_iterations, batch_size, batch_count):
    u, i, j, iv, jv, auc, loss, train_op = self.u, self.i, self.j, self.iv, self.jv, self.auc, self.loss, self.train_op
    user_count = self.corpus.user_count
    item_count = self.corpus.item_count
    user_items = self.corpus.user_items
    image_features = self.corpus.image_features
    item_dist = self.corpus.item_dist
    
    val_ratings = self.val_ratings
    test_ratings = self.test_ratings
    
    for epoch in range(1, max_iterations+1):
        epoch_start = time.time()
        train_loss_vals=[]
        for d, _iv, _jv in self.generate_train_batch(user_items, val_ratings, test_ratings, item_count, image_features, sample_count=batch_count, batch_size=batch_size ):
            _loss, _ = self.session.run([loss, train_op], feed_dict={ u:d[:,0], i:d[:,1], j:d[:,2], iv:_iv, jv:_jv})
            train_loss_vals.append(_loss)
            
        duration = time.time() - epoch_start
        
        yield epoch, duration, np.mean(train_loss_vals)
            
            

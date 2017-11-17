from model import Model
from corpus import Corpus
from sampling import Uniform
import tensorflow as tf
import time
import numpy as np

class BPR(Model):
  def __init__(self, session, corpus, sampler, k, factor_reg, bias_reg):
    self.sampler = sampler
    
    self.lfactor_reg = factor_reg
    self.bias_reg = bias_reg
    self.K=k
    
    #build model
    self.u, self.i, self.j, self.mf_auc, self.bprloss, self.train_op = BPR.bpr_mf(corpus.user_count, corpus.item_count, k, regulation_rate=factor_reg, bias_reg=bias_reg)
    
    Model.__init__(self, corpus, session) #this needs to go after model construcion b/c it needs TF variables to exist
    
    print "BPR - K=%d, reg_lf: %.2f, reg_bias=%.2f"%(k, factor_reg, bias_reg)
  
  def train(self, max_iterations, batch_size, batch_count):
    print "max_iterations: %d, batch_size: %d, batch_count: %d"%(max_iterations, batch_size, batch_count)
    corpus = self.corpus
    user_count = self.corpus.user_count
    item_count = self.corpus.item_count
    user_items = self.corpus.user_items
    item_dist = self.corpus.item_dist
    
    val_ratings = self.val_ratings
    test_ratings = self.test_ratings
    
    u, i, j, mf_auc, bprloss, train_op = self.u, self.i, self.j, self.mf_auc, self.bprloss, self.train_op
    
    for epoch in range(1, max_iterations+1):
        epoch_start_time = time.time()
        train_loss_vals=[]
        
        for batch in self.sampler.generate_train_batch(user_items, val_ratings, test_ratings, item_count, None, sample_count=batch_count, batch_size=batch_size ):
          _batch_loss, _ = self.session.run([bprloss, train_op], feed_dict={u:batch[:,0], i:batch[:,1], j:batch[:,2]})
          train_loss_vals.append(_batch_loss)

        duration = time.time() - epoch_start_time
        
        yield epoch, duration, np.mean(train_loss_vals)
        
  
  @classmethod
  def bpr_mf(cls, user_count, item_count, hidden_dim, lr=0.1, regulation_rate = 0.0001, bias_reg=.01):
      
      #model input
      u = tf.placeholder(tf.int32, [None])
      i = tf.placeholder(tf.int32, [None])
      j = tf.placeholder(tf.int32, [None])
      
      #model paramenters
      #latent factors
      #hidden_dim is the k hyper parameter
      user_emb_w = tf.get_variable("user_emb_w", [user_count+1, hidden_dim], initializer=tf.random_normal_initializer(0, 0.1))
      item_emb_w = tf.get_variable("item_emb_w", [item_count+1, hidden_dim], initializer=tf.random_normal_initializer(0, 0.1))
      item_b = tf.get_variable("item_b", [item_count+1, 1], initializer=tf.random_normal_initializer(0, 0.1))    #item bias
      user_b = tf.get_variable("user_b", [user_count+1, 1], initializer=tf.random_normal_initializer(0, 0.1))    #user bias
      
      u_emb = tf.nn.embedding_lookup(user_emb_w, u) #lookup the latent factor for user u
      i_emb = tf.nn.embedding_lookup(item_emb_w, i) #lookup the latent factor fo item i
      j_emb = tf.nn.embedding_lookup(item_emb_w, j) #lookup the latent factor for item j
      i_b = tf.nn.embedding_lookup(item_b, i)       #lookup the bias vector for item i
      j_b = tf.nn.embedding_lookup(item_b, j)       #lookup the bias vector for item js
      
      
      # MF predict: u_i > u_j
      # xuij = xui - xuj
      xui = i_b + tf.reduce_sum(tf.multiply(u_emb, i_emb), 1, keep_dims=True)
      xuj = j_b + tf.reduce_sum(tf.multiply(u_emb, j_emb), 1, keep_dims=True)
      xuij = xui-xuj
      
      # AUC for one user:
      # reasonable iff all (u,i,j) pairs are from the same user
      #
      # average AUC = mean( auc for each user in test set)
      mf_auc = tf.reduce_mean(tf.to_float(xuij > 0)) # xui - xui > 0 == xui > xuj
      tf.summary.scalar('user_auc', mf_auc)
      
      l2_norm = tf.add_n([
              regulation_rate * tf.reduce_sum(tf.multiply(u_emb, u_emb)),
              regulation_rate * tf.reduce_sum(tf.multiply(i_emb, i_emb)),
              regulation_rate * tf.reduce_sum(tf.multiply(j_emb, j_emb)),
              #reg for biases
              bias_reg * tf.reduce_sum(tf.multiply(i_b, i_b)),
              bias_reg/10.0 * tf.reduce_sum(tf.multiply(j_b, j_b)),
          ])
      
      
      bprloss =  l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(xuij))) #BPR loss
      
      global_step = tf.Variable(0, trainable=False)
      learning_rate = tf.train.exponential_decay(lr, global_step, 400, 0.8, staircase=True)
      #.1 ... .001
      
      #optimizer updates 62 parameters
      # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(bprloss, global_step=global_step)
      train_op =  tf.train.AdamOptimizer().minimize(bprloss, global_step=global_step)
      return u, i, j, mf_auc, bprloss, train_op
  
  def evaluate(self, eval_set, sample_size=1000 , cold_start=False):
      u, i, j, auc, loss, train_op = self.u, self.i, self.j, self.mf_auc, self.bprloss, self.train_op
      
      loss_vals=[]
      auc_vals=[]
      for uij in self.generate_user_eval_batch(self.corpus.user_items, eval_set, self.corpus.item_count, self.corpus.item_dist, None, sample_size=sample_size, cold_start=cold_start):
          _loss, user_auc = self.session.run([loss, auc], feed_dict={u: uij[:,0], i: uij[:,1], j: uij[:,2]})
          loss_vals.append(_loss)
          auc_vals.append(user_auc)
      
      auc = np.mean(auc_vals)
      loss = np.mean(loss_vals)
      return auc, loss
    

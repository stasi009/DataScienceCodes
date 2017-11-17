import random
import numpy 
import tensorflow as tf
from datetime import datetime

class Model(object):
  
  def __init__(self, corpus, session):
    self.corpus = corpus
    self.session = session
    
    self.merged = tf.summary.merge_all()
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    self.train_writer = tf.summary.FileWriter('logs/train/run-%s'%now, self.session.graph)
    self.test_writer  = tf.summary.FileWriter('logs/test/run-%s'%now, self.session.graph)
    self.saver = tf.train.Saver()
    self.session.run(tf.global_variables_initializer())
    
    self.val_ratings, self.test_ratings = self.generate_val_and_test()
    
    
  def generate_val_and_test(self):
      '''
      for each user, random select one rating into test set
      '''
      user_test = dict()
      user_val = dict()
      for u, i_list in self.corpus.user_items.iteritems():
          samples = random.sample(i_list, 2)
          user_test[u] = samples[0]
          user_val[u] = samples[1]
      return user_val, user_test
      
  #TODO save this model to path and return path
  def save(self):
    self.saver.save(self.session, "logs/")
  
  def restore(self):
    self.saver.restore(self.session, "logs/")
      
  def train(self):
    raise Exception("Not implemented yet!")
    
  def evaluate(self):
    raise Exception("Not implemented yet!")
    
  def export(filename):
    raise Exception("Not implemented yet!")
    
  
  
  def generate_user_eval_batch(self, user_items, test_ratings, item_count, item_dist, image_features, sample_size=3000, neg_sample_size=1000, cold_start=False):
      # using leave one cv
      for u in random.sample(test_ratings.keys(), sample_size): #uniform random sampling w/o replacement
          t = []
          ilist = []
          jlist = []
        
          i = test_ratings[u]
          #check if we have an image for i, sometimes we dont...
          if image_features and i not in image_features:
            continue
        
          #filter for cold start
          if cold_start and item_dist[i] > 5:
            continue
        
          for _ in xrange(neg_sample_size):
              j = random.randint(1, item_count)
              if j != test_ratings[u] and not (j in user_items[u]):
                  # find negative item not in train or test set
                
                  #sometimes there will not be an image for given product
                  if image_features:
                    try:
                      image_features[i]
                      image_features[j]
                    except KeyError:
                      continue  #if image not found, skip item
                
                  t.append([u, i, j])
                  
                  if image_features:
                    ilist.append(image_features[i])
                    jlist.append(image_features[j])
        
          if image_features:
            yield numpy.asarray(t), numpy.vstack(tuple(ilist)), numpy.vstack(tuple(jlist))
          else:
            yield numpy.asarray(t)
    
    
if __name__ == '__main__':
  import os
  import corpus
  print "Loading dataset..."
  data_dir = os.path.join("data", "amzn")
  simple_path = os.path.join(data_dir, 'reviews_Women_5.txt')


  
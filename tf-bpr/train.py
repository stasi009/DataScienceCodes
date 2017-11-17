from models.vbpr import VBPR
from models.bpr import BPR
from models.hbpr import HBPR
from corpus import Corpus
import argparse
import os
import sys
import tensorflow as tf
import sampling
import numpy as np

def main(_):
  corpus = Corpus()
  corpus.load_data(FLAGS.reviews_path, FLAGS.images_path, FLAGS.user_min, 0);
  
  sampler = sampling.Uniform()
  session = tf.Session()
  
  model = None
  if FLAGS.model == "BPR":
    model = BPR(session, corpus, sampler, FLAGS.K, FLAGS.reg, FLAGS.bias_reg)
  elif FLAGS.model == "VBPR": 
    model = VBPR(session, corpus, sampler, FLAGS.K, FLAGS.K2, FLAGS.reg, FLAGS.bias_reg)
  elif FLAGS.model == "HBPR":
    model = HBPR(session, corpus, sampler, FLAGS.K, FLAGS.K2, FLAGS.reg, FLAGS.bias_reg)
  else:
    raise Exception("Could not find model %s"%FLAGS.model)
  
  evaluate(model) #find the best model
  
  session.close()

def evaluate(model):
  #start training loop
  epoch_durations = []
  best_auc=-1
  best_iter=-1
  
  for iteration, duration, train_loss in model.train(FLAGS.max_iterations, FLAGS.batch_size, FLAGS.batch_count ):
    epoch_durations.append(duration)
    
    if iteration % 5 != 0:
      print "iteration: %d (%.2fs), train loss: %.2f"%(iteration, np.mean(epoch_durations), train_loss )
      continue
    
    #train eval
    val_auc, val_loss = model.evaluate(model.val_ratings,  sample_size=500)
    print "iteration: %d (%.2fs), train loss: %.2f, val loss: %.2f, val auc: %.2f"%(iteration, np.mean(epoch_durations), train_loss ,val_loss, val_auc ),
    
    #early termination/checks for convergance
    if val_auc > best_auc:
      best_auc = val_auc
      best_iter = iteration
      print "*"
      model.save()
    elif val_auc < best_iter and iteration >= best_iter+21: #overfitting
      print "Overfitted. Exiting..."
      break
    else:
      print
  
  #restore best model from checkpoint
  model.restore()
  #test auc
  test_auc, test_loss = model.evaluate(model.test_ratings, sample_size=1000)
  print "Best model iteration %d, test: %.3f, val: %.3f"%(best_iter, test_auc, best_auc)
  
  
  #cold auc
  cold_auc, cold_loss = model.evaluate(model.test_ratings, sample_size=1000, cold_start=True)
  print "cold auc: %.2f"%(cold_auc)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model',
      type=str,
      default="VBPR",
      help='Which model to evaluate?: bpr, vbpr or hbpr'
  )
  parser.add_argument(
      '--K',
      type=int,
      default=20,
      help='Dimension of Latent Factors'
  )
  parser.add_argument(
      '--K2',
      type=int,
      default=20,
      help='Dimension of Visual Factors'
  )
  parser.add_argument(
      '--reg',
      type=float,
      default=10,
      help='L2 Regularization constant for latent factors'
  )
  parser.add_argument(
      '--bias_reg',
      type=float,
      default=0.01,
      help='L2 regularization constant for item i,j bias vectors'
  )
  parser.add_argument(
      '--max_iterations',
      type=int,
      default=2000,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=512,
      help='Batch size. Must divide evenly into the dataset sizes.'
  )
  
  parser.add_argument(
      '--batch_count',
      type=int,
      default=400,
      help='Batch Count. How many samples of batch_size per GD iteration. Arbitrary amount as sampling is random...'
  )
  parser.add_argument(
      '--images_path',
      type=str,
      default="data/amzn/image_features_Women.b",
      help='Path to image features file'
  )
  parser.add_argument(
      '--reviews_path',
      type=str,
      default="data/amzn/reviews_Women_ALL_scraped.csv",
      help='Path to reviews file'
  )
  parser.add_argument(
      '--name',
      type=str,
      default="BPR",
      help='The name of this trainging session'
  )
  parser.add_argument(
      '--user_min',
      type=int,
      default=5,
      help='users in training set have at least this many reviews'
  )
  
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


# nohup time python -u vbpr.py > vbpr3-test005.log 2>&1 &

# nohup time ./train ../tf-bpr/data/amzn/reviews_Women.txt ../tf-bpr/data/amzn/image_features_Women.b 10 10 na 0.01 9.5 0.01 na 20 "women" > logs/vbpr-test001.log 2>&1 &
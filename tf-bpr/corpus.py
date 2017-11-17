from collections import defaultdict
import numpy as np
import random 
import csv

class Corpus(object):
  """docstring for Corpus"""
  def __init__(self):
    super(Corpus, self).__init__()
    
    
  @staticmethod
  def stats(reviews):
    #frequency distributions
    user_dist=defaultdict(int)
    item_dist=defaultdict(int)
  
    #cores
    user_items = defaultdict(list)
    item_users = defaultdict(list)
  
    for review in reviews:
       u=review[0]
       i=review[1]
       user_dist[u]+=1
       item_dist[i]+=1
       user_items[u].append(i)
       item_users[i].append(u)
     
    return user_dist, item_dist, user_items, item_users
  
  @staticmethod
  def load_complex(path, user_min=5):
    print "load_complex"
    #load raw from disk
    reviews=[]
    with open(path, 'r') as f:
      next(f)
      csvreader = csv.reader(f)
      for auid, asin, _, brand, price in csvreader:
        reviews.append([auid,asin, brand, price])
    
    #stats
    user_dist, item_dist, user_ratings, item_users = Corpus.stats(reviews)
  
    #filter out based on distribution of users
    reviews_reduced=[]
    for auid, asin, brand, price in reviews:
      if user_dist[auid] >=user_min:
        reviews_reduced.append([auid, asin, brand, price])
    
    #map to sequential ids
    users = {}
    items = {}
    brands = {}
    prices = {}
    user_count=0
    item_count=0
    triples=[]
    for auid, asin, brand, price in reviews_reduced:
      if auid in users:
        u = users[auid]
      else:
        user_count+=1 #new user so increment
        users[auid]=user_count
        u = user_count
    
      if asin in items:
        i = items[asin]
      else:
        item_count+=1 #new user so increment
        items[asin]=item_count
        i = item_count
        
      brands[i] = brand
      if (price=='' or price=='\r\n' or price=='\n'):
          prices[i] = 0
      else:
          prices[i] = float(price.rstrip())
    
      triples.append([u, i])
  
    return users, items, np.array(triples), brands, prices
            
  @staticmethod
  def load_simple(path, user_min=5):
    #load raw from disk
    reviews=[]
    with open((path), 'r') as f:
      for line in f.readlines():
        auid, asin, _ = line.split(",", 2)
        reviews.append([auid,asin])
  
    #stats
    user_dist, item_dist, user_ratings, item_users = Corpus.stats(reviews)
  
    #filter out based on distribution of users
    reviews_reduced=[]
    for auid, asin in reviews:
      if user_dist[auid] >=user_min:
        reviews_reduced.append([auid, asin])
  
    #map to sequential ids
    users = {}
    items = {}
    user_count=0
    item_count=0
    triples=[]
    for auid, asin in reviews_reduced:
      if auid in users:
        u = users[auid]
      else:
        user_count+=1 #new user so increment
        users[auid]=user_count
        u = user_count
    
      if asin in items:
        i = items[asin]
      else:
        item_count+=1 #new user so increment
        items[asin]=item_count
        i = item_count
    
      triples.append([u, i])
  
    return users, items, np.array(triples)
    
  #merges image features w/ meta generated features
  
  def load_heuristics(self):
    image_features_plus = Corpus.merge_image_features_and_meta(self.brands, self.prices, self.image_features)
    #overwrite
    self.image_features = image_features_plus
    
  @staticmethod
  def merge_image_features_and_meta(brands, prices, image_features):
    #one-hot encode prices
    prices_features= {}
    prices_all = list(set(prices.values()))
    price_quant_level = 10
    price_max = float(max(prices.values()))
    for key, value in prices.iteritems():
        prices_vec = np.zeros(price_quant_level+1)
        idx = int(np.ceil(float(value)/(price_max/price_quant_level)))
        prices_vec[idx]=1
        prices_features[key] = prices_vec

    #one-hot encode brands
    brands_features = {}
    brands_all = list(set(brands.values()))
    for key, value in brands.iteritems():
        brands_vec = np.zeros(len(brands_all))
        brands_vec[brands_all.index(value)] = 1
        brands_features[key] = brands_vec

    a = prices_features
    b = brands_features
    
    #merge user price and brand vector
    c = dict([(k, np.append(a[k],b[k])) for k in set(b) & set(a)])
    
    #merge user image and c vectors
    f=image_features
    image_features_p = dict([(k, np.append(c[k],f[k])) for k in set(c) & set(f)])
    return image_features_p
    
  NORM_FACTOR = 58.388599

  #load image features for the given asin collection into dictionary
  @staticmethod
  def load_image_features(path, items):
    count=0
    image_features = {}
    f = open(path, 'rb')
    while True: 
      asin = f.read(10)
      if asin == '': break
      features_bytes = f.read(16384) # 4 * 4096 = 16KB, fast read, don't unpack
  
      if asin in items: #only unpack 4096 bytes if w need it -- big speed up
        features = np.fromstring(features_bytes, dtype=np.float32)/Corpus.NORM_FACTOR
        iid=items[asin]
        image_features[iid] = features
  
      if count%100000==0:
        print count
      count+=1

    return image_features
    
  def load_reviews(self, path, user_min):
    print "Loading dataset from: ",path

    users, items, reviews_all, brands, prices = Corpus.load_complex(path, user_min=user_min)
    print "generating stats..."
    user_dist, item_dist, train_ratings, item_users = Corpus.stats(reviews_all)

    user_count = len(train_ratings)
    item_count = len(item_users)
    reviews_count = len(reviews_all)
    print user_count,item_count,reviews_count
  
    # return users, items, reviews_all,user_dist, item_dist, train_ratings, item_users
    
    self.users=users
    self.items=items
    self.reviews = reviews_all
    self.brands = brands
    self.prices = prices
    
    
    self.user_dist = user_dist
    self.item_dist = item_dist
    self.user_items = train_ratings
    self.item_users = item_users
    
    self.user_count = len(self.user_items)
    self.item_count = len(self.item_users)

  def load_images(self, path, items):
    print "Loading image features from: ",path
    self.image_features = Corpus.load_image_features(path, items)

    print "extracted image feature count: ",len(self.image_features)
  
  def load_data(self, reviews_path, images_path, user_min, item_min):
    #load reviews
    self.load_reviews(reviews_path, user_min)
    if images_path:
      self.load_images(images_path, self.items)
    
if __name__ == '__main__':
  import os
  data_dir = os.path.join("data", "amzn")
  simple_path = os.path.join(data_dir, 'reviews_Women_ALL_scraped.csv')

  users, items, reviews, brands, prices = Corpus.load_complex(simple_path)
  image_features = Corpus.load_image_features("data/amzn/image_features_Women.b", items)
  image_features_plus = Corpus.merge_image_features_and_meta(brands, prices, image_features)
  
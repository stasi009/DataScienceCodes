import tensorflow as tf
import os
import numpy
import random
import time
import numpy as np
from corpus import Corpus
from model import Model
from vbpr import VBPR


class HBPR(VBPR):
  def __init__(self, session, corpus, sampler, k, k2, factor_reg, bias_reg):
    #I have the image features in the corpus
    #now load the heuristic features: brand and price
    corpus.load_heuristics()
    #now just run VBPR:
    
    VBPR.__init__(self, session, corpus, sampler, k, k2, factor_reg, bias_reg)
  

  
 
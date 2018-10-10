# AUTHOR: Preetham V V
# DATE: FEB, 2017
# THE AUTHOUR OR HIS ASSOCIATED COMPANIES/SERVICES HOLDS NO RESPONSIBILITY OR LIABIILTY IF YOU DECIDE TO USE IT.
# THIS CODE IS SHARED AS AN EXAMPLE ONLY.

# FEEL FREE TO USE AND DISTRIBUTE AS NECESSARY.
# APPRECIATE THE ATTRIBUTION IF ANY

# THE CODE IN SOME PLACES IS A TEMPLATE AND IS NOT THOROUGHLY TESTED. 

# MY TEST ENV: (CUDA 8) — 3units of Titan-X Pascal 
# on Ubuntu 16.04 with TensorFlow v0.12. (Nvidia Driver Version: 370.28)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Input, Dense, Reshape
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.layers import Merge
from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model

import keras.backend.tensorflow_backend as KTF
import scipy.sparse as spsp

import xxhash
import keras
import numbers
import math
import random
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy as sp
import sklearn.preprocessing as pp

import tempfile
import datetime
import time
import re
import os
import itertools
import gc
import hashlib

from keras.models import load_model

from datetime import date

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn_pandas import DataFrameMapper

from os import listdir
from os.path import isfile, join
from os import walk

# A TEMPLATE METHOD ONLY    
# INCOMPLETE AND USED ONLY AS AN EXAMPLE
# ALL further DEFs below this method is mostly working on NVIDIA Tesla-X

# Define all your features as a name:value pair in a dict. HAS TO BE A DICT.
wide_col_len_dict = {
    'dow': 7, 'hour': 24, 'pricing':7, 
    }

def main(_):
    np.random.seed(1975)
    KTF.set_session(get_session())
    df_train = #LOAD YOUR TRAINING DATA AS A DATAFRAME WHICH HAS COLS SAME AS KEYS IN wide_col_len_dict
    labels = #LOAD YOUR TARGETS
    model = wide_model(wide_col_len_dict,middle_layer=False, data_parallel=IS_PARALLEL)
    # NUMERICAL HASH FOR THE FEATURES IN df_train
    df_train_hash = hash_features(df_train,wide_col_len_dict)
    # OBTAINS THE 4_BIT_HASHED input data from a sparse matrix
    input_dense_matrix, _, _ = get_input_data(df_train_hash, wide_col_len_dict)
    # RUN THE model
    hist = model.fit(input_dense_matrix, labels, nb_epoch=train_steps, verbose=0, batch_size=32)


#Adds the hash of the features into the features dataframe as 'featurename'_hash
def hash_features(features_df, wide_col_len_dict, split_hash=True):
    #4-bit-on split-feature-hash for avoiding collisions and compacting the one-hot to 4-hot
    def four_bit_split_hash(val,bucket_len):
        array_width=int(bucket_len/4)
        hash_str = xxhash.xxh64(str(val).encode('utf-8'), seed=0).hexdigest()
        split = int(len(hash_str)/4)
        first_hash = int(hash_str[:split],16) % array_width
        second_hash = int(hash_str[split:split*2],16) % array_width
        third_hash = int(hash_str[split*2:split*3],16) % array_width
        fourth_hash = int(hash_str[split*3:],16) % array_width
        return (first_hash, second_hash, third_hash, fourth_hash)

    #single-bit-on split-feature-hash
    def one_hot_hash(val,bucket_len):
        hash_str = xxhash.xxh64(str(val).encode('utf-8'), seed=0).hexdigest()
        one_hot = int(hash_str,16) % bucket_len
        return one_hot

    keys_to_hash = wide_col_len_dict.keys()
    for feature_name in keys_to_hash:
        feature_time = time.time()    
        bucket_len = bucket_size(int(wide_col_len_dict[feature_name]))
        unique_feature_values = features_df[feature_name].to_frame().drop_duplicates().copy()
        unique_feature_hash =[]
        hash_function = four_bit_split_hash if split_hash else one_hot_hash
        unique_feature_hash = unique_feature_values.apply(hash_function, args=[bucket_len],axis=1)
        unique_feature_values[feature_name+'_hash']=unique_feature_hash
        features_df = pd.merge(features_df,unique_feature_values, 
                                   left_on=feature_name, 
                                   right_on=feature_name,
                                   how='left')
    return features_df
    
def bucket_size(feature_length):
    #Determine a bucket size that is large enough to avoid collisions (minimum is 32)
    bucket_size = math.ceil(pow(feature_length,0.5))*4
    if(bucket_size<32):
        bucket_size=32
    return bucket_size
    
#Takes a hash_array and returns sparse_matrix of the hash
def sparse_hash(features_df_hash_array, feature_name, feature_len, split_hash=True):
    def get_splits_sparse(hash_array, bucket_len):
        #Encode the split-array    
        def encode_splits(split_array, array_width):
            arr = np.array(split_array)
            row = np.arange(arr.size)
            data = np.ones(arr.size)
            encoded_array = spsp.csr_matrix((data, (row, arr)), shape=(arr.size, array_width))            
            return encoded_array

        first_array=[]
        second_array=[]
        third_array=[]
        fourth_array=[]
        array_width = int(bucket_len/4)
        for hash_value in hash_array:
            first_array.append(hash_value[0])
            second_array.append(hash_value[1])
            third_array.append(hash_value[2])
            fourth_array.append(hash_value[3])

        #Convert to compressed sparse format
        first_encoded_sparse_array = encode_splits(first_array, array_width)
        second_encoded_sparse_array = encode_splits(second_array, array_width)
        third_encoded_sparse_array = encode_splits(third_array, array_width)
        fourth_encoded_sparse_array = encode_splits(fourth_array, array_width)
        concat_encoded_sparse = spsp.hstack((first_encoded_sparse_array,
                                             second_encoded_sparse_array,
                                             third_encoded_sparse_array,
                                             fourth_encoded_sparse_array), 
                                            format='csr')
        return concat_encoded_sparse

    def get_one_hot_sparse(hash_array):
        a = np.array(hash_array)
        encoded_array = np.zeros((a.size, a.max()+1))
        encoded_array[np.arange(a.size), a] = 1
        return spsp.csr_matrix(encoded_array)
    
    bucket_len = bucket_size(feature_len)
    ret = []
    if(split_hash):
        ret = get_splits_sparse(features_df_hash_array[feature_name+'_hash'], bucket_len)
    else:
        ret = get_one_hot_sparse(features_df_hash_array[feature_name+'_hash'], bucket_len)
    return ret
    
def wide_model(wide_col_len_dict, middle_layer=False, data_parallel=False):
    def wide_reduce(feature_length):
        reduced = int(pow(feature_length,REDUCE_RATIO))
        if(reduced<=0):
            reduced = 1
        return reduced
    
    def wide_array_width(wide_col_len_dict):
        total_width=0
        for key in wide_col_len_dict:
            total_width=total_width+bucket_size(wide_col_len_dict[key])
        return total_width

    def parallelize_tower(model, gpu_count):
        def prediction_mean(total_outputs, gpu_count):
            return tf.realdiv(tf.add_n(total_outputs), gpu_count)
        
        total_outputs=[]
        with tf.device('/cpu:0'):
            input_ops_placeholder = tf.placeholder(tf.float32, shape=(None, width))

        for gpu_id in range(gpu_count):
            with tf.device('/gpu:%d' % gpu_id):
                total_outputs.append(model(input_ops_placeholder))
                
        with tf.device('/cpu:0'):
            pred_mean = Lambda(prediction_mean, output_shape=model.output_shape, arguments={'gpu_count':gpu_count})(total_outputs)
            model.outputs.append(pred_mean)
            return model

    def parallelize_data(model, gpu_count):
        #Extracts a slice of size from a tensor input starting at the location specified by begin_from. 
        def batch_slice(batch, gpu_id, gpu_count):
            batch_shape = tf.shape(batch)
            size = tf.concat(0, [tf.floor_div(batch_shape[:1],gpu_count), batch_shape[1:]])
            skip = tf.concat(0, [tf.floor_div(batch_shape[:1],gpu_count), batch_shape[1:]*0])
            begin_from = skip * gpu_id
            return tf.slice(batch, begin_from, size)

        final_outputs = []
        final_outputs.extend([[]]*len(model.outputs))
        for gpu_id in range(gpu_count):
            with tf.device('/gpu:%d' % gpu_id):
                with tf.name_scope('tower_%d' % gpu_id) as scope:
                    inputs = []
                    for feature in model.inputs:
                        input_shape = tuple(feature.get_shape().as_list())[1:]
                        slice_n = Lambda(batch_slice, output_shape=input_shape, arguments={'gpu_id':gpu_id,'gpu_count':gpu_count})(feature)
                        inputs.append(slice_n)                
                    outputs = model(inputs)
                    if not isinstance(outputs, list):
                        outputs = [outputs]
                    for output_num in range(len(outputs)):
                        final_outputs[output_num].append(outputs[output_num])
        with tf.device('/cpu:0'):
            merged = []
            for outputs in final_outputs:
                merged.append(merge(outputs, mode='concat',concat_axis=0))
            return Model(input=model.inputs, output=merged)
    
    # Begin with a CPU Replica
    with tf.device('/cpu:0'):
        width = wide_array_width(wide_col_len_dict)
        reduction = wide_reduce(width)        
        print("REDUCED LAYER : " + str(reduction))
        
        model = Sequential()
        model.add(Dense(reduction, input_dim=width))         
        if(middle_layer):
            model.add(Dense(wide_reduction(reduction))) #Exponentially Reduce
        #final_layer              
        model.add(Dense(1, init='normal'))        
    
    #Parllelize Data
    if(data_parallel):
        model = parallelize_data(model, NUM_OF_GPUS)
    else:
        model = parallelize_tower(model, NUM_OF_GPUS)
        
    if(FTRL):
        model.compile(loss='mean_squared_error', optimizer=tf.train.FtrlOptimizer(
            learning_rate=FTRL_LEARNING_RATE,
            l1_regularization_strength=1.0,
            l2_regularization_strength=1.0))
    else:
        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=0.002,
                                                                                       beta_1=0.9, 
                                                                                       beta_2=0.999, 
                                                                                       epsilon=1e-08, 
                                                                                       decay=0.0))
    return model
    
def get_session(gpu_fraction=0.2):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Strips the dataframe into 3 parts. 
# (1) Dense Matrix
# (2) feed_list of arrays for branching and merging 
# (3) Dictionary of Sparse Arrays for any other manipulation later
def get_input_data(features_df, wide_col_len_dict):
    wide_col_data_dict = {}
    for feature_name in SUMMARY_KEY_COLS:
        start_time = time.time()
        sparse_matrix = sparse_hash(features_df, feature_name, int(wide_col_len_dict[feature_name]))
        wide_col_data_dict[feature_name] = sparse_matrix 
        del sparse_matrix
        
    feed_list = []
    for key in SUMMARY_KEY_COLS:
        arr = wide_col_data_dict[key].toarray()
        feed_list.append(arr)

    input_dense_matrix = np.hstack(tuple(feed_list))

    return input_dense_matrix, feed_list, wide_col_data_dict
    
    
# ALL CONSTANTS ARE DEFINED HERE
tf.logging.set_verbosity(tf.logging.ERROR)

#GPU Related
NUM_OF_GPUS=3
GPU_FRACTION=0.7

#Data Sampling
TRAINING_DATA_SAMPLE=0.3
TEST_DATA_SAMPLE=0.075

#NN Arch
REDUCE_RATIO=0.3
FTRL=True
IS_PARALLEL=False
FTRL_LEARNING_RATE=0.2
train_steps = 3
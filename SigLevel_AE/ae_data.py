#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#Import Data from the .mat files

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

NUM_CLASSES = 7
VALID_SIZE = 10000
TEST_SIZE = 20000
TRAIN_SIZE = 70000

def read_and_decode(filename ,batch_size, shuffle_batch=True):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename])
    _,serialized_example = reader.read(filename_queue)
    features =tf.parse_single_example(serialized_example,features={'sample':tf.VarLenFeature(tf.float32),'label':tf.FixedLenFeature([],tf.int64)})
    
    #sample = tf.decode_raw(features['sample'],tf.uint8)
    #sample = tf.cast(sample, tf.float32)
    # make it dense tensor
    sample = tf.sparse_tensor_to_dense(features['sample'])

    #sample = features['sample']
    sample = tf.reshape(sample, [2*1024]) 
#    sample = sample[:1024]
    
    label = features['label']
    
    if shuffle_batch:
        samples, labels = tf.train.shuffle_batch([sample,label],batch_size=batch_size,capacity=8000,num_threads=4,min_after_dequeue=2000)
    else:
        samples, labels = tf.train.batch([sample,label],batch_size=batch_size,capacity=8000,num_threads=4)
    
    return samples, labels
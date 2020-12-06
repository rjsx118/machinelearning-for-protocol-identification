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

def read_and_decode(filename ,batch_size, shuffle_batch=True, length = 1024):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename])
    _,serialized_example = reader.read(filename_queue)
    features =tf.parse_single_example(serialized_example,features={'sample':tf.VarLenFeature(tf.float32),'label':tf.FixedLenFeature([],tf.int64)})
    
    #sample = tf.decode_raw(features['sample'],tf.uint8)
    #sample = tf.cast(sample, tf.float32)
    # make it dense tensor
    sample = tf.sparse_tensor_to_dense(features['sample'])

    #sample = features['sample']
    sample = tf.reshape(sample, [2,-1])
    
    label = features['label']
    
    if shuffle_batch:
        samples, labels = tf.train.shuffle_batch([sample,label],batch_size=batch_size,capacity=8000,num_threads=4,min_after_dequeue=2000,shapes=([2,length],[]), allow_smaller_final_batch=True)
    else:
        samples, labels = tf.train.batch([sample,label],batch_size=batch_size,capacity=8000,num_threads=4,shapes=([2,length],[]),dynamic_pad=True, allow_smaller_final_batch=True)
    
    return samples, labels


def read_and_decode_var(filename, batch_size, shuffle, length = 1024):
    def parse(example_proto):
        features = {'sample':tf.VarLenFeature(tf.float32),'label':tf.FixedLenFeature([],tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, features)
        
        sample = tf.sparse_tensor_to_dense(parsed_features['sample'])
        sample = tf.reshape(sample, [2,-1])
        label = parsed_features['label']
        
        return sample, label
    
    dataset = tf.data.TFRecordDataset(filename).map(parse)
    
    if shuffle:
        dataset = dataset.shuffle(batch_size*2)
    dataset = dataset.repeat()
    dataset = dataset.padded_batch(batch_size,padded_shapes=([2,length],[]))
        
    iterator = dataset.make_one_shot_iterator()
    
    batch_features, batch_labels = iterator.get_next()

    return batch_features, batch_labels





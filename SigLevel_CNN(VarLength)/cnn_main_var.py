#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# This model is consist of a autoencoder-based feature extractor 
#  and a SVM-based classifier

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#import numpy as np
import argparse
import sys
from six.moves import xrange
import os.path
import time
import scipy.io as sio
import tensorflow.contrib.slim as slim
import cnn_data
import cnn_inference as cnninf
#import data

TEST_LENGTH = [640,720,800,880,960,1024]
TEST_NUMBER = [20176,2,77100,88,32133,10501]
TRAIN_NUMBER = [68985,9,264258,408,107671,48669] #68985

FLAGS = None

root_path = '/TFRawdata_varlength'
#tfrecords_train = os.path.join(root_path, 'train_var.tfrecords')
##   dataset for validation
##tfrecords_valid = os.path.join(root_path, 'valid.tfrecords')
##   dataset for test
#tfrecords_test = os.path.join(root_path, 'test_var.tfrecords')
trainfile = [os.path.join(root_path, 'train_var','train_var_%d.tfrecords' % (k+1)) for k in range(6)]
def main(_):
    for i in range(len(TEST_LENGTH)*10):
        tf.reset_default_graph()
        print('File number :%d' % i)
        train_samples, train_labels = cnn_data.read_and_decode(trainfile[i%6], cnninf.BATCH_SIZE, True, TEST_LENGTH[i%6])
        #test_samples, test_labels = cnn_data.read_and_decode_var(tfrecords_test, cnninf.BATCH_SIZE, False, 1024)
        X = tf.placeholder(dtype=tf.float32, shape=train_samples.get_shape())
        Y = tf.placeholder(dtype=tf.float32, shape=[cnninf.BATCH_SIZE])
        
        global_step = tf.Variable(0,trainable=False)
        
        ## The structure of cnn model
        
        features, logits_cnn = cnninf.inference(X,cnninf.BATCH_SIZE)
        
        loss_cnn = cnninf.loss(logits_cnn,Y)
        train_cnn = cnninf.train(loss_cnn,global_step,1000,cnninf.BATCH_SIZE)
        correct_cnn = cnninf.evaluation(logits_cnn, Y)
        
        epoch_train = (TRAIN_NUMBER[i%6] // 10000 + 1) * 10
        
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
        test_accuracy = []
        test_loss = []
        
        with tf.Session() as sess:
            sess.run(init)
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord) 
            
            if os.path.exists('logfile/chcom.ckpt'):
                saver.restore(sess, 'logfile/chcom.ckpt')
#                tf.get_variable_scope().reuse_variables()
                
            for _ in xrange(epoch_train):
                sam_batch, label_batch = sess.run([train_samples, train_labels])
                sess.run(train_cnn,feed_dict={X:sam_batch,Y:label_batch})
                err, acc = sess.run([loss_cnn, correct_cnn],feed_dict={X:sam_batch,Y:label_batch})
                print("Loss of File %d ,training CNN, loss: %f, and accuracy:%f" % (i,err,acc))
                
            saver.save(sess, 'logfile/chcom.ckpt')
            coord.request_stop()
            coord.join(threads)
        sess.close()
        del sess

    
    
   
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.01, help='The initial learning rate')
    parser.add_argument('--max_steps', type=int, default=50000, help='The max steps for training')
    parser.add_argument('--train_dir', type=str, default='logfile', help='The road of logfile')
    FLAGS,unparsed = parser.parse_known_args()
    tf.app.run(main=main,argv=[sys.argv[0]]+unparsed)

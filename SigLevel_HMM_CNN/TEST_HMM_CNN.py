#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# This model is consist of a autoencoder-based feature extractor 
#  and a SVM-based classifier

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import sys
from six.moves import xrange
import os.path
import time
from hmmlearn import hmm
from sklearn.externals import joblib

import hmm_data
import hmm_cnn_inference as hmminf
#import data

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2000

FLAGS = None

root_path = '/Dataset/SigLevel/TFRawdata'
tfrecords_train = os.path.join(root_path, 'train.tfrecords')
#   dataset for validation
tfrecords_valid = os.path.join(root_path, 'valid.tfrecords')
#   dataset for test
tfrecords_test = os.path.join(root_path, 'test.tfrecords')

def main(_):
    X = tf.placeholder(dtype=tf.float32, shape=[None, 2048], name='X_input')
    Y = tf.placeholder(dtype=tf.float32, shape=[None], name='Y_input')
    
    global_step = tf.Variable(0,trainable=False)
    
    ## The structure of cnn model
    features, logits_cnn = hmminf.inference(X,hmminf.BATCH_SIZE)
    loss_cnn = hmminf.loss(logits_cnn,Y)
    train_cnn = hmminf.train(loss_cnn,global_step,NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,hmminf.BATCH_SIZE)
    correct_cnn = hmminf.evaluation(logits_cnn, Y)
    
    '''Construct the dataset'''
    train_samples, train_labels=hmm_data.read_and_decode(tfrecords_test,hmminf.BATCH_SIZE,False)
    test_samples, test_labels=hmm_data.read_and_decode(tfrecords_train,hmminf.BATCH_SIZE,False)       
    epoch_train = hmm_data.TRAIN_SIZE *hmm_data.NUM_CLASSES // hmminf.BATCH_SIZE
    epoch_test = hmm_data.TEST_SIZE *hmm_data.NUM_CLASSES // hmminf.BATCH_SIZE 

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        
        sess.run(init)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        try:
            saver.restore(sess, 'logfile\\chcom.ckpt')
            
            '''The training of classification model, such as hmm.'''

            for i in range(epoch_train):
                train_data, train_target = sess.run([train_samples, train_labels])
                train_feature = sess.run(features, feed_dict={X:train_data,Y:train_target})
                if i == 0:
                    feature_train = train_feature
                    label_train = train_target
                else:
                    feature_train = np.vstack((feature_train, train_feature))
                    label_train = np.hstack((label_train, train_target))

            print('Fit a Hidden Markov Model model to the data.')
            state_num = 7
            model = hmm.GaussianHMM(n_components=state_num)
            rf = model.fit(feature_train)
            predicted = model.predict(feature_train)
            accuracy = hmminf.accuracy_score(label_train, predicted)
            print("Accuracy: %.3f" % accuracy)
            joblib.dump(rf,'rf.model')

            n_acc, n_batch = 0, 0
            for i in range(epoch_test):
                test_data, test_target = sess.run([test_samples, test_labels])
                feature_test = sess.run(features, feed_dict={X:test_data,Y:test_target})
                test_predicted = model.predict(feature_test)
                test_accuracy_value = hmminf.accuracy_score(test_target,test_predicted)
                n_acc += test_accuracy_value
                n_batch+=1
                #print("Testing Classification report of batch:%d, accuracy %g:\n" % (i+1,test_accuracy_value.eval()))
            print("The Evaluation of CNN_HMM based model %g:\n" % (n_acc/n_batch))
                
#                if n_acc/n_batch > 0.953:
#                    joblib.dump(rf,'rf1.model')
#                    break
                

        except tf.errors.OutOfRangeError:
            print('Done training --epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)
 
 
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.01, help='The initial learning rate')
    parser.add_argument('--max_steps', type=int, default=200000, help='The max steps for training')
    parser.add_argument('--train_dir', type=str, default='logfile', help='The road of logfile')
    FLAGS,unparsed = parser.parse_known_args()
    tf.app.run(main=main,argv=[sys.argv[0]]+unparsed)
    
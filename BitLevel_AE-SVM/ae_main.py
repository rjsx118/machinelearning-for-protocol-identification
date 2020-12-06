#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# This model is consist of a autoencoder-based feature extractor 
#  and a SVM-based classifier

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import sys
from six.moves import xrange
import os.path
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.svm import SVC
import time
import scipy.io as sio
import numpy as np

import ae_data
import ae_inference

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2000
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY_FACTOR = 0.95

FLAGS = None

root_path = '/DataSet/BitLevel/TFRawdata'
tfrecords_train = os.path.join(root_path, 'train.tfrecords')
#   dataset for validation
tfrecords_valid = os.path.join(root_path, 'valid.tfrecords')
#   dataset for test
tfrecords_test = os.path.join(root_path, 'test.tfrecords')

def main(_):
    X = tf.placeholder(dtype=tf.float32,shape=[None,1024],name='X_input')
    Y = tf.placeholder(dtype=tf.float32,shape=[None],name='Y_input')
    
    '''The model of autoencoder'''
    dimensions = [1024,512,256,128,ae_inference.FEATURE_DIMENSION]
    autoencoder = ae_inference.autoencoder(X,dimensions)
    
    global_step = tf.Variable(0,trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, FLAGS.max_steps//10, LEARNING_RATE_DECAY_FACTOR)
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(autoencoder['loss'],global_step=global_step)
    tf.summary.scalar('loss', autoencoder['loss'])
    
    '''Construct the dataset'''
    train_samples, train_labels=ae_data.read_and_decode(tfrecords_train,FLAGS.batch_size,True)
    test_samples, test_labels=ae_data.read_and_decode(tfrecords_test,FLAGS.batch_size,False)    
    epoch_train = ae_data.TRAIN_SIZE *ae_data.NUM_CLASSES // FLAGS.batch_size
    epoch_test = ae_data.TEST_SIZE *ae_data.NUM_CLASSES // FLAGS.batch_size
    
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        try:
            '''
            The training of autoencoder model to extract the features of protocol.
            '''
            saver.restore(sess, 'logfile\\chcom.ckpt')       
            
            '''The training of classification model, such as svm.''' 
            start_time = time.time()
            for i in range(100):
                train_data, train_target = sess.run([train_samples, train_labels])
                ae_train = sess.run(autoencoder, feed_dict={X:train_data,Y:train_target})
                if i == 0:
                    feature_train = ae_train['hidden']
                    label_train = train_target
                else:
                    feature_train = np.vstack((feature_train, ae_train['hidden']))
                    label_train = np.hstack((label_train, train_target))
            classifier = SVC(decision_function_shape='ovo')
            rf = classifier.fit(feature_train,label_train)
            train_predicted = classifier.predict(feature_train)
            train_accuracy_value = tf.reduce_mean(tf.cast(tf.equal(train_predicted,label_train),tf.float32))
            print("Training Classification report for classifier %g:\n" % train_accuracy_value.eval())
#            cm1 = confusion_matrix(label_train,train_predicted)
#            print(cm1)
            joblib.dump(rf,'logfile\\rf.model')

            '''The testing of svm model.''' 
            n_acc, n_batch, n_cm = 0, 0, 0
            for i in range(epoch_test):
                test_data, test_target = sess.run([test_samples, test_labels])
                ae_test = sess.run(autoencoder, feed_dict={X:test_data,Y:test_target})
                feature_test = ae_test['hidden']
                test_predicted = rf.predict(feature_test)
                test_accuracy_value = tf.reduce_mean(tf.cast(tf.equal(test_predicted,test_target),tf.float32))
                n_acc += test_accuracy_value.eval()
                cm = confusion_matrix(test_target, test_predicted, labels=[0,1,2,3,4,5,6])
                n_cm += cm
                n_batch+=1
#                print("Testing Classification report of batch:%d, accuracy %g:" % (i+1,test_accuracy_value.eval()))
            print("Testing Classification report for classifier %g:\n" % (n_acc/n_batch))
            
#            cm2 = confusion_matrix(label_test,test_predicted)
            #print(cm2)
         
            
#            sio.savemat('logfile\\accuracy.mat', {'train_accuracy': train_accuracy_value.eval(), 'test_accuracy': test_accuracy_value.eval(),'confusion_matrix1':cm1,'confusion_matrix2':cm2}) 


        except tf.errors.OutOfRangeError:
            print('Done training --epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='data_ae',help='Path to the train/test data ')
    parser.add_argument('--batch_size',type=int,default=1000,help='The size of minibatch')
    parser.add_argument('--learning_rate',type=float,default=0.01,help='The initial learning rate')
    parser.add_argument('--max_steps',type=int,default=10000,help='The max steps for training')
    parser.add_argument('--train_dir',type=str,default='logfile',help='The road of logfile')
    FLAGS,unparsed = parser.parse_known_args()
    tf.app.run(main=main,argv=[sys.argv[0]]+unparsed)
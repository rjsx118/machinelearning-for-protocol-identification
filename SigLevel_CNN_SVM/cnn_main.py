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
import scipy.io as sio
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import cnn_data
import cnn_inference as cnninf
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
    X = tf.placeholder(dtype=tf.float32, shape=[None, 2*1024], name='X_input')
    Y = tf.placeholder(dtype=tf.float32, shape=[None], name='Y_input')
    
    global_step = tf.Variable(0,trainable=False)
    
    ## The structure of cnn model
    features, logits_cnn = cnninf.inference(X,cnninf.BATCH_SIZE)
    loss_cnn = cnninf.loss(logits_cnn,Y)
    train_cnn = cnninf.train(loss_cnn,global_step,NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,cnninf.BATCH_SIZE)
    correct_cnn = cnninf.evaluation(logits_cnn, Y)
    
    loss_val = tf.Variable(0.0)  
    acc_val = tf.Variable(0.0) 
    tf.summary.scalar('loss', loss_val)
    tf.summary.scalar('accuracy', acc_val)
    
    train_samples, train_labels=cnn_data.read_and_decode(tfrecords_train,cnninf.BATCH_SIZE,True)
    test_samples, test_labels=cnn_data.read_and_decode(tfrecords_test,cnninf.BATCH_SIZE,True)  
    
    epoch_train = cnn_data.TRAIN_SIZE *cnn_data.NUM_CLASSES // cnninf.BATCH_SIZE
    epoch_test = cnn_data.TEST_SIZE *cnn_data.NUM_CLASSES // cnninf.BATCH_SIZE
    
    saver = tf.train.Saver() 
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord) 
        
        try:
            saver.restore(sess, 'logfile\\chcom.ckpt-29999')
            
            '''The training of classification model, such as svm.''' 
            start_time = time.time()

            for i in range(epoch_train):
                train_data, train_target = sess.run([train_samples, train_labels])
                fea = sess.run(features, feed_dict={X:train_data,Y:train_target})
                if i == 0:
                    feature_train = fea
                    label_train = train_target
                else:
                    feature_train = np.vstack((feature_train, fea))
                    label_train = np.hstack((label_train, train_target))
            classifier = SVC(decision_function_shape='ovo')
            rf=classifier.fit(feature_train,label_train)
            train_predicted = classifier.predict(feature_train)
            train_accuracy_value = tf.reduce_mean(tf.cast(tf.equal(train_predicted,label_train),tf.float32))
            print("Training Classification report for classifier %g:\n" % train_accuracy_value.eval())
            #cm1 = confusion_matrix(label_train,train_predicted)
            joblib.dump(rf,'rf.model')

            '''The testing of svm model.''' 
            n_acc, n_batch = 0, 0
            for i in range(epoch_test):
                test_data, test_target = sess.run([test_samples, test_labels])
                feature_test = sess.run(features, feed_dict={X:test_data,Y:test_target})
                test_predicted = classifier.predict(feature_test)
                test_accuracy_value = tf.reduce_mean(tf.cast(tf.equal(test_predicted,test_target),tf.float32))
                n_acc += test_accuracy_value.eval()
                n_batch+=1
                #print("Testing Classification report of batch:%d, accuracy %g:\n" % (i+1,test_accuracy_value.eval()))
            print("The Evaluation of CNN_SVM based model %g:\n" % (n_acc/n_batch))
            #cm2 = confusion_matrix(label_test, test_predicted)
            #print(cm2)
            
            duration=time.time()-start_time
            print("The time of training the svm model is:  %f" % duration)            
            
#            sio.savemat('logfile\\accuracy.mat', {'train_accuracy': train_accuracy_value, 'test_accuracy': test_accuracy_value,'duration': duration,'confusion_matrix1':cm1,'confusion_matrix2':cm2}) 
#                    #print("Loss of training CNN, step: %d, loss: %f, and accuracy:%f" % (i+1,err,ac))
        except tf.errors.OutOfRangeError:
            print('Done training --epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)
 
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.01, help='The initial learning rate')
    parser.add_argument('--max_steps', type=int, default=100000, help='The max steps for training')
    parser.add_argument('--train_dir', type=str, default='logfile', help='The road of logfile')
    FLAGS,unparsed = parser.parse_known_args()
    tf.app.run(main=main,argv=[sys.argv[0]]+unparsed)
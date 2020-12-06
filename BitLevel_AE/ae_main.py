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
import time
import scipy.io as sio
from sklearn.metrics import confusion_matrix

import ae_data
import ae_inference

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2000
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY_FACTOR = 0.95

FLAGS = None

root_path = '/Dataset/BitLevel/TFRawdata'
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
    
    '''The model of softmax'''
    #global_step_fully = tf.Variable(0,trainable=False)
    logits = ae_inference.fully_inference(autoencoder['hidden'])    
    loss_ae = ae_inference.fully_loss(logits,Y)    
    train_ae = ae_inference.fully_train(loss_ae,global_step,FLAGS.max_steps//10,FLAGS.batch_size)    
    correct_ae = ae_inference.fully_evaluation(logits, Y)
    
    '''Construct the dataset'''
    train_samples, train_labels=ae_data.read_and_decode(tfrecords_train,FLAGS.batch_size,True)
    valid_samples, valid_labels=ae_data.read_and_decode(tfrecords_valid,FLAGS.batch_size,False)
    test_samples, test_labels=ae_data.read_and_decode(tfrecords_test,FLAGS.batch_size,False)    

    saver = tf.train.Saver(max_to_keep = 50) 
    init = tf.global_variables_initializer()
    epoch_train = ae_data.TRAIN_SIZE *ae_data.NUM_CLASSES // FLAGS.batch_size
    epoch_valid = ae_data.VALID_SIZE *ae_data.NUM_CLASSES // FLAGS.batch_size
    epoch_test = ae_data.TEST_SIZE *ae_data.NUM_CLASSES // FLAGS.batch_size
    
    train_accuracy = []
    test_accuracy = []
    train_loss = []
    test_loss = []
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        try:
            '''
            The training of autoencoder model to extract the features of protocol.
            '''
            start_time = time.time()
            for i in xrange(FLAGS.max_steps):
                sam_batch, label_batch = sess.run([train_samples, train_labels])
                sess.run(optimizer, feed_dict={X: sam_batch, Y: label_batch}) 
            
                if (i+1) % 1000 == 0:
            # 基于softmax的识别模型
                    for j in xrange(FLAGS.max_steps_fully_epoch):
                        sam_batch, label_batch = sess.run([train_samples, train_labels])
                        sess.run(train_ae, feed_dict={X: sam_batch, Y: label_batch}) 
#                if i % 20 == 0:
#                    #########   Validation evaluation   ##########
#                    n_cost, n_acc, n_batch = 0, 0, 0
#                    for _ in range(epoch_valid):
#                        sam_batch, label_batch = sess.run([valid_samples, valid_labels])
#                        ae, acc = sess.run([autoencoder, correct_ae], feed_dict={X: sam_batch, Y: label_batch})
#                        n_cost += ae['loss']
#                        n_acc += acc
#                        n_batch += 1
#                    print("Evaluation of validation data, loss: %f, and accuracy:%f" % ((n_cost / n_batch), (n_acc / n_batch / FLAGS.batch_size)))                    
                if (i+1) % 1000 == 0:
                    #########   Training evaluation   ##########
                    n_cost, n_acc, n_batch = 0, 0, 0
                    for _ in range(epoch_train):
                        sam_batch, label_batch = sess.run([train_samples, train_labels])
                        ae, acc = sess.run([autoencoder, correct_ae], feed_dict={X: sam_batch, Y: label_batch})
                        n_cost += ae['loss']
                        n_acc += acc
                        n_batch += 1
                    print("Evaluation of training data, loss: %f, and accuracy:%f" % ((n_cost/n_batch), (n_acc/n_batch/FLAGS.batch_size)))
                    train_accuracy.append(n_acc/n_batch/FLAGS.batch_size)
                    train_loss.append(n_cost/n_batch/FLAGS.batch_size)
                    
                    #########   Testing evaluation   ##########
                    n_cost, n_acc, n_batch = 0, 0, 0
                    for j in range(epoch_test):
                        sam_batch, label_batch = sess.run([test_samples, test_labels])
                        ae, lgs, acc = sess.run([autoencoder,logits, correct_ae], feed_dict={X: sam_batch, Y: label_batch})
                        n_cost += ae['loss']
                        n_acc += acc
                        n_batch += 1
#                        cm = confusion_matrix(label_batch, tf.argmax(lgs,1).eval(), labels=[0,1,2,3,4,5,6])
#                        n_cm += cm
                        #print("Evaluation of testing data, batch:%d, accuracy:%f" % (j,acc / FLAGS.batch_size))
                    print("Evaluation of testing data, loss: %f, and accuracy:%f" % ((n_cost / n_batch), (n_acc / n_batch / FLAGS.batch_size)))
                    test_accuracy.append(n_acc / n_batch / FLAGS.batch_size)
                    test_loss.append(n_cost / n_batch/FLAGS.batch_size)
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'chcom.ckpt')
                    saver.save(sess, checkpoint_path, global_step=i)
                
            duration=time.time()-start_time
            print("The time of training the softmax model is:  %f" % duration)
            
            
            sio.savemat('logfile/accuracy.mat', {'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy, 
                                                 'train_loss': train_loss, 'test_loss': test_loss, 'duration': duration})


        except tf.errors.OutOfRangeError:
            print('Done training --epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=100,help='The size of minibatch')
    parser.add_argument('--learning_rate',type=float,default=0.01,help='The initial learning rate')
    parser.add_argument('--max_steps',type=int,default=50000,help='The max steps for training ae')
    parser.add_argument('--max_steps_fully_epoch',type=int,default=100,help='The max steps for training fully layer')
    parser.add_argument('--train_dir',type=str,default='logfile',help='The road of logfile')
    FLAGS,unparsed = parser.parse_known_args()
    tf.app.run(main=main,argv=[sys.argv[0]]+unparsed)

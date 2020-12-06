#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# This model is consist of a autoencoder-based feature extractor 
#  and a SVM-based classifier

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
#import numpy as np
import argparse
import sys
from six.moves import xrange
import os.path
import time
import scipy.io as sio
import lstm_data
import lstm_cnn_inference as lstminf
#import data

#NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2000

FLAGS = None

root_path ='/Dataset/SigLevel/TFRawdata'
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
    features, logits_cnn = lstminf.inference(X,lstminf.BATCH_SIZE)
    loss_cnn = lstminf.loss(logits_cnn,Y)
    train_cnn = lstminf.train(loss_cnn,global_step,FLAGS.max_steps//10,lstminf.BATCH_SIZE)
    correct_cnn = lstminf.evaluation(logits_cnn, Y)
    
    ## The structure of lstm model
    features_batch = tf.reshape(features,[-1, lstminf.N_STEPS, lstminf.FEATURE_DIM])
#    #logits_rnn = lstminf.RNN(features_batch, FLAGS.rnn_batch_size) # 单层LSTM网络
    logits_rnn = lstminf.RNN_MULTILAYER(features_batch, FLAGS.rnn_batch_size)  # 多层LSTM网络
    loss_rnn = lstminf.loss(logits_rnn, Y)
    train_rnn = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss_rnn)  
    correct_rnn = lstminf.evaluation(logits_rnn, Y)    
    
    train_samples, train_labels = lstm_data.read_and_decode(tfrecords_train, lstminf.BATCH_SIZE, True)
    train_samples_f, train_labels_f = lstm_data.read_and_decode(tfrecords_train, lstminf.BATCH_SIZE, False)
    valid_samples, valid_labels = lstm_data.read_and_decode(tfrecords_valid, lstminf.BATCH_SIZE, False)
    test_samples, test_labels = lstm_data.read_and_decode(tfrecords_test, lstminf.BATCH_SIZE, False)
    
    init = tf.global_variables_initializer()
    variables = slim.get_variables_to_restore()
    variables_to_restore = [v for v in variables if 'conv' in v.name or 'fc' in v.name or 'softmax' in v.name ]
   
#    variables_to_restore = tf.contrib.framework.get_variables_to_restore(include=['conv1/weights','conv2/weights','conv3/weights','conv4/weights',
#                                                                                  'fc1/weights','fc2/weights','softmax_linear/weights',
#                                                                                  'conv1/biases','conv2/biases','conv3/biases','conv4/biases',
#                                                                                  'fc1/biases','fc2/biases','softmax_linear/biases'])
    saver = tf.train.Saver(variables_to_restore)
    saver_out = tf.train.Saver(max_to_keep = 100)
    
    epoch_train = lstm_data.TRAIN_SIZE *lstm_data.NUM_CLASSES // lstminf.BATCH_SIZE
    epoch_valid = lstm_data.VALID_SIZE *lstm_data.NUM_CLASSES // lstminf.BATCH_SIZE
    epoch_test = lstm_data.TEST_SIZE *lstm_data.NUM_CLASSES // lstminf.BATCH_SIZE
    
    train_accuracy = []
    valid_accuracy = []
    test_accuracy = []
    train_loss = []
    valid_loss = []
    test_loss = []
    #cnn_acc = []
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        try:
            '''
            The training of cnn model to extract the features of protocol.
            '''
            saver.restore(sess, 'cnnmodel/chcom.ckpt')
            n_cost, n_acc, n_batch = 0, 0, 0
            for _ in range(epoch_test):
                sam_batch, label_batch = sess.run([test_samples, test_labels])
                err,acc = sess.run([loss_cnn, correct_cnn],feed_dict={X:sam_batch,Y:label_batch})
                n_cost += err
                n_acc += acc
                n_batch += 1
            cnn_acc = n_acc/n_batch/lstminf.BATCH_SIZE
            print("Evaluation of cnn model accuracy:%f" % cnn_acc)
#                    
            
            start_time = time.time()

            for j in xrange(FLAGS.max_steps_rnn):
        # 基于LSTM的识别模型
                sam_batch, label_batch = sess.run([train_samples_f, train_labels_f])
                sess.run(train_rnn, feed_dict={X: sam_batch, Y: label_batch})
                if (j+1)>100000 and (j+1) % 4000 == 0:
                    #########   Training evaluation   ##########
#                    n_cost, n_acc, n_batch = 0, 0, 0
#                    for _ in range(epoch_train):
#                        sam_batch, label_batch = sess.run([train_samples_f, train_labels_f])
#                        err, acc = sess.run([loss_rnn, correct_rnn], feed_dict={X:sam_batch, Y:label_batch})
#                        n_cost += err
#                        n_acc += acc
#                        n_batch += 1
#                    print("Evaluation of training data, loss: %f, and accuracy:%f" % ((n_cost/n_batch), (n_acc/n_batch/lstminf.BATCH_SIZE)))
#                    train_accuracy.append(n_acc/n_batch/lstminf.BATCH_SIZE)
#                    train_loss.append(n_cost/n_batch/lstminf.BATCH_SIZE)
                    #########   Testing evaluation   ##########
                    n_cost, n_acc, n_batch = 0, 0, 0
                    for _ in range(epoch_test):
                        sam_batch, label_batch = sess.run([test_samples, test_labels])
                        err, acc = sess.run([loss_rnn, correct_rnn], feed_dict={X: sam_batch, Y: label_batch})
                        n_cost += err
                        n_acc += acc
                        n_batch += 1
                        # print("Evaluation of testing data, batch: %d, and accuracy:%f" % (j, acc / lstminf.BATCH_SIZE))
                    print("Evaluation of testing data, loss: %f, and accuracy:%f" % ((n_cost / n_batch), (n_acc / n_batch / lstminf.BATCH_SIZE)))
                    test_accuracy.append(n_acc / n_batch / lstminf.BATCH_SIZE)
                    test_loss.append(n_cost / n_batch)
                    
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'chcom.ckpt')
                    saver_out.save(sess, checkpoint_path, global_step=j)

                
            during_time = time.time() - start_time
            sio.savemat('logfile/accuracy.mat', {'train_accuracy': train_accuracy, 
                                                 'test_accuracy': test_accuracy, 'train_loss': train_loss, 
                                                 'test_loss': test_loss,
                                                 'duration': during_time,'cnn_acc':cnn_acc})
#
#            print("the time of training lstm： %f" % during_time)
#            checkpoint_path = os.path.join(FLAGS.train_dir, 'chcom.ckpt')
#            saver.save(sess, checkpoint_path, global_step=i)

        except tf.errors.OutOfRangeError:
            print('Done training --epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)
 
 
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_batch_size', type=int, default=1, help='The size of minibatch of rnn')
    parser.add_argument('--max_steps_rnn', type=int, default=300000, help='The max steps for training lstm')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='The initial learning rate')
    parser.add_argument('--max_steps', type=int, default=100000, help='The max steps for training')
    parser.add_argument('--train_dir', type=str, default='logfile', help='The road of logfile')
    FLAGS,unparsed = parser.parse_known_args()
    tf.app.run(main=main,argv=[sys.argv[0]]+unparsed)
    

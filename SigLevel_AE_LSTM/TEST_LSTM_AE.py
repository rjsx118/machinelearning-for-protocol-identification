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

import lstm_data
import lstm_ae_inference as lstminf

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2000
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY_FACTOR = 0.9
FLAGS = None

root_path = '/Dataset/SigLevel/TFRawdata'
tfrecords_train = os.path.join(root_path, 'train.tfrecords')
#   dataset for validation
tfrecords_valid = os.path.join(root_path, 'valid.tfrecords')
#   dataset for test
tfrecords_test = os.path.join(root_path, 'test.tfrecords')

def main(_):
    X = tf.placeholder(dtype=tf.float32, shape=[None, 2048], name='X_input')
    X_lstm = tf.placeholder(dtype=tf.float32, shape=[None, lstminf.N_STEPS, lstminf.FEATURE_DIMENSION], name='X_input')
    Y = tf.placeholder(dtype=tf.float32, shape=[None], name='Y_input')
    
    '''The model of autoencoder'''
    dimensions = [2048, 1024, 256, 128, lstminf.FEATURE_DIMENSION]
    autoencoder = lstminf.autoencoder(X, dimensions)
    
    ## The structure of autoencoder model
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, FLAGS.max_steps//10, LEARNING_RATE_DECAY_FACTOR)
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(autoencoder['loss'], global_step=global_step)
    tf.summary.scalar('loss', autoencoder['loss'])
    
    '''The model of softmax'''
    logits = lstminf.fully_inference(autoencoder['hidden'])    
    loss_ae = lstminf.fully_loss(logits,Y)    
    train_ae = lstminf.fully_train(loss_ae,global_step,FLAGS.max_steps//10,lstminf.BATCH_SIZE) 
    correct_ae = lstminf.fully_evaluation(logits, Y)
    
    ## The structure of lstm model
    logits_rnn = lstminf.RNN_MULTILAYER(X_lstm, FLAGS.rnn_batch_size)  # 多层LSTM网络
    loss_rnn = lstminf.loss(logits_rnn, Y)
    train_rnn = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss_rnn)  
    correct_rnn = lstminf.evaluation(logits_rnn, Y)    
    
    train_samples, train_labels = lstm_data.read_and_decode(tfrecords_train, lstminf.BATCH_SIZE, True)
    train_samples_f, train_labels_f = lstm_data.read_and_decode(tfrecords_train, lstminf.BATCH_SIZE, False)
    valid_samples, valid_labels = lstm_data.read_and_decode(tfrecords_valid, lstminf.BATCH_SIZE, False)
    test_samples, test_labels = lstm_data.read_and_decode(tfrecords_test, lstminf.BATCH_SIZE, False)

    saver = tf.train.Saver(max_to_keep = 50)
    init = tf.global_variables_initializer()
    epoch_train = lstm_data.TRAIN_SIZE *lstm_data.NUM_CLASSES // lstminf.BATCH_SIZE
    epoch_valid = lstm_data.VALID_SIZE *lstm_data.NUM_CLASSES // lstminf.BATCH_SIZE
    epoch_test = lstm_data.TEST_SIZE *lstm_data.NUM_CLASSES // lstminf.BATCH_SIZE
    
    train_accuracy = []
    valid_accuracy = []
    test_accuracy = []
    train_loss = []
    valid_loss = []
    test_loss = []
    ae_acc = []
    
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
            
                if (i+1) % 2000 == 0:
                    # 基于softmax的识别模型
                    for _ in xrange(1000):
                        sam_batch, label_batch = sess.run([train_samples, train_labels])
                        sess.run(train_ae, feed_dict={X: sam_batch, Y: label_batch}) 
                    #########   Validation evaluation   ##########
                    n_acc, n_batch = 0, 0
                    for _ in range(epoch_test):
                        sam_batch, label_batch = sess.run([test_samples, test_labels])
                        acc = sess.run(correct_ae, feed_dict={X: sam_batch, Y: label_batch})
                        n_acc += acc
                        n_batch += 1
                    ae_acc.append(n_acc / n_batch / lstminf.BATCH_SIZE)
                
                    
                    
                    ####  The training of 2l-lstm model.
                    '''The training of 2l-lstm model.'''
                    for _ in xrange(1000):
                        sam_batch, label_batch = sess.run([train_samples_f, train_labels_f])
                        ae_output = sess.run(autoencoder, feed_dict={X: sam_batch, Y: label_batch})
                        sam_batch = ae_output['hidden'].reshape([lstminf.BATCH_SIZE, lstminf.N_STEPS, lstminf.FEATURE_DIMENSION])
                        sess.run(train_rnn, feed_dict={X_lstm: sam_batch, Y: label_batch})
                    #########   Training evaluation   ##########
                    n_cost, n_acc, n_batch = 0, 0, 0
                    for _ in range(epoch_train):
                        sam_batch, label_batch = sess.run([train_samples_f, train_labels_f])
                        ae_output = sess.run(autoencoder, feed_dict={X: sam_batch, Y: label_batch})
                        sam_batch = ae_output['hidden'].reshape([lstminf.BATCH_SIZE, lstminf.N_STEPS, lstminf.FEATURE_DIMENSION])
                        err, acc = sess.run([loss_rnn, correct_rnn], feed_dict={X_lstm:sam_batch, Y:label_batch})
                        n_cost += err
                        n_acc += acc
                        n_batch += 1
                    print("Evaluation of training data, loss: %f, and accuracy:%f" % ((n_cost/n_batch), (n_acc/n_batch/lstminf.BATCH_SIZE)))
                    train_accuracy.append(n_acc/n_batch/lstminf.BATCH_SIZE)
                    train_loss.append(n_cost/n_batch/lstminf.BATCH_SIZE)

                    #########   Validation evaluation   ##########
                    n_cost, n_acc, n_batch = 0, 0, 0
                    for _ in range(epoch_valid):
                        sam_batch, label_batch = sess.run([valid_samples, valid_labels])
                        ae_output = sess.run(autoencoder, feed_dict={X: sam_batch, Y: label_batch})
                        sam_batch = ae_output['hidden'].reshape([lstminf.BATCH_SIZE, lstminf.N_STEPS, lstminf.FEATURE_DIMENSION])
                        err, acc = sess.run([loss_rnn, correct_rnn], feed_dict={X_lstm: sam_batch, Y: label_batch})
                        n_cost += err
                        n_acc += acc
                        n_batch += 1
                    print("Evaluation of validation data, loss: %f, and accuracy:%f" % ((n_cost / n_batch), (n_acc / n_batch / lstminf.BATCH_SIZE)))
                    valid_accuracy.append(n_acc / n_batch / lstminf.BATCH_SIZE)
                    valid_loss.append(n_cost / n_batch)

                    #########   Testing evaluation   ##########
                    n_cost, n_acc, n_batch = 0, 0, 0
                    for _ in range(epoch_test):
                        sam_batch, label_batch = sess.run([test_samples, test_labels])
                        ae_output = sess.run(autoencoder, feed_dict={X: sam_batch, Y: label_batch})
                        sam_batch = ae_output['hidden'].reshape([lstminf.BATCH_SIZE, lstminf.N_STEPS, lstminf.FEATURE_DIMENSION])
                        err, acc = sess.run([loss_rnn, correct_rnn], feed_dict={X_lstm: sam_batch, Y: label_batch})
                        n_cost += err
                        n_acc += acc
                        n_batch += 1
                    print("Evaluation of testing data, loss: %f, and accuracy:%f" % ((n_cost / n_batch), (n_acc / n_batch / lstminf.BATCH_SIZE)))
                    test_accuracy.append(n_acc / n_batch / lstminf.BATCH_SIZE)
                    test_loss.append(n_cost / n_batch)
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'chcom.ckpt')
                    saver.save(sess, checkpoint_path, global_step=i)                    
            during_time = time.time() - start_time
            sio.savemat('logfile/accuracy.mat', {'train_accuracy': train_accuracy, 'valid_accuracy': valid_accuracy, 
                                                 'test_accuracy': test_accuracy, 'train_loss': train_loss, 
                                                 'valid_loss': valid_loss, 'test_loss': test_loss, 'duration': during_time})

            print("the time of training lstm： %f" % during_time)

        except tf.errors.OutOfRangeError:
            print('Done training --epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_batch_size', type=int, default=1, help='The size of minibatch of rnn')
    parser.add_argument('--max_steps_rnn', type=int, default=10000, help='The max steps for training lstm')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='The initial learning rate')
    parser.add_argument('--max_steps', type=int, default=50000, help='The max steps for training')
    parser.add_argument('--train_dir', type=str, default='logfile', help='The road of logfile')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)

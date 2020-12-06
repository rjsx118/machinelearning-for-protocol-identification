#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# This model is used to determine the structure of the lstm model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import sys
import os.path
import time
import scipy.io as sio

import lstm_data
import lstm_inference

FLAGS = None

root_path = '/Dataset/SigLevel/TFRawdata'
tfrecords_train = os.path.join(root_path, 'train.tfrecords')
#   dataset for validation
#tfrecords_valid = os.path.join(root_path, 'valid.tfrecords')
#   dataset for test
tfrecords_test = os.path.join(root_path, 'test.tfrecords')


def main(_):
    Samples_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, lstm_inference.N_STEPS, lstm_inference.N_INPUTS], name='X_input')
    Labels_placeholder = tf.placeholder(dtype=tf.float32, shape=[None], name='Y_input')

    #logits_rnn = lstm_inference.RNN(Samples_placeholder, FLAGS.rnn_batch_size) # 单层LSTM网络
    logits_rnn = lstm_inference.RNN_MULTILAYER(Samples_placeholder, FLAGS.rnn_batch_size)  # 多层LSTM网络
    loss_rnn = lstm_inference.loss(logits_rnn, Labels_placeholder)
    train_rnn = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss_rnn)
    
    correct_rnn = lstm_inference.evaluation(logits_rnn, Labels_placeholder)
    
    loss_val = tf.Variable(0.0)  
    acc_val = tf.Variable(0.0) 
    tf.summary.scalar('loss', loss_val)
    tf.summary.scalar('accuracy', acc_val)
#        
    train_samples, train_labels = lstm_data.read_and_decode(tfrecords_train, lstm_inference.BATCH_SIZE, False)
    test_samples, test_labels = lstm_data.read_and_decode(tfrecords_test, lstm_inference.BATCH_SIZE, False)

    saver = tf.train.Saver(max_to_keep = 50)
    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    train_accuracy = []
    test_accuracy = []
    train_loss = []
    test_loss = []
    
    epoch_train = lstm_data.TRAIN_SIZE *lstm_data.NUM_CLASSES // lstm_inference.BATCH_SIZE
    epoch_test = lstm_data.TEST_SIZE *lstm_data.NUM_CLASSES // lstm_inference.BATCH_SIZE
    
    writer_test = tf.summary.FileWriter("logfile/plot_3")    #test
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        try:
            start_time = time.time()

            step = 0
            while step  < FLAGS.max_steps_rnn:
                sam_batch, label_batch = sess.run([train_samples, train_labels])
                sam_batch = sam_batch.reshape([lstm_inference.BATCH_SIZE, lstm_inference.N_STEPS, lstm_inference.N_INPUTS])
                sess.run(train_rnn, feed_dict={Samples_placeholder: sam_batch, Labels_placeholder: label_batch})
                if (step+1) % 100 == 0:
                    #########   Training evaluation   ##########
                    n_cost, n_acc, n_batch = 0, 0, 0
                    for _ in range(epoch_train):
                        sam_batch, label_batch = sess.run([train_samples, train_labels])
                        sam_batch = sam_batch.reshape([lstm_inference.BATCH_SIZE, lstm_inference.N_STEPS, lstm_inference.N_INPUTS])
                        err, acc = sess.run([loss_rnn, correct_rnn], feed_dict={Samples_placeholder:sam_batch, Labels_placeholder:label_batch})
                        n_cost += err
                        n_acc += acc
                        n_batch += 1
                    print("Evaluation of training data, loss: %f, and accuracy:%f" % ((n_cost/n_batch), (n_acc/n_batch/lstm_inference.BATCH_SIZE)))
                    train_accuracy.append(n_acc/n_batch/lstm_inference.BATCH_SIZE)
                    train_loss.append(n_cost/n_batch/lstm_inference.BATCH_SIZE)


                    #########   Testing evaluation   ##########
                    n_cost, n_acc, n_batch = 0, 0, 0
                    for _ in range(epoch_test):
                        sam_batch, label_batch = sess.run([test_samples, test_labels])
                        sam_batch = sam_batch.reshape([lstm_inference.BATCH_SIZE, lstm_inference.N_STEPS, lstm_inference.N_INPUTS])
                        err, acc = sess.run([loss_rnn, correct_rnn], feed_dict={Samples_placeholder: sam_batch, Labels_placeholder: label_batch})
                        n_cost += err
                        n_acc += acc
                        n_batch += 1
                    print("Evaluation of testing data, loss: %f, and accuracy:%f" % ((n_cost / n_batch), (n_acc / n_batch / lstm_inference.BATCH_SIZE)))
                    test_accuracy.append(n_acc / n_batch / lstm_inference.BATCH_SIZE)
                    test_loss.append(n_cost / n_batch)

                    write_op = sess.run(summary, {loss_val: n_cost / n_batch, acc_val: n_acc / n_batch / lstm_inference.BATCH_SIZE})
                    writer_test.add_summary(write_op, step)

                    checkpoint_path = os.path.join(FLAGS.train_dir, 'chcom.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                step += 1
                
            during_time = time.time() - start_time
            sio.savemat('logfile/accuracy.mat', {'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy, 'train_loss': train_loss, 
                                                 'test_loss': test_loss, 'duration': during_time})

            print("the training time of lstm： %f" % during_time)

        except tf.errors.OutOfRangeError:
            print('Done training --epoch limit reached')
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_batch_size', type=int, default=1, help='The size of minibatch of rnn')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='The initial learning rate')
    parser.add_argument('--max_steps_rnn', type=int, default=5000, help='The max steps for training lstm')
    parser.add_argument('--train_dir', type=str, default='logfile', help='The road of logfile')
    parser.add_argument('--use_fp16', type=bool, default=False, help='Train the model using fp16.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)

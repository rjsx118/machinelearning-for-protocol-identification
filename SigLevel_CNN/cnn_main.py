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
    
    writer_train = tf.summary.FileWriter("logfile/plot_1")    #train
    writer_test = tf.summary.FileWriter("logfile/plot_2")    #test
#        
    train_samples, train_labels = cnn_data.read_and_decode(tfrecords_train, cnninf.BATCH_SIZE, True)
    train_samples_f, train_labels_f = cnn_data.read_and_decode(tfrecords_train, cnninf.BATCH_SIZE, False)
    valid_samples, valid_labels = cnn_data.read_and_decode(tfrecords_valid, cnninf.BATCH_SIZE, False)
    test_samples, test_labels = cnn_data.read_and_decode(tfrecords_test, cnninf.BATCH_SIZE, False)
    
    epoch_train = cnn_data.TRAIN_SIZE *cnn_data.NUM_CLASSES // cnninf.BATCH_SIZE
    epoch_valid = cnn_data.VALID_SIZE *cnn_data.NUM_CLASSES // cnninf.BATCH_SIZE
    epoch_test = cnn_data.TEST_SIZE *cnn_data.NUM_CLASSES // cnninf.BATCH_SIZE    

    saver = tf.train.Saver(max_to_keep = 50)
    summary = tf.summary.merge_all()    
    init = tf.global_variables_initializer()

    train_accuracy = []
    test_accuracy = []
    train_loss = []
    test_loss = []
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord) 
        
        try:
            start_time = time.time()
            for i in xrange(FLAGS.max_steps):
                if coord.should_stop():
                    break
                
                sam_batch, label_batch = sess.run([train_samples, train_labels])
                sess.run(train_cnn,feed_dict={X:sam_batch,Y:label_batch})
                
#                if (i+1) % 500 == 0:  
#                    #########   Validation evaluation   ##########
#                    n_cost, n_acc, n_batch = 0, 0, 0 
#                    for _ in range(epoch_valid):
#                        sam_batch, label_batch = sess.run([valid_samples, valid_labels])
#                        err,acc = sess.run([loss_cnn, correct_cnn],feed_dict={X:sam_batch,Y:label_batch})
#                        n_cost += err
#                        n_acc += acc
#                        n_batch += 1
#                    print("Loss of validation dataset, step: %d, loss: %f, and accuracy:%f" % (i+1,(n_cost/n_batch),(n_acc/n_batch/cnninf.BATCH_SIZE)))
#                    
                if (i+1) % 2000 == 0: 
                    #########   Training evaluation   ##########
                    n_cost, n_acc, n_batch = 0, 0, 0 
                    for _ in range(epoch_train):
                        sam_batch, label_batch = sess.run([train_samples_f, train_labels_f])
                        err,acc = sess.run([loss_cnn, correct_cnn],feed_dict={X:sam_batch,Y:label_batch})
                        n_cost += err
                        n_acc += acc
                        n_batch += 1
                    print("Loss of training CNN, loss: %f, and accuracy:%f" % ((n_cost/n_batch),(n_acc/n_batch/cnninf.BATCH_SIZE)))
                    train_accuracy.append(n_acc/n_batch/cnninf.BATCH_SIZE)
                    train_loss.append(n_cost/n_batch)
                    write_op = sess.run(summary, {loss_val: n_cost/n_batch, acc_val:n_acc/n_batch/cnninf.BATCH_SIZE})  
                    writer_train.add_summary(write_op, i)  
                    writer_train.flush() 
                    #########   Testing evaluation   ##########
                    n_cost, n_acc, n_batch = 0, 0, 0
                    for j in range(epoch_test):
                        sam_batch, label_batch = sess.run([test_samples, test_labels])
                        logits, err, acc = sess.run([logits_cnn, loss_cnn, correct_cnn],feed_dict={X:sam_batch,Y:label_batch})
                        n_cost += err
                        n_acc += acc
                        n_batch += 1
                        #print("Loss of testing CNN of batch: %d, loss: %f, and accuracy:%f" % (j, err, acc/cnninf.BATCH_SIZE))
                    print("Loss of testing CNN, loss: %f, and accuracy:%f" % ((n_cost/n_batch),(n_acc/n_batch/cnninf.BATCH_SIZE)))
                    test_accuracy.append(n_acc/n_batch/cnninf.BATCH_SIZE)
                    test_loss.append(n_cost/n_batch)
                    write_op = sess.run(summary, {loss_val: n_cost/n_batch, acc_val:n_acc/n_batch/cnninf.BATCH_SIZE})  
                    writer_test.add_summary(write_op, i)  
                    writer_test.flush()                     

                    checkpoint_path = os.path.join(FLAGS.train_dir, 'chcom.ckpt')
                    saver.save(sess, checkpoint_path, global_step=i)
            duration=time.time()-start_time        
            sio.savemat('logfile/accuracy.mat', {'train_accuracy': train_accuracy,'test_accuracy': test_accuracy,
                                                 'train_loss': train_loss,'test_loss': test_loss,'duration':duration})
                   
                    #print("Loss of training CNN, step: %d, loss: %f, and accuracy:%f" % (i+1,err,ac))
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

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
The model of feature extraction
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import ae_data

import math
FEATURE_DIMENSION = 30
###  FEATURE_DIMENSION 的设置：5，10，15，20，25，30，40，50，60，70，80

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.衰减呈阶梯函数，控制衰减周期（阶梯宽度）
LEARNING_RATE_DECAY_FACTOR = 0.9  # Learning rate decay factor.学习率衰减因子
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.初始学习率


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
      
      
def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
    """
    dtype = tf.float32
    var = _variable_on_cpu(name,shape,tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def autoencoder(datasets,dimensions):
    # Define the archtecture of autoencoder for feature extraction
    current_input=datasets
    with tf.variable_scope('encoder_1') as scope:
        n_input = dimensions[0]
        n_output = dimensions[1]
        W = tf.get_variable(name="weights",shape=[n_input,n_output],initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        b = tf.get_variable(name="biases",shape=[n_output],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
        out1 = tf.nn.tanh(tf.matmul(current_input, W) + b)
    
    with tf.variable_scope('encoder_2') as scope:
        n_input = dimensions[1]
        n_output = dimensions[2]
        W = tf.get_variable(name="weights",shape=[n_input,n_output],initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        b = tf.get_variable(name="biases",shape=[n_output],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
        out2 = tf.nn.tanh(tf.matmul(out1, W) + b)
        
    with tf.variable_scope('encoder_3') as scope:
        n_input = dimensions[2]
        n_output = dimensions[3]
        W = tf.get_variable(name="weights",shape=[n_input,n_output],initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        b = tf.get_variable(name="biases",shape=[n_output],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
        out3 = tf.nn.tanh(tf.matmul(out2, W) + b)
        
    with tf.variable_scope('encoder_4') as scope:
        n_input = dimensions[3]
        n_output = dimensions[4]
        W = tf.get_variable(name="weights",shape=[n_input,n_output],initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        b = tf.get_variable(name="biases",shape=[n_output],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
        out4 = tf.nn.tanh(tf.matmul(out3, W) + b)
        
    with tf.variable_scope('decoder_4') as scope:
        n_input = dimensions[4]
        n_output = dimensions[3]
        W = tf.get_variable(name="weights",shape=[n_input,n_output],initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        b = tf.get_variable(name="biases",shape=[n_output],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
        out5 = tf.nn.tanh(tf.matmul(out4, W) + b)
        
    with tf.variable_scope('decoder_3') as scope:
        n_input = dimensions[3]
        n_output = dimensions[2]
        W = tf.get_variable(name="weights",shape=[n_input,n_output],initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        b = tf.get_variable(name="biases",shape=[n_output],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
        out6 = tf.nn.tanh(tf.matmul(out5, W) + b)
        
    with tf.variable_scope('decoder_2') as scope:
        n_input = dimensions[2]
        n_output = dimensions[1]
        W = tf.get_variable(name="weights",shape=[n_input,n_output],initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        b = tf.get_variable(name="biases",shape=[n_output],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
        out7 = tf.nn.tanh(tf.matmul(out6, W) + b)
        
    with tf.variable_scope('decoder_1') as scope:
        n_input = dimensions[1]
        n_output = dimensions[0]
        W = tf.get_variable(name="weights",shape=[n_input,n_output],initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        b = tf.get_variable(name="biases",shape=[n_output],initializer=tf.constant_initializer(value=0.0, dtype=tf.float32))
        out8 = tf.nn.tanh(tf.matmul(out7, W) + b)

    cost = tf.reduce_mean(tf.square(out8-datasets))
    
    return {'hidden':out4,'loss':cost}
'''    
def classifier(datasets,labels):
    # define the structure of the classifier
    model = SVC(decision_function_shape='ovo')
    classifier.fit(datasets,labels)
    
    return model
'''
def fully_inference(features):
    with tf.variable_scope('fully_connect') as scope:
        weights = _variable_with_weight_decay('weights',shape=[FEATURE_DIMENSION,ae_data.NUM_CLASSES],stddev=0.1,wd=0.0)
        biases  = _variable_on_cpu('biases',[ae_data.NUM_CLASSES],tf.constant_initializer(0.1))
        output = tf.add(tf.matmul(features,weights),biases,name=scope.name)
    return output

def fully_loss(logits,labels):
    """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch. #强制类型转换，使符合sparse_softmax_cross_entropy_with_logits输入参数格式要求
    labels = tf.cast(labels,tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
    return cross_entropy_mean


def fully_train(total_loss, global_step, decay_steps, batch_size):
    """Train cnn-based protocol identification model.
    
    Create an optimizer and apply to all trainable variables. Add moving  average for all trainable variables.
    Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps processed.
    Returns:
    train_op: op for training.
    """    
    #num_batches_per_epoch = num_train / batch_size#求训练块的个数
    #decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)#每经过decay_step步训练，衰减lr 
    # Decay the learning rate exponentially based on the number of steps.
    #decay_steps =1000
    #lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,global_step,decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
    lr = INITIAL_LEARNING_RATE
    #tf.scalar_summary('learning_rate', lr)
    # Generate moving averages of all losses and associated summaries.
    # loss_averages_op = _add_loss_summaries(total_loss)
    # Compute gradients.
    with tf.control_dependencies([total_loss]):  #tf.control_dependencies是一个context manager,控制节点执行顺序，先执行control_inputs中的操作，再执行context中的操作
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss) #返回计算出的（gradient, variable） pairs 

    # Apply gradients.#返回一步梯度更新操作
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    #for var in tf.trainable_variables():
        #tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    #for grad, var in grads:
        #if grad is not None:
            #tf.histogram_summary(var.op.name + '/gradients', grad)
        
    # Track the moving averages of all trainable variables. #num_updates参数用于动态调整衰减率，真实的decay_rate =min(decay, (1 + num_updates) / (10 + num_updates) 
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())# #返回模型参数变量的滑动更新操作,tf.trainable_variables() 方法返回所有`trainable=True`的变量，列表结构 
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')  #Does nothing. Only useful as a placeholder for control edges  

    return train_op


def fully_evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the range [0, NUM_CLASSES).
    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size) that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    labels = tf.cast(labels, dtype=tf.int32)
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))

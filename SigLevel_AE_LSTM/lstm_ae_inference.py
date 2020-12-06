#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
The model of feature extraction
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import lstm_data

import math
FEATURE_DIMENSION = 30
LSTM_LAYER = 2 # The number of lstm layer, 1,2,3
HIDDEN_UNITS = 20  # The number of output feature dimension, 10，15，20，25，30，35，40，50，60，70，80
N_STEPS = 1
BATCH_SIZE = 100

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.衰减呈阶梯函数，控制衰减周期（阶梯宽度）
LEARNING_RATE_DECAY_FACTOR = 0.9  # Learning rate decay factor.学习率衰减因子
INITIAL_LEARNING_RATE = 0.01       # Initial learning rate.初始学习率

def autoencoder(datasets,dimensions):
    # Define the archtecture of autoencoder for feature extraction
    current_input=datasets
    
    encoder = []
    for layer_i, n_output in enumerate(dimensions[1:]):
        #n_input = int(current_input.get_shape()[1])
        n_input = dimensions[layer_i]
        '''
        with tf.variable_scope('encoder_%d' % layer_i) as scope:
            W = tf.get_variable(name="weights",shape=[n_input,n_output],initializer=tf.truncated_normal_initializer(mean=-1.0 / math.sqrt(n_input), stddev=1.0 / math.sqrt(n_input), seed=None, dtype=tf.float32))
            b = tf.get_variable(name="biases",shape=[n_output],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))
        
        with tf.name_scope('encoder_%d' % layer_i) as scope:
            W = tf.truncated_normal(shape=[n_input,n_output],mean=-1.0 / math.sqrt(n_input), stddev=1.0 / math.sqrt(n_input))
            b = tf.constant(0.1,shape=[n_output])
            encoder.append(W)
            output = tf.nn.sigmoid(tf.matmul(current_input,W)+b)
        current_input = output
        '''
        W = tf.Variable(tf.random_uniform([n_input, n_output],-1.0 / math.sqrt(n_input),1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output
    z = current_input
    encoder.reverse()
    
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        '''
        with variable_scope('decoder_%d' % layer_i) as scope:
            W = tf.transpose(encoder[layer_i])
            b = tf.get_variable(name="biases",shape=[n_output],initializer=tf.constant_initializer(value=0.1, dtype=tf.float32))
        
        with tf.name_scope('decoder_%d' % layer_i) as scope:
            W = tf.transpose(encoder[layer_i])
            b = tf.constant(0.1,shape=[n_output])
            output = tf.nn.sigmoid(tf.matmul(current_input,W)+b)
        
        current_input = output
        '''
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output
    y = current_input
    cost = tf.reduce_mean(tf.square(y-datasets))
    
    return {'hidden':z,'loss':cost}


def loss(logits,labels):
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
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.int64),logits=logits,name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
    tf.add_to_collection('losses',cross_entropy_mean) #把张量cross_entropy_loss添加到字典集合中key='losses'的子集中

    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')  #返回字典集合中key='losses'的子集中元素之和


def evaluation(logits, labels):
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


def fully_inference(features):
    with tf.variable_scope('fully_connect') as scope:
        weights = _variable_with_weight_decay('weights',shape=[FEATURE_DIMENSION,lstm_data.NUM_CLASSES],stddev=0.1,wd=0.0)
        biases  = _variable_on_cpu('biases',[lstm_data.NUM_CLASSES],tf.constant_initializer(0.1))
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


def fully_train(total_loss, global_step, num_train, batch_size):
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
    decay_steps =1000
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,global_step,decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
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


def RNN_MULTILAYER(X, batch_size):
    # hidden layer for input
    X = tf.reshape(X, [-1, FEATURE_DIMENSION])
    with tf.variable_scope('RNN_in') as scope:
        weights = _variable_with_weight_decay('weights',shape=[FEATURE_DIMENSION,HIDDEN_UNITS],stddev=0.1,wd=0.0)
        biases  = _variable_on_cpu('biases', [HIDDEN_UNITS], tf.constant_initializer(0.1))
    X_in = tf.matmul(X, weights) + biases
    X_in = tf.reshape(X_in, [-1,N_STEPS, HIDDEN_UNITS])

    # cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_UNITS, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell for _ in range(LSTM_LAYER)])
    _init_state = lstm_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
    outputs,states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)
    
    #hidden layer for output as the final results
    #results = tf.matmul(states[1], weights['out']) + biases['out']
    # or
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    with tf.variable_scope('RNN_out') as scope:
        weights = _variable_with_weight_decay('weights',shape=[HIDDEN_UNITS,lstm_data.NUM_CLASSES],stddev=0.1,wd=0.0)
        biases  = _variable_on_cpu('biases',[lstm_data.NUM_CLASSES],tf.constant_initializer(0.1))
    results = tf.matmul(outputs[-1], weights) + biases

    return results
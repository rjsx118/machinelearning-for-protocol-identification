#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
The model of feature extraction, the model is based on CNN
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import lstm_data

import tensorflow as tf

LSTM_LAYER = 3 # The number of lstm layer, 1,2,3
HIDDEN_UNITS = 30  # The number of output feature dimension, 10，15，20，25，30，35，40，50，60，70，80
N_INPUTS = 2048 # The dimension of input data
N_STEPS = 1
BATCH_SIZE = 100

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


def _add_loss_summaries(total_loss):
    """Add summaries for losses in protocol identification model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
    """
    #通过使用指数衰减，来维护变量的滑动均值。当训练模型时，维护训练参数的滑动均值是有好处的。在测试过程中使用滑动参数比最终训练的参数值本身，  
    #会提高模型的实际性能（准确率）。apply()方法会添加trained variables的shadow copies，并添加操作来维护变量的滑动均值到shadow copies。average  
    #方法可以访问shadow variables，在创建evaluation model时非常有用。  
    #滑动均值是通过指数衰减计算得到的。shadow variable的初始化值和trained variables相同，其更新公式为  
    # shadow_variable = decay * shadow_variable + (1 - decay) * variable  

    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')#创建一个新的指数滑动均值对象 
    losses = tf.get_collection('losses')# 从字典集合中返回关键字'losses'对应的所有变量，包括交叉熵损失和正则项损失   # 创建‘shadow variables’,并添加维护滑动均值的操作  
    loss_averages_op = loss_averages.apply(losses + [total_loss])#维护变量的滑动均值，返回一个能够更新shadow variables的操作  

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    #for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
        #tf.scalar_summary(l.op.name +' (raw)', l)#保存变量到Summary缓存对象，以便写入到文件中  
        #tf.scalar_summary(l.op.name, loss_averages.average(l))#average() returns the shadow variable for a given variable.  

    return loss_averages_op  #返回损失变量的更新操作


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


def RNN(X, batch_size):
    # hidden layer for input
    X = tf.reshape(X, [-1, N_INPUTS])
    with tf.variable_scope('RNN_in') as scope:
        weights = _variable_with_weight_decay('weights',shape=[N_INPUTS,HIDDEN_UNITS],stddev=0.1,wd=0.0)
        biases  = _variable_on_cpu('biases', [HIDDEN_UNITS], tf.constant_initializer(0.1))
    X_in = tf.matmul(X, weights) + biases
    X_in = tf.reshape(X_in, [-1,N_STEPS, HIDDEN_UNITS])

    # cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_UNITS, forget_bias=1.0, state_is_tuple=True)
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

def RNN_MULTILAYER(X, batch_size):
    # hidden layer for input
    X = tf.reshape(X, [-1, N_INPUTS])
    with tf.variable_scope('RNN_in') as scope:
        weights = _variable_with_weight_decay('weights',shape=[N_INPUTS,HIDDEN_UNITS],stddev=0.1,wd=0.0)
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

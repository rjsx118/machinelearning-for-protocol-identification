#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
The model of feature extraction, the model is based on CNN
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cnn_data

import tensorflow as tf
#from six.moves import xrange

CONV_KERNEL_LENGTH = 4
POOL_KERNEL_LENGTH = 4
CONV1_KERNEL = 128
CONV2_KERNEL = 64
CONV3_KERNEL = 32
CONV4_KERNEL = 16
FC1_UNITS = 1024
FC2_UNITS = 25  #10, 20, 25, 28, 30, 31, 32, 35, 40, 50, 60, 65, 70, 75, 80
BATCH_SIZE = 100

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.衰减呈阶梯函数，控制衰减周期（阶梯宽度）
LEARNING_RATE_DECAY_FACTOR = 0.95  # Learning rate decay factor.学习率衰减因子
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


def inference(datasets, batch_size):
    # Define the archtecture of cnn for feature extraction

    data = tf.reshape(datasets,[-1,2,1024,1])
    
    with tf.variable_scope('conv1') as scope:
        weights = _variable_with_weight_decay('weights',shape=[2,CONV_KERNEL_LENGTH,1,CONV1_KERNEL],stddev=0.1,wd=0.0)
        biases  = _variable_on_cpu('biases',[CONV1_KERNEL],tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(data,weights,strides=[1,1,1,1],padding='SAME')
        bias = tf.nn.bias_add(conv,biases)
        conv1= tf.nn.relu(bias)
        variable_summaries(weights)
    
    pool1 = tf.nn.max_pool(conv1,ksize=[1,1,POOL_KERNEL_LENGTH,1],strides=[1,1,POOL_KERNEL_LENGTH,1],padding='SAME',name='pool1')
    
    with tf.variable_scope('conv2') as scope:
        weights = _variable_with_weight_decay('weights',shape=[1,CONV_KERNEL_LENGTH,CONV1_KERNEL,CONV2_KERNEL],stddev=0.1,wd=0.0)
        biases  = _variable_on_cpu('biases',[CONV2_KERNEL],tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool1,weights,strides=[1,1,1,1],padding='SAME')
        bias = tf.nn.bias_add(conv,biases)
        conv2= tf.nn.relu(bias)
        variable_summaries(weights)
    
    pool2 = tf.nn.max_pool(conv2,ksize=[1,1,POOL_KERNEL_LENGTH,1],strides=[1,1,POOL_KERNEL_LENGTH,1],padding='SAME',name='pool2')
    
    with tf.variable_scope('conv3') as scope:
        weights = _variable_with_weight_decay('weights',shape=[1,CONV_KERNEL_LENGTH,CONV2_KERNEL,CONV3_KERNEL],stddev=0.1,wd=0.0)
        biases  = _variable_on_cpu('biases',[CONV3_KERNEL],tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool2,weights,strides=[1,1,1,1],padding='SAME')
        bias = tf.nn.bias_add(conv,biases)
        conv3= tf.nn.relu(bias)
        variable_summaries(weights)
        
    pool3 = tf.nn.max_pool(conv3,ksize=[1,1,POOL_KERNEL_LENGTH,1],strides=[1,1,POOL_KERNEL_LENGTH,1],padding='SAME',name='pool2')

    with tf.variable_scope('conv4') as scope:
        weights = _variable_with_weight_decay('weights',shape=[1,CONV_KERNEL_LENGTH,CONV3_KERNEL,CONV4_KERNEL],stddev=0.1,wd=0.0)
        biases  = _variable_on_cpu('biases',[CONV4_KERNEL],tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool3,weights,strides=[1,1,1,1],padding='SAME')
        bias = tf.nn.bias_add(conv,biases)
        conv4= tf.nn.relu(bias)
        variable_summaries(weights)
        
    pool4 = tf.nn.max_pool(conv4,ksize=[1,1,POOL_KERNEL_LENGTH,1],strides=[1,1,POOL_KERNEL_LENGTH,1],padding='SAME',name='pool2')
    
    with tf.variable_scope('fc1') as scope:
        shape=pool4.get_shape().as_list()
        dim = shape[1]*shape[2]*shape[3]        
        reshape = tf.reshape(pool4,[-1,dim])
        weights = _variable_with_weight_decay('weights',shape=[dim,FC1_UNITS],stddev=0.1,wd=0.0)
        biases  = _variable_on_cpu('biases',[FC1_UNITS],tf.constant_initializer(0.1))
        local5 = tf.nn.relu(tf.matmul(reshape,weights)+biases,name=scope.name)
        variable_summaries(weights)
        
    with tf.variable_scope('fc2') as scope:
        weights = _variable_with_weight_decay('weights',shape=[1024,FC2_UNITS],stddev=0.1,wd=0.0)
        biases  = _variable_on_cpu('biases',[FC2_UNITS],tf.constant_initializer(0.1))
        local6 = tf.nn.relu(tf.matmul(local5,weights)+biases,name=scope.name)
        variable_summaries(weights)
        
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights',shape=[FC2_UNITS,cnn_data.NUM_CLASSES],stddev=0.1,wd=0.0)
        biases  = _variable_on_cpu('biases',[cnn_data.NUM_CLASSES],tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local6,weights),biases,name=scope.name)
        variable_summaries(weights)
        
    return local6, softmax_linear

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
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
    tf.add_to_collection('losses',cross_entropy_mean) #把张量cross_entropy_loss添加到字典集合中key='losses'的子集中

    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')  #返回字典集合中key='losses'的子集中元素之和
    
def train(total_loss, global_step, num_train, batch_size):
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
    decay_steps = 1000
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,global_step,decay_steps,LEARNING_RATE_DECAY_FACTOR,staircase=True)
    #tf.scalar_summary('learning_rate', lr)
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):  #tf.control_dependencies是一个context manager,控制节点执行顺序，先执行control_inputs中的操作，再执行context中的操作
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

def do_eval(sess, eval_correct, Samples_placeholder, Labels_placeholder, filename_queue, batch_size, total_size):
    """Runs one evaluation against the full epoch of data.
    Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    #true_count = np.zeros(NUM_CLASSES)
    steps_per_epoch = total_size // batch_size
    num_examples = steps_per_epoch * batch_size
    for _ in range(steps_per_epoch):
        x_batch,y_batch=cnn_data.read_and_decode(filename_queue, batch_size=batch_size, shuffle_batch=False)
        feed_dict = {Samples_placeholder:x_batch, Labels_placeholder:y_batch}
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
        
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))
    return precision
    

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
import tensorflow as tf
import pdb

def max_pool(x, ksize, stride, padding='VALID', 
		scope='max_pool'):

    in_shape = x.get_shape().as_list()
    # assume to be [batch, height, width, channels]
    assert len(in_shape)==4     # B,H,W,C
    padding = padding.upper()
    kshape = [1, ksize, ksize, 1]
    sshape = [1, stride, stride, 1]
    with tf.variable_scope(scope):
    	return tf.nn.max_pool(x, ksize=kshape, strides=sshape, 
		padding=padding, name='output')

def conv2d(x, out_channel, kernel_shape, 
	   padding='SAME', stride=1,
	   W_init=None, b_init=None,
	   nl=tf.identity, use_bias=True, 
	   scope='conv2d'):

    in_shape = x.get_shape().as_list()
    assert len(in_shape)==4	# B,H,W,C
    in_channel = in_shape[-1]

    filter_shape = [kernel_shape, kernel_shape] + [in_channel, out_channel]
    padding = padding.upper()
    stride = [1, stride, stride, 1]
    if W_init is None:
	W_init = tf.contrib.layers.variance_scaling_initializer()
    if b_init is None:
	b_init = tf.constant_initializer()
    with tf.variable_scope(scope):
    	# weight init
    	W = tf.get_variable('W', filter_shape, initializer=W_init)
    	if use_bias:
	    b = tf.get_variable('b', [out_channel], initializer=b_init)
    	# conv2d
    	conv = tf.nn.conv2d(x, W, stride, padding)
    	return nl(tf.nn.bias_add(conv, b) if use_bias else conv, name='output')

def fc(x, out_dim, 
       W_init=None, b_init=None,
       nl=tf.identity, use_bias=True, 
       scope='fc'):

    x = tf.contrib.layers.flatten(x)
    in_shape = x.get_shape().as_list()
    assert len(in_shape)==2
    in_dim = in_shape[1]
    if W_init is None:
	W_init = tf.contrib.layers.variance_scaling_initializer()
    if b_init is None:
	b_init = tf.constant_initializer()
    with tf.variable_scope(scope):
    	W = tf.get_variable('W', [in_dim, out_dim], initializer=W_init)
    	if use_bias:
	    b = tf.get_variable('b', [out_dim], initializer=b_init)
    	out = tf.matmul(x, W)+b
    	return nl(out, name='output')

    


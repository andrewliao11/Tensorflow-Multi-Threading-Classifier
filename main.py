import tensorflow as tf
import numpy as np
import os, argparse, pdb, multiprocessing
import nn, construct_binary, reader
from utils import *

class Model(object):

    def __init__(self, num_classes=None, scope='mnist'):
	assert num_classes!=None
	self.num_classes = num_classes
	self.scope = scope

    def _get_logits(self, x):
	x = tf.to_float(x)/255.
	x = tf.reshape(x, [-1, 28, 28, 1])
	conv1 = nn.max_pool(nn.conv2d(x, 32, 5, nl=tf.nn.relu, scope='conv1'), 
			2, 2, scope='mp1')
	conv2 = nn.max_pool(nn.conv2d(conv1, 64, 5, nl=tf.nn.relu, scope='conv2'), 
			2, 2, scope='mp2')
	fc1 = nn.fc(conv2, 100, nl=tf.nn.relu, scope='fc1')
	prob = nn.fc(fc1, self.num_classes, scope='fc2')
	return prob

    def _build_graph(self, x, y):
	with tf.variable_scope(self.scope):
	    logits = self._get_logits(x)
	    y_onehot = tf.one_hot(y, self.num_classes, 1.0, 0.0)
	    self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_onehot))
	    prediction = tf.argmax(logits, 1)
	    self.accuracy = tf.reduce_mean(tf.to_float(tf.equal(prediction, y)))
	    tf.summary.scalar("cost", self.cost)
	    tf.summary.scalar("accuracy", self.accuracy)

    def _get_all_params(self):
	params = []
	for param in tf.global_variables():
	    if self.scope in param.name:
		params.append(param)
	return params

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-thread mnist classifier')
    parser.add_argument('--lr', type=float, default=3e-4,
                    help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    bin_filepath = 'mnist.tfrecords'
    if not os.path.exists(bin_filepath):
    	construct_binary.mnist(bin_filepath)
    # convert the string into tensor (represent a single tensor)
    single_label, single_image = reader.read_to_tensor(bin_filepath) 
    images_batch, labels_batch = tf.train.shuffle_batch(
    	[single_image, single_label], batch_size=args.batch_size,
   	capacity=2000,
    	min_after_dequeue=1000
    )

    # build graph
    M = Model(num_classes=10)
    M._build_graph(images_batch, labels_batch)
    global_step = tf.get_variable('global_step', [], 
			initializer=tf.constant_initializer(0), trainable=False)
    train_op = tf.train.AdamOptimizer(args.lr).minimize(M.cost, global_step=global_step)
    if not os.path.exists(bin_filepath):
	os.makdir('./logs')
    summary_writer = tf.summary.FileWriter('./logs')
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(max_to_keep=20)

    config = get_session_config(0.3, multiprocessing.cpu_count()/2)
    sess = tf.Session(config=config)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # creates threads to start all queue runners collected in the graph
    # [remember] always call init_op before start the runner
    tf.train.start_queue_runners(sess=sess)
    step = 0
    while True:
  	_, summary_str, loss, accuracy = sess.run([train_op, summary_op, M.cost, M.accuracy])
	summary_writer.add_summary(summary_str, step)
	if step%100 == 0:
	    if not os.path.exists('./checkpoints'):
		os.makdir('./checkpoints')
	    saver.save(sess, os.path.join('./checkpoints', 'mnist'), global_step=global_step)
	    print "==================================="
	    print "[#] Iter", step
	    print "[L] Loss =", loss
	    print "[A] Accuracy =", accuracy
	step += 1

import tensorflow as tf
import numpy as np
import os, argparse, pdb
import nn, construct_binary

class Model(object):

    def __init__(self, args):
	self.num_outputs = args.classes

    def _get_logits(self, x):
	fc1 = nn.fc(x, 100, nl=tf.nn.relu, scope='fc1')
	prob = nn.fc(fc1, self.num_outputs, scope='fc2')
	return tf.log(prob+1e-8)

    def _build_graph(self, x, y):
	logits = self._get_logits(x)
	y_onehot = tf.one_hot(y, self.num_outputs, 1.0, 0.0)
	self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_onehot))
	tf.summary.scalar("cost_cls", self.cost)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-thread mnist classifier')
    parser.add_argument('--lr', type=float, default=3e-4,
                    help='initial learning rate')
    parser.add_argument('--ds_dir', type=str, default='./data/mnist',
                    help='specify where the dataset is')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--classes', type=int, default=10)
    args = parser.parse_args()

    M = Model(args)
    bin_filepath = construct_binary.mnist()
    pdb.set_trace()


    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
    	while not coord.should_stop():
            # Run training steps or whatever
            sess.run(train_op)
    except tf.errors.OutOfRangeError:
    	print('Done training -- epoch limit reached')
    finally:
    	# When done, ask the threads to stop.
    	coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()




import tensorflow as tf
import logging


class ImageClassifier():

    def __init__(self, train_b_2_r, test_b_2_r):
        self.train_b_2_r = train_b_2_r
        self.test_b_2_r = test_b_2_r
        (b, r) = train_b_2_r[0]
        self.width = b.n
        self.height = b.m

    def neural_net(self):  # from "Machine Learning with Tensorflow", chapter 9
        W = tf.Variable(tf.random_normal([5, 5, 1, 32]))
        b = tf.Variable(tf.random_normal([32]))
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        conv_with_b = tf.nn.bias_add(conv, b)
        conv_out = tf.nn.relu(conv_with_b)

    def train(self, n_training_epochs):
        logging.info('About to start training with {} epochs'.format(n_training_epochs))
        with tf.compat.v1.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(n_training_epochs):
                logging.debug('epoch {}'.format(epoch))
                for bitmap, result in self.train_b_2_r:
                    pass

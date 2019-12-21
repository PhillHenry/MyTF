import tensorflow as tf
import logging
import numpy as np


class ImageClassifier():

    def __init__(self, train_b_2_r, test_b_2_r):
        self.train_b_2_r = train_b_2_r
        self.test_b_2_r = test_b_2_r
        (b, r) = train_b_2_r[0]
        self.width = b.n
        self.height = b.m

    def conv_layer(self, x, W, b):
        conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')
        conv_with_b = tf.nn.bias_add(conv, b)
        conv_out = tf.nn.relu(conv_with_b)
        return conv_out

    def maxpool_layer(self, conv, k=2):
        return tf.nn.max_pool(conv, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def model(self, x):
        n_cats = 2

        # Applies 64 convolutions
        # of window size 5 × 5
        W1 = tf.Variable(tf.random_normal([5, 5, 1, 64]))
        b1 = tf.Variable(tf.random_normal([64]))

        # Applies 64 more convolutions
        # of window size 5 × 5
        W2 = tf.Variable(tf.random_normal([5, 5, 64, 64]))
        b2 = tf.Variable(tf.random_normal([64]))

        # Introduces a fully
        # connected layer
        W3 = tf.Variable(tf.random_normal([5*21*64, 1024]))
        b3 = tf.Variable(tf.random_normal([1024]))

        # Defines the variables
        # for a fully connected
        # linear layer
        W_out = tf.Variable(tf.random_normal([1024, n_cats]))
        b_out = tf.Variable(tf.random_normal([n_cats]))

        x_reshaped = tf.reshape(x, shape=[-1, self.width, self.height, 1])
        print('x_reshaped = {}'.format(np.shape(x_reshaped)))

        conv_out1 = self.conv_layer(x_reshaped, W1, b1)
        maxpool_out1 = self.maxpool_layer(conv_out1)
        norm1 = tf.nn.lrn(maxpool_out1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        conv_out2 = self.conv_layer(norm1, W2, b2)
        norm2 = tf.nn.lrn(conv_out2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        maxpool_out2 = self.maxpool_layer(norm2)

        maxpool_reshaped = tf.reshape(maxpool_out2, [-1, W3.get_shape().as_list()[0]])
        local = tf.add(tf.matmul(maxpool_reshaped, W3), b3)
        local_out = tf.nn.relu(local)
        out = tf.add(tf.matmul(local_out, W_out), b_out)

        print('out = {}'.format(np.shape(out)))

        return out

    def one_hot_encode(self, b_2_r, n_cats):
        xs = map(lambda b2r: b2r[1], b_2_r)
        labels = np.asarray(list(xs))
        return tf.one_hot(labels, n_cats, on_value=1., off_value=0., axis=-1)

    def as_numpy_array(self, b_2_r):
        xs = map(lambda b2r: b2r[0].matrix, b_2_r)
        return np.asarray(list(xs))

    def flatten(self, xs):
        xs_reshaped = map(lambda x: np.reshape(x, self.width * self.height), xs)
        return np.asarray(list(xs_reshaped))

    def matrix_2d_to_1d(self, m):
        shape = np.shape(m)
        return np.reshape(m, shape[0] * shape[1])

    def stack_data(self, matrices):
        flattened = []
        for matrix in matrices:
            flattened.append(self.matrix_2d_to_1d(matrix))
        return np.asarray(flattened)

    def structure_data(self):
        samples_width_height = self.as_numpy_array(self.train_b_2_r)
        return self.stack_data(samples_width_height)

    def train(self, n_training_epochs):  # from "Machine Learning with Tensorflow", chapter 9
        n_cats = 2
        x = tf.placeholder(dtype=tf.float32, shape=[None, self.width * self.height], name="x")
        y = tf.placeholder(tf.float32, [None, n_cats])

        data = self.structure_data()

        model_op = self.model(x)
        print('model_op = {}'.format(np.shape(model_op)))
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=model_op, labels=y)
        )
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
        correct_pred = tf.equal(tf.argmax(model_op, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        logging.info('About to start training with {} epochs'.format(n_training_epochs))

        with tf.compat.v1.Session() as sess:
            sess.run(tf.global_variables_initializer())
            onehot_labels = self.one_hot_encode(self.train_b_2_r, n_cats)
            onehot_vals = sess.run(onehot_labels)
            batch_size = len(data) // 100
            print('batch size', batch_size)
            for j in range(0, n_training_epochs):
                print('EPOCH', j)
                for i in range(0, len(data), batch_size):
                    batch_data = data[i:i+batch_size]
                    batch_onehot_vals = onehot_vals[i:i+batch_size]
                    print('x shape {}, y shape{}'.format(np.shape(batch_data), np.shape(batch_onehot_vals)))
                    _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: batch_data, y:
                        batch_onehot_vals})
                    if i % 1000 == 0:
                        print(i, accuracy_val)
                    print('DONE WITH EPOCH')


if __name__ == "__main__":
    from data import training
    from tf import anns
    n_samples = 100

    i = 40
    j = 100

    n_pts = int((i * j) / 20)
    class_ratio = int(n_samples / 20)

    test_data = training.create_data(n_samples, j, i, class_ratio, n_pts, n_pts * 0.0001)
    training_data = training.create_data(n_samples, j, i, class_ratio, n_pts, n_pts * 0.0001)

    classifier = anns.ImageClassifier(training_data, test_data)
    classifier.train(100)
import tensorflow as tf
import logging


class ImageClassifier():

    def __init__(self, train_b_2_r, test_b_2_r):
        self.train_b_2_r = train_b_2_r
        self.test_b_2_r = test_b_2_r
        (b, r) = train_b_2_r[0]
        self.width = b.n
        self.height = b.m

    def conv_layer(self, x, W, b):
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        conv_with_b = tf.nn.bias_add(conv, b)
        conv_out = tf.nn.relu(conv_with_b)
        return conv_out

    def maxpool_layer(self, conv, k=2):
        return tf.nn.max_pool(conv, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def model(self, x):
        n_cats = 2

        W1 = tf.Variable(tf.random_normal([5, 5, 1, 64]))
        b1 = tf.Variable(tf.random_normal([64]))

        W2 = tf.Variable(tf.random_normal([5, 5, 64, 64]))
        b2 = tf.Variable(tf.random_normal([64]))

        W3 = tf.Variable(tf.random_normal([6*6*64, 1024]))
        b3 = tf.Variable(tf.random_normal([1024]))

        W_out = tf.Variable(tf.random_normal([1024, n_cats]))
        b_out = tf.Variable(tf.random_normal([n_cats]))

        x_reshaped = tf.reshape(x, shape=[-1, 24, 24, 1])

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
        return out

    def train(self, n_training_epochs):  # from "Machine Learning with Tensorflow", chapter 9
        n_cats = 2
        x = tf.placeholder(dtype=tf.float32, shape=[None, self.width * self.height], name="x")
        y = tf.placeholder(tf.float32, [None, n_cats])
        labels = map(lambda b2r: b2r[1], self.train_b_2_r)
        data = self.train_b_2_r

        model_op = self.model(x)
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=model_op, labels=y)
        )
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
        correct_pred = tf.equal(tf.argmax(model_op, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        logging.info('About to start training with {} epochs'.format(n_training_epochs))

        with tf.compat.v1.Session() as sess:
            sess.run(tf.global_variables_initializer())
            onehot_labels = tf.one_hot(labels, n_cats, on_value=1., off_value=0., axis=-1)
            onehot_vals = sess.run(onehot_labels)
            batch_size = len(data) // 200
            print('batch size', batch_size)
            for j in range(0, n_training_epochs):
                print('EPOCH', j)
                for i in range(0, len(data), batch_size):
                    batch_data = data[i:i+batch_size, :]
                    batch_onehot_vals = onehot_vals[i:i+batch_size, :]
                    _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: batch_data, y:
                        batch_onehot_vals})
                    if i % 1000 == 0:
                        print(i, accuracy_val)
                    print('DONE WITH EPOCH')

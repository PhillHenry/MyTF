import tensorflow as tf


class ImageClassifier():

    def __init__(self, bitmaps_to_results, bitmaps):
        self.bitmaps_to_results = bitmaps_to_results
        self.bitmaps = bitmaps

    def train(self, bitmap_to_results, n_training_epochs):
        with tf.session as sess:
            for epoch in range(n_training_epochs):
                for bitmap, result in bitmap_to_results:
                    pass

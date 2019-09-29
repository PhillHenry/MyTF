import numpy as np
from tests.tf import image_factory
import tensorflow as tf

factory = image_factory.ImageClassifierFactory()


def test_one_hot_encoding():
    encoded = factory.create_image_classifier().one_hot_encode(factory.training_data, 2)
    shape = tf.shape(encoded)
    assert factory.n_samples == shape


def test_bitmaps_as_numpy_arrays():
    as_array = factory.create_image_classifier().as_numpy_array(factory.training_data)
    assert len(factory.training_data) == np.shape(as_array)[0]



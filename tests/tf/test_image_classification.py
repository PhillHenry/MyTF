from data import training
from tf import anns
import numpy as np

n_samples = 10

i = 42
j = 101

n_pts = int((i * j) / 20)
class_ratio = int(n_samples / 20)

test_data = training.create_data(n_samples, j, i, class_ratio, n_pts, n_pts * 0.0001)
training_data = training.create_data(n_samples, j, i, class_ratio, n_pts, n_pts * 0.0001)

classifier = anns.ImageClassifier(training_data, test_data)


def test_one_hot_encoding():
    encoded = classifier.one_hot_encode(training_data, 2)
    assert n_samples == np.shape(encoded)[0]


def test_bitmaps_as_numpy_arrays():
    as_array = classifier.as_numpy_array(training_data)
    assert len(training_data) == np.shape(as_array)[0]



import my_keras.input_manipulation as to_test
import numpy as np


def test_rotate_array():
    xs = [1, 2, 3, 4, 5]
    assert np.array_equal(to_test.rotate(xs, 2), np.asarray([3, 4, 5, 1, 2]))

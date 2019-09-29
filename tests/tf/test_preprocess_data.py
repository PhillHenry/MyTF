from tests.tf import image_factory
import numpy as np

factory = image_factory.ImageClassifierFactory()


def test_2d_matrix_to_1d():
    matrix = factory.training_data[0][0].matrix
    xs = factory.create_image_classifier().matrix_2d_to_1d(matrix)
    shape = np.shape(matrix)
    assert xs.shape[0] == (shape[0] * shape[1])


def test_as_numpy_array():
    xs = factory.create_image_classifier().as_numpy_array(factory.training_data)
    assert np.shape(xs) == (factory.n_samples, factory.i, factory.j)


def test_stack_data():
    xs = factory.create_image_classifier().structure_data()
    assert np.shape(xs) == (factory.n_samples, factory.i * factory.j)
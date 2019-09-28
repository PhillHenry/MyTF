from tests.tf import image_factory
import numpy as np

factory = image_factory.ImageClassifierFactory()


def test_as_numpy_array():
    xs = factory.create_image_classifier().as_numpy_array(factory.training_data)
    assert np.shape(xs) == [factory.n_samples, factory.i, factory.j]


def test_stack_data():
    xs = factory.create_image_classifier().structure_data()
    assert np.shape(xs) == [factory.n_samples, factory.i * factory.j]
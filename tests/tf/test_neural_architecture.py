import unittest
from tests.tf import image_factory

factory = image_factory.ImageClassifierFactory()


class MyTestCase(unittest.TestCase):
    def test_neural_net_size(self):
        ann = factory.create_image_classifier()
        dim = ann.size_after(2)
        self.assertEqual(dim, 3 * 7)


if __name__ == '__main__':
    unittest.main()

from data import training
from tf import anns


class ImageClassifierFactory:
    n_samples = 10

    i = 42
    j = 101

    n_pts = int((i * j) / 20)
    class_ratio = int(n_samples / 20)

    test_data = training.create_data(n_samples, j, i, class_ratio, n_pts, n_pts * 0.0001)
    training_data = training.create_data(n_samples, j, i, class_ratio, n_pts, n_pts * 0.0001)

    def create_image_classifier(self):
        classifier = anns.ImageClassifier(self.training_data, self.test_data)
        return classifier


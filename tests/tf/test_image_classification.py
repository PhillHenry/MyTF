from data import training
from tf import anns


def test_expect_better_than_guessing():
    n_samples = 100

    i = 42
    j = 101

    n_pts = int((i * j) / 20)
    class_ratio = int(n_samples / 20)

    test_data = training.create_data(n_samples, j, i, class_ratio, n_pts, n_pts * 0.0001)
    training_data = training.create_data(n_samples, j, i, class_ratio, n_pts, n_pts * 0.0001)

    classifier = anns.ImageClassifier(training_data, test_data)
    classifier.train(100)



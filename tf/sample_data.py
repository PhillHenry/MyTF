from data import training


class Sample:

    def __init__(self, i, j, n_samples):
        self.n_pts = int((i * j) / 20)
        self.class_ratio = int(n_samples / 20)
        self.test_data = training.create_data(n_samples, j, i, self.class_ratio, self.n_pts, self.n_pts * 0.0001)
        self.training_data = training.create_data(n_samples, j, i, self.class_ratio, self.n_pts, self.n_pts * 0.0001)


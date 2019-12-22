from data import training
import matplotlib.pyplot as plt


class Sample:

    def __init__(self, i, j, n_samples):
        self.n_pts = int((i * j) / 20)
        self.class_ratio = 20 / n_samples
        self.test_data = training.create_data(n_samples, j, i, self.class_ratio, self.n_pts, self.n_pts * 0.0001)
        self.training_data = training.create_data(n_samples, j, i, self.class_ratio, self.n_pts, self.n_pts * 0.0001)
        print('Number of positive training data = {}'.format(len(Sample.matrices_of(self.training_data, 1))))
        print('Number of negative training data = {}'.format(len(Sample.matrices_of(self.training_data, 0))))

    @staticmethod
    def matrices_of(b2cs, c):
        bs = filter(lambda b2c: b2c[1] == c, b2cs)
        matrices = map(lambda b2c: b2c[0].matrix, bs)
        return list(matrices)


if __name__ == '__main__':
    samples = Sample(40, 100, 100)
    fig = plt.figure(0)

    nOfEach = 4
    n_rows = 2

    positive_matrices = Sample.matrices_of(samples.training_data, 1)
    for i in range(nOfEach):
        fig.add_subplot(n_rows, nOfEach, i + 1)
        plt.imshow(positive_matrices[i])

    negative_matrices = Sample.matrices_of(samples.training_data, 0)
    for i in range(nOfEach):
        fig.add_subplot(n_rows, nOfEach, i + 1 + nOfEach)
        plt.imshow(negative_matrices[i])

    plt.show()

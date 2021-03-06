from data import training
import matplotlib.pyplot as plt


class Sample:

    def __init__(self, i, j, n_samples):
        self.n_pts = int((i * j) / 20)
        self.class_ratio = 20 / n_samples
        self.pts_ration = self.n_pts * 0.001
        self.test_data = training.create_data(n_samples, j, i, self.class_ratio, self.n_pts, self.pts_ration)
        self.training_data = training.create_data(n_samples, j, i, self.class_ratio, self.n_pts, self.pts_ration)
        print('Number of positive training data = {}'.format(len(Sample.matrices_of(self.training_data, 1))))
        print('Number of negative training data = {}'.format(len(Sample.matrices_of(self.training_data, 0))))

    @staticmethod
    def matrices_of(b2cs, c):
        bs = filter(lambda b2c: b2c[1] == c, b2cs)
        matrices = map(lambda b2c: b2c[0].matrix, bs)
        return list(matrices)


def do_plot(fig, offset, n_rows, matrics):
    nOfEach = len(matrics)
    for i in range(nOfEach):
        ax = fig.add_subplot(n_rows, nOfEach, i + 1 + offset)
        do_axes(ax)
        plt.imshow(matrics[i])


def do_axes(ax):
    ax.set_yticklabels([])  # https://stackoverflow.com/questions/37039685/hide-axis-values-in-matplotlib
    ax.set_xticklabels([])
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('none')
    ax.set_axis_off()


if __name__ == '__main__':
    samples = Sample(40, 100, 100)
    nOfEach = 3
    n_rows = 2
    fig, (ax1, ax2) = plt.subplots(n_rows, 1)
    plt.title("Simulated network connections (machines vs ports)", pad=-60.0)
    ax1.set_title('Port scanned', pad=-60.0)
    do_axes(ax1)
    ax2.set_title('Normal activity', pad=-60.0)
    do_axes(ax2)

    positive_matrices = Sample.matrices_of(samples.training_data, 1)
    do_plot(fig, 0, n_rows, positive_matrices[0:nOfEach])

    negative_matrices = Sample.matrices_of(samples.training_data, 0)
    do_plot(fig, nOfEach, n_rows, negative_matrices[0:nOfEach])

    plt.show()
    fig.savefig('/tmp/test.png')

import numpy as np


class RandomBitmap:

    def __init__(self, n, m):
        self.matrix = np.zeros([m, n])
        self.n = n
        self.m = m

    def add_random_points(self, npts):
        if npts > (self.n * self.m):
            raise Exception('n = {}, m = {}, npts = {}'.format(self.n, self.m, npts))
        xs = np.random.randint(0, self.m, npts)
        ys = np.random.randint(0, self.n, npts)
        coords = np.vstack([xs,ys]).T  # https://stackoverflow.com/questions/26193386/numpy-zip-function
        for (x, y) in coords:
            self.matrix[x][y] = 1
        total = self.num_points()
        if total < npts:
            self.add_random_points(npts - total)
        return self.matrix

    def num_points(self):
        total = 0
        for row in self.matrix:
            total += len(list(filter(lambda x: x == 1, row)))
        return total
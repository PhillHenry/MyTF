import numpy as np
import random


class RandomBitmap:

    def __init__(self, n, m):
        self.matrix = np.zeros([m, n])
        self.n = n
        self.m = m

    def add_line(self, n_pts):
        if n_pts > self.n or n_pts > self.m:
            raise Exception('A straight line of size {} will not fit into a matrix of {} x {}'
                            .format(n_pts, self.n, self.m))
        rand_x = random.randint(0, self.n - 1)
        for y in range(n_pts):
            self.matrix[y][rand_x] = 1
        return self.matrix

    def add_random_points(self, npts):
        if npts > (self.n * self.m):
            raise Exception('n = {}, m = {}, npts = {}'.format(self.n, self.m, npts))
        self._add_random_points(npts)
        total = self.num_points()
        while total < npts:
            self.add_random_points(npts - total)
            total = self.num_points()
        return self.matrix

    def _add_random_points(self, npts):
        xs = np.random.randint(0, self.m, npts)
        ys = np.random.randint(0, self.n, npts)
        coords = np.vstack([xs, ys]).T  # https://stackoverflow.com/questions/26193386/numpy-zip-function
        for (x, y) in coords:
            self.matrix[x][y] = 1

    def num_points(self):
        total = 0
        for row in self.matrix:
            total += len(list(filter(lambda x: x == 1, row)))
        return total
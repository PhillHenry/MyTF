import numpy as np


class RandomBitmap:

    def __init__(self, n, m):
        self.matrix = np.zeros([m, n])
        self.n = n
        self.m = m

    def add_random_points(self, npts):
        xs = np.random.randint(0, self.m, npts)
        ys = np.random.randint(0, self.n, npts)
        coords = np.vstack([xs,ys]).T
        for (x, y) in coords:
            self.matrix[x][y] = 1
        return self.matrix


from data import bitmaps
from random import shuffle


def create_data(n, width, height, class_ratio, npts, pts_ratio):
    bitmap_to_class = []
    n_pos = int(class_ratio * n)
    n_neg = n - n_pos

    for i in range(n_pos):
        b = bitmaps.RandomBitmap(width, height)
        n_non_random = int(npts * pts_ratio)
        b.add_line(n_non_random)
        b.add_random_points(npts - n_non_random)
        bitmap_to_class.append((b, 1))

    for i in range(n_neg):
        b = bitmaps.RandomBitmap(width, height)
        b.add_random_points(npts)
        bitmap_to_class.append((b, 0))

    shuffle(bitmap_to_class)
    return bitmap_to_class



from data import bitmaps
from random import shuffle


def create_data(n_samples, width, height, class_ratio, n_pts, pts_ratio):
    bitmap_to_class = []
    n_pos = int(class_ratio * n_samples)
    n_neg = n_samples - n_pos

    for i in range(n_pos):
        b = bitmaps.RandomBitmap(width, height)
        n_non_random = int(n_pts * pts_ratio)
        b.add_line(n_non_random)
        b.add_random_points(n_pts - n_non_random)
        bitmap_to_class.append((b, 1))

    for i in range(n_neg):
        b = bitmaps.RandomBitmap(width, height)
        b.add_random_points(n_pts)
        bitmap_to_class.append((b, 0))

    shuffle(bitmap_to_class)
    return bitmap_to_class



from data import bitmaps
from random import shuffle


def create(n, width, height, ratio):
    bitmap_to_class = []
    n_x = int(ratio * n)
    n_y = n - n_x
    for i in range(n_x):
        bitmap_to_class.append((bitmaps.RandomBitmap(width, height), 1))
    for i in range(n_y):
        bitmap_to_class.append((bitmaps.RandomBitmap(width, height), 0))
    shuffle(bitmap_to_class)
    return bitmap_to_class



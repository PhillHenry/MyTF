from data import bitmaps
import numpy as np

def test_bitmap_contains_expected_num_of_points():
    n_pts = 1000
    bitmap = bitmaps.RandomBitmap(42, 99)
    matrix = bitmap.add_random_points(n_pts)
    total = 0
    for row in matrix:
        total += len(list(filter(lambda x: x == 1, row)))
    assert len == total


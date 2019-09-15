from data import bitmaps


def test_bitmap_contains_expected_num_of_points():
    bitmap = bitmaps.RandomBitmap(100, 100, 1000)
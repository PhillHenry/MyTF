from data import bitmaps


def test_bitmap_contains_expected_num_of_points():
    bitmap = bitmaps.RandomBitmap(9, 99)
    bitmap.add_random_points(1000)
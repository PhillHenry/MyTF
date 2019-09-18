from data import bitmaps


def test_bitmap_contains_expected_num_of_points():
    i = 42
    j = 99
    npts = int((i * j) / 5)
    bitmap = bitmaps.RandomBitmap(i, j)
    matrix = bitmap.add_random_points(npts)
    assert bitmap.num_points() == npts



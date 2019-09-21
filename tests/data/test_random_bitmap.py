from data import bitmaps
import pytest


i = 42
j = 99


def test_bitmap_contains_expected_num_of_points():
    npts = int((i * j) / 5)
    bitmap = bitmaps.RandomBitmap(i, j)
    bitmap.add_random_points(npts)
    assert bitmap.num_points() == npts


def test_add_line_has_expected_num_of_points():
    bitmap = bitmaps.RandomBitmap(i, j)
    n_pts = int(i/5)
    bitmap.add_line(n_pts)
    assert bitmap.num_points() == n_pts


def test_line_can_fit_in_rectangle():
    bitmap = bitmaps.RandomBitmap(i, j)
    with pytest.raises(Exception):
        bitmap.add_line(i + 1)
    with pytest.raises(Exception):
        bitmap.add_line(j + 1)





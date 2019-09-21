from data import training


def test_create_training_data():
    w = 13
    h = 23
    n = 101
    n_pts = int(w * h / 10)
    ratio = 0.1
    b2cs = training.create_data(n, w, h, ratio, n_pts, 0.1)
    assert len(b2cs) == n
    assert len(list(filter(lambda b2c: b2c[1] == 1, b2cs))) == int(n * ratio)
    for (b, c) in b2cs:
        assert b.num_points() > 0

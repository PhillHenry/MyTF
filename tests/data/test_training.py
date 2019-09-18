from data import training


def test_create_training_data():
    w = 13
    h = 23
    n = 101
    ratio = 0.1
    b2cs = training.create(n, w, h, ratio)
    assert len(b2cs) == n
    assert len(list(filter(lambda b2c: b2c[1] == 1, b2cs))) == int(n * ratio)

import numpy as np

def rotate(xs, n):
    length = len(xs)
    n = n % length
    head = xs[n:length]
    tail = xs[0:n]
    return np.append(head, tail, axis=0)



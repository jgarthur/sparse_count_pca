"""Independent dense oracle for current count-shifted PFlog."""

import numpy as np


def count_shifted_clr(mtx, count_shift):
    values = np.asarray(mtx.toarray(), dtype=np.float64)
    logged = np.log(values + count_shift)
    return logged - logged.mean(axis=1, keepdims=True)


def pflog(mtx, alpha):
    values = np.asarray(mtx.toarray(), dtype=np.float64)
    logged = np.log1p(4.0 * alpha * values)
    return logged - logged.mean(axis=1, keepdims=True)

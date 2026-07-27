"""Independent dense oracle for current count-scale shifted PFlog."""

import numpy as np


def count_shifted_clr(mtx, count_shift):
    values = mtx.toarray() if hasattr(mtx, "toarray") else mtx
    values = np.asarray(values, dtype=np.float64)
    logged = np.log(values + count_shift)
    return logged - logged.mean(axis=1, keepdims=True)


def pflog(mtx, alpha):
    values = mtx.toarray() if hasattr(mtx, "toarray") else mtx
    values = np.asarray(values, dtype=np.float64)
    logged = np.log1p(4.0 * alpha * values)
    return logged - logged.mean(axis=1, keepdims=True)

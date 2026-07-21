"""Independent dense oracle for historical proportion-shifted CLR."""

import numpy as np


def proportion_shifted_clr(mtx, composition_shift):
    values = mtx.toarray() if hasattr(mtx, "toarray") else mtx
    values = np.asarray(values, dtype=np.float64)
    proportions = values / values.sum(axis=1, keepdims=True)
    logged = np.log(proportions + composition_shift)
    return logged - logged.mean(axis=1, keepdims=True)

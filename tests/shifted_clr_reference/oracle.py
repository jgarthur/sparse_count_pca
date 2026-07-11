"""Pinned shifted CLR oracle copied from the BHGP_2022 reference repository."""

import numpy as np
import scipy as sp


# Copied from scripts/norm_sparse.py at the revision documented in README.md.
def do_pf(mtx, sf=None):
    pf = np.asarray(mtx.sum(axis=1)).ravel()
    if not sf:
        sf = pf.mean()
    pf = sp.sparse.diags(sf / pf) @ mtx
    return pf


def norm_clr(mtx, c=1.0):
    # u = x / s (each row sums to 1.0).
    u = do_pf(mtx, sf=1.0)
    D = mtx.shape[1]
    log_c = np.log(c)

    log_term = u.copy()
    if c == 1.0:
        log_term.data = np.log1p(log_term.data)
    else:
        log_term.data = np.log(log_term.data + c) - log_c

    row_sum = np.asarray(log_term.sum(axis=1)).ravel()
    per_cell_mean = row_sum / D + log_c
    dense_log = log_term.toarray() + log_c
    return dense_log - per_cell_mean[:, None]

"""Regression test against pinned Townes residual-PCA singular values."""

import numpy as np
import pytest

from sparse_count_pca import residual_pca_matrix

EXPECTED_SINGULAR_VALUES = {
    ("poisson", "pearson"): [
        4.26640006240745,
        2.87725497981510,
        1.31325556575074,
    ],
    ("poisson", "deviance"): [
        4.65082547716202,
        3.13188657560929,
        1.20899966390989,
    ],
    ("binomial", "pearson"): [
        4.93440039372825,
        3.31080439192454,
        1.51912783790216,
    ],
    ("binomial", "deviance"): [
        5.33828565260273,
        3.58431544182494,
        1.41440244585374,
    ],
}


@pytest.mark.parametrize(
    ("model", "residual", "expected"),
    [
        (model, residual, expected)
        for (model, residual), expected in EXPECTED_SINGULAR_VALUES.items()
    ],
)
def test_singular_values_match_townes_reference(
    counts,
    model,
    residual,
    expected,
):
    """Unclipped residual PCA singular values match the pinned Townes reference."""
    result = residual_pca_matrix(
        counts,
        n_comps=3,
        model=model,
        residual=residual,
        clip=None,
        dtype="float64",
    )
    np.testing.assert_allclose(
        result.singular_values,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )

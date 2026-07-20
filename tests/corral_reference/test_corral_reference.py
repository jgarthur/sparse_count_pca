"""Regression test against the pinned corral correspondence-analysis output."""

import numpy as np

from sparse_count_pca import correspondence_analysis_matrix

EXPECTED_SINGULAR_VALUES = np.array([0.636028804403228, 0.428984435895091])
EXPECTED_TOTAL_INERTIA = 0.626893939393939
EXPECTED_ROW_STANDARD = np.array(
    [
        [1.28153192925511, -0.294713295713232],
        [-0.902647560092095, -1.50156224598234],
        [-1.41362099798205, 0.741841266497165],
        [0.968726990207983, 1.05732199392307],
        [0.408636982083447, -1.14435545118083],
        [-0.462147046541116, 0.926546308926599],
    ]
)
EXPECTED_ROW_PRINCIPAL = np.array(
    [
        [0.815091220768688, -0.126427416912324],
        [-0.574109848442866, -0.644146833054098],
        [-0.899103673225819, 0.318238357231986],
        [0.616138269375121, 0.453574679122559],
        [0.259904891149478, -0.490910677688282],
        [-0.293938833470029, 0.397473945665555],
    ]
)
EXPECTED_COLUMN_STANDARD = np.array(
    [
        [1.14857481214476, -0.226373274239128],
        [-0.627835545834432, -1.55203761561838],
        [-1.32800078395355, 0.962747952666988],
        [0.702845625630056, 0.836242325757717],
    ]
)
EXPECTED_COLUMN_PRINCIPAL = np.array(
    [
        [0.730526664536097, -0.0971106113511968],
        [-0.399321491578922, -0.665799981024014],
        [-0.844646750864526, 0.413003887384001],
        [0.447030062949524, 0.358734942386773],
    ]
)


def _assert_axes_equal_up_to_sign(actual, expected):
    signs = np.where(np.sum(actual * expected, axis=0) < 0, -1.0, 1.0)
    np.testing.assert_allclose(actual * signs, expected, rtol=1e-9, atol=1e-9)


def test_correspondence_analysis_matches_corral(counts):
    """Correspondence coordinates and inertias match the corral reference."""
    result = correspondence_analysis_matrix(counts, n_comps=2)

    np.testing.assert_allclose(
        result.singular_values,
        EXPECTED_SINGULAR_VALUES,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.total_inertia,
        EXPECTED_TOTAL_INERTIA,
        rtol=1e-12,
        atol=1e-14,
    )
    _assert_axes_equal_up_to_sign(
        result.row_principal_coordinates / result.singular_values,
        EXPECTED_ROW_STANDARD,
    )
    _assert_axes_equal_up_to_sign(
        result.row_principal_coordinates, EXPECTED_ROW_PRINCIPAL
    )
    _assert_axes_equal_up_to_sign(
        result.column_principal_coordinates / result.singular_values,
        EXPECTED_COLUMN_STANDARD,
    )
    _assert_axes_equal_up_to_sign(
        result.column_principal_coordinates, EXPECTED_COLUMN_PRINCIPAL
    )

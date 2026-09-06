import numpy as np
import pytest

from spellcaster.ml.rejection import (
    learn_rejection_threshold,
    nearest_same_class_distances,
)


def test_nearest_same_class_distances():

    features = np.asarray(
        [
            # Class A
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            # Class B
            [10.0, 0.0],
            [12.0, 0.0],
            [14.0, 0.0],
        ],
        dtype=np.float64,
    )

    labels = np.asarray(
        [
            "a",
            "a",
            "a",
            "b",
            "b",
            "b",
        ],
        dtype=np.str_,
    )

    distances = nearest_same_class_distances(
        features,
        labels,
    )

    expected = np.asarray(
        [
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        distances,
        expected,
    )


def test_rejection_threshold_uses_largest_known_distance():

    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [10.0, 0.0],
            [12.0, 0.0],
            [14.0, 0.0],
        ],
        dtype=np.float64,
    )

    labels = np.asarray(
        [
            "a",
            "a",
            "a",
            "b",
            "b",
            "b",
        ],
        dtype=np.str_,
    )

    threshold = learn_rejection_threshold(
        features,
        labels,
    )

    assert threshold == pytest.approx(2.0)

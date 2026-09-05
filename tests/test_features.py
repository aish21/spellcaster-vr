import numpy as np
import pytest

from spellcaster.gestures.models import (
    GestureSample,
    Point2D,
)
from spellcaster.gestures.spells import Spell
from spellcaster.ml.features import (
    sample_to_feature_vector,
    trajectory_to_feature_vector,
)


def test_trajectory_to_feature_vector():
    trajectory = (
        Point2D(0.1, 0.2),
        Point2D(0.3, 0.4),
        Point2D(0.5, 0.6),
    )

    features = trajectory_to_feature_vector(trajectory)

    expected = np.array(
        [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        features,
        expected,
    )


def test_feature_vector_has_two_values_per_point():
    trajectory = tuple(
        Point2D(
            x=float(index),
            y=float(index),
        )
        for index in range(32)
    )

    features = trajectory_to_feature_vector(trajectory)

    assert features.shape == (64,)


def test_empty_trajectory_cannot_create_features():
    with pytest.raises(
        ValueError,
        match="empty trajectory",
    ):

        trajectory_to_feature_vector(())


def test_sample_feature_vector_is_translation_and_scale_invariant():
    small = GestureSample(
        gesture_id="small",
        spell=Spell.FIREBALL,
        duration_ms=500,
        trajectory=(
            Point2D(0.10, 0.20),
            Point2D(0.20, 0.10),
            Point2D(0.30, 0.20),
        ),
    )

    large_shifted = GestureSample(
        gesture_id="large",
        spell=Spell.FIREBALL,
        duration_ms=900,
        trajectory=(
            Point2D(0.40, 0.70),
            Point2D(0.60, 0.50),
            Point2D(0.80, 0.70),
        ),
    )

    small_features = sample_to_feature_vector(
        small,
        target_points=8,
    )

    large_features = sample_to_feature_vector(
        large_shifted,
        target_points=8,
    )

    np.testing.assert_allclose(
        small_features,
        large_features,
    )

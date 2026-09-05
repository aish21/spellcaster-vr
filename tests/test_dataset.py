import numpy as np
import pytest

from spellcaster.gestures.models import (
    GestureSample,
    Point2D,
)
from spellcaster.gestures.spells import Spell
from spellcaster.ml.dataset import (
    build_ml_dataset,
)
from spellcaster.ml.features import (
    sample_to_feature_vector,
)


def test_build_ml_dataset_has_expected_shapes():

    samples = (
        GestureSample(
            gesture_id="fireball-1",
            spell=Spell.FIREBALL,
            duration_ms=500,
            trajectory=(
                Point2D(0.1, 0.3),
                Point2D(0.2, 0.1),
                Point2D(0.3, 0.3),
            ),
        ),
        GestureSample(
            gesture_id="shield-1",
            spell=Spell.SHIELD,
            duration_ms=700,
            trajectory=(
                Point2D(0.2, 0.3),
                Point2D(0.3, 0.1),
                Point2D(0.4, 0.3),
            ),
        ),
    )

    dataset = build_ml_dataset(
        samples,
        target_points=8,
    )

    assert dataset.features.shape == (
        2,
        16,
    )

    assert dataset.labels.shape == (2,)

    assert len(dataset.sample_ids) == 2


def test_dataset_preserves_sample_alignment():

    fireball = GestureSample(
        gesture_id="fireball-123",
        spell=Spell.FIREBALL,
        duration_ms=500,
        trajectory=(
            Point2D(0.1, 0.3),
            Point2D(0.2, 0.1),
            Point2D(0.3, 0.3),
        ),
    )

    shield = GestureSample(
        gesture_id="shield-456",
        spell=Spell.SHIELD,
        duration_ms=700,
        trajectory=(
            Point2D(0.2, 0.4),
            Point2D(0.4, 0.2),
            Point2D(0.6, 0.4),
        ),
    )

    dataset = build_ml_dataset(
        (
            fireball,
            shield,
        ),
        target_points=8,
    )

    assert dataset.labels.tolist() == [
        "fireball",
        "shield",
    ]

    assert dataset.sample_ids == (
        "fireball-123",
        "shield-456",
    )


def test_dataset_feature_row_matches_sample_features():

    sample = GestureSample(
        gesture_id="sample-1",
        spell=Spell.LIGHTNING,
        duration_ms=500,
        trajectory=(
            Point2D(0.1, 0.1),
            Point2D(0.4, 0.2),
            Point2D(0.2, 0.4),
        ),
    )

    expected = sample_to_feature_vector(
        sample,
        target_points=8,
    )

    dataset = build_ml_dataset(
        (sample,),
        target_points=8,
    )

    np.testing.assert_allclose(
        dataset.features[0],
        expected,
    )


def test_empty_samples_cannot_build_ml_dataset():

    with pytest.raises(
        ValueError,
        match="zero samples",
    ):

        build_ml_dataset(
            (),
            target_points=8,
        )

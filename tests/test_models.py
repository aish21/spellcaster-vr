import pytest
from dataclasses import FrozenInstanceError

from spellcaster.gestures.models import (
    GestureSample,
    Point2D,
)
from spellcaster.gestures.spells import Spell


def test_gesture_sample_contains_expected_data():
    trajectory = (
        Point2D(0.1, 0.2),
        Point2D(0.3, 0.4),
    )

    sample = GestureSample(
        gesture_id="test-id",
        spell=Spell.FIREBALL,
        duration_ms=500,
        trajectory=trajectory,
    )

    assert sample.gesture_id == "test-id"
    assert sample.spell == Spell.FIREBALL
    assert sample.duration_ms == 500
    assert len(sample.trajectory) == 2


def test_gesture_sample_is_immutable():
    sample = GestureSample(
        gesture_id="test-id",
        spell=Spell.SHIELD,
        duration_ms=500,
        trajectory=(Point2D(0.1, 0.2),),
    )

    with pytest.raises(FrozenInstanceError):
        sample.duration_ms = 999

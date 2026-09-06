from dataclasses import FrozenInstanceError

import pytest

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
        session_id="session-123",
        spell=Spell.FIREBALL,
        duration_ms=500,
        trajectory=trajectory,
    )

    assert sample.gesture_id == "test-id"

    assert sample.session_id == "session-123"

    assert sample.spell == Spell.FIREBALL

    assert sample.duration_ms == 500

    assert sample.trajectory == trajectory


def test_gesture_sample_is_immutable():

    sample = GestureSample(
        gesture_id="test-id",
        session_id="session-123",
        spell=Spell.SHIELD,
        duration_ms=500,
        trajectory=(Point2D(0.1, 0.2),),
    )

    with pytest.raises(FrozenInstanceError):

        sample.duration_ms = 600

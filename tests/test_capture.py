import pytest

from spellcaster.gestures.capture import GestureCapture
from spellcaster.gestures.models import (
    HandObservation,
    Point2D,
)


def create_landmarks() -> list[Point2D]:
    return [Point2D(0.0, 0.0) for _ in range(21)]


def test_calculate_pinch_ratio():
    landmarks = create_landmarks()

    landmarks[0] = Point2D(0.0, 0.0)
    landmarks[4] = Point2D(0.0, 0.0)
    landmarks[8] = Point2D(0.3, 0.0)
    landmarks[9] = Point2D(0.0, 1.0)

    observation = HandObservation(
        landmarks=tuple(landmarks),
        handedness="Right",
        confidence=1.0,
    )

    capture = GestureCapture()

    ratio = capture.calculate_pinch_ratio(observation)

    assert ratio == 0.3


def test_pinch_ratio_is_scale_normalized():
    capture = GestureCapture()

    small_landmarks = create_landmarks()

    small_landmarks[0] = Point2D(0.0, 0.0)
    small_landmarks[4] = Point2D(0.0, 0.0)
    small_landmarks[8] = Point2D(0.3, 0.0)
    small_landmarks[9] = Point2D(0.0, 1.0)

    large_landmarks = create_landmarks()

    large_landmarks[0] = Point2D(0.0, 0.0)
    large_landmarks[4] = Point2D(0.0, 0.0)
    large_landmarks[8] = Point2D(0.6, 0.0)
    large_landmarks[9] = Point2D(0.0, 2.0)

    small_observation = HandObservation(
        landmarks=tuple(small_landmarks),
        handedness="Right",
        confidence=1.0,
    )

    large_observation = HandObservation(
        landmarks=tuple(large_landmarks),
        handedness="Right",
        confidence=1.0,
    )

    small_ratio = capture.calculate_pinch_ratio(small_observation)

    large_ratio = capture.calculate_pinch_ratio(large_observation)

    assert small_ratio == pytest.approx(0.3)
    assert large_ratio == pytest.approx(0.3)

import pytest

from spellcaster.gestures.capture import (
    CaptureEvent,
    GestureCapture,
)
from spellcaster.gestures.models import (
    HandObservation,
    Point2D,
)


def create_landmarks() -> list[Point2D]:
    """Create a neutral set of 21 fake hand landmarks."""
    return [Point2D(0.0, 0.0) for _ in range(21)]


def create_observation(
    pinch_distance: float,
    hand_scale: float = 1.0,
) -> HandObservation:
    """
    Create a synthetic hand observation.

    The geometry is deliberately simple:

    wrist      = (0, 0)
    thumb tip  = (0, 0)
    index tip  = (pinch_distance, 0)
    middle MCP = (0, hand_scale)

    Therefore:

        pinch ratio
        = pinch_distance / hand_scale
    """

    landmarks = create_landmarks()

    # Wrist
    landmarks[0] = Point2D(0.0, 0.0)

    # Thumb tip
    landmarks[4] = Point2D(0.0, 0.0)

    # Index fingertip
    landmarks[8] = Point2D(
        pinch_distance,
        0.0,
    )

    # Middle MCP
    landmarks[9] = Point2D(
        0.0,
        hand_scale,
    )

    return HandObservation(
        landmarks=tuple(landmarks),
        handedness="Right",
        confidence=1.0,
    )


def create_capture(
    *,
    minimum_points: int = 3,
    minimum_duration_ms: int = 200,
    max_lost_frames: int = 2,
    smoothing_alpha: float = 1.0,
) -> GestureCapture:
    """
    Create a GestureCapture configured for predictable tests.

    smoothing_alpha=1.0 means no smoothing, which makes
    trajectory values deterministic in unit tests.
    """

    return GestureCapture(
        pinch_start_ratio=0.35,
        pinch_end_ratio=0.50,
        minimum_points=minimum_points,
        minimum_duration_ms=minimum_duration_ms,
        max_lost_frames=max_lost_frames,
        smoothing_alpha=smoothing_alpha,
    )


# ============================================================
# Pinch-ratio tests
# ============================================================


def test_calculate_pinch_ratio():
    capture = create_capture()

    observation = create_observation(
        pinch_distance=0.3,
        hand_scale=1.0,
    )

    ratio = capture.calculate_pinch_ratio(observation)

    assert ratio == pytest.approx(0.3)


def test_pinch_ratio_is_scale_normalized():
    capture = create_capture()

    small_observation = create_observation(
        pinch_distance=0.3,
        hand_scale=1.0,
    )

    large_observation = create_observation(
        pinch_distance=0.6,
        hand_scale=2.0,
    )

    small_ratio = capture.calculate_pinch_ratio(small_observation)

    large_ratio = capture.calculate_pinch_ratio(large_observation)

    assert small_ratio == pytest.approx(0.3)
    assert large_ratio == pytest.approx(0.3)


# ============================================================
# Pinch state / hysteresis tests
# ============================================================


def test_pinch_starts_below_start_threshold():
    capture = create_capture()

    result = capture.update(
        create_observation(0.30),
        timestamp_ms=0,
    )

    assert result.event == CaptureEvent.STARTED
    assert capture.is_pinching is True


def test_open_hand_does_not_start_pinch():
    capture = create_capture()

    result = capture.update(
        create_observation(0.60),
        timestamp_ms=0,
    )

    assert result.event == CaptureEvent.NONE
    assert capture.is_pinching is False


def test_pinch_uses_hysteresis():
    capture = create_capture()

    # OPEN -> PINCHING
    result = capture.update(
        create_observation(0.30),
        timestamp_ms=0,
    )

    assert result.event == CaptureEvent.STARTED
    assert capture.is_pinching is True

    # Above START threshold (0.35),
    # but below END threshold (0.50).
    #
    # Since we are already pinching, we stay pinching.
    result = capture.update(
        create_observation(0.42),
        timestamp_ms=50,
    )

    assert result.event == CaptureEvent.NONE
    assert capture.is_pinching is True

    # Now cross END threshold.
    result = capture.update(
        create_observation(0.55),
        timestamp_ms=100,
    )

    # This particular gesture is too short to satisfy our
    # validity rules, so ending the pinch produces REJECTED,
    # rather than a separate ENDED event.
    assert result.event == CaptureEvent.REJECTED
    assert capture.is_pinching is False


def test_invalid_hysteresis_thresholds_raise_error():
    with pytest.raises(ValueError):
        GestureCapture(
            pinch_start_ratio=0.60,
            pinch_end_ratio=0.40,
            minimum_points=3,
            minimum_duration_ms=200,
            max_lost_frames=2,
            smoothing_alpha=1.0,
        )


# ============================================================
# Gesture lifecycle tests
# ============================================================


def test_valid_gesture_completes():
    capture = create_capture(
        minimum_points=3,
        minimum_duration_ms=200,
    )

    # Point 1: start
    result = capture.update(
        create_observation(0.30),
        timestamp_ms=0,
    )

    assert result.event == CaptureEvent.STARTED

    # Point 2
    result = capture.update(
        create_observation(0.30),
        timestamp_ms=100,
    )

    assert result.event == CaptureEvent.NONE

    # Point 3
    result = capture.update(
        create_observation(0.30),
        timestamp_ms=200,
    )

    assert result.event == CaptureEvent.NONE

    # Release
    result = capture.update(
        create_observation(0.60),
        timestamp_ms=250,
    )

    assert result.event == CaptureEvent.COMPLETED
    assert result.duration_ms == 250
    assert len(result.trajectory) == 3
    assert capture.is_pinching is False


def test_short_gesture_is_rejected():
    capture = create_capture(
        minimum_points=3,
        minimum_duration_ms=200,
    )

    capture.update(
        create_observation(0.30),
        timestamp_ms=0,
    )

    # Immediate release:
    # only one point and 50 ms.
    result = capture.update(
        create_observation(0.60),
        timestamp_ms=50,
    )

    assert result.event == CaptureEvent.REJECTED
    assert result.duration_ms == 50
    assert len(result.trajectory) == 1
    assert capture.is_pinching is False


# ============================================================
# Tracking-loss tests
# ============================================================


def test_temporary_tracking_loss_does_not_cancel():
    capture = create_capture(
        max_lost_frames=2,
    )

    capture.update(
        create_observation(0.30),
        timestamp_ms=0,
    )

    first_loss = capture.update(
        None,
        timestamp_ms=30,
    )

    second_loss = capture.update(
        None,
        timestamp_ms=60,
    )

    assert first_loss.event == CaptureEvent.NONE
    assert second_loss.event == CaptureEvent.NONE

    assert capture.is_pinching is True


def test_extended_tracking_loss_cancels():
    capture = create_capture(
        max_lost_frames=2,
    )

    capture.update(
        create_observation(0.30),
        timestamp_ms=0,
    )

    capture.update(
        None,
        timestamp_ms=30,
    )

    capture.update(
        None,
        timestamp_ms=60,
    )

    result = capture.update(
        None,
        timestamp_ms=90,
    )

    assert result.event == CaptureEvent.CANCELLED
    assert result.duration_ms == 90
    assert capture.is_pinching is False


def test_tracking_can_recover_within_grace_period():
    capture = create_capture(
        max_lost_frames=2,
    )

    capture.update(
        create_observation(0.30),
        timestamp_ms=0,
    )

    capture.update(
        None,
        timestamp_ms=30,
    )

    # Hand returns before grace period expires.
    result = capture.update(
        create_observation(0.30),
        timestamp_ms=60,
    )

    assert result.event == CaptureEvent.NONE
    assert capture.is_pinching is True


def test_pinch_start_requires_confirmation_frames():
    capture = GestureCapture(
        pinch_start_ratio=0.35,
        pinch_end_ratio=0.50,
        minimum_points=1,
        minimum_duration_ms=0,
        max_lost_frames=2,
        smoothing_alpha=1.0,
        pinch_start_confirm_frames=2,
        pinch_end_confirm_frames=2,
    )

    first = capture.update(
        create_observation(0.30),
        timestamp_ms=0,
    )

    assert first.event == CaptureEvent.NONE
    assert capture.is_pinching is False

    second = capture.update(
        create_observation(0.30),
        timestamp_ms=33,
    )

    assert second.event == CaptureEvent.STARTED
    assert capture.is_pinching is True


def test_single_release_frame_does_not_end_cast():
    capture = GestureCapture(
        pinch_start_ratio=0.35,
        pinch_end_ratio=0.50,
        minimum_points=1,
        minimum_duration_ms=0,
        max_lost_frames=2,
        smoothing_alpha=1.0,
        pinch_start_confirm_frames=1,
        pinch_end_confirm_frames=2,
    )

    capture.update(
        create_observation(0.30),
        timestamp_ms=0,
    )

    first_release = capture.update(
        create_observation(0.60),
        timestamp_ms=33,
    )

    assert first_release.event == CaptureEvent.NONE
    assert capture.is_pinching is True

    second_release = capture.update(
        create_observation(0.60),
        timestamp_ms=66,
    )

    assert second_release.event == CaptureEvent.COMPLETED

    assert capture.is_pinching is False


def test_tracking_loss_breaks_release_confirmation():
    capture = GestureCapture(
        pinch_start_ratio=0.35,
        pinch_end_ratio=0.50,
        minimum_points=1,
        minimum_duration_ms=0,
        max_lost_frames=2,
        smoothing_alpha=1.0,
        pinch_start_confirm_frames=1,
        pinch_end_confirm_frames=2,
    )

    # Start casting.
    result = capture.update(
        create_observation(0.30),
        timestamp_ms=0,
    )

    assert result.event == CaptureEvent.STARTED
    assert capture.is_pinching is True

    # First potential release frame.
    result = capture.update(
        create_observation(0.60),
        timestamp_ms=33,
    )

    assert result.event == CaptureEvent.NONE
    assert capture.is_pinching is True

    # Tracking disappears.
    #
    # This must reset the release confirmation counter.
    result = capture.update(
        None,
        timestamp_ms=66,
    )

    assert result.event == CaptureEvent.NONE
    assert capture.is_pinching is True

    # Hand comes back and still looks released.
    #
    # This should now be release candidate #1 again,
    # NOT candidate #2.
    result = capture.update(
        create_observation(0.60),
        timestamp_ms=99,
    )

    assert result.event == CaptureEvent.NONE
    assert capture.is_pinching is True

    # Second consecutive release observation.
    result = capture.update(
        create_observation(0.60),
        timestamp_ms=132,
    )

    assert result.event == CaptureEvent.COMPLETED
    assert capture.is_pinching is False


def test_capture_preserves_raw_and_smoothed_trajectories():
    capture = GestureCapture(
        pinch_start_ratio=0.35,
        pinch_end_ratio=0.50,
        minimum_points=2,
        minimum_duration_ms=0,
        max_lost_frames=2,
        smoothing_alpha=0.5,
        pinch_start_confirm_frames=1,
        pinch_end_confirm_frames=1,
    )

    # Start at x = 0.10.
    capture.update(
        create_observation(0.10),
        timestamp_ms=0,
    )

    # Move to x = 0.30 while still pinching.
    capture.update(
        create_observation(0.30),
        timestamp_ms=33,
    )

    # Release.
    result = capture.update(
        create_observation(0.60),
        timestamp_ms=66,
    )

    assert result.event == CaptureEvent.COMPLETED

    # ------------------------------------------------
    # Raw point should remain exactly where MediaPipe
    # reported it.
    # ------------------------------------------------

    assert result.trajectory[1].x == pytest.approx(0.30)

    # ------------------------------------------------
    # EMA:
    #
    # 0.5 * 0.30 + 0.5 * 0.10 = 0.20
    # ------------------------------------------------

    assert result.smoothed_trajectory[1].x == pytest.approx(0.20)

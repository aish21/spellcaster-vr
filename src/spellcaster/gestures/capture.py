from dataclasses import dataclass
from enum import Enum, auto

from spellcaster.gestures.models import (
    HandObservation,
    Point2D,
)
from spellcaster.vision.geometry import (
    distance_2d,
    exponential_smooth,
)
from spellcaster.vision.landmarks import HandLandmark


class CaptureEvent(Enum):
    NONE = auto()
    STARTED = auto()
    COMPLETED = auto()
    REJECTED = auto()
    CANCELLED = auto()


@dataclass(frozen=True)
class CaptureResult:
    event: CaptureEvent
    trajectory: tuple[Point2D, ...] = ()
    duration_ms: int | None = None


class GestureCapture:

    def __init__(
        self,
        pinch_start_ratio: float,
        pinch_end_ratio: float,
        minimum_points: int,
        minimum_duration_ms: int,
        max_lost_frames: int,
        smoothing_alpha: float,
        pinch_start_confirm_frames: int = 1,
        pinch_end_confirm_frames: int = 1,
    ) -> None:

        # --------------------------------------------
        # Validate configuration
        # --------------------------------------------

        if pinch_start_ratio >= pinch_end_ratio:
            raise ValueError("pinch_start_ratio must be less than " "pinch_end_ratio")

        if minimum_points <= 0:
            raise ValueError("minimum_points must be greater than zero")

        if minimum_duration_ms < 0:
            raise ValueError("minimum_duration_ms cannot be negative")

        if max_lost_frames < 0:
            raise ValueError("max_lost_frames cannot be negative")

        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in the range (0, 1]")

        if pinch_start_confirm_frames <= 0:
            raise ValueError("pinch_start_confirm_frames must be greater than zero")

        if pinch_end_confirm_frames <= 0:
            raise ValueError("pinch_end_confirm_frames must be greater than zero")

        # --------------------------------------------
        # Configuration
        # --------------------------------------------

        self._pinch_start_ratio = pinch_start_ratio
        self._pinch_end_ratio = pinch_end_ratio

        self._minimum_points = minimum_points
        self._minimum_duration_ms = minimum_duration_ms

        self._max_lost_frames = max_lost_frames

        self._smoothing_alpha = smoothing_alpha

        self._pinch_start_confirm_frames = pinch_start_confirm_frames

        self._pinch_end_confirm_frames = pinch_end_confirm_frames

        # --------------------------------------------
        # Persistent state
        # --------------------------------------------

        self._is_pinching = False

        self._trajectory: list[Point2D] = []

        self._cast_start_time_ms: int | None = None

        self._smoothed_index: Point2D | None = None

        self._lost_frames = 0

        self._start_candidate_frames = 0
        self._end_candidate_frames = 0

    @property
    def is_pinching(self) -> bool:
        return self._is_pinching

    @property
    def current_trajectory(self) -> tuple[Point2D, ...]:
        return tuple(self._trajectory)

    def calculate_pinch_ratio(
        self,
        observation: HandObservation,
    ) -> float:
        landmarks = observation.landmarks

        wrist = landmarks[HandLandmark.WRIST]

        thumb_tip = landmarks[HandLandmark.THUMB_TIP]

        index_tip = landmarks[HandLandmark.INDEX_TIP]

        middle_mcp = landmarks[HandLandmark.MIDDLE_MCP]

        pinch_distance = distance_2d(
            thumb_tip,
            index_tip,
        )

        hand_scale = distance_2d(
            wrist,
            middle_mcp,
        )

        if hand_scale <= 1e-6:
            return float("inf")

        return pinch_distance / hand_scale

    def update(
        self,
        observation: HandObservation | None,
        timestamp_ms: int,
    ) -> CaptureResult:

        # --------------------------------------------
        # No hand detected
        # --------------------------------------------

        if observation is None:
            return self._handle_missing_hand(timestamp_ms)

        # Detection recovered, so the consecutive
        # missing-frame counter resets.
        self._lost_frames = 0

        pinch_ratio = self.calculate_pinch_ratio(observation)

        index_tip = observation.landmarks[HandLandmark.INDEX_TIP]

        # --------------------------------------------
        # State: OPEN
        # --------------------------------------------

        if not self._is_pinching:

            if pinch_ratio < self._pinch_start_ratio:

                self._start_candidate_frames += 1

                if self._start_candidate_frames >= self._pinch_start_confirm_frames:
                    self._start_candidate_frames = 0

                    self._start_cast(
                        index_tip=index_tip,
                        timestamp_ms=timestamp_ms,
                    )

                    return CaptureResult(event=CaptureEvent.STARTED)

            else:
                self._start_candidate_frames = 0

            return CaptureResult(event=CaptureEvent.NONE)

        # --------------------------------------------
        # State: PINCHING
        # --------------------------------------------

        if pinch_ratio > self._pinch_end_ratio:

            self._end_candidate_frames += 1

            if self._end_candidate_frames >= self._pinch_end_confirm_frames:
                self._end_candidate_frames = 0

                return self._finish_cast(timestamp_ms)

            return CaptureResult(event=CaptureEvent.NONE)

        self._end_candidate_frames = 0

        self._record_point(index_tip)

        return CaptureResult(event=CaptureEvent.NONE)

    def _start_cast(
        self,
        index_tip: Point2D,
        timestamp_ms: int,
    ) -> None:
        self._is_pinching = True

        self._cast_start_time_ms = timestamp_ms

        self._smoothed_index = index_tip

        self._trajectory = [index_tip]
        self._start_candidate_frames = 0
        self._end_candidate_frames = 0

    def _record_point(
        self,
        index_tip: Point2D,
    ) -> None:

        if self._smoothed_index is None:
            self._smoothed_index = index_tip

        else:
            self._smoothed_index = exponential_smooth(
                current=index_tip,
                previous=self._smoothed_index,
                alpha=self._smoothing_alpha,
            )

        self._trajectory.append(self._smoothed_index)

    def _finish_cast(
        self,
        timestamp_ms: int,
    ) -> CaptureResult:

        if self._cast_start_time_ms is None:
            duration_ms = 0
        else:
            duration_ms = timestamp_ms - self._cast_start_time_ms

        completed_trajectory = tuple(self._trajectory)

        enough_points = len(completed_trajectory) >= self._minimum_points

        long_enough = duration_ms >= self._minimum_duration_ms

        if enough_points and long_enough:
            event = CaptureEvent.COMPLETED
        else:
            event = CaptureEvent.REJECTED

        self._reset_cast()

        return CaptureResult(
            event=event,
            trajectory=completed_trajectory,
            duration_ms=duration_ms,
        )

    def _handle_missing_hand(
        self,
        timestamp_ms: int,
    ) -> CaptureResult:

        # ------------------------------------------------
        # IDLE
        #
        # If we were considering starting a cast,
        # losing the hand breaks that consecutive sequence.
        # ------------------------------------------------

        if not self._is_pinching:
            self._lost_frames = 0

            self._start_candidate_frames = 0

            return CaptureResult(event=CaptureEvent.NONE)

        # ------------------------------------------------
        # CASTING
        #
        # If we were considering ending a cast, a missing
        # frame breaks that consecutive release sequence.
        # ------------------------------------------------

        self._end_candidate_frames = 0

        self._lost_frames += 1

        # ------------------------------------------------
        # Temporary tracking loss
        #
        # Do nothing yet. The existing trajectory and
        # casting state remain alive.
        # ------------------------------------------------

        if self._lost_frames <= self._max_lost_frames:
            return CaptureResult(event=CaptureEvent.NONE)

        # ------------------------------------------------
        # Tracking was gone too long -> cancel cast
        # ------------------------------------------------

        if self._cast_start_time_ms is None:
            duration_ms = 0

        else:
            duration_ms = timestamp_ms - self._cast_start_time_ms

        cancelled_trajectory = tuple(self._trajectory)

        self._reset_cast()

        return CaptureResult(
            event=CaptureEvent.CANCELLED,
            trajectory=cancelled_trajectory,
            duration_ms=duration_ms,
        )

    def _reset_cast(self) -> None:
        self._is_pinching = False

        self._trajectory = []

        self._cast_start_time_ms = None

        self._smoothed_index = None

        self._lost_frames = 0

        self._start_candidate_frames = 0
        self._end_candidate_frames = 0

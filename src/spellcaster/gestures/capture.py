from dataclasses import dataclass
from enum import Enum, auto

from spellcaster.gestures.models import Point2D, HandObservation
from spellcaster.vision.geometry import distance_2d
from spellcaster.vision.landmarks import HandLandmark


class CaptureEvent(Enum):
    NONE = auto()
    STARTED = auto()
    COMPLETED = auto()
    CANCELLED = auto()
    REJECTED = auto()


@dataclass
class CaptureResult:
    event: CaptureEvent
    trajectory: list[Point2D] | None = None
    duration_ms: int | None = None


class GestureCapture:

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

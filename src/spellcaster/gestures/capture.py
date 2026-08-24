from dataclasses import dataclass
from enum import Enum, auto

from spellcaster.gestures.models import Point2D


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

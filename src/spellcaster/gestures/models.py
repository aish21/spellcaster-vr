from dataclasses import dataclass
from spellcaster.gestures.spells import Spell


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class HandObservation:
    landmarks: tuple[Point2D, ...]
    handedness: str
    confidence: float


@dataclass(frozen=True)
class GestureSample:
    gesture_id: str
    spell: Spell
    duration_ms: int
    trajectory: tuple[Point2D, ...]

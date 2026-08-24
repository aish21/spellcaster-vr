import math

from spellcaster.gestures.models import Point2D


def distance_2d(point_a: Point2D, point_b: Point2D) -> float:
    """Calculate the Euclidean distance between two 2D points."""
    return math.hypot((point_a.x - point_b.x), (point_a.y - point_b.y))


def normalized_to_pixel(
    point: Point2D,
    width: int,
    height: int,
) -> tuple[int, int]:
    x = int(point.x * width)
    y = int(point.y * height)

    return x, y


def exponential_smooth(
    current: Point2D,
    previous: Point2D,
    alpha: float,
) -> Point2D:
    return Point2D(
        x=(alpha * current.x + (1.0 - alpha) * previous.x),
        y=(alpha * current.y + (1.0 - alpha) * previous.y),
    )

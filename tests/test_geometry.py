from spellcaster.gestures.models import Point2D
from spellcaster.vision.geometry import (
    distance_2d,
    normalized_to_pixel,
)


def test_distance_2d():
    point_a = Point2D(0.0, 0.0)
    point_b = Point2D(3.0, 4.0)

    assert distance_2d(point_a, point_b) == 5.0


def test_normalized_to_pixel():
    point = Point2D(0.5, 0.25)

    pixel = normalized_to_pixel(
        point,
        width=640,
        height=480,
    )

    assert pixel == (320, 120)

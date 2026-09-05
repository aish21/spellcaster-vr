import pytest

from spellcaster.gestures.models import Point2D
from spellcaster.ml.preprocessing import (
    center_trajectory,
)


def test_center_trajectory_around_origin():
    trajectory = (
        Point2D(0.20, 0.30),
        Point2D(0.40, 0.30),
        Point2D(0.40, 0.70),
        Point2D(0.20, 0.70),
    )

    centered = center_trajectory(trajectory)

    assert centered[0].x == pytest.approx(-0.10)

    assert centered[0].y == pytest.approx(-0.20)

    assert centered[2].x == pytest.approx(0.10)

    assert centered[2].y == pytest.approx(0.20)


def test_centering_removes_translation():
    first = (
        Point2D(0.10, 0.20),
        Point2D(0.20, 0.10),
        Point2D(0.30, 0.20),
    )

    second = (
        Point2D(0.50, 0.70),
        Point2D(0.60, 0.60),
        Point2D(0.70, 0.70),
    )

    centered_first = center_trajectory(first)

    centered_second = center_trajectory(second)

    for point_a, point_b in zip(
        centered_first,
        centered_second,
    ):
        assert point_a.x == pytest.approx(point_b.x)

        assert point_a.y == pytest.approx(point_b.y)


def test_center_empty_trajectory_raises_error():
    with pytest.raises(
        ValueError,
        match="empty trajectory",
    ):
        center_trajectory(())

import pytest

from spellcaster.gestures.models import Point2D
from spellcaster.ml.preprocessing import (
    center_trajectory,
    normalize_translation_and_scale,
    scale_trajectory,
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


def test_scale_normalization_removes_size_difference():
    small = (
        Point2D(0.0, 0.0),
        Point2D(0.1, -0.2),
        Point2D(0.2, 0.0),
    )

    large = (
        Point2D(0.0, 0.0),
        Point2D(0.2, -0.4),
        Point2D(0.4, 0.0),
    )

    normalized_small = normalize_translation_and_scale(small)

    normalized_large = normalize_translation_and_scale(large)

    for point_a, point_b in zip(
        normalized_small,
        normalized_large,
    ):
        assert point_a.x == pytest.approx(point_b.x)

        assert point_a.y == pytest.approx(point_b.y)


def test_scale_trajectory_preserves_aspect_ratio():
    trajectory = (
        Point2D(-0.1, -0.2),
        Point2D(0.1, 0.2),
    )

    scaled = scale_trajectory(trajectory)

    xs = [point.x for point in scaled]

    ys = [point.y for point in scaled]

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    assert width == pytest.approx(0.5)

    assert height == pytest.approx(1.0)


def test_scale_zero_size_trajectory_raises_error():
    trajectory = (
        Point2D(0.0, 0.0),
        Point2D(0.0, 0.0),
    )

    with pytest.raises(
        ValueError,
        match="zero-size trajectory",
    ):
        scale_trajectory(trajectory)

from collections.abc import Sequence

from spellcaster.gestures.models import Point2D


def center_trajectory(
    trajectory: Sequence[Point2D],
) -> tuple[Point2D, ...]:
    """
    Remove absolute screen position from a trajectory.

    The trajectory is translated so that the centre of its
    axis-aligned bounding box lies at (0, 0).

    No scaling, rotation, smoothing, or resampling is applied.
    """

    if not trajectory:
        raise ValueError("Cannot center an empty trajectory")

    min_x = min(point.x for point in trajectory)

    max_x = max(point.x for point in trajectory)

    min_y = min(point.y for point in trajectory)

    max_y = max(point.y for point in trajectory)

    center_x = (min_x + max_x) / 2.0

    center_y = (min_y + max_y) / 2.0

    return tuple(
        Point2D(
            x=point.x - center_x,
            y=point.y - center_y,
        )
        for point in trajectory
    )


def scale_trajectory(
    trajectory: Sequence[Point2D],
) -> tuple[Point2D, ...]:
    """
    Normalize trajectory size using one uniform scale factor.

    The larger bounding-box dimension becomes 1.0.

    x and y use the same scale factor so aspect ratio and
    geometric shape are preserved.
    """

    if not trajectory:
        raise ValueError("Cannot scale an empty trajectory")

    min_x = min(point.x for point in trajectory)

    max_x = max(point.x for point in trajectory)

    min_y = min(point.y for point in trajectory)

    max_y = max(point.y for point in trajectory)

    width = max_x - min_x

    height = max_y - min_y

    scale = max(
        width,
        height,
    )

    if scale <= 1e-12:
        raise ValueError("Cannot scale a zero-size trajectory")

    return tuple(
        Point2D(
            x=point.x / scale,
            y=point.y / scale,
        )
        for point in trajectory
    )


def normalize_translation_and_scale(
    trajectory: Sequence[Point2D],
) -> tuple[Point2D, ...]:
    """
    Remove absolute position and absolute size while preserving
    trajectory shape, aspect ratio, orientation, and point order.
    """

    centered = center_trajectory(trajectory)

    return scale_trajectory(centered)

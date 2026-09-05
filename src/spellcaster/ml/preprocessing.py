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

    # ========================================================
    # Find spatial bounds
    # ========================================================

    min_x = min(point.x for point in trajectory)

    max_x = max(point.x for point in trajectory)

    min_y = min(point.y for point in trajectory)

    max_y = max(point.y for point in trajectory)

    # ========================================================
    # Bounding-box centre
    # ========================================================

    center_x = (min_x + max_x) / 2.0

    center_y = (min_y + max_y) / 2.0

    # ========================================================
    # Translate every point
    # ========================================================

    return tuple(
        Point2D(
            x=point.x - center_x,
            y=point.y - center_y,
        )
        for point in trajectory
    )

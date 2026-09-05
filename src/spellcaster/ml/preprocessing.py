import math

from collections.abc import Sequence

from spellcaster.gestures.models import Point2D


def center_trajectory(
    trajectory: Sequence[Point2D],
) -> tuple[Point2D, ...]:
    """
    Remove absolute screen position using the centre of the
    trajectory's axis-aligned bounding box.
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
    Normalize size using one uniform scale factor.

    The larger bounding-box dimension becomes 1.0 while aspect
    ratio is preserved.
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
    aspect ratio, orientation, point order, and shape.
    """

    centered = center_trajectory(trajectory)

    return scale_trajectory(centered)


def trajectory_length(
    trajectory: Sequence[Point2D],
) -> float:
    """
    Calculate total Euclidean distance travelled along a
    trajectory.
    """

    if len(trajectory) < 2:
        return 0.0

    total = 0.0

    for previous, current in zip(
        trajectory,
        trajectory[1:],
    ):
        total += math.hypot(
            current.x - previous.x,
            current.y - previous.y,
        )

    return total


def interpolate_point(
    start: Point2D,
    end: Point2D,
    fraction: float,
) -> Point2D:
    """
    Linearly interpolate between two points.

    fraction=0 returns start.
    fraction=1 returns end.
    """

    return Point2D(
        x=(start.x + fraction * (end.x - start.x)),
        y=(start.y + fraction * (end.y - start.y)),
    )


def resample_trajectory(
    trajectory: Sequence[Point2D],
    target_points: int,
) -> tuple[Point2D, ...]:
    """
    Resample a trajectory to a fixed number of spatially
    equidistant points.

    Sampling is based on cumulative path length rather than
    original frame/index position.
    """

    if not trajectory:
        raise ValueError("Cannot resample an empty trajectory")

    if target_points < 2:
        raise ValueError("target_points must be at least 2")

    if len(trajectory) < 2:
        raise ValueError("Cannot resample a trajectory " "with fewer than 2 points")

    # ========================================================
    # Calculate cumulative path distance for each original
    # trajectory point.
    # ========================================================

    cumulative_distances = [0.0]

    for previous, current in zip(
        trajectory,
        trajectory[1:],
    ):

        segment_length = math.hypot(
            current.x - previous.x,
            current.y - previous.y,
        )

        cumulative_distances.append(cumulative_distances[-1] + segment_length)

    total_length = cumulative_distances[-1]

    if total_length <= 1e-12:
        raise ValueError("Cannot resample a zero-length trajectory")

    # ========================================================
    # Generate equally spaced target distances.
    # ========================================================

    step = total_length / (target_points - 1)

    target_distances = [step * index for index in range(target_points)]

    # ========================================================
    # Interpolate each desired point along the original path.
    # ========================================================

    resampled: list[Point2D] = []

    segment_index = 0

    for target_distance in target_distances:

        # Move forward until we locate the original segment
        # containing this target path distance.
        while (
            segment_index < len(trajectory) - 2
            and cumulative_distances[segment_index + 1] < target_distance
        ):
            segment_index += 1

        segment_start_distance = cumulative_distances[segment_index]

        segment_end_distance = cumulative_distances[segment_index + 1]

        segment_length = segment_end_distance - segment_start_distance

        # Duplicate adjacent points can create a zero-length
        # segment. In that case stay at the segment start.
        if segment_length <= 1e-12:

            fraction = 0.0

        else:

            fraction = (target_distance - segment_start_distance) / segment_length

        point = interpolate_point(
            trajectory[segment_index],
            trajectory[segment_index + 1],
            fraction,
        )

        resampled.append(point)

    # ========================================================
    # Preserve exact original boundaries.
    # ========================================================

    resampled[0] = trajectory[0]

    resampled[-1] = trajectory[-1]

    return tuple(resampled)


def preprocess_trajectory(
    trajectory: Sequence[Point2D],
    target_points: int,
) -> tuple[Point2D, ...]:
    """
    Current ML preprocessing pipeline.

    1. translation normalization
    2. uniform scale normalization
    3. path-distance resampling
    """

    normalized = normalize_translation_and_scale(trajectory)

    return resample_trajectory(
        normalized,
        target_points,
    )

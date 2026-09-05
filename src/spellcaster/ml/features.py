from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from spellcaster.gestures.models import (
    GestureSample,
    Point2D,
)
from spellcaster.ml.preprocessing import (
    preprocess_trajectory,
)

# ============================================================
# Type aliases
# ============================================================


FloatArray = NDArray[np.float64]


# ============================================================
# Trajectory → feature vector
# ============================================================


def trajectory_to_feature_vector(
    trajectory: Sequence[Point2D],
) -> FloatArray:
    """
    Flatten an ordered 2D trajectory into a one-dimensional
    numeric feature vector.

    Example:

        (
            Point2D(x0, y0),
            Point2D(x1, y1),
        )

    becomes:

        [
            x0,
            y0,
            x1,
            y1,
        ]

    Point order is deliberately preserved.
    """

    if not trajectory:

        raise ValueError("Cannot create features " "from an empty trajectory")

    values: list[float] = []

    for point in trajectory:

        values.extend(
            (
                point.x,
                point.y,
            )
        )

    return np.asarray(
        values,
        dtype=np.float64,
    )


# ============================================================
# GestureSample → model-ready feature vector
# ============================================================


def sample_to_feature_vector(
    sample: GestureSample,
    target_points: int,
) -> FloatArray:
    """
    Convert one raw GestureSample into one fixed-length numeric
    ML feature vector.

    Pipeline:

        raw GestureSample trajectory

        → translation normalization

        → uniform scale normalization

        → path-distance resampling

        → flatten x/y coordinates

    The spell label and duration are deliberately NOT included
    in the feature vector.
    """

    processed = preprocess_trajectory(
        sample.trajectory,
        target_points=target_points,
    )

    return trajectory_to_feature_vector(processed)

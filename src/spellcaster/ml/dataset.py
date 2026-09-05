from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spellcaster.gestures.models import GestureSample
from spellcaster.ml.features import (
    FloatArray,
    sample_to_feature_vector,
)

# ============================================================
# Type aliases
# ============================================================


StringArray = NDArray[np.str_]


# ============================================================
# Model-ready dataset
# ============================================================


@dataclass(frozen=True)
class MLDataset:
    """
    Model-ready supervised-learning dataset.

    features:
        Numeric feature matrix X.

        Shape:
            (number_of_samples, number_of_features)

    labels:
        Target labels y.

        Shape:
            (number_of_samples,)

    sample_ids:
        Source GestureSample IDs.

        These are diagnostic metadata and are NOT supplied to
        the model as features.
    """

    features: FloatArray
    labels: StringArray
    sample_ids: tuple[str, ...]


# ============================================================
# Dataset construction
# ============================================================


def build_ml_dataset(
    samples: Sequence[GestureSample],
    target_points: int,
) -> MLDataset:
    """
    Convert raw GestureSamples into one supervised-learning
    dataset.

    For every sample:

        raw trajectory

        → translation normalization

        → uniform scale normalization

        → path-distance resampling

        → flattened numeric feature vector

    Labels and sample IDs remain aligned with feature rows.
    """

    if not samples:

        raise ValueError("Cannot build an ML dataset " "from zero samples")

    # ========================================================
    # Build one feature vector per gesture
    # ========================================================

    feature_vectors = [
        sample_to_feature_vector(
            sample,
            target_points=target_points,
        )
        for sample in samples
    ]

    # Stack:
    #
    #     (64,)
    #     (64,)
    #     (64,)
    #
    # into:
    #
    #     (N, 64)
    #
    features = np.stack(
        feature_vectors,
        axis=0,
    )

    # ========================================================
    # Target labels
    # ========================================================

    labels = np.asarray(
        [sample.spell.value for sample in samples],
        dtype=np.str_,
    )

    # ========================================================
    # Diagnostic source metadata
    # ========================================================

    sample_ids = tuple(sample.gesture_id for sample in samples)

    # ========================================================
    # Dataset alignment invariant
    # ========================================================

    if not (features.shape[0] == labels.shape[0] == len(sample_ids)):

        raise RuntimeError("ML dataset rows, labels, and " "sample IDs are not aligned")

    return MLDataset(
        features=features,
        labels=labels,
        sample_ids=sample_ids,
    )

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedKFold,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
)

from spellcaster.gestures.spells import Spell
from spellcaster.ml.dataset import MLDataset

# ============================================================
# Type aliases
# ============================================================


IntArray = NDArray[np.int64]


KNNWeights = Literal[
    "uniform",
    "distance",
]


# ============================================================
# Evaluation domain models
# ============================================================


@dataclass(frozen=True)
class KNNPrediction:
    """
    One out-of-fold kNN prediction.

    Neighbour metadata allows a prediction to be traced back
    to the exact training gestures that influenced it.
    """

    sample_id: str

    actual_label: str

    predicted_label: str

    neighbor_ids: tuple[str, ...]

    neighbor_labels: tuple[str, ...]

    neighbor_distances: tuple[float, ...]


@dataclass(frozen=True)
class KNNEvaluation:
    """
    Result of stratified cross-validation for one kNN
    configuration.
    """

    predictions: tuple[KNNPrediction, ...]

    accuracy: float

    labels: tuple[str, ...]

    confusion_matrix: IntArray


# ============================================================
# kNN cross-validation
# ============================================================


def evaluate_knn_cross_validation(
    dataset: MLDataset,
    n_neighbors: int,
    n_splits: int,
    weights: KNNWeights = "uniform",
    random_state: int = 42,
) -> KNNEvaluation:
    """
    Evaluate a k-nearest-neighbours classifier using stratified
    cross-validation.

    Every GestureSample is predicted only while it belongs to a
    held-out fold.

    The nearest-neighbour metadata therefore contains only
    examples from that fold's training data.
    """

    # ========================================================
    # Validate experiment configuration
    # ========================================================

    if n_neighbors <= 0:

        raise ValueError("n_neighbors must be " "greater than zero")

    if n_splits < 2:

        raise ValueError("n_splits must be " "at least 2")

    if weights not in (
        "uniform",
        "distance",
    ):

        raise ValueError("weights must be either " "'uniform' or 'distance'")

    # ========================================================
    # Validate class counts
    #
    # Stratified n-fold CV requires at least n examples in
    # every class.
    # ========================================================

    unique_labels, label_counts = np.unique(
        dataset.labels,
        return_counts=True,
    )

    minimum_class_count = int(label_counts.min())

    if minimum_class_count < n_splits:

        raise ValueError(
            "Stratified cross-validation "
            f"with {n_splits} folds requires "
            f"at least {n_splits} samples "
            "in every class"
        )

    # ========================================================
    # Preserve our domain's Spell ordering in reports.
    # ========================================================

    dataset_label_set = set(str(label) for label in unique_labels)

    labels = tuple(spell.value for spell in Spell if spell.value in dataset_label_set)

    # ========================================================
    # Cross-validation splitter
    # ========================================================

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    prediction_records: list[KNNPrediction] = []

    actual_labels: list[str] = []

    predicted_labels: list[str] = []

    # ========================================================
    # Cross-validation folds
    # ========================================================

    for train_indices, test_indices in splitter.split(
        dataset.features,
        dataset.labels,
    ):

        x_train = dataset.features[train_indices]

        y_train = dataset.labels[train_indices]

        x_test = dataset.features[test_indices]

        # ====================================================
        # Fit classifier
        # ====================================================

        classifier = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
        )

        classifier.fit(
            x_train,
            y_train,
        )

        # ====================================================
        # Predict held-out fold
        # ====================================================

        fold_predictions = classifier.predict(x_test)

        # ====================================================
        # Nearest-neighbour diagnostics
        # ====================================================

        (
            neighbor_distances,
            neighbor_positions,
        ) = classifier.kneighbors(
            x_test,
            n_neighbors=n_neighbors,
        )

        # ====================================================
        # Build one diagnostic record per held-out gesture
        # ====================================================

        for (
            test_position,
            dataset_index,
        ) in enumerate(test_indices):

            actual_label = str(dataset.labels[dataset_index])

            predicted_label = str(fold_predictions[test_position])

            # ------------------------------------------------
            # kneighbors() returns row positions relative to
            # X_train.
            #
            # Convert those positions back into indices from
            # the complete MLDataset.
            # ------------------------------------------------

            neighbor_dataset_indices = train_indices[neighbor_positions[test_position]]

            neighbor_ids = tuple(
                dataset.sample_ids[int(neighbor_index)]
                for neighbor_index in neighbor_dataset_indices
            )

            neighbor_labels = tuple(
                str(dataset.labels[int(neighbor_index)])
                for neighbor_index in neighbor_dataset_indices
            )

            distances = tuple(
                float(distance) for distance in neighbor_distances[test_position]
            )

            prediction_records.append(
                KNNPrediction(
                    sample_id=(dataset.sample_ids[int(dataset_index)]),
                    actual_label=(actual_label),
                    predicted_label=(predicted_label),
                    neighbor_ids=(neighbor_ids),
                    neighbor_labels=(neighbor_labels),
                    neighbor_distances=(distances),
                )
            )

            actual_labels.append(actual_label)

            predicted_labels.append(predicted_label)

    # ========================================================
    # Overall metrics
    # ========================================================

    accuracy = float(
        accuracy_score(
            actual_labels,
            predicted_labels,
        )
    )

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=list(labels),
    )

    return KNNEvaluation(
        predictions=tuple(prediction_records),
        accuracy=accuracy,
        labels=labels,
        confusion_matrix=matrix,
    )

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
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


FloatArray = NDArray[np.float64]


IntArray = NDArray[np.int64]


# ============================================================
# Open-set prediction
# ============================================================


@dataclass(frozen=True)
class OpenSetPrediction:
    """
    One held-out open-set prediction.

    nearest_known_* describes the genuine known-spell training
    sample that was closest to the test gesture.
    """

    sample_id: str

    actual_label: str

    predicted_label: str

    nearest_known_id: str

    nearest_known_label: str

    nearest_known_distance: float

    rejection_threshold: float

    @property
    def distance_ratio(
        self,
    ) -> float:
        """
        Distance relative to the threshold learned from the
        current training fold.

        ratio <= 1:
            inside known-spell distance range

        ratio > 1:
            reject
        """

        return self.nearest_known_distance / self.rejection_threshold


# ============================================================
# Complete evaluation result
# ============================================================


@dataclass(frozen=True)
class OpenSetEvaluation:

    predictions: tuple[OpenSetPrediction, ...]

    accuracy: float

    valid_accuracy: float

    invalid_recall: float

    macro_f1: float

    labels: tuple[str, ...]

    confusion_matrix: IntArray


# ============================================================
# Threshold learning
# ============================================================


def nearest_same_class_distances(
    features: FloatArray,
    labels: NDArray[np.str_],
) -> FloatArray:
    """
    For every training gesture, calculate the Euclidean
    distance to its nearest OTHER gesture from the same class.

    INVALID should not be included in this training data.
    """

    if features.shape[0] != labels.shape[0]:

        raise ValueError(
            "features and labels must contain " "the same number of samples"
        )

    distances: list[float] = []

    for label in np.unique(labels):

        class_features = features[labels == label]

        if class_features.shape[0] < 2:

            raise ValueError(
                "Every known spell needs at least "
                "2 training samples to learn "
                "a rejection threshold"
            )

        # ----------------------------------------------------
        # For each gesture in this class, calculate its
        # distance to every other gesture in the same class.
        # ----------------------------------------------------

        for index, feature_vector in enumerate(class_features):

            differences = class_features - feature_vector

            candidate_distances = np.linalg.norm(
                differences,
                axis=1,
            )

            # A sample has distance 0 to itself.
            #
            # Replace that value with infinity so it cannot
            # become its own nearest neighbour.
            candidate_distances[index] = np.inf

            nearest_distance = float(np.min(candidate_distances))

            distances.append(nearest_distance)

    return np.asarray(
        distances,
        dtype=np.float64,
    )


def learn_rejection_threshold(
    features: FloatArray,
    labels: NDArray[np.str_],
) -> float:
    """
    Learn a global known-spell rejection threshold using only
    training examples.

    The threshold is the largest nearest-same-class distance
    observed among the genuine spell training samples.

    This gives the first baseline rule:

        accept a new spell only if it is no farther away than
        the widest within-class variation seen during training.
    """

    distances = nearest_same_class_distances(
        features,
        labels,
    )

    if not np.isfinite(distances).all():

        raise ValueError("Threshold distances contain " "non-finite values")

    threshold = float(np.max(distances))

    if threshold <= 0.0:

        raise ValueError("Learned rejection threshold " "must be greater than zero")

    return threshold


# ============================================================
# Open-set cross-validation
# ============================================================


def evaluate_open_set_cross_validation(
    dataset: MLDataset,
    n_splits: int,
    random_state: int = 42,
) -> OpenSetEvaluation:
    """
    Evaluate known-spell recognition with distance-based
    rejection.

    INVALID samples participate in the stratified test folds,
    but are NEVER used to:

        - train the classifier
        - learn the rejection threshold
    """

    if n_splits < 2:

        raise ValueError("n_splits must be at least 2")

    # ========================================================
    # Validate that every class can participate in the
    # stratified split.
    # ========================================================

    unique_labels, label_counts = np.unique(
        dataset.labels,
        return_counts=True,
    )

    if int(label_counts.min()) < n_splits:

        raise ValueError("Every class must contain at least " f"{n_splits} samples")

    dataset_label_set = set(str(label) for label in unique_labels)

    labels = tuple(spell.value for spell in Spell if spell.value in dataset_label_set)

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    predictions: list[OpenSetPrediction] = []

    # ========================================================
    # CV folds
    # ========================================================

    for train_indices, test_indices in splitter.split(
        dataset.features,
        dataset.labels,
    ):

        # ----------------------------------------------------
        # Remove INVALID examples from training.
        # ----------------------------------------------------

        training_labels = dataset.labels[train_indices]

        known_training_mask = training_labels != Spell.INVALID.value

        known_train_indices = train_indices[known_training_mask]

        x_train = dataset.features[known_train_indices]

        y_train = dataset.labels[known_train_indices]

        # ----------------------------------------------------
        # Learn rejection boundary using genuine spells only.
        # ----------------------------------------------------

        threshold = learn_rejection_threshold(
            x_train,
            y_train,
        )

        # ----------------------------------------------------
        # Known-spell classifier.
        #
        # k=1 means the closest known gesture determines the
        # candidate spell.
        # ----------------------------------------------------

        classifier = KNeighborsClassifier(n_neighbors=1)

        classifier.fit(
            x_train,
            y_train,
        )

        x_test = dataset.features[test_indices]

        (
            nearest_distances,
            nearest_positions,
        ) = classifier.kneighbors(
            x_test,
            n_neighbors=1,
        )

        candidate_labels = classifier.predict(x_test)

        # ----------------------------------------------------
        # Apply rejection rule.
        # ----------------------------------------------------

        for test_position, dataset_index in enumerate(test_indices):

            nearest_distance = float(
                nearest_distances[
                    test_position,
                    0,
                ]
            )

            nearest_training_position = int(
                nearest_positions[
                    test_position,
                    0,
                ]
            )

            nearest_dataset_index = int(known_train_indices[nearest_training_position])

            candidate_label = str(candidate_labels[test_position])

            if nearest_distance > threshold:

                predicted_label = Spell.INVALID.value

            else:

                predicted_label = candidate_label

            predictions.append(
                OpenSetPrediction(
                    sample_id=(dataset.sample_ids[int(dataset_index)]),
                    actual_label=str(dataset.labels[dataset_index]),
                    predicted_label=(predicted_label),
                    nearest_known_id=(dataset.sample_ids[nearest_dataset_index]),
                    nearest_known_label=str(dataset.labels[nearest_dataset_index]),
                    nearest_known_distance=(nearest_distance),
                    rejection_threshold=(threshold),
                )
            )

    # ========================================================
    # Metrics
    # ========================================================

    actual_labels = [prediction.actual_label for prediction in predictions]

    predicted_labels = [prediction.predicted_label for prediction in predictions]

    accuracy = float(
        accuracy_score(
            actual_labels,
            predicted_labels,
        )
    )

    # --------------------------------------------------------
    # Valid-spell accuracy
    # --------------------------------------------------------

    valid_predictions = [
        prediction
        for prediction in predictions
        if (prediction.actual_label != Spell.INVALID.value)
    ]

    correct_valid = sum(
        prediction.actual_label == prediction.predicted_label
        for prediction in valid_predictions
    )

    valid_accuracy = correct_valid / len(valid_predictions)

    # --------------------------------------------------------
    # Invalid recall
    # --------------------------------------------------------

    invalid_predictions = [
        prediction
        for prediction in predictions
        if (prediction.actual_label == Spell.INVALID.value)
    ]

    correct_invalid = sum(
        prediction.predicted_label == Spell.INVALID.value
        for prediction in invalid_predictions
    )

    invalid_recall = correct_invalid / len(invalid_predictions)

    # --------------------------------------------------------
    # Macro F1
    # --------------------------------------------------------

    macro_f1 = float(
        f1_score(
            actual_labels,
            predicted_labels,
            labels=list(labels),
            average="macro",
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
        labels=list(labels),
    )

    return OpenSetEvaluation(
        predictions=tuple(predictions),
        accuracy=accuracy,
        valid_accuracy=float(valid_accuracy),
        invalid_recall=float(invalid_recall),
        macro_f1=macro_f1,
        labels=labels,
        confusion_matrix=matrix,
    )

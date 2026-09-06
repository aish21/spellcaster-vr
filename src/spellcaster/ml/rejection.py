from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    LeaveOneGroupOut,
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


StringArray = NDArray[np.str_]


IntArray = NDArray[np.int_]


# ============================================================
# Individual prediction
# ============================================================


@dataclass(frozen=True)
class OpenSetPrediction:
    """
    One held-out open-set prediction.

    nearest_known_* describes the genuine known-spell training
    sample that was closest to this test gesture.
    """

    sample_id: str

    session_id: str

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
        Distance relative to the rejection threshold learned
        from the current training fold.

        ratio <= 1:
            accepted as a known spell

        ratio > 1:
            rejected as INVALID
        """

        return self.nearest_known_distance / self.rejection_threshold


# ============================================================
# Overall evaluation result
# ============================================================


@dataclass(frozen=True)
class OpenSetEvaluation:
    """
    Aggregate metrics for a complete open-set evaluation.
    """

    predictions: tuple[OpenSetPrediction, ...]

    accuracy: float

    valid_accuracy: float

    invalid_recall: float

    macro_f1: float

    labels: tuple[str, ...]

    confusion_matrix: IntArray


# ============================================================
# Session-level metrics
# ============================================================


@dataclass(frozen=True)
class SessionMetrics:
    """
    Metrics for one completely held-out collection session.
    """

    session_id: str

    sample_count: int

    accuracy: float

    valid_accuracy: float

    invalid_recall: float

    macro_f1: float

    rejection_threshold: float


@dataclass(frozen=True)
class SessionOpenSetEvaluation:
    """
    Leave-one-session-out evaluation.

    overall:
        Metrics across every held-out prediction.

    sessions:
        Metrics calculated independently for each held-out
        session.
    """

    overall: OpenSetEvaluation

    sessions: tuple[SessionMetrics, ...]


# ============================================================
# Label ordering
# ============================================================


def _ordered_labels(
    labels: StringArray,
) -> tuple[str, ...]:
    """
    Preserve Spell enum ordering in reports and confusion
    matrices.
    """

    label_set = {str(label) for label in np.unique(labels)}

    return tuple(spell.value for spell in Spell if spell.value in label_set)


# ============================================================
# Threshold learning
# ============================================================


def nearest_same_class_distances(
    features: FloatArray,
    labels: StringArray,
) -> FloatArray:
    """
    For every known-spell training sample, calculate the
    Euclidean distance to its nearest OTHER training sample
    from the same class.

    INVALID examples must not be supplied here.
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

        for (
            index,
            feature_vector,
        ) in enumerate(class_features):

            differences = class_features - feature_vector

            candidate_distances = np.linalg.norm(
                differences,
                axis=1,
            )

            # Exclude the sample itself.
            candidate_distances[index] = np.inf

            nearest_distance = float(np.min(candidate_distances))

            distances.append(nearest_distance)

    return np.asarray(
        distances,
        dtype=np.float64,
    )


def learn_rejection_threshold(
    features: FloatArray,
    labels: StringArray,
) -> float:
    """
    Learn a global rejection threshold using genuine known
    spells from the TRAINING data only.

    Baseline rule:

        threshold =
            largest nearest-same-class training distance
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
# One train/test fold
# ============================================================


def _evaluate_open_set_fold(
    dataset: MLDataset,
    train_indices: NDArray[np.int_],
    test_indices: NDArray[np.int_],
) -> list[OpenSetPrediction]:
    """
    Train and evaluate one open-set train/test split.

    The splitting strategy is deliberately kept outside this
    function.

    This allows both:

        StratifiedKFold

    and:

        LeaveOneGroupOut

    to use exactly the same classification and rejection logic.
    """

    # ========================================================
    # Remove INVALID samples from training.
    # ========================================================

    training_labels = dataset.labels[train_indices]

    known_training_mask = training_labels != Spell.INVALID.value

    known_train_indices = train_indices[known_training_mask]

    if known_train_indices.size == 0:

        raise ValueError("Training fold contains no " "known-spell samples")

    x_train = dataset.features[known_train_indices]

    y_train = dataset.labels[known_train_indices]

    # ========================================================
    # Learn rejection boundary from training data only.
    # ========================================================

    threshold = learn_rejection_threshold(
        x_train,
        y_train,
    )

    # ========================================================
    # Known-spell classifier
    # ========================================================

    classifier = KNeighborsClassifier(n_neighbors=1)

    classifier.fit(
        x_train,
        y_train,
    )

    # ========================================================
    # Held-out samples
    # ========================================================

    x_test = dataset.features[test_indices]

    candidate_labels = classifier.predict(x_test)

    (
        nearest_distances,
        nearest_positions,
    ) = classifier.kneighbors(
        x_test,
        n_neighbors=1,
    )

    predictions: list[OpenSetPrediction] = []

    # ========================================================
    # Apply rejection rule
    # ========================================================

    for (
        test_position,
        dataset_index,
    ) in enumerate(test_indices):

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
                session_id=(dataset.session_ids[int(dataset_index)]),
                actual_label=str(dataset.labels[dataset_index]),
                predicted_label=(predicted_label),
                nearest_known_id=(dataset.sample_ids[nearest_dataset_index]),
                nearest_known_label=str(dataset.labels[nearest_dataset_index]),
                nearest_known_distance=(nearest_distance),
                rejection_threshold=(threshold),
            )
        )

    return predictions


# ============================================================
# Metric aggregation
# ============================================================


def _summarize_predictions(
    predictions: list[OpenSetPrediction],
    labels: tuple[str, ...],
) -> OpenSetEvaluation:
    """
    Convert a collection of held-out predictions into aggregate
    metrics.
    """

    if not predictions:

        raise ValueError("Cannot summarize zero predictions")

    actual_labels = [prediction.actual_label for prediction in predictions]

    predicted_labels = [prediction.predicted_label for prediction in predictions]

    # ========================================================
    # Valid-spell accuracy
    # ========================================================

    valid_predictions = [
        prediction
        for prediction in predictions
        if (prediction.actual_label != Spell.INVALID.value)
    ]

    if not valid_predictions:

        raise ValueError("Evaluation contains no " "valid-spell samples")

    correct_valid = sum(
        prediction.actual_label == prediction.predicted_label
        for prediction in valid_predictions
    )

    valid_accuracy = correct_valid / len(valid_predictions)

    # ========================================================
    # INVALID recall
    # ========================================================

    invalid_predictions = [
        prediction
        for prediction in predictions
        if (prediction.actual_label == Spell.INVALID.value)
    ]

    if not invalid_predictions:

        raise ValueError("Evaluation contains no " "INVALID samples")

    correct_invalid = sum(
        prediction.predicted_label == Spell.INVALID.value
        for prediction in invalid_predictions
    )

    invalid_recall = correct_invalid / len(invalid_predictions)

    # ========================================================
    # Standard metrics
    # ========================================================

    accuracy = float(
        accuracy_score(
            actual_labels,
            predicted_labels,
        )
    )

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


# ============================================================
# Ordinary stratified cross-validation
# ============================================================


def evaluate_open_set_cross_validation(
    dataset: MLDataset,
    n_splits: int,
    random_state: int = 42,
) -> OpenSetEvaluation:
    """
    Evaluate open-set recognition using ordinary stratified
    cross-validation.

    This remains useful as the controlled within-dataset
    baseline.
    """

    if n_splits < 2:

        raise ValueError("n_splits must be at least 2")

    unique_labels, label_counts = np.unique(
        dataset.labels,
        return_counts=True,
    )

    if int(label_counts.min()) < n_splits:

        raise ValueError("Every class must contain at least " f"{n_splits} samples")

    labels = _ordered_labels(dataset.labels)

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    predictions: list[OpenSetPrediction] = []

    for (
        train_indices,
        test_indices,
    ) in splitter.split(
        dataset.features,
        dataset.labels,
    ):

        predictions.extend(
            _evaluate_open_set_fold(
                dataset=dataset,
                train_indices=(train_indices),
                test_indices=(test_indices),
            )
        )

    return _summarize_predictions(
        predictions,
        labels,
    )


# ============================================================
# Session-aware validation
# ============================================================


def evaluate_open_set_leave_one_session_out(
    dataset: MLDataset,
) -> SessionOpenSetEvaluation:
    """
    Perform Leave-One-Session-Out open-set evaluation.

    Every collection session becomes the test set exactly once.

    No gesture from the held-out session may participate in:

        - known-spell classifier training
        - rejection threshold learning
    """

    if len(dataset.session_ids) != dataset.features.shape[0]:

        raise ValueError("session_ids must align with " "dataset feature rows")

    groups = np.asarray(
        dataset.session_ids,
        dtype=np.str_,
    )

    unique_sessions = np.unique(groups)

    if len(unique_sessions) < 2:

        raise ValueError(
            "Leave-one-session-out evaluation " "requires at least 2 sessions"
        )

    labels = _ordered_labels(dataset.labels)

    # ========================================================
    # Validate session coverage
    #
    # Our current collection protocol expects every session to
    # contain every class. This gives directly comparable
    # session-level metrics.
    # ========================================================

    required_labels = set(labels)

    for session_id in unique_sessions:

        session_mask = groups == session_id

        session_labels = {
            str(label) for label in np.unique(dataset.labels[session_mask])
        }

        missing_labels = required_labels - session_labels

        if missing_labels:

            missing_text = ", ".join(sorted(missing_labels))

            raise ValueError(
                f"Session {session_id} is missing " f"labels: {missing_text}"
            )

    # ========================================================
    # Leave-One-Group-Out
    # ========================================================

    splitter = LeaveOneGroupOut()

    all_predictions: list[OpenSetPrediction] = []

    session_results: list[SessionMetrics] = []

    for (
        train_indices,
        test_indices,
    ) in splitter.split(
        dataset.features,
        dataset.labels,
        groups=groups,
    ):

        held_out_sessions = np.unique(groups[test_indices])

        if len(held_out_sessions) != 1:

            raise RuntimeError(
                "LeaveOneGroupOut produced " "more than one held-out session"
            )

        held_out_session = str(held_out_sessions[0])

        fold_predictions = _evaluate_open_set_fold(
            dataset=dataset,
            train_indices=(train_indices),
            test_indices=(test_indices),
        )

        # ----------------------------------------------------
        # Ensure the group boundary really held.
        # ----------------------------------------------------

        if any(
            prediction.session_id != held_out_session for prediction in fold_predictions
        ):

            raise RuntimeError("Held-out predictions contain " "multiple sessions")

        fold_evaluation = _summarize_predictions(
            fold_predictions,
            labels,
        )

        # Every prediction in this fold used the same threshold.
        rejection_threshold = fold_predictions[0].rejection_threshold

        session_results.append(
            SessionMetrics(
                session_id=(held_out_session),
                sample_count=len(fold_predictions),
                accuracy=(fold_evaluation.accuracy),
                valid_accuracy=(fold_evaluation.valid_accuracy),
                invalid_recall=(fold_evaluation.invalid_recall),
                macro_f1=(fold_evaluation.macro_f1),
                rejection_threshold=(rejection_threshold),
            )
        )

        all_predictions.extend(fold_predictions)

    overall = _summarize_predictions(
        all_predictions,
        labels,
    )

    return SessionOpenSetEvaluation(
        overall=overall,
        sessions=tuple(session_results),
    )

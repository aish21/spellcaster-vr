from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from spellcaster.config import (
    CROSS_VALIDATION_FOLDS,
    INSPECTION_PLOTS_PATH,
    RAW_DATA_PATH,
    RESAMPLED_GESTURE_POINTS,
)
from spellcaster.gestures.repository import (
    GestureRepository,
)
from spellcaster.gestures.spells import Spell
from spellcaster.ml.dataset import (
    build_ml_dataset,
)
from spellcaster.ml.evaluation import (
    KNNEvaluation,
    KNNWeights,
    evaluate_knn_cross_validation,
)

# ============================================================
# Experiment search space
# ============================================================


NEIGHBOR_OPTIONS = (
    1,
    3,
    5,
)


WEIGHT_OPTIONS: tuple[KNNWeights, ...] = (
    "uniform",
    "distance",
)


# ============================================================
# Experiment result
# ============================================================


@dataclass(frozen=True)
class ExperimentResult:
    """
    Summary metrics for one kNN configuration.
    """

    n_neighbors: int

    weights: KNNWeights

    overall_accuracy: float

    valid_accuracy: float

    invalid_recall: float

    macro_f1: float


# ============================================================
# Metric calculation
# ============================================================


def summarize_evaluation(
    evaluation: KNNEvaluation,
    n_neighbors: int,
    weights: KNNWeights,
) -> ExperimentResult:
    """
    Calculate metrics that distinguish ordinary spell
    recognition from INVALID rejection.
    """

    predictions = evaluation.predictions

    # ========================================================
    # Overall labels
    # ========================================================

    actual_labels = [prediction.actual_label for prediction in predictions]

    predicted_labels = [prediction.predicted_label for prediction in predictions]

    # ========================================================
    # Valid-spell accuracy
    #
    # INVALID examples are deliberately excluded.
    # ========================================================

    valid_predictions = [
        prediction
        for prediction in predictions
        if (prediction.actual_label != Spell.INVALID.value)
    ]

    if not valid_predictions:

        raise ValueError(
            "Cannot calculate valid-spell " "accuracy without valid samples"
        )

    correct_valid = sum(
        prediction.actual_label == prediction.predicted_label
        for prediction in valid_predictions
    )

    valid_accuracy = correct_valid / len(valid_predictions)

    # ========================================================
    # INVALID recall
    #
    # Of all actual invalid gestures, how many were rejected
    # as INVALID?
    # ========================================================

    invalid_predictions = [
        prediction
        for prediction in predictions
        if (prediction.actual_label == Spell.INVALID.value)
    ]

    if not invalid_predictions:

        raise ValueError("Cannot calculate INVALID recall " "without INVALID samples")

    correct_invalid = sum(
        prediction.predicted_label == Spell.INVALID.value
        for prediction in invalid_predictions
    )

    invalid_recall = correct_invalid / len(invalid_predictions)

    # ========================================================
    # Macro F1
    #
    # Every class contributes equally.
    # ========================================================

    macro_f1 = float(
        f1_score(
            actual_labels,
            predicted_labels,
            labels=list(evaluation.labels),
            average="macro",
            zero_division=0,
        )
    )

    return ExperimentResult(
        n_neighbors=n_neighbors,
        weights=weights,
        overall_accuracy=(evaluation.accuracy),
        valid_accuracy=float(valid_accuracy),
        invalid_recall=float(invalid_recall),
        macro_f1=macro_f1,
    )


# ============================================================
# Terminal table
# ============================================================


def print_results(
    results: list[ExperimentResult],
) -> None:
    """
    Print the complete kNN comparison table.
    """

    print()

    print("=========================================================")

    print("kNN CONFIGURATION COMPARISON")

    print("=========================================================")

    print()

    print(
        f"{'k':>3}  "
        f"{'weights':<10}  "
        f"{'overall':>9}  "
        f"{'valid':>9}  "
        f"{'invalid':>9}  "
        f"{'macro F1':>9}"
    )

    print("-" * 61)

    for result in results:

        print(
            f"{result.n_neighbors:>3}  "
            f"{result.weights:<10}  "
            f"{result.overall_accuracy:>8.1%}  "
            f"{result.valid_accuracy:>8.1%}  "
            f"{result.invalid_recall:>8.1%}  "
            f"{result.macro_f1:>9.3f}"
        )

    print()


# ============================================================
# Experiment visualization
# ============================================================


def save_comparison_plot(
    results: list[ExperimentResult],
) -> None:
    """
    Save a grouped bar chart comparing the four evaluation
    metrics for every kNN configuration.
    """

    labels = [(f"k={result.n_neighbors}\n" f"{result.weights}") for result in results]

    overall_values = [result.overall_accuracy for result in results]

    valid_values = [result.valid_accuracy for result in results]

    invalid_values = [result.invalid_recall for result in results]

    macro_f1_values = [result.macro_f1 for result in results]

    x_positions = np.arange(len(results))

    bar_width = 0.20

    figure, axis = plt.subplots(figsize=(12, 7))

    axis.bar(
        x_positions - 1.5 * bar_width,
        overall_values,
        width=bar_width,
        label="Overall accuracy",
    )

    axis.bar(
        x_positions - 0.5 * bar_width,
        valid_values,
        width=bar_width,
        label="Valid-spell accuracy",
    )

    axis.bar(
        x_positions + 0.5 * bar_width,
        invalid_values,
        width=bar_width,
        label="Invalid recall",
    )

    axis.bar(
        x_positions + 1.5 * bar_width,
        macro_f1_values,
        width=bar_width,
        label="Macro F1",
    )

    axis.set_title("kNN configuration comparison")

    axis.set_ylabel("Score")

    axis.set_ylim(
        0.0,
        1.05,
    )

    axis.set_xticks(x_positions)

    axis.set_xticklabels(labels)

    axis.grid(
        True,
        axis="y",
    )

    axis.legend()

    figure.tight_layout()

    output_path = INSPECTION_PLOTS_PATH / "knn_comparison.png"

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(f"Saved comparison plot: " f"{output_path}")


# ============================================================
# Application
# ============================================================


def main() -> None:

    # ========================================================
    # Load raw gestures
    # ========================================================

    repository = GestureRepository(RAW_DATA_PATH)

    samples = repository.load_all()

    if not samples:

        print("Dataset contains no " "gesture samples.")

        return

    # ========================================================
    # Build the same X / y dataset used by every experiment.
    # ========================================================

    dataset = build_ml_dataset(
        samples,
        target_points=(RESAMPLED_GESTURE_POINTS),
    )

    print(f"Samples: " f"{dataset.features.shape[0]}")

    print(f"Features per sample: " f"{dataset.features.shape[1]}")

    print(f"CV folds: " f"{CROSS_VALIDATION_FOLDS}")

    # ========================================================
    # Evaluate every configuration.
    # ========================================================

    results: list[ExperimentResult] = []

    for weights in WEIGHT_OPTIONS:

        for n_neighbors in NEIGHBOR_OPTIONS:

            evaluation = evaluate_knn_cross_validation(
                dataset=dataset,
                n_neighbors=(n_neighbors),
                n_splits=(CROSS_VALIDATION_FOLDS),
                weights=weights,
                random_state=42,
            )

            result = summarize_evaluation(
                evaluation=evaluation,
                n_neighbors=n_neighbors,
                weights=weights,
            )

            results.append(result)

    # ========================================================
    # Report
    # ========================================================

    print_results(results)

    # ========================================================
    # Save comparison visualization
    # ========================================================

    INSPECTION_PLOTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_comparison_plot(results)

    plt.show()


if __name__ == "__main__":
    main()

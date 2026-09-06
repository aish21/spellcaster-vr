import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
)

from spellcaster.config import (
    CROSS_VALIDATION_FOLDS,
    INSPECTION_PLOTS_PATH,
    KNN_NEIGHBORS,
    RAW_DATA_PATH,
    RESAMPLED_GESTURE_POINTS,
)
from spellcaster.gestures.repository import (
    GestureRepository,
)
from spellcaster.ml.dataset import (
    build_ml_dataset,
)
from spellcaster.ml.evaluation import (
    KNNEvaluation,
    evaluate_knn_cross_validation,
)

# ============================================================
# Terminal reporting
# ============================================================


def print_evaluation_summary(
    evaluation: KNNEvaluation,
) -> None:
    """
    Print overall accuracy and per-class classification metrics.
    """

    actual_labels = [prediction.actual_label for prediction in evaluation.predictions]

    predicted_labels = [
        prediction.predicted_label for prediction in evaluation.predictions
    ]

    print()

    print("========================================")

    print("kNN CROSS-VALIDATION RESULTS")

    print("========================================")

    print()

    print(
        f"Accuracy: "
        f"{evaluation.accuracy:.3f} "
        f"({evaluation.accuracy * 100:.1f}%)"
    )

    print()

    print(
        classification_report(
            actual_labels,
            predicted_labels,
            labels=list(evaluation.labels),
            zero_division=0,
        )
    )


# ============================================================
# Misclassification inspection
# ============================================================


def print_misclassifications(
    evaluation: KNNEvaluation,
) -> None:
    """
    Print every incorrect held-out prediction together with the
    training gestures that were nearest to it.
    """

    mistakes = [
        prediction
        for prediction in evaluation.predictions
        if (prediction.actual_label != prediction.predicted_label)
    ]

    print("========================================")

    print("MISCLASSIFICATIONS")

    print("========================================")

    print()

    if not mistakes:

        print("No out-of-fold " "misclassifications.")

        print()

        return

    for prediction in mistakes:

        print(f"Sample: " f"{prediction.sample_id}")

        print(f"Actual: " f"{prediction.actual_label}")

        print(f"Predicted: " f"{prediction.predicted_label}")

        print("Nearest training gestures:")

        for (
            neighbor_id,
            neighbor_label,
            distance,
        ) in zip(
            prediction.neighbor_ids,
            prediction.neighbor_labels,
            prediction.neighbor_distances,
        ):

            print(
                f"  {neighbor_id[:8]}  "
                f"{neighbor_label:<14} "
                f"distance="
                f"{distance:.4f}"
            )

        print()


# ============================================================
# Confusion matrix
# ============================================================


def save_confusion_matrix(
    evaluation: KNNEvaluation,
) -> None:
    """
    Save and retain an interactive confusion-matrix figure.
    """

    figure, axis = plt.subplots(figsize=(8, 7))

    display = ConfusionMatrixDisplay(
        confusion_matrix=(evaluation.confusion_matrix),
        display_labels=(evaluation.labels),
    )

    display.plot(
        ax=axis,
        colorbar=False,
    )

    axis.set_title("kNN — 5-Fold Cross-Validation")

    figure.tight_layout()

    output_path = INSPECTION_PLOTS_PATH / "knn_confusion_matrix.png"

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(f"Saved confusion matrix: " f"{output_path}")


# ============================================================
# Application
# ============================================================


def main() -> None:

    # ========================================================
    # Load raw dataset
    # ========================================================

    repository = GestureRepository(RAW_DATA_PATH)

    samples = repository.load_all()

    if not samples:

        print("Dataset contains no " "gesture samples.")

        return

    # ========================================================
    # Build X / y
    # ========================================================

    dataset = build_ml_dataset(
        samples,
        target_points=(RESAMPLED_GESTURE_POINTS),
    )

    print(f"Samples: " f"{dataset.features.shape[0]}")

    print(f"Features per sample: " f"{dataset.features.shape[1]}")

    print(f"k: " f"{KNN_NEIGHBORS}")

    print(f"CV folds: " f"{CROSS_VALIDATION_FOLDS}")

    # ========================================================
    # Evaluate
    # ========================================================

    evaluation = evaluate_knn_cross_validation(
        dataset=dataset,
        n_neighbors=(KNN_NEIGHBORS),
        n_splits=(CROSS_VALIDATION_FOLDS),
    )

    # ========================================================
    # Terminal diagnostics
    # ========================================================

    print_evaluation_summary(evaluation)

    print_misclassifications(evaluation)

    # ========================================================
    # Confusion matrix
    # ========================================================

    INSPECTION_PLOTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_confusion_matrix(evaluation)

    plt.show()


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
)

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
from spellcaster.ml.rejection import (
    OpenSetEvaluation,
    evaluate_open_set_cross_validation,
)

# ============================================================
# Terminal summary
# ============================================================


def print_summary(
    evaluation: OpenSetEvaluation,
) -> None:

    actual_labels = [prediction.actual_label for prediction in evaluation.predictions]

    predicted_labels = [
        prediction.predicted_label for prediction in evaluation.predictions
    ]

    print()

    print("========================================")

    print("OPEN-SET SPELL RECOGNITION")

    print("========================================")

    print()

    print(f"Overall accuracy: " f"{evaluation.accuracy:.1%}")

    print(f"Valid-spell accuracy: " f"{evaluation.valid_accuracy:.1%}")

    print(f"Invalid recall: " f"{evaluation.invalid_recall:.1%}")

    print(f"Macro F1: " f"{evaluation.macro_f1:.3f}")

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
# Prediction diagnostics
# ============================================================


def print_predictions(
    evaluation: OpenSetEvaluation,
) -> None:

    print("========================================")

    print("DISTANCE / REJECTION DIAGNOSTICS")

    print("========================================")

    print()

    for prediction in evaluation.predictions:

        interesting = (
            prediction.actual_label == Spell.INVALID.value
            or prediction.actual_label != prediction.predicted_label
        )

        if not interesting:

            continue

        print(f"Sample: " f"{prediction.sample_id}")

        print(f"Actual: " f"{prediction.actual_label}")

        print(f"Nearest known spell: " f"{prediction.nearest_known_label}")

        print(f"Nearest sample: " f"{prediction.nearest_known_id[:8]}")

        print(f"Distance: " f"{prediction.nearest_known_distance:.4f}")

        print(f"Threshold: " f"{prediction.rejection_threshold:.4f}")

        print(f"Distance ratio: " f"{prediction.distance_ratio:.3f}")

        print(f"Final prediction: " f"{prediction.predicted_label}")

        print()


# ============================================================
# Confusion matrix
# ============================================================


def save_confusion_matrix(
    evaluation: OpenSetEvaluation,
) -> None:

    figure, axis = plt.subplots(figsize=(8, 7))

    display = ConfusionMatrixDisplay(
        confusion_matrix=(evaluation.confusion_matrix),
        display_labels=(evaluation.labels),
    )

    display.plot(
        ax=axis,
        colorbar=False,
    )

    axis.set_title("Open-Set Spell Recognition")

    figure.tight_layout()

    output_path = INSPECTION_PLOTS_PATH / "open_set_confusion_matrix.png"

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(f"Saved confusion matrix: " f"{output_path}")


# ============================================================
# Distance-ratio plot
# ============================================================


def save_distance_ratio_plot(
    evaluation: OpenSetEvaluation,
) -> None:
    """
    Plot each gesture's nearest-known distance divided by the
    rejection threshold learned in its CV fold.

    ratio > 1:
        rejected

    ratio <= 1:
        accepted as a known spell
    """

    valid_ratios = [
        prediction.distance_ratio
        for prediction in evaluation.predictions
        if (prediction.actual_label != Spell.INVALID.value)
    ]

    invalid_ratios = [
        prediction.distance_ratio
        for prediction in evaluation.predictions
        if (prediction.actual_label == Spell.INVALID.value)
    ]

    figure, axis = plt.subplots(figsize=(9, 6))

    axis.scatter(
        [0] * len(valid_ratios),
        valid_ratios,
        label="Valid spells",
    )

    axis.scatter(
        [1] * len(invalid_ratios),
        invalid_ratios,
        label="Invalid gestures",
    )

    # Rejection boundary.
    axis.axhline(
        1.0,
        linestyle="--",
        label="Rejection threshold",
    )

    axis.set_xticks(
        [
            0,
            1,
        ]
    )

    axis.set_xticklabels(
        [
            "Valid spells",
            "Invalid gestures",
        ]
    )

    axis.set_ylabel("Nearest-known distance / threshold")

    axis.set_title("Open-set distance ratios")

    axis.grid(
        True,
        axis="y",
    )

    axis.legend()

    figure.tight_layout()

    output_path = INSPECTION_PLOTS_PATH / "open_set_distance_ratios.png"

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(f"Saved distance-ratio plot: " f"{output_path}")


# ============================================================
# Application
# ============================================================


def main() -> None:

    repository = GestureRepository(RAW_DATA_PATH)

    samples = repository.load_all()

    if not samples:

        print("Dataset contains no gesture samples.")

        return

    dataset = build_ml_dataset(
        samples,
        target_points=(RESAMPLED_GESTURE_POINTS),
    )

    print(f"Samples: " f"{dataset.features.shape[0]}")

    print(f"Features per sample: " f"{dataset.features.shape[1]}")

    print(f"CV folds: " f"{CROSS_VALIDATION_FOLDS}")

    print("Training classes: " "fireball, lightning, shield, " "telekinesis")

    print("INVALID samples are evaluation-only.")

    evaluation = evaluate_open_set_cross_validation(
        dataset=dataset,
        n_splits=(CROSS_VALIDATION_FOLDS),
        random_state=42,
    )

    print_summary(evaluation)

    print_predictions(evaluation)

    INSPECTION_PLOTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_confusion_matrix(evaluation)

    save_distance_ratio_plot(evaluation)

    plt.show()


if __name__ == "__main__":
    main()

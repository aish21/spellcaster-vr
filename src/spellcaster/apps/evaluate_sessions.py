import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
)

from spellcaster.config import (
    INSPECTION_PLOTS_PATH,
    RAW_DATA_PATH,
    RESAMPLED_GESTURE_POINTS,
)
from spellcaster.gestures.repository import (
    GestureRepository,
)
from spellcaster.ml.dataset import (
    build_ml_dataset,
)
from spellcaster.ml.rejection import (
    SessionOpenSetEvaluation,
    evaluate_open_set_leave_one_session_out,
)

# ============================================================
# Display helpers
# ============================================================


def short_session_id(
    session_id: str,
) -> str:
    """
    Shorten UUID-style session IDs for terminal/plot display
    while keeping descriptive legacy IDs readable.
    """

    if len(session_id) <= 18:

        return session_id

    return session_id[:8]


# ============================================================
# Overall results
# ============================================================


def print_overall_summary(
    evaluation: SessionOpenSetEvaluation,
) -> None:

    overall = evaluation.overall

    actual_labels = [prediction.actual_label for prediction in overall.predictions]

    predicted_labels = [
        prediction.predicted_label for prediction in overall.predictions
    ]

    print()

    print("============================================================")

    print("LEAVE-ONE-SESSION-OUT OPEN-SET EVALUATION")

    print("============================================================")

    print()

    print(f"Overall accuracy: " f"{overall.accuracy:.1%}")

    print(f"Valid-spell accuracy: " f"{overall.valid_accuracy:.1%}")

    print(f"Invalid recall: " f"{overall.invalid_recall:.1%}")

    print(f"Macro F1: " f"{overall.macro_f1:.3f}")

    print()

    print(
        classification_report(
            actual_labels,
            predicted_labels,
            labels=list(overall.labels),
            zero_division=0,
        )
    )


# ============================================================
# Per-session results
# ============================================================


def print_session_summary(
    evaluation: SessionOpenSetEvaluation,
) -> None:

    print("============================================================")

    print("HELD-OUT SESSION RESULTS")

    print("============================================================")

    print()

    print(
        f"{'session':<20}"
        f"{'N':>5}"
        f"{'overall':>11}"
        f"{'valid':>11}"
        f"{'invalid':>11}"
        f"{'macro F1':>11}"
        f"{'threshold':>12}"
    )

    print("-" * 81)

    for session in evaluation.sessions:

        print(
            f"{short_session_id(session.session_id):<20}"
            f"{session.sample_count:>5}"
            f"{session.accuracy:>10.1%}"
            f"{session.valid_accuracy:>10.1%}"
            f"{session.invalid_recall:>10.1%}"
            f"{session.macro_f1:>11.3f}"
            f"{session.rejection_threshold:>12.4f}"
        )

    print()


# ============================================================
# Error diagnostics
# ============================================================


def print_misclassifications(
    evaluation: SessionOpenSetEvaluation,
) -> None:

    mistakes = [
        prediction
        for prediction in evaluation.overall.predictions
        if (prediction.actual_label != prediction.predicted_label)
    ]

    print("============================================================")

    print("MISCLASSIFICATIONS")

    print("============================================================")

    print()

    if not mistakes:

        print("No leave-one-session-out " "misclassifications.")

        print()

        return

    for prediction in mistakes:

        print(f"Sample: " f"{prediction.sample_id}")

        print(f"Session: " f"{prediction.session_id}")

        print(f"Actual: " f"{prediction.actual_label}")

        print(f"Predicted: " f"{prediction.predicted_label}")

        print(f"Nearest known spell: " f"{prediction.nearest_known_label}")

        print(f"Nearest known sample: " f"{prediction.nearest_known_id[:8]}")

        print(f"Distance: " f"{prediction.nearest_known_distance:.4f}")

        print(f"Threshold: " f"{prediction.rejection_threshold:.4f}")

        print(f"Distance ratio: " f"{prediction.distance_ratio:.3f}")

        print()


# ============================================================
# Confusion matrix
# ============================================================


def save_confusion_matrix(
    evaluation: SessionOpenSetEvaluation,
) -> None:

    overall = evaluation.overall

    figure, axis = plt.subplots(
        figsize=(
            8,
            7,
        )
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=(overall.confusion_matrix),
        display_labels=(overall.labels),
    )

    display.plot(
        ax=axis,
        colorbar=False,
    )

    axis.set_title("Leave-One-Session-Out " "Open-Set Recognition")

    figure.tight_layout()

    output_path = INSPECTION_PLOTS_PATH / ("session_open_set_" "confusion_matrix.png")

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(f"Saved confusion matrix: " f"{output_path}")


# ============================================================
# Per-session plot
# ============================================================


def save_session_metrics_plot(
    evaluation: SessionOpenSetEvaluation,
) -> None:

    sessions = list(evaluation.sessions)

    labels = [short_session_id(session.session_id) for session in sessions]

    overall_values = [session.accuracy for session in sessions]

    valid_values = [session.valid_accuracy for session in sessions]

    invalid_values = [session.invalid_recall for session in sessions]

    macro_f1_values = [session.macro_f1 for session in sessions]

    x_positions = np.arange(len(sessions))

    bar_width = 0.20

    figure, axis = plt.subplots(
        figsize=(
            12,
            7,
        )
    )

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

    axis.set_title("Performance on completely unseen sessions")

    axis.set_ylabel("Score")

    axis.set_ylim(
        0.0,
        1.05,
    )

    axis.set_xlabel("Held-out collection session")

    axis.set_xticks(x_positions)

    axis.set_xticklabels(labels)

    axis.grid(
        True,
        axis="y",
    )

    axis.legend()

    figure.tight_layout()

    output_path = INSPECTION_PLOTS_PATH / "session_open_set_metrics.png"

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(f"Saved session metrics: " f"{output_path}")


# ============================================================
# Application
# ============================================================


def main() -> None:

    # ========================================================
    # Load source data
    # ========================================================

    repository = GestureRepository(RAW_DATA_PATH)

    samples = repository.load_all()

    if not samples:

        print("Dataset contains no " "gesture samples.")

        return

    # ========================================================
    # Construct ML dataset
    # ========================================================

    dataset = build_ml_dataset(
        samples,
        target_points=(RESAMPLED_GESTURE_POINTS),
    )

    unique_sessions = sorted(set(dataset.session_ids))

    print(f"Samples: " f"{dataset.features.shape[0]}")

    print(f"Features per sample: " f"{dataset.features.shape[1]}")

    print(f"Collection sessions: " f"{len(unique_sessions)}")

    print()

    for session_id in unique_sessions:

        count = sum(
            stored_session == session_id for stored_session in dataset.session_ids
        )

        print(f"  " f"{short_session_id(session_id)}: " f"{count} samples")

    # ========================================================
    # LOGO is impossible with a single session.
    # ========================================================

    if len(unique_sessions) < 2:

        print()

        print(
            "Leave-one-session-out evaluation "
            "requires at least 2 collection sessions."
        )

        print("Collect another complete session, " "then run this app again.")

        return

    # ========================================================
    # Evaluate
    # ========================================================

    evaluation = evaluate_open_set_leave_one_session_out(dataset)

    # ========================================================
    # Reports
    # ========================================================

    print_overall_summary(evaluation)

    print_session_summary(evaluation)

    print_misclassifications(evaluation)

    # ========================================================
    # Saved diagnostics
    # ========================================================

    INSPECTION_PLOTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_confusion_matrix(evaluation)

    save_session_metrics_plot(evaluation)

    plt.show()


if __name__ == "__main__":
    main()

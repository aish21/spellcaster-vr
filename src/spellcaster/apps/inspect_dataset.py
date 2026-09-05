from statistics import mean, median

import matplotlib.pyplot as plt

from spellcaster.config import (
    INSPECTION_PLOTS_PATH,
    RAW_DATA_PATH,
)
from spellcaster.gestures.models import GestureSample
from spellcaster.gestures.repository import (
    GestureRepository,
)
from spellcaster.gestures.spells import Spell

# ============================================================
# Dataset grouping
# ============================================================


def group_by_spell(
    samples: list[GestureSample],
) -> dict[Spell, list[GestureSample]]:
    """
    Group GestureSamples by their spell label.

    Every Spell is included in the result, even when no samples
    currently exist for that spell.
    """

    grouped = {spell: [] for spell in Spell}

    for sample in samples:

        grouped[sample.spell].append(sample)

    return grouped


# ============================================================
# Statistics helpers
# ============================================================


def describe_values(
    values: list[int],
) -> tuple[str, str, str, str]:
    """
    Return printable minimum, median, mean, and maximum values.

    Strings are returned because this helper is currently used
    only for terminal presentation.
    """

    if not values:

        return (
            "-",
            "-",
            "-",
            "-",
        )

    return (
        str(min(values)),
        f"{median(values):.1f}",
        f"{mean(values):.1f}",
        str(max(values)),
    )


# ============================================================
# Terminal summary
# ============================================================


def print_dataset_summary(
    samples: list[GestureSample],
) -> None:
    """
    Print class counts plus duration and trajectory-length
    statistics.
    """

    grouped = group_by_spell(samples)

    print()

    print(f"Dataset: {RAW_DATA_PATH}")

    print(f"Total samples: " f"{len(samples)}")

    print()

    header = f"{'SPELL':<14}" f"{'N':>4}    " f"{'DURATION (ms)':<29}" f"{'POINTS'}"

    print(header)

    print(
        " " * 22
        + "min    median    mean    max"
        + "      "
        + "min    median    mean    max"
    )

    print("-" * 91)

    for spell in Spell:

        spell_samples = grouped[spell]

        durations = [sample.duration_ms for sample in spell_samples]

        point_counts = [len(sample.trajectory) for sample in spell_samples]

        (
            duration_min,
            duration_median,
            duration_mean,
            duration_max,
        ) = describe_values(durations)

        (
            points_min,
            points_median,
            points_mean,
            points_max,
        ) = describe_values(point_counts)

        print(
            f"{spell.value.upper():<14}"
            f"{len(spell_samples):>4}    "
            f"{duration_min:>4}    "
            f"{duration_median:>6}    "
            f"{duration_mean:>6}    "
            f"{duration_max:>4}"
            f"      "
            f"{points_min:>4}    "
            f"{points_median:>6}    "
            f"{points_mean:>6}    "
            f"{points_max:>4}"
        )

    print()


# ============================================================
# Plotting
# ============================================================


def plot_spell_samples(
    spell: Spell,
    samples: list[GestureSample],
) -> None:
    """
    Plot every raw trajectory belonging to one spell.

    Coordinates remain in original normalized camera space.

    No:
        - centering
        - scaling
        - resampling
        - smoothing

    is performed here.

    The resulting plot is both saved to disk and displayed
    interactively later by plt.show().
    """

    figure, axis = plt.subplots(figsize=(7, 7))

    # --------------------------------------------------------
    # Draw every sample belonging to this spell.
    # --------------------------------------------------------

    for sample in samples:

        x_values = [point.x for point in sample.trajectory]

        y_values = [point.y for point in sample.trajectory]

        short_id = sample.gesture_id[:8]

        axis.plot(
            x_values,
            y_values,
            marker="o",
            markersize=2,
            linewidth=1,
            label=short_id,
        )

    # --------------------------------------------------------
    # Plot metadata
    # --------------------------------------------------------

    axis.set_title(f"{spell.value.upper()} " f"— {len(samples)} raw samples")

    axis.set_xlabel("Normalized x")

    axis.set_ylabel("Normalized y")

    # --------------------------------------------------------
    # Preserve the complete original camera coordinate space.
    #
    # This is deliberate. At this stage we want to SEE
    # translation and scale differences between gestures.
    # --------------------------------------------------------

    axis.set_xlim(
        0.0,
        1.0,
    )

    # Camera/image coordinates have +y downward.
    #
    # Reverse the plot's y-axis so the gesture appears the same
    # way it appeared in the webcam.
    axis.set_ylim(
        1.0,
        0.0,
    )

    # Equal scaling prevents geometric distortion.
    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    axis.grid(True)

    if samples:

        axis.legend(
            title="Sample ID",
            fontsize=8,
        )

    figure.tight_layout()

    # --------------------------------------------------------
    # Save a persistent copy of this raw-data diagnostic.
    #
    # Examples:
    #
    # fireball_raw.png
    # lightning_raw.png
    # shield_raw.png
    # --------------------------------------------------------

    output_path = INSPECTION_PLOTS_PATH / f"{spell.value}_raw.png"

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(f"Saved plot: " f"{output_path}")


# ============================================================
# Application
# ============================================================


def main() -> None:

    repository = GestureRepository(RAW_DATA_PATH)

    samples = repository.load_all()

    # --------------------------------------------------------
    # No dataset yet.
    # --------------------------------------------------------

    if not samples:

        print("Dataset contains no gesture samples.")

        return

    # --------------------------------------------------------
    # Ensure generated plot directory exists.
    # --------------------------------------------------------

    INSPECTION_PLOTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Terminal statistics
    # --------------------------------------------------------

    print_dataset_summary(samples)

    # --------------------------------------------------------
    # Group once instead of repeatedly scanning the full
    # dataset for every plot.
    # --------------------------------------------------------

    grouped = group_by_spell(samples)

    # --------------------------------------------------------
    # Generate and save one raw-data plot per class.
    # --------------------------------------------------------

    for spell in Spell:

        plot_spell_samples(
            spell,
            grouped[spell],
        )

    print()

    print(f"Inspection plots saved to: " f"{INSPECTION_PLOTS_PATH}")

    # --------------------------------------------------------
    # The figures have already been written to disk.
    #
    # Now open the same figures interactively so we can zoom,
    # inspect individual samples, and visually compare them.
    # --------------------------------------------------------

    plt.show()


if __name__ == "__main__":
    main()

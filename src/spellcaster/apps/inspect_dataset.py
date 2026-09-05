from collections.abc import (
    Callable,
    Sequence,
)
from statistics import mean, median

import matplotlib.pyplot as plt

from spellcaster.config import (
    INSPECTION_PLOTS_PATH,
    RAW_DATA_PATH,
)
from spellcaster.gestures.models import (
    GestureSample,
    Point2D,
)
from spellcaster.gestures.repository import (
    GestureRepository,
)
from spellcaster.gestures.spells import Spell
from spellcaster.ml.preprocessing import (
    center_trajectory,
    normalize_translation_and_scale,
)

TrajectoryTransform = Callable[
    [Sequence[Point2D]],
    tuple[Point2D, ...],
]


def group_by_spell(
    samples: list[GestureSample],
) -> dict[Spell, list[GestureSample]]:

    grouped = {spell: [] for spell in Spell}

    for sample in samples:
        grouped[sample.spell].append(sample)

    return grouped


def describe_values(
    values: list[int],
) -> tuple[str, str, str, str]:

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


def print_dataset_summary(
    samples: list[GestureSample],
) -> None:

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

        duration_stats = describe_values(durations)

        point_stats = describe_values(point_counts)

        print(
            f"{spell.value.upper():<14}"
            f"{len(spell_samples):>4}    "
            f"{duration_stats[0]:>4}    "
            f"{duration_stats[1]:>6}    "
            f"{duration_stats[2]:>6}    "
            f"{duration_stats[3]:>4}"
            f"      "
            f"{point_stats[0]:>4}    "
            f"{point_stats[1]:>6}    "
            f"{point_stats[2]:>6}    "
            f"{point_stats[3]:>4}"
        )

    print()


def unchanged_trajectory(
    trajectory: Sequence[Point2D],
) -> tuple[Point2D, ...]:

    return tuple(trajectory)


def plot_spell_samples(
    spell: Spell,
    samples: list[GestureSample],
    transform: TrajectoryTransform,
    filename_suffix: str,
    representation_name: str,
    centered_coordinates: bool,
) -> None:

    figure, axis = plt.subplots(figsize=(7, 7))

    for sample in samples:

        trajectory = transform(sample.trajectory)

        x_values = [point.x for point in trajectory]

        y_values = [point.y for point in trajectory]

        short_id = sample.gesture_id[:8]

        axis.plot(
            x_values,
            y_values,
            marker="o",
            markersize=2,
            linewidth=1,
            label=short_id,
        )

    axis.set_title(
        f"{spell.value.upper()} " f"— {len(samples)} " f"{representation_name} samples"
    )

    axis.set_xlabel("x")

    axis.set_ylabel("y")

    if centered_coordinates:

        axis.set_xlim(
            -0.5,
            0.5,
        )

        axis.set_ylim(
            0.5,
            -0.5,
        )

        axis.axvline(
            0.0,
            linewidth=0.8,
        )

        axis.axhline(
            0.0,
            linewidth=0.8,
        )

    else:

        axis.set_xlim(
            0.0,
            1.0,
        )

        axis.set_ylim(
            1.0,
            0.0,
        )

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

    output_path = INSPECTION_PLOTS_PATH / (f"{spell.value}_" f"{filename_suffix}.png")

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    print(f"Saved plot: " f"{output_path}")


def main() -> None:

    repository = GestureRepository(RAW_DATA_PATH)

    samples = repository.load_all()

    if not samples:

        print("Dataset contains no gesture samples.")

        return

    INSPECTION_PLOTS_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_dataset_summary(samples)

    grouped = group_by_spell(samples)

    # ========================================================
    # RAW
    # ========================================================

    for spell in Spell:

        plot_spell_samples(
            spell=spell,
            samples=grouped[spell],
            transform=unchanged_trajectory,
            filename_suffix="raw",
            representation_name="raw",
            centered_coordinates=False,
        )

    # ========================================================
    # TRANSLATION NORMALIZED
    # ========================================================

    for spell in Spell:

        plot_spell_samples(
            spell=spell,
            samples=grouped[spell],
            transform=center_trajectory,
            filename_suffix="centered",
            representation_name="centered",
            centered_coordinates=True,
        )

    # ========================================================
    # TRANSLATION + SCALE NORMALIZED
    # ========================================================

    for spell in Spell:

        plot_spell_samples(
            spell=spell,
            samples=grouped[spell],
            transform=(normalize_translation_and_scale),
            filename_suffix="normalized",
            representation_name=("translation + scale normalized"),
            centered_coordinates=True,
        )

    print()

    print(f"Inspection plots saved to: " f"{INSPECTION_PLOTS_PATH}")

    plt.show()


if __name__ == "__main__":
    main()

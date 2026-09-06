import json

from pathlib import Path

from spellcaster.gestures.models import (
    GestureSample,
    Point2D,
)
from spellcaster.gestures.spells import Spell

SCHEMA_VERSION = 3

TRAJECTORY_REPRESENTATION = "raw_pre_ema"

LEGACY_SESSION_ID = "legacy-pilot-session"


# ============================================================
# Serialization
# ============================================================


def _sample_to_dict(
    sample: GestureSample,
) -> dict:

    return {
        "gesture_id": sample.gesture_id,
        "session_id": sample.session_id,
        "spell": sample.spell.value,
        "duration_ms": sample.duration_ms,
        "trajectory": [
            [
                point.x,
                point.y,
            ]
            for point in sample.trajectory
        ],
    }


def _sample_from_dict(
    data: dict,
) -> GestureSample:

    trajectory = tuple(
        Point2D(
            x=point[0],
            y=point[1],
        )
        for point in data["trajectory"]
    )

    return GestureSample(
        gesture_id=data["gesture_id"],
        session_id=data["session_id"],
        spell=Spell(data["spell"]),
        duration_ms=data["duration_ms"],
        trajectory=trajectory,
    )


# ============================================================
# Schema migration
# ============================================================


def _migrate_schema_2_sample(
    data: dict,
) -> dict:
    """
    Convert one schema-v2 sample into schema-v3 form.

    All samples from the original pilot dataset are deliberately
    assigned to the SAME legacy session. Assigning each sample
    its own session would falsely imply independent collection
    sessions during future group-aware evaluation.
    """

    migrated = dict(data)

    migrated["session_id"] = LEGACY_SESSION_ID

    return migrated


# ============================================================
# Repository
# ============================================================


class GestureRepository:

    def __init__(
        self,
        path: Path,
    ) -> None:

        self._path = path

    def load_all(
        self,
    ) -> list[GestureSample]:

        if not self._path.exists():

            return []

        with self._path.open(
            "r",
            encoding="utf-8",
        ) as file:

            document = json.load(file)

        schema_version = document.get("schema_version")

        trajectory_representation = document.get("trajectory_representation")

        if trajectory_representation != TRAJECTORY_REPRESENTATION:

            raise ValueError(
                "Unsupported trajectory "
                "representation: "
                f"{trajectory_representation}"
            )

        # ====================================================
        # Current schema
        # ====================================================

        if schema_version == SCHEMA_VERSION:

            sample_data = document.get(
                "samples",
                [],
            )

        # ====================================================
        # Schema 2 → 3 migration
        # ====================================================

        elif schema_version == 2:

            sample_data = [
                _migrate_schema_2_sample(sample)
                for sample in document.get(
                    "samples",
                    [],
                )
            ]

        # ====================================================
        # Unsupported schema
        # ====================================================

        else:

            raise ValueError(
                "Unsupported gesture dataset " "schema version: " f"{schema_version}"
            )

        return [_sample_from_dict(sample) for sample in sample_data]

    def save(
        self,
        sample: GestureSample,
    ) -> None:

        samples = self.load_all()

        samples.append(sample)

        self._write_all(samples)

    def count_by_spell(
        self,
    ) -> dict[Spell, int]:

        counts = {spell: 0 for spell in Spell}

        for sample in self.load_all():

            counts[sample.spell] += 1

        return counts

    def _write_all(
        self,
        samples: list[GestureSample],
    ) -> None:

        self._path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        document = {
            "schema_version": (SCHEMA_VERSION),
            "trajectory_representation": (TRAJECTORY_REPRESENTATION),
            "samples": [_sample_to_dict(sample) for sample in samples],
        }

        temporary_path = self._path.with_suffix(self._path.suffix + ".tmp")

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                document,
                file,
                indent=2,
            )

        temporary_path.replace(self._path)

import json
from pathlib import Path

from spellcaster.gestures.models import (
    GestureSample,
    Point2D,
)
from spellcaster.gestures.spells import Spell

SCHEMA_VERSION = 2


def _sample_to_dict(
    sample: GestureSample,
) -> dict:
    return {
        "gesture_id": sample.gesture_id,
        "spell": sample.spell.value,
        "duration_ms": sample.duration_ms,
        "trajectory": [[point.x, point.y] for point in sample.trajectory],
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
        spell=Spell(data["spell"]),
        duration_ms=data["duration_ms"],
        trajectory=trajectory,
    )


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

        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                "Unsupported gesture dataset " f"schema version: {schema_version}"
            )

        trajectory_representation = document.get("trajectory_representation")

        if trajectory_representation != "raw_pre_ema":
            raise ValueError(
                "Unsupported trajectory "
                "representation: "
                f"{trajectory_representation}"
            )

        return [_sample_from_dict(sample_data) for sample_data in document["samples"]]

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
            "schema_version": SCHEMA_VERSION,
            "trajectory_representation": "raw_pre_ema",
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

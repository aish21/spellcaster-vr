import json

from spellcaster.gestures.models import (
    GestureSample,
    Point2D,
)
from spellcaster.gestures.repository import GestureRepository, LEGACY_SESSION_ID
from spellcaster.gestures.spells import Spell

TEST_SESSION_ID = "test-session"


def test_missing_repository_is_empty(
    tmp_path,
):

    repository = GestureRepository(tmp_path / "gestures.json")

    assert repository.load_all() == []


def test_sample_can_be_saved_and_loaded(
    tmp_path,
):

    repository = GestureRepository(tmp_path / "gestures.json")

    original = GestureSample(
        gesture_id="sample-001",
        session_id=TEST_SESSION_ID,
        spell=Spell.FIREBALL,
        duration_ms=742,
        trajectory=(
            Point2D(0.1, 0.2),
            Point2D(0.3, 0.4),
            Point2D(0.5, 0.6),
        ),
    )

    repository.save(original)

    loaded = repository.load_all()

    assert loaded == [original]


def test_save_appends_samples(
    tmp_path,
):

    repository = GestureRepository(tmp_path / "gestures.json")

    first = GestureSample(
        gesture_id="sample-001",
        session_id="session-a",
        spell=Spell.FIREBALL,
        duration_ms=500,
        trajectory=(Point2D(0.1, 0.2),),
    )

    second = GestureSample(
        gesture_id="sample-002",
        session_id="session-b",
        spell=Spell.SHIELD,
        duration_ms=700,
        trajectory=(Point2D(0.3, 0.4),),
    )

    repository.save(first)

    repository.save(second)

    assert repository.load_all() == [
        first,
        second,
    ]


def test_count_by_spell(
    tmp_path,
):

    repository = GestureRepository(tmp_path / "gestures.json")

    repository.save(
        GestureSample(
            gesture_id="1",
            session_id=TEST_SESSION_ID,
            spell=Spell.FIREBALL,
            duration_ms=500,
            trajectory=(Point2D(0.1, 0.1),),
        )
    )

    repository.save(
        GestureSample(
            gesture_id="2",
            session_id=TEST_SESSION_ID,
            spell=Spell.FIREBALL,
            duration_ms=600,
            trajectory=(Point2D(0.2, 0.2),),
        )
    )

    repository.save(
        GestureSample(
            gesture_id="3",
            session_id=TEST_SESSION_ID,
            spell=Spell.SHIELD,
            duration_ms=700,
            trajectory=(Point2D(0.3, 0.3),),
        )
    )

    counts = repository.count_by_spell()

    assert counts[Spell.FIREBALL] == 2

    assert counts[Spell.SHIELD] == 1

    assert counts[Spell.LIGHTNING] == 0

    assert counts[Spell.TELEKINESIS] == 0

    assert counts[Spell.INVALID] == 0


def test_schema_2_dataset_is_migrated_to_legacy_session(
    tmp_path,
):

    path = tmp_path / "gestures.json"

    document = {
        "schema_version": 2,
        "trajectory_representation": ("raw_pre_ema"),
        "samples": [
            {
                "gesture_id": "legacy-1",
                "spell": "fireball",
                "duration_ms": 500,
                "trajectory": [
                    [
                        0.1,
                        0.2,
                    ],
                    [
                        0.3,
                        0.4,
                    ],
                ],
            },
        ],
    }

    path.write_text(
        json.dumps(document),
        encoding="utf-8",
    )

    repository = GestureRepository(path)

    samples = repository.load_all()

    assert len(samples) == 1

    assert samples[0].session_id == LEGACY_SESSION_ID

    assert samples[0].gesture_id == "legacy-1"

from spellcaster.gestures.repository import (
    GestureRepository,
)
from spellcaster.gestures.models import (
    GestureSample,
    Point2D,
)
from spellcaster.gestures.spells import Spell


def test_missing_dataset_returns_empty_list(
    tmp_path,
):
    repository = GestureRepository(tmp_path / "gestures.json")

    samples = repository.load_all()

    assert samples == []


def test_sample_can_be_saved_and_loaded(
    tmp_path,
):
    repository = GestureRepository(tmp_path / "gestures.json")

    original = GestureSample(
        gesture_id="sample-001",
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

    assert len(loaded) == 1
    assert loaded[0] == original


def test_save_appends_samples(
    tmp_path,
):
    repository = GestureRepository(tmp_path / "gestures.json")

    first = GestureSample(
        gesture_id="sample-001",
        spell=Spell.FIREBALL,
        duration_ms=500,
        trajectory=(Point2D(0.1, 0.2),),
    )

    second = GestureSample(
        gesture_id="sample-002",
        spell=Spell.SHIELD,
        duration_ms=700,
        trajectory=(Point2D(0.3, 0.4),),
    )

    repository.save(first)
    repository.save(second)

    loaded = repository.load_all()

    assert loaded == [
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
            spell=Spell.FIREBALL,
            duration_ms=500,
            trajectory=(Point2D(0.1, 0.1),),
        )
    )

    repository.save(
        GestureSample(
            gesture_id="2",
            spell=Spell.FIREBALL,
            duration_ms=600,
            trajectory=(Point2D(0.2, 0.2),),
        )
    )

    repository.save(
        GestureSample(
            gesture_id="3",
            spell=Spell.SHIELD,
            duration_ms=700,
            trajectory=(Point2D(0.3, 0.3),),
        )
    )

    counts = repository.count_by_spell()

    assert counts[Spell.FIREBALL] == 2
    assert counts[Spell.SHIELD] == 1
    assert counts[Spell.LIGHTNING] == 0

from spellcaster.config import RAW_DATA_PATH
from spellcaster.gestures.models import (
    GestureSample,
    Point2D,
)
from spellcaster.gestures.repository import (
    GestureRepository,
)
from spellcaster.gestures.spells import Spell

repository = GestureRepository(RAW_DATA_PATH)


sample = GestureSample(
    gesture_id="synthetic-demo-001",
    spell=Spell.FIREBALL,
    duration_ms=742,
    trajectory=(
        Point2D(0.2, 0.4),
        Point2D(0.3, 0.3),
        Point2D(0.4, 0.4),
    ),
)


repository.save(sample)


print("Saved sample.")

print(repository.load_all())

print(repository.count_by_spell())

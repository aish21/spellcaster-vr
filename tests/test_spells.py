from spellcaster.gestures.spells import Spell


def test_spell_values_are_stable():
    assert Spell.FIREBALL.value == "fireball"
    assert Spell.LIGHTNING.value == "lightning"
    assert Spell.SHIELD.value == "shield"
    assert Spell.TELEKINESIS.value == "telekinesis"
    assert Spell.INVALID.value == "invalid"

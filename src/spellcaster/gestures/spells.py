from enum import Enum


class Spell(str, Enum):
    FIREBALL = "fireball"
    LIGHTNING = "lightning"
    SHIELD = "shield"
    TELEKINESIS = "telekinesis"
    INVALID = "invalid"

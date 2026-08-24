from pathlib import Path

MODEL_PATH = Path("models/hand_landmarker.task")
RAW_DATA_PATH = Path("data/raw/gestures.json")

PINCH_START_RATIO = 0.35
PINCH_END_RATIO = 0.50

MIN_GESTURE_POINTS = 10
MIN_GESTURE_DURATION_MS = 200

MAX_LOST_FRAMES = 5

SMOOTHING_ALPHA = 0.35

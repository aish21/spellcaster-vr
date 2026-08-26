import time

import cv2

from spellcaster.config import (
    CASTING_HAND,
    MAX_LOST_FRAMES,
    MIN_GESTURE_DURATION_MS,
    MIN_GESTURE_POINTS,
    MODEL_PATH,
    NUM_HANDS,
    PINCH_END_CONFIRM_FRAMES,
    PINCH_END_RATIO,
    PINCH_START_CONFIRM_FRAMES,
    PINCH_START_RATIO,
    RAW_DATA_PATH,
    SMOOTHING_ALPHA,
)
from spellcaster.gestures.capture import (
    CaptureEvent,
    GestureCapture,
)
from spellcaster.gestures.repository import (
    GestureRepository,
)
from spellcaster.gestures.spells import Spell
from spellcaster.vision.hand_tracker import HandTracker
from spellcaster.vision.rendering import (
    draw_hand,
    draw_trajectory,
)

SPELL_KEYS = {
    ord("1"): Spell.FIREBALL,
    ord("2"): Spell.LIGHTNING,
    ord("3"): Spell.SHIELD,
    ord("4"): Spell.TELEKINESIS,
    ord("5"): Spell.INVALID,
}


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam")


tracker = HandTracker(
    model_path=str(MODEL_PATH),
    num_hands=NUM_HANDS,
    preferred_handedness=CASTING_HAND,
)


capture = GestureCapture(
    pinch_start_ratio=PINCH_START_RATIO,
    pinch_end_ratio=PINCH_END_RATIO,
    minimum_points=MIN_GESTURE_POINTS,
    minimum_duration_ms=MIN_GESTURE_DURATION_MS,
    max_lost_frames=MAX_LOST_FRAMES,
    smoothing_alpha=SMOOTHING_ALPHA,
    pinch_start_confirm_frames=(PINCH_START_CONFIRM_FRAMES),
    pinch_end_confirm_frames=(PINCH_END_CONFIRM_FRAMES),
)


repository = GestureRepository(RAW_DATA_PATH)


current_spell = Spell.FIREBALL

display_trajectory = ()


try:

    while True:

        # ----------------------------------------------------
        # Read / mirror frame
        # ----------------------------------------------------

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(
            frame,
            1,
        )

        # ----------------------------------------------------
        # Detect casting hand
        # ----------------------------------------------------

        observation = tracker.detect(frame)

        # ----------------------------------------------------
        # Update capture state
        # ----------------------------------------------------

        timestamp_ms = time.monotonic_ns() // 1_000_000

        result = capture.update(
            observation,
            timestamp_ms,
        )

        # ----------------------------------------------------
        # Draw hand
        # ----------------------------------------------------

        if observation is not None:

            draw_hand(
                frame,
                observation,
            )

        # ----------------------------------------------------
        # Handle capture events
        # ----------------------------------------------------

        if result.event == CaptureEvent.STARTED:

            print(f"Started " f"{current_spell.value}")

        elif result.event == CaptureEvent.COMPLETED:

            print(
                f"Completed "
                f"{current_spell.value}: "
                f"{len(result.trajectory)} points, "
                f"{result.duration_ms} ms"
            )

            display_trajectory = result.trajectory

        elif result.event == CaptureEvent.REJECTED:

            print(
                f"Rejected gesture: "
                f"{len(result.trajectory)} points, "
                f"{result.duration_ms} ms"
            )

            display_trajectory = ()

        elif result.event == CaptureEvent.CANCELLED:

            print("Gesture cancelled")

            display_trajectory = ()

        # ----------------------------------------------------
        # Live trajectory
        # ----------------------------------------------------

        if capture.is_pinching:

            display_trajectory = capture.current_trajectory

        draw_trajectory(
            frame,
            display_trajectory,
        )

        # ----------------------------------------------------
        # UI
        # ----------------------------------------------------

        cv2.putText(
            frame,
            (f"Spell: " f"{current_spell.value.upper()}"),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        if capture.is_pinching:

            capture_status = "CASTING"

        elif observation is None:

            capture_status = "NO CASTING HAND"

        else:

            capture_status = "READY"

        cv2.putText(
            frame,
            capture_status,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            (
                "1 Fireball | "
                "2 Lightning | "
                "3 Shield | "
                "4 Telekinesis | "
                "5 Invalid"
            ),
            (
                20,
                frame.shape[0] - 20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            "SpellCaster Gesture Collector",
            frame,
        )

        # ----------------------------------------------------
        # Keyboard
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key in SPELL_KEYS and not capture.is_pinching:

            current_spell = SPELL_KEYS[key]

            display_trajectory = ()

            print(f"Selected spell: " f"{current_spell.value}")


finally:

    tracker.close()

    camera.release()

    cv2.destroyAllWindows()

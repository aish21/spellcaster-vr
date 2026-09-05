import time
import uuid

from enum import Enum, auto

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
    TARGET_SAMPLES_PER_SPELL,
)
from spellcaster.gestures.capture import (
    CaptureEvent,
    GestureCapture,
)
from spellcaster.gestures.models import GestureSample
from spellcaster.gestures.repository import (
    GestureRepository,
)
from spellcaster.gestures.spells import Spell
from spellcaster.vision.hand_tracker import HandTracker
from spellcaster.vision.rendering import (
    draw_hand,
    draw_trajectory,
)

# ============================================================
# Collector state
# ============================================================


class CollectorState(Enum):
    COLLECTING = auto()
    REVIEW = auto()


# ============================================================
# Keyboard mappings
# ============================================================


SPELL_KEYS = {
    ord("1"): Spell.FIREBALL,
    ord("2"): Spell.LIGHTNING,
    ord("3"): Spell.SHIELD,
    ord("4"): Spell.TELEKINESIS,
    ord("5"): Spell.INVALID,
}


# ============================================================
# Collector UI helpers
# ============================================================


def draw_dataset_counts(
    frame,
    sample_counts: dict[Spell, int],
    target: int,
) -> None:

    x = frame.shape[1] - 210
    y = 40

    cv2.putText(
        frame,
        "DATASET",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    y += 30

    for spell in Spell:

        count = sample_counts[spell]

        text = f"{spell.value[:12].upper()}: " f"{count}/{target}"

        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        y += 25


# ============================================================
# Camera setup
# ============================================================


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam")


# ============================================================
# SpellCaster components
# ============================================================


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


sample_counts = repository.count_by_spell()


# ============================================================
# Collector application state
# ============================================================


current_spell = Spell.FIREBALL

collector_state = CollectorState.COLLECTING


# Smoothed trajectory shown to the user.
display_trajectory = ()


# Raw data waiting to become a GestureSample.
pending_trajectory = ()

pending_duration_ms: int | None = None

pending_spell: Spell | None = None


# ============================================================
# Main application
# ============================================================


try:

    while True:

        # ====================================================
        # 1. Read camera frame
        # ====================================================

        success, frame = camera.read()

        if not success:
            break

        frame = cv2.flip(
            frame,
            1,
        )

        # ====================================================
        # 2. Detect designated casting hand
        # ====================================================

        observation = tracker.detect(frame)

        # ====================================================
        # 3. Update GestureCapture
        # ====================================================

        timestamp_ms = time.monotonic_ns() // 1_000_000

        if collector_state == CollectorState.COLLECTING:

            result = capture.update(
                observation,
                timestamp_ms,
            )

        else:

            result = None

        # ====================================================
        # 4. Draw detected hand
        # ====================================================

        if observation is not None:

            draw_hand(
                frame,
                observation,
            )

        # ====================================================
        # 5. Handle capture events
        # ====================================================

        if result is not None:

            if result.event == CaptureEvent.STARTED:

                print(f"Started " f"{current_spell.value}")

            elif result.event == CaptureEvent.COMPLETED:

                # --------------------------------------------
                # Persistable representation:
                #
                # RAW pre-EMA trajectory.
                # --------------------------------------------

                pending_trajectory = result.trajectory

                pending_duration_ms = result.duration_ms

                pending_spell = current_spell

                # --------------------------------------------
                # Presentation representation:
                #
                # EMA-smoothed trajectory.
                # --------------------------------------------

                display_trajectory = result.smoothed_trajectory

                collector_state = CollectorState.REVIEW

                print(
                    f"Reviewing "
                    f"{pending_spell.value}: "
                    f"{len(pending_trajectory)} points, "
                    f"{pending_duration_ms} ms"
                )

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

        # ====================================================
        # 6. Display live smoothed trajectory
        # ====================================================

        if collector_state == CollectorState.COLLECTING and capture.is_pinching:

            display_trajectory = capture.current_trajectory

        # ====================================================
        # 7. Draw trajectory
        # ====================================================

        draw_trajectory(
            frame,
            display_trajectory,
        )

        # ====================================================
        # 8. Draw selected spell
        # ====================================================

        cv2.putText(
            frame,
            (f"Spell: " f"{current_spell.value.upper()}"),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # ====================================================
        # 9. Determine status
        # ====================================================

        if collector_state == CollectorState.REVIEW:

            capture_status = "REVIEW - S SAVE | R REJECT"

        elif capture.is_pinching:

            capture_status = "CASTING"

        elif observation is None:

            capture_status = "NO CASTING HAND"

        else:

            capture_status = "READY"

        # ====================================================
        # 10. Draw status
        # ====================================================

        cv2.putText(
            frame,
            capture_status,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # ====================================================
        # 11. Draw selected-spell progress
        # ====================================================

        current_count = sample_counts[current_spell]

        if current_count >= TARGET_SAMPLES_PER_SPELL:

            progress_text = (
                f"Samples: "
                f"{current_count} / "
                f"{TARGET_SAMPLES_PER_SPELL} "
                f"- TARGET REACHED"
            )

        else:

            progress_text = (
                f"Samples: " f"{current_count} / " f"{TARGET_SAMPLES_PER_SPELL}"
            )

        cv2.putText(
            frame,
            progress_text,
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        # ====================================================
        # 12. Draw review metadata
        # ====================================================

        if collector_state == CollectorState.REVIEW:

            if pending_spell is not None and pending_duration_ms is not None:

                review_text = (
                    f"{pending_spell.value.upper()} | "
                    f"{len(pending_trajectory)} points | "
                    f"{pending_duration_ms} ms"
                )

                cv2.putText(
                    frame,
                    review_text,
                    (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        # ====================================================
        # 13. Draw full dataset counts
        # ====================================================

        draw_dataset_counts(
            frame,
            sample_counts,
            TARGET_SAMPLES_PER_SPELL,
        )

        # ====================================================
        # 14. Draw controls
        # ====================================================

        cv2.putText(
            frame,
            (
                "1 Fireball | "
                "2 Lightning | "
                "3 Shield | "
                "4 Telekinesis | "
                "5 Invalid | "
                "S Save | "
                "R Reject"
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

        # ====================================================
        # 15. Show frame
        # ====================================================

        cv2.imshow(
            "SpellCaster Gesture Collector",
            frame,
        )

        # ====================================================
        # 16. Read keyboard input
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # ====================================================
        # 17. Spell selection
        # ====================================================

        if (
            key in SPELL_KEYS
            and collector_state == CollectorState.COLLECTING
            and not capture.is_pinching
        ):

            current_spell = SPELL_KEYS[key]

            display_trajectory = ()

            print(f"Selected spell: " f"{current_spell.value}")

        # ====================================================
        # 18. Reject
        # ====================================================

        if key == ord("r") and collector_state == CollectorState.REVIEW:

            if pending_spell is not None:

                print(f"Rejected " f"{pending_spell.value} sample")

            pending_trajectory = ()

            pending_duration_ms = None

            pending_spell = None

            display_trajectory = ()

            collector_state = CollectorState.COLLECTING

        # ====================================================
        # 19. Save
        # ====================================================

        if key == ord("s") and collector_state == CollectorState.REVIEW:

            if (
                pending_spell is None
                or pending_duration_ms is None
                or not pending_trajectory
            ):

                raise RuntimeError(
                    "Collector entered REVIEW without " "a complete pending gesture"
                )

            sample = GestureSample(
                gesture_id=str(uuid.uuid4()),
                spell=pending_spell,
                duration_ms=(pending_duration_ms),
                trajectory=(pending_trajectory),
            )

            repository.save(sample)

            sample_counts[sample.spell] += 1

            print(f"Saved " f"{sample.spell.value} sample " f"{sample.gesture_id}")

            print(
                f"{sample.spell.value}: "
                f"{sample_counts[sample.spell]} / "
                f"{TARGET_SAMPLES_PER_SPELL}"
            )

            pending_trajectory = ()

            pending_duration_ms = None

            pending_spell = None

            display_trajectory = ()

            collector_state = CollectorState.COLLECTING


# ============================================================
# Cleanup
# ============================================================

finally:

    tracker.close()

    camera.release()

    cv2.destroyAllWindows()

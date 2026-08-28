import time

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

# ============================================================
# Collector state
# ============================================================


class CollectorState(Enum):
    """
    High-level state of the gesture collection application.

    COLLECTING:
        The user is free to perform a new gesture.

    REVIEW:
        A technically valid gesture has completed and is waiting
        for the user to save or reject it.
    """

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
# Camera setup
# ============================================================


camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam")


# ============================================================
# RuneCaster components
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


# We create the repository now because the collector owns the
# persistence dependency.
#
# It is intentionally NOT used to save anything during this step.
repository = GestureRepository(RAW_DATA_PATH)


# ============================================================
# Collector application state
# ============================================================


# Label currently selected by the user.
current_spell = Spell.FIREBALL


# Application initially accepts gestures.
collector_state = CollectorState.COLLECTING


# ------------------------------------------------------------
# Display state
#
# This controls what trajectory is currently visible.
#
# While casting:
#     live GestureCapture trajectory
#
# During review:
#     completed trajectory
#
# After rejection/cancellation:
#     empty
# ------------------------------------------------------------

display_trajectory = ()


# ------------------------------------------------------------
# Pending sample state
#
# When GestureCapture returns COMPLETED, we snapshot the
# gesture here and enter REVIEW.
#
# It is NOT yet persisted.
# ------------------------------------------------------------

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

        # ----------------------------------------------------
        # Mirror the webcam.
        #
        # This makes interaction feel like looking in a mirror:
        #
        # move right -> trail moves right
        # move left  -> trail moves left
        #
        # We flip BEFORE detection so rendered coordinates and
        # detected coordinates use the same coordinate system.
        # ----------------------------------------------------

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

        # GestureCapture is deliberately paused during REVIEW.
        #
        # Otherwise an accidental pinch while inspecting the
        # previous gesture could begin another capture internally.
        if collector_state == CollectorState.COLLECTING:

            result = capture.update(
                observation,
                timestamp_ms,
            )

        else:

            result = None

        # ====================================================
        # 4. Render detected hand
        # ====================================================

        if observation is not None:

            draw_hand(
                frame,
                observation,
            )

        # ====================================================
        # 5. Handle GestureCapture events
        # ====================================================

        if result is not None:

            # ------------------------------------------------
            # Gesture started
            # ------------------------------------------------

            if result.event == CaptureEvent.STARTED:

                print(f"Started " f"{current_spell.value}")

            # ------------------------------------------------
            # Valid gesture completed
            #
            # Important:
            #
            # COMPLETED does NOT mean SAVED.
            #
            # GestureCapture has only decided that the gesture
            # satisfies technical requirements such as duration
            # and point count.
            #
            # Human review happens next.
            # ------------------------------------------------

            elif result.event == CaptureEvent.COMPLETED:

                pending_trajectory = result.trajectory

                pending_duration_ms = result.duration_ms

                # Snapshot the label NOW.
                #
                # This guarantees that the future sample keeps
                # the spell selected when the gesture was drawn.
                pending_spell = current_spell

                display_trajectory = result.trajectory

                collector_state = CollectorState.REVIEW

                print(
                    f"Reviewing "
                    f"{pending_spell.value}: "
                    f"{len(pending_trajectory)} points, "
                    f"{pending_duration_ms} ms"
                )

            # ------------------------------------------------
            # Technically invalid gesture
            #
            # Never enters REVIEW.
            # ------------------------------------------------

            elif result.event == CaptureEvent.REJECTED:

                print(
                    f"Rejected gesture: "
                    f"{len(result.trajectory)} points, "
                    f"{result.duration_ms} ms"
                )

                display_trajectory = ()

            # ------------------------------------------------
            # Tracking was lost for too long
            #
            # Never enters REVIEW.
            # ------------------------------------------------

            elif result.event == CaptureEvent.CANCELLED:

                print("Gesture cancelled")

                display_trajectory = ()

        # ====================================================
        # 6. Display live trajectory while actively casting
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
        # 9. Determine application status
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
        # 10. Draw application status
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
        # 11. Draw REVIEW metadata
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
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        # ====================================================
        # 12. Draw controls
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
        # 13. Display frame
        # ====================================================

        cv2.imshow(
            "SpellCaster Gesture Collector",
            frame,
        )

        # ====================================================
        # 14. Read keyboard input
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        if key == ord("q"):
            break

        # ====================================================
        # 15. Spell selection
        #
        # Only available while:
        #
        # - COLLECTING
        # - not currently casting
        #
        # This prevents accidental relabeling.
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
        # 16. Reject reviewed gesture
        # ====================================================

        if key == ord("r") and collector_state == CollectorState.REVIEW:

            if pending_spell is not None:

                print(f"Rejected " f"{pending_spell.value} sample")

            # Clear every piece of pending sample state.
            pending_trajectory = ()

            pending_duration_ms = None

            pending_spell = None

            display_trajectory = ()

            collector_state = CollectorState.COLLECTING

        # ====================================================
        # 17. Save placeholder
        #
        # Persistence deliberately arrives in Step 9D.
        # ====================================================

        if key == ord("s") and collector_state == CollectorState.REVIEW:

            print("Save not implemented yet")


# ============================================================
# Cleanup
# ============================================================

finally:

    tracker.close()

    camera.release()

    cv2.destroyAllWindows()

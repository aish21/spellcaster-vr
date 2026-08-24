# import math
# import time

# import cv2
# import mediapipe as mp

# # ============================================================
# # Configuration
# # ============================================================

# MODEL_PATH = "models/hand_landmarker.task"

# # We now use:
# #
# #     thumb-index distance
# #     --------------------
# #          hand size
# #
# # instead of the raw thumb-index distance.
# #
# # These values are starting points and may need calibration
# # for your hand/camera.
# PINCH_START_RATIO = 0.35
# PINCH_END_RATIO = 0.50

# # Reject extremely short accidental gestures.
# MIN_GESTURE_POINTS = 10
# MIN_GESTURE_DURATION_MS = 200

# # Allow MediaPipe to temporarily lose the hand for a few
# # frames without immediately cancelling the spell.
# MAX_LOST_FRAMES = 5

# # Exponential moving average smoothing.
# #
# # 1.0 -> no smoothing
# # smaller -> smoother but more lag
# SMOOTHING_ALPHA = 0.35


# HAND_CONNECTIONS = [
#     # Thumb
#     (0, 1),
#     (1, 2),
#     (2, 3),
#     (3, 4),
#     # Index
#     (0, 5),
#     (5, 6),
#     (6, 7),
#     (7, 8),
#     # Middle
#     (0, 9),
#     (9, 10),
#     (10, 11),
#     (11, 12),
#     # Ring
#     (0, 13),
#     (13, 14),
#     (14, 15),
#     (15, 16),
#     # Pinky
#     (0, 17),
#     (17, 18),
#     (18, 19),
#     (19, 20),
#     # Palm
#     (5, 9),
#     (9, 13),
#     (13, 17),
# ]


# # ============================================================
# # MediaPipe setup
# # ============================================================

# BaseOptions = mp.tasks.BaseOptions
# HandLandmarker = mp.tasks.vision.HandLandmarker
# HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# VisionRunningMode = mp.tasks.vision.RunningMode


# options = HandLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=MODEL_PATH),
#     running_mode=VisionRunningMode.VIDEO,
#     num_hands=1,
# )


# # ============================================================
# # Main application
# # ============================================================

# with HandLandmarker.create_from_options(options) as landmarker:

#     camera = cv2.VideoCapture(0)

#     if not camera.isOpened():
#         raise RuntimeError("Could not open webcam")

#     # ========================================================
#     # Persistent application state
#     #
#     # These variables survive between camera frames.
#     # ========================================================

#     was_pinching = False

#     trajectory = []

#     cast_start_time_ms = None

#     lost_frames = 0

#     # Previous smoothed fingertip position.
#     #
#     # Stored as:
#     # (x, y)
#     #
#     # using normalized coordinates.
#     smoothed_index = None

#     previous_frame_time = time.perf_counter()

#     while True:

#         # ====================================================
#         # 1. Read webcam frame
#         # ====================================================

#         success, frame = camera.read()

#         if not success:
#             break

#         height, width, _ = frame.shape

#         # ====================================================
#         # 2. Convert BGR -> RGB
#         # ====================================================

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # ====================================================
#         # 3. Convert OpenCV image -> MediaPipe image
#         # ====================================================

#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

#         # ====================================================
#         # 4. Timestamp
#         # ====================================================

#         timestamp_ms = time.monotonic_ns() // 1_000_000

#         # ====================================================
#         # 5. Hand landmark inference
#         # ====================================================

#         result = landmarker.detect_for_video(mp_image, timestamp_ms)

#         # ====================================================
#         # 6. Hand detected
#         # ====================================================

#         if result.hand_landmarks:

#             # Hand has reappeared, so reset the missing-frame
#             # counter.
#             lost_frames = 0

#             hand = result.hand_landmarks[0]

#             # =================================================
#             # Convert normalized landmarks -> pixels
#             # =================================================

#             points = []

#             for landmark in hand:

#                 x = int(landmark.x * width)
#                 y = int(landmark.y * height)

#                 points.append((x, y))

#             # =================================================
#             # Draw skeleton
#             # =================================================

#             for start_index, end_index in HAND_CONNECTIONS:

#                 cv2.line(
#                     frame, points[start_index], points[end_index], (255, 255, 255), 2
#                 )

#             # =================================================
#             # Draw landmark points
#             # =================================================

#             for point in points:

#                 cv2.circle(frame, point, 5, (0, 255, 0), -1)

#             # =================================================
#             # Important landmarks
#             # =================================================

#             wrist = hand[0]

#             thumb_tip = hand[4]

#             index_tip = hand[8]

#             middle_mcp = hand[9]

#             # Highlight index fingertip.
#             cv2.circle(frame, points[8], 10, (0, 0, 255), -1)

#             # =================================================
#             # 7. Calculate raw pinch distance
#             # =================================================

#             pinch_distance = math.hypot(
#                 thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y
#             )

#             # =================================================
#             # 8. Estimate hand scale
#             #
#             # Wrist -> middle-finger MCP gives us an approximate
#             # measure of how large the hand currently appears.
#             #
#             # As the hand moves closer to the webcam:
#             #
#             # pinch distance grows
#             # AND
#             # hand scale grows
#             #
#             # Dividing them reduces sensitivity to camera
#             # distance.
#             # =================================================

#             hand_scale = math.hypot(wrist.x - middle_mcp.x, wrist.y - middle_mcp.y)

#             # Avoid division by zero in the extremely unlikely
#             # case that MediaPipe returns overlapping points.
#             if hand_scale > 1e-6:

#                 pinch_ratio = pinch_distance / hand_scale

#             else:

#                 pinch_ratio = float("inf")

#             # =================================================
#             # 9. Hysteresis
#             #
#             # Starting and ending a pinch use DIFFERENT
#             # thresholds.
#             #
#             # This prevents noisy values near one threshold from
#             # rapidly producing:
#             #
#             # PINCH -> OPEN -> PINCH -> OPEN
#             # =================================================

#             if was_pinching:

#                 # Once we're already pinching, tolerate more
#                 # separation before ending the pinch.
#                 is_pinching = pinch_ratio < PINCH_END_RATIO

#             else:

#                 # Starting a pinch requires the fingers to be
#                 # closer together.
#                 is_pinching = pinch_ratio < PINCH_START_RATIO

#             # =================================================
#             # Display detection information
#             # =================================================

#             status = "PINCH" if is_pinching else "OPEN"

#             cv2.putText(
#                 frame,
#                 (f"{status} " f"raw={pinch_distance:.3f} " f"ratio={pinch_ratio:.3f}"),
#                 (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (0, 255, 0),
#                 2,
#             )

#             # =================================================
#             # 10. Casting state machine
#             # =================================================

#             # -------------------------------------------------
#             # OPEN -> PINCH
#             #
#             # A new cast begins.
#             # -------------------------------------------------

#             if is_pinching and not was_pinching:

#                 cast_start_time_ms = timestamp_ms

#                 # The first point doesn't need smoothing because
#                 # we don't have any previous point yet.
#                 smoothed_index = (index_tip.x, index_tip.y)

#                 trajectory = [smoothed_index]

#                 print("CAST START")

#             # -------------------------------------------------
#             # PINCH -> PINCH
#             #
#             # Cast continues.
#             # -------------------------------------------------

#             elif is_pinching and was_pinching:

#                 current_x = index_tip.x
#                 current_y = index_tip.y

#                 # =============================================
#                 # Exponential moving average smoothing
#                 #
#                 # smoothed =
#                 #     alpha * current
#                 #   + (1-alpha) * previous
#                 # =============================================

#                 if smoothed_index is None:

#                     smoothed_index = (current_x, current_y)

#                 else:

#                     previous_x, previous_y = smoothed_index

#                     smoothed_x = (
#                         SMOOTHING_ALPHA * current_x + (1 - SMOOTHING_ALPHA) * previous_x
#                     )

#                     smoothed_y = (
#                         SMOOTHING_ALPHA * current_y + (1 - SMOOTHING_ALPHA) * previous_y
#                     )

#                     smoothed_index = (smoothed_x, smoothed_y)

#                 trajectory.append(smoothed_index)

#             # -------------------------------------------------
#             # PINCH -> OPEN
#             #
#             # User released their fingers.
#             # -------------------------------------------------

#             elif not is_pinching and was_pinching:

#                 if cast_start_time_ms is None:

#                     duration_ms = 0

#                 else:

#                     duration_ms = timestamp_ms - cast_start_time_ms

#                 # =============================================
#                 # Gesture validation
#                 # =============================================

#                 enough_points = len(trajectory) >= MIN_GESTURE_POINTS

#                 long_enough = duration_ms >= MIN_GESTURE_DURATION_MS

#                 if enough_points and long_enough:

#                     print("CAST END")

#                     print("Points recorded:", len(trajectory))

#                     print("Duration:", f"{duration_ms} ms")

#                     print("First five points:", trajectory[:5])

#                 else:

#                     print(
#                         "CAST REJECTED"
#                         f" - points={len(trajectory)},"
#                         f" duration={duration_ms} ms"
#                     )

#                     # Invalid gestures disappear immediately.
#                     trajectory = []

#                 # Reset cast-specific state.
#                 cast_start_time_ms = None

#                 smoothed_index = None

#             # =================================================
#             # Update state for NEXT frame
#             # =================================================

#             was_pinching = is_pinching

#         # ====================================================
#         # 11. No hand detected
#         # ====================================================

#         else:

#             lost_frames += 1

#             # ------------------------------------------------
#             # We deliberately DO NOT immediately reset
#             # was_pinching.
#             #
#             # A temporary MediaPipe detection failure should
#             # not destroy the spell.
#             # ------------------------------------------------

#             if was_pinching:

#                 cv2.putText(
#                     frame,
#                     ("TRACKING LOST " f"{lost_frames}/{MAX_LOST_FRAMES}"),
#                     (20, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (0, 165, 255),
#                     2,
#                 )

#                 # --------------------------------------------
#                 # Only cancel after several consecutive
#                 # missing frames.
#                 # --------------------------------------------

#                 if lost_frames > MAX_LOST_FRAMES:

#                     print("CAST CANCELLED" " - hand tracking lost")

#                     was_pinching = False

#                     trajectory = []

#                     cast_start_time_ms = None

#                     smoothed_index = None

#             else:

#                 cv2.putText(
#                     frame,
#                     "NO HAND",
#                     (20, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (0, 165, 255),
#                     2,
#                 )

#         # ====================================================
#         # 12. Draw spell trajectory
#         #
#         # This intentionally happens OUTSIDE the
#         # hand-detection block.
#         #
#         # Therefore the trail stays visible even if tracking
#         # briefly disappears.
#         # ====================================================

#         trajectory_pixels = []

#         for x, y in trajectory:

#             pixel_x = int(x * width)
#             pixel_y = int(y * height)

#             trajectory_pixels.append((pixel_x, pixel_y))

#         for i in range(1, len(trajectory_pixels)):

#             cv2.line(
#                 frame, trajectory_pixels[i - 1], trajectory_pixels[i], (255, 0, 255), 4
#             )

#         # ====================================================
#         # 13. Calculate/display FPS
#         # ====================================================

#         current_frame_time = time.perf_counter()

#         frame_duration = current_frame_time - previous_frame_time

#         if frame_duration > 0:

#             fps = 1.0 / frame_duration

#         else:

#             fps = 0.0

#         previous_frame_time = current_frame_time

#         cv2.putText(
#             frame,
#             f"FPS: {fps:.1f}",
#             (20, 75),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (255, 255, 255),
#             2,
#         )

#         # ====================================================
#         # 14. Show final frame
#         # ====================================================

#         cv2.imshow("RuneCaster Hand Tracking", frame)

#         # ====================================================
#         # 15. Quit on Q
#         # ====================================================

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     # ========================================================
#     # Cleanup
#     # ========================================================

#     camera.release()

#     cv2.destroyAllWindows()

import cv2

from spellcaster.config import MODEL_PATH
from spellcaster.vision.hand_tracker import HandTracker
from spellcaster.vision.rendering import draw_hand

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam")


tracker = HandTracker(str(MODEL_PATH))


try:

    while True:

        success, frame = camera.read()

        if not success:
            break

        observation = tracker.detect(frame)

        if observation is not None:
            draw_hand(frame, observation)

        cv2.imshow(
            "RuneCaster Tracker Test",
            frame,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


finally:

    tracker.close()
    camera.release()
    cv2.destroyAllWindows()

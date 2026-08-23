import time

import cv2
import mediapipe as mp

MODEL_PATH = "models/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
)

with HandLandmarker.create_from_options(options) as landmarker:
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam")

    while True:
        success, frame = camera.read()

        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        print(type(frame))
        print(type(rgb_frame))

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = time.monotonic_ns() // 1_000_000

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        print(result)
        print("Hands detected:", len(result.hand_landmarks))

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]

            index_tip = hand[8]

            print("Index:", index_tip.x, index_tip.y, index_tip.z)
            cv2.imshow("SpellCaster Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

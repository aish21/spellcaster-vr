import time

import cv2
import mediapipe as mp
import numpy as np

from spellcaster.gestures.models import (
    HandObservation,
    Point2D,
)


class HandTracker:

    def __init__(self, model_path: str):

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
        )

        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(
        self,
        frame: np.ndarray,
    ) -> HandObservation | None:

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        timestamp_ms = time.monotonic_ns() // 1_000_000

        result = self._landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        if not result.hand_landmarks:
            return None

        media_pipe_hand = result.hand_landmarks[0]

        landmarks = tuple(
            Point2D(
                x=landmark.x,
                y=landmark.y,
            )
            for landmark in media_pipe_hand
        )

        handedness_result = result.handedness[0][0]

        return HandObservation(
            landmarks=landmarks,
            handedness=handedness_result.category_name,
            confidence=handedness_result.score,
        )

    def close(self) -> None:
        self._landmarker.close()

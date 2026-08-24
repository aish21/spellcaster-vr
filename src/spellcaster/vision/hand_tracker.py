import time

import cv2
import mediapipe as mp
import numpy as np

from spellcaster.gestures.models import (
    HandObservation,
    Point2D,
)


class HandTracker:

    def __init__(
        self,
        model_path: str,
        num_hands: int = 1,
        preferred_handedness: str | None = None,
    ) -> None:
        self._preferred_handedness = preferred_handedness

        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=num_hands,
        )

        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(
        self,
        frame: np.ndarray,
    ) -> HandObservation | None:

        # ----------------------------------------------------
        # 1. OpenCV uses BGR.
        #    MediaPipe expects RGB.
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        # ----------------------------------------------------
        # 2. VIDEO mode requires monotonically increasing time.
        # ----------------------------------------------------

        timestamp_ms = time.monotonic_ns() // 1_000_000

        result = self._landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )

        # ----------------------------------------------------
        # 3. Nothing detected.
        # ----------------------------------------------------

        if not result.hand_landmarks:
            return None

        # ----------------------------------------------------
        # 4. Convert ALL MediaPipe hands into our own
        #    HandObservation objects first.
        #
        #    Important:
        #    do not choose a hand inside this loop.
        # ----------------------------------------------------

        observations: list[HandObservation] = []

        for index, media_pipe_hand in enumerate(result.hand_landmarks):

            landmarks = tuple(
                Point2D(
                    x=landmark.x,
                    y=landmark.y,
                )
                for landmark in media_pipe_hand
            )

            handedness_result = result.handedness[index][0]

            observation = HandObservation(
                landmarks=landmarks,
                handedness=(handedness_result.category_name),
                confidence=handedness_result.score,
            )

            observations.append(observation)

        # ----------------------------------------------------
        # 5. If a preferred casting hand was requested,
        #    filter AFTER we have processed every detected hand.
        # ----------------------------------------------------

        if self._preferred_handedness is not None:

            matching_hands = [
                observation
                for observation in observations
                if (observation.handedness == self._preferred_handedness)
            ]

            # The preferred hand is not currently visible.
            #
            # We intentionally do NOT fall back to the other
            # hand, otherwise control could suddenly jump.
            if not matching_hands:
                return None

            # There should normally only be one matching hand,
            # but choosing by confidence is a safe policy.
            return max(
                matching_hands,
                key=lambda observation: observation.confidence,
            )

        # ----------------------------------------------------
        # 6. No preferred hand configured.
        #
        #    Return whichever detected hand has the highest
        #    confidence.
        # ----------------------------------------------------

        return max(
            observations,
            key=lambda observation: observation.confidence,
        )

    def close(self) -> None:
        self._landmarker.close()

import cv2
import numpy as np

from spellcaster.gestures.models import (
    HandObservation,
    Point2D,
)
from spellcaster.vision.geometry import (
    normalized_to_pixel,
)
from spellcaster.vision.landmarks import (
    HAND_CONNECTIONS,
    HandLandmark,
)


def draw_hand(
    frame: np.ndarray,
    observation: HandObservation,
) -> None:

    height, width, _ = frame.shape

    pixel_points = [
        normalized_to_pixel(
            point,
            width,
            height,
        )
        for point in observation.landmarks
    ]

    for start, end in HAND_CONNECTIONS:

        cv2.line(
            frame,
            pixel_points[start],
            pixel_points[end],
            (255, 255, 255),
            2,
        )

    for point in pixel_points:

        cv2.circle(
            frame,
            point,
            5,
            (0, 255, 0),
            -1,
        )

    index_tip = pixel_points[HandLandmark.INDEX_TIP]

    cv2.circle(
        frame,
        index_tip,
        10,
        (0, 0, 255),
        -1,
    )

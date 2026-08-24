from enum import IntEnum


class HandLandmark(IntEnum):
    WRIST = 0

    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4

    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8

    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12

    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16

    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


HAND_CONNECTIONS = [
    # Thumb
    (HandLandmark.WRIST, HandLandmark.THUMB_CMC),
    (HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP),
    (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP),
    (HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP),
    # Index
    (HandLandmark.WRIST, HandLandmark.INDEX_MCP),
    (HandLandmark.INDEX_MCP, HandLandmark.INDEX_PIP),
    (HandLandmark.INDEX_PIP, HandLandmark.INDEX_DIP),
    (HandLandmark.INDEX_DIP, HandLandmark.INDEX_TIP),
    # Middle
    (HandLandmark.WRIST, HandLandmark.MIDDLE_MCP),
    (HandLandmark.MIDDLE_MCP, HandLandmark.MIDDLE_PIP),
    (HandLandmark.MIDDLE_PIP, HandLandmark.MIDDLE_DIP),
    (HandLandmark.MIDDLE_DIP, HandLandmark.MIDDLE_TIP),
    # Ring
    (HandLandmark.WRIST, HandLandmark.RING_MCP),
    (HandLandmark.RING_MCP, HandLandmark.RING_PIP),
    (HandLandmark.RING_PIP, HandLandmark.RING_DIP),
    (HandLandmark.RING_DIP, HandLandmark.RING_TIP),
    # Pinky
    (HandLandmark.WRIST, HandLandmark.PINKY_MCP),
    (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP),
    (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP),
    (HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP),
    # Palm
    (HandLandmark.INDEX_MCP, HandLandmark.MIDDLE_MCP),
    (HandLandmark.MIDDLE_MCP, HandLandmark.RING_MCP),
    (HandLandmark.RING_MCP, HandLandmark.PINKY_MCP),
]

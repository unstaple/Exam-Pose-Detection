import cv2

from .types import BoundingBox, HeadPose, BehaviorResult


def draw_bbox(frame, bbox: BoundingBox, color=(0, 255, 0)):

    cv2.rectangle(
        frame,
        (bbox.x1, bbox.y1),
        (bbox.x2, bbox.y2),
        color,
        2,
    )

    label = f"Face {bbox.confidence:.2f}"

    cv2.putText(
        frame,
        label,
        (bbox.x1, max(20, bbox.y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def draw_pose(
    frame,
    pose: HeadPose,
    origin=(20, 40),
):

    x, y = origin

    lines = [
        f"Yaw:   {pose.yaw:+.1f} deg",
        f"Pitch: {pose.pitch:+.1f} deg",
        f"Roll:  {pose.roll:+.1f} deg",
    ]

    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (x, y + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )


def draw_behavior(
    frame,
    result: BehaviorResult,
    origin=(20, 125),
):

    x, y = origin

    text = (
        f"{result.state} | "
        f"score={result.score:.2f}"
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        result.reason,
        (x, y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )
from dataclasses import dataclass
from typing import List

import cv2
from ultralytics import YOLO


@dataclass
class TrackedFace:
    tracker_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height


class FaceTracker:
    """
    YOLO face detector + multi-object tracker.

    ByteTrack is used to maintain identities between frames.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.35,
        image_size: int = 1280,
        device: str = "cuda:0",
        tracker: str = "bytetrack.yaml",
    ):
        self.model = YOLO(model_path)

        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.tracker = tracker

    def track(self, frame) -> List[TrackedFace]:

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        if not results:
            return []

        boxes = results[0].boxes

        if boxes is None:
            return []

        # A detection may exist without a tracker ID yet.
        if boxes.id is None:
            return []

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        faces = []

        for tracker_id, box, confidence in zip(
            ids,
            xyxy,
            confs,
        ):
            x1, y1, x2, y2 = map(
                int,
                box,
            )

            faces.append(
                TrackedFace(
                    tracker_id=int(tracker_id),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(confidence),
                )
            )

        return faces

    @staticmethod
    def crop_face(
        frame,
        face: TrackedFace,
        padding: float = 0.25,
    ):
        h, w = frame.shape[:2]

        pad_x = int(face.width * padding)
        pad_y = int(face.height * padding)

        x1 = max(0, face.x1 - pad_x)
        y1 = max(0, face.y1 - pad_y)

        x2 = min(w, face.x2 + pad_x)
        y2 = min(h, face.y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]
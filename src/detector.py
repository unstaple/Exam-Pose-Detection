from pathlib import Path
from typing import List

import cv2
from ultralytics import YOLO

from .types import BoundingBox


class FaceDetector:
    """
    YOLOv8 face detector wrapper.

    Detects faces and returns bounding boxes.
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5,
        image_size: int = 640,
        device: str = "cuda:0",
    ):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Face model not found: {model_path}"
            )

        self.model = YOLO(str(model_path))

        self.confidence = confidence
        self.image_size = image_size
        self.device = device

    def detect(self, frame) -> List[BoundingBox]:
        """
        Detect all faces in a frame.
        """

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        detections = []

        if not results:
            return detections

        result = results[0]

        if result.boxes is None:
            return detections

        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().item())

            x1, y1, x2, y2 = map(int, xyxy)

            detections.append(
                BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                )
            )

        return detections

    @staticmethod
    def crop_face(
        frame,
        bbox: BoundingBox,
        padding: float = 0.20,
    ):
        """
        Crop a face with additional padding.

        Padding is important for head-pose estimation because
        6DRepNet benefits from having some context around the face.
        """

        h, w = frame.shape[:2]

        face_width = bbox.width
        face_height = bbox.height

        pad_x = int(face_width * padding)
        pad_y = int(face_height * padding)

        x1 = max(0, bbox.x1 - pad_x)
        y1 = max(0, bbox.y1 - pad_y)

        x2 = min(w, bbox.x2 + pad_x)
        y2 = min(h, bbox.y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]
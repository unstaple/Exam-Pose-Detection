import argparse
import time
from pathlib import Path

import cv2

from src.detector import FaceDetector
from src.pose_estimator import HeadPoseEstimator
from src.smoothing import PoseSmoother
from src.behavior_analyzer import BehaviorAnalyzer
from src.visualization import (
    draw_bbox,
    draw_pose,
    draw_behavior,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Exam Head Pose Detection System"
    )

    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, video path, or stream URL",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Inference device, e.g. cuda:0 or cpu",
    )

    parser.add_argument(
        "--face-model",
        default="Models/yolov8n-face.pt",
    )

    parser.add_argument(
        "--pose-model",
        default="Models/6DRepNet_300W_LP_AFLW2000.pth",
    )

    parser.add_argument(
        "--face-conf",
        type=float,
        default=0.5,
    )

    return parser.parse_args()


def open_source(source):

    # Camera index
    if source.isdigit():
        return cv2.VideoCapture(int(source))

    # Video / network stream
    return cv2.VideoCapture(source)


def main():

    args = parse_args()

    face_detector = FaceDetector(
        model_path=args.face_model,
        confidence=args.face_conf,
        device=args.device,
    )

    pose_estimator = HeadPoseEstimator(
        model_path=args.pose_model,
        device=args.device,
    )

    smoother = PoseSmoother(
        alpha=0.35,
    )

    analyzer = BehaviorAnalyzer(
        yaw_threshold=35.0,
        pitch_up_threshold=-20.0,
        roll_threshold=35.0,
        min_duration=1.0,
        alert_duration=3.0,
    )

    cap = open_source(args.source)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open source: {args.source}"
        )

    previous_time = time.perf_counter()

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame.")
            break

        current_time = time.perf_counter()

        fps = 1.0 / max(
            current_time - previous_time,
            1e-6,
        )

        previous_time = current_time

        # -------------------------------------------------
        # 1. Face detection
        # -------------------------------------------------

        faces = face_detector.detect(frame)

        # -------------------------------------------------
        # Multiple faces:
        # For a first prototype, select the largest face.
        # -------------------------------------------------

        if len(faces) == 0:

            result = analyzer.update(None)

            cv2.putText(
                frame,
                "NO FACE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        else:

            # Largest face = primary examinee
            bbox = max(
                faces,
                key=lambda b: b.width * b.height,
            )

            draw_bbox(frame, bbox)

            # -------------------------------------------------
            # 2. Crop face
            # -------------------------------------------------

            face_crop = face_detector.crop_face(
                frame,
                bbox,
                padding=0.20,
            )

            # -------------------------------------------------
            # 3. Head pose estimation
            # -------------------------------------------------

            pose = pose_estimator.predict(
                face_crop
            )

            # -------------------------------------------------
            # 4. Temporal smoothing
            # -------------------------------------------------

            pose = smoother.update(pose)

            # -------------------------------------------------
            # 5. Behavioral analysis
            # -------------------------------------------------

            result = analyzer.update(
                pose,
                timestamp=current_time,
            )

            # -------------------------------------------------
            # 6. Visualization
            # -------------------------------------------------

            draw_pose(frame, pose)

            draw_behavior(
                frame,
                result,
            )

        # -------------------------------------------------
        # FPS
        # -------------------------------------------------

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "Exam Pose Detection",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
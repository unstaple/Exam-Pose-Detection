from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2


class ClassroomLogger:
    """
    Event-only logger.

    Only SUSPICIOUS / ALERT events are written.

    Each event receives:
        - JSONL record
        - Annotated JPEG snapshot
    """

    def __init__(
        self,
        base_dir: str = "logs",
        jpeg_quality: int = 90,
    ):

        self.base_dir = Path(base_dir)

        self.session_id = (
            uuid.uuid4().hex[:8]
        )

        start_time = (
            datetime.now()
            .strftime(
                "%Y-%m-%d_%H-%M-%S"
            )
        )

        self.session_dir = (
            self.base_dir
            / (
                f"session_"
                f"{start_time}_"
                f"{self.session_id}"
            )
        )

        self.snapshot_dir = (
            self.session_dir
            / "snapshots"
        )

        self.snapshot_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.events_path = (
            self.session_dir
            / "events.jsonl"
        )

        self.events_file = open(
            self.events_path,
            "a",
            encoding="utf-8",
            buffering=1,
        )

        self.jpeg_quality = (
            jpeg_quality
        )

        self.closed = False

    # =========================================================
    # Timestamp
    # =========================================================

    @staticmethod
    def timestamp():

        return datetime.now(
            timezone.utc
        ).isoformat()

    # =========================================================
    # Filename safety
    # =========================================================

    @staticmethod
    def safe_filename(
        text: str,
    ):

        allowed = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            "_-."
        )

        return "".join(
            char
            if char in allowed
            else "_"
            for char in text
        )

    # =========================================================
    # Behavior event
    # =========================================================

    def log_behavior_event(
        self,
        frame,
        person_id: str,
        tracker_id: int,
        event_type: str,
        behavior,
        pose=None,
        bbox=None,
    ):

        if behavior is None:
            return None, None

        if behavior.state not in {
            "SUSPICIOUS",
            "ALERT",
        }:
            return None, None

        timestamp = self.timestamp()

        annotated = frame.copy()

        # -----------------------------------------------------
        # Bounding box
        # -----------------------------------------------------

        if bbox is not None:

            if behavior.state == "ALERT":
                color = (
                    0,
                    0,
                    255,
                )
            else:
                color = (
                    0,
                    255,
                    255,
                )

            cv2.rectangle(
                annotated,

                (
                    bbox.x1,
                    bbox.y1,
                ),

                (
                    bbox.x2,
                    bbox.y2,
                ),

                color,
                3,
            )

            cv2.putText(
                annotated,

                (
                    f"{person_id} | "
                    f"{behavior.state}"
                ),

                (
                    bbox.x1,
                    max(
                        30,
                        bbox.y1 - 12,
                    ),
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.75,

                color,

                2,
            )

        # -----------------------------------------------------
        # Pose text
        # -----------------------------------------------------

        if (
            bbox is not None
            and pose is not None
        ):

            text = (
                f"Yaw: {pose.yaw:+.1f}  "
                f"Pitch: {pose.pitch:+.1f}  "
                f"Roll: {pose.roll:+.1f}"
            )

            cv2.putText(
                annotated,

                text,

                (
                    bbox.x1,
                    bbox.y2 + 25,
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                color,

                2,
            )

        # -----------------------------------------------------
        # Snapshot filename
        # -----------------------------------------------------

        filename = (
            f"{self.safe_filename(person_id)}_"
            f"{self.safe_filename(event_type)}_"
            f"{datetime.now().strftime('%H-%M-%S-%f')}"
            ".jpg"
        )

        snapshot_path = (
            self.snapshot_dir
            / filename
        )

        success = cv2.imwrite(
            str(snapshot_path),
            annotated,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                self.jpeg_quality,
            ],
        )

        if not success:
            snapshot_path = None

        # -----------------------------------------------------
        # Relative path for dashboard
        # -----------------------------------------------------

        snapshot_relative = None

        if snapshot_path is not None:

            snapshot_relative = (
                snapshot_path
                .relative_to(
                    self.base_dir
                )
                .as_posix()
            )

        # -----------------------------------------------------
        # JSONL event
        # -----------------------------------------------------

        record = {

            "record_type":
                "behavior_event",

            "timestamp":
                timestamp,

            "session_id":
                self.session_id,

            "person_id":
                person_id,

            "tracker_id":
                int(tracker_id),

            "event_type":
                event_type,

            "state":
                behavior.state,

            "score":
                float(
                    behavior.score
                ),

            "reason":
                behavior.reason,

            "duration":
                float(
                    behavior.duration
                ),

            "bbox":
                None,

            "pose":
                None,

            "snapshot":
                snapshot_relative,
        }

        if bbox is not None:

            record["bbox"] = {

                "x1": int(
                    bbox.x1
                ),

                "y1": int(
                    bbox.y1
                ),

                "x2": int(
                    bbox.x2
                ),

                "y2": int(
                    bbox.y2
                ),

                "confidence":
                    float(
                        bbox.confidence
                    ),
            }

        if pose is not None:

            record["pose"] = {

                "yaw":
                    float(pose.yaw),

                "pitch":
                    float(pose.pitch),

                "roll":
                    float(pose.roll),
            }

        self.events_file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

        self.events_file.flush()

        return (
            snapshot_relative,
            record,
        )

    # =========================================================
    # Close
    # =========================================================

    def close(self):

        if self.closed:
            return

        self.events_file.flush()

        self.events_file.close()

        self.closed = True
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2


class ClassroomLogger:
    """
    Event-only classroom logger.

    Saves:
        events.jsonl

    and event screenshots:

        snapshots/
            face01_...jpg
            face02_...jpg
            ...

    Only SUSPICIOUS / ALERT events are logged.
    """

    def __init__(
        self,
        base_dir: str = "logs",
        jpeg_quality: int = 90,
    ):

        self.base_dir = Path(base_dir)

        self.session_id = uuid.uuid4().hex[:8]

        start_time = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        self.session_dir = (
            self.base_dir
            / f"session_{start_time}_{self.session_id}"
        )

        self.snapshot_dir = (
            self.session_dir / "snapshots"
        )

        self.snapshot_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.events_path = (
            self.session_dir / "events.jsonl"
        )

        self.events_file = open(
            self.events_path,
            "a",
            encoding="utf-8",
            buffering=1,
        )

        self.jpeg_quality = jpeg_quality

        self.closed = False

        # Used to avoid saving the same state every frame.
        self.last_logged_state = {}

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    @staticmethod
    def timestamp() -> str:

        return datetime.now(
            timezone.utc
        ).isoformat()

    @staticmethod
    def safe_filename(text: str) -> str:

        allowed = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            "_-."
        )

        return "".join(
            c if c in allowed else "_"
            for c in text
        )

    # ---------------------------------------------------------
    # Session
    # ---------------------------------------------------------

    def log_session_start(
        self,
        source: str,
        device: str,
        config: Optional[dict] = None,
    ):

        # Optional informational record.
        # This isn't a person event.
        record = {
            "record_type": "session_start",
            "timestamp": self.timestamp(),
            "session_id": self.session_id,
            "source": source,
            "device": device,
            "config": config or {},
        }

        self._write(record)

    # ---------------------------------------------------------
    # Suspicious / Alert event
    # ---------------------------------------------------------

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
        """
        Save an event and a corresponding annotated screenshot.

        event_type examples:

            suspicious_start
            alert_start
            alert_repeat
        """

        if behavior is None:
            return None

        if behavior.state not in {
            "SUSPICIOUS",
            "ALERT",
        }:
            return None

        timestamp = self.timestamp()

        # -----------------------------------------------------
        # Annotate image
        # -----------------------------------------------------

        image = frame.copy()

        if bbox is not None:

            color = (
                (0, 255, 255)
                if behavior.state == "SUSPICIOUS"
                else (0, 0, 255)
            )

            cv2.rectangle(
                image,
                (bbox.x1, bbox.y1),
                (bbox.x2, bbox.y2),
                color,
                3,
            )

            cv2.putText(
                image,
                (
                    f"{person_id} | "
                    f"{behavior.state}"
                ),
                (
                    bbox.x1,
                    max(30, bbox.y1 - 12),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
            )

            if pose is not None:

                pose_text = (
                    f"Yaw: {pose.yaw:+.1f}  "
                    f"Pitch: {pose.pitch:+.1f}  "
                    f"Roll: {pose.roll:+.1f}"
                )

                cv2.putText(
                    image,
                    pose_text,
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
        # Save snapshot
        # -----------------------------------------------------

        filename = (
            f"{self.safe_filename(person_id)}_"
            f"{event_type}_"
            f"{datetime.now().strftime('%H-%M-%S-%f')}.jpg"
        )

        snapshot_path = (
            self.snapshot_dir / filename
        )

        success = cv2.imwrite(
            str(snapshot_path),
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                self.jpeg_quality,
            ],
        )

        if not success:
            snapshot_path = None

        # -----------------------------------------------------
        # JSON record
        # -----------------------------------------------------

        record = {
            "record_type": "behavior_event",

            "timestamp": timestamp,

            "session_id": self.session_id,

            "person_id": person_id,

            "tracker_id": tracker_id,

            "event_type": event_type,

            "state": behavior.state,

            "score": behavior.score,

            "reason": behavior.reason,

            "duration": behavior.duration,

            "bbox": None,

            "pose": None,

            "snapshot": (
                str(snapshot_path)
                if snapshot_path
                else None
            ),
        }

        if bbox is not None:

            record["bbox"] = {
                "x1": bbox.x1,
                "y1": bbox.y1,
                "x2": bbox.x2,
                "y2": bbox.y2,
                "confidence": bbox.confidence,
            }

        if pose is not None:

            record["pose"] = {
                "yaw": pose.yaw,
                "pitch": pose.pitch,
                "roll": pose.roll,
            }

        self._write(record)

        return snapshot_path

    # ---------------------------------------------------------
    # Internal
    # ---------------------------------------------------------

    def _write(self, record: dict[str, Any]):

        if self.closed:
            return

        self.events_file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

        self.events_file.flush()

    # ---------------------------------------------------------
    # Shutdown
    # ---------------------------------------------------------

    def log_session_end(
        self,
        reason: str = "normal_shutdown",
    ):

        self._write({
            "record_type": "session_end",
            "timestamp": self.timestamp(),
            "session_id": self.session_id,
            "reason": reason,
        })

    def close(self):

        if self.closed:
            return

        self.events_file.flush()
        self.events_file.close()

        self.closed = True

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()
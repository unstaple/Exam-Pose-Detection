from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    send_from_directory,
)


class DashboardState:
    """
    Thread-safe state shared between the inference loop
    and the Flask dashboard.

    Contains:
        - Latest annotated frame
        - Current tracked people
        - Recent suspicious/alert events
    """

    def __init__(
        self,
        session_id: str,
        max_events: int = 50,
        jpeg_quality: int = 80,
    ):
        self.session_id = session_id
        self.max_events = max_events
        self.jpeg_quality = jpeg_quality

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

        self._frame_jpeg: Optional[bytes] = None
        self._frame_seq = 0
        self._last_frame_encode = 0.0

        self.people: dict[str, dict] = {}

        self.events = deque(
            maxlen=max_events
        )

        self.started_at = time.time()

    # =========================================================
    # Frame
    # =========================================================

    def update_frame(
        self,
        frame,
        max_fps: float = 10.0,
    ):
        """
        Convert the latest annotated frame to JPEG.

        The dashboard does not need the full inference FPS,
        so we deliberately limit dashboard encoding.
        """

        now = time.time()

        min_interval = 1.0 / max(
            max_fps,
            1.0,
        )

        if (
            now - self._last_frame_encode
            < min_interval
        ):
            return

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                self.jpeg_quality,
            ],
        )

        if not success:
            return

        jpeg_bytes = encoded.tobytes()

        with self._condition:

            self._frame_jpeg = jpeg_bytes

            self._frame_seq += 1

            self._last_frame_encode = now

            self._condition.notify_all()

    # =========================================================
    # People
    # =========================================================

    def update_people(
        self,
        results,
    ):
        """
        Update the dashboard's current view of tracked people.

        results:
            [
                (
                    face,
                    person_id,
                    pose,
                    behavior,
                    pose_updated,
                ),
                ...
            ]
        """

        now = time.time()

        people = {}

        for (
            face,
            person_id,
            pose,
            behavior,
            _pose_updated,
        ) in results:

            state = (
                behavior.state
                if behavior is not None
                else "NO_POSE"
            )

            score = (
                float(behavior.score)
                if behavior is not None
                else 0.0
            )

            reason = (
                behavior.reason
                if behavior is not None
                else ""
            )

            duration = (
                float(behavior.duration)
                if behavior is not None
                else 0.0
            )

            people[person_id] = {
                "person_id": person_id,

                "tracker_id": int(
                    face.tracker_id
                ),

                "confidence": round(
                    float(face.confidence),
                    3,
                ),

                "bbox": {
                    "x1": int(face.x1),
                    "y1": int(face.y1),
                    "x2": int(face.x2),
                    "y2": int(face.y2),
                },

                "pose": (
                    {
                        "yaw": round(
                            float(pose.yaw),
                            2,
                        ),
                        "pitch": round(
                            float(pose.pitch),
                            2,
                        ),
                        "roll": round(
                            float(pose.roll),
                            2,
                        ),
                    }
                    if pose is not None
                    else None
                ),

                "state": state,

                "score": round(
                    score,
                    3,
                ),

                "reason": reason,

                "duration": round(
                    duration,
                    2,
                ),

                "last_seen": now,
            }

        with self._lock:
            self.people = people

    # =========================================================
    # Events
    # =========================================================

    def add_event(
        self,
        person_id: str,
        tracker_id: int,
        event_type: str,
        behavior,
        pose=None,
        snapshot_url: Optional[str] = None,
    ):
        """
        Add one suspicious/alert event to the dashboard.
        """

        event = {
            "timestamp": time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(),
            ),

            "person_id": person_id,

            "tracker_id": int(
                tracker_id
            ),

            "event_type": event_type,

            "state": (
                behavior.state
                if behavior
                else None
            ),

            "score": (
                round(
                    float(behavior.score),
                    3,
                )
                if behavior
                else None
            ),

            "reason": (
                behavior.reason
                if behavior
                else ""
            ),

            "duration": (
                round(
                    float(behavior.duration),
                    2,
                )
                if behavior
                else 0.0
            ),

            "pose": (
                {
                    "yaw": round(
                        float(pose.yaw),
                        2,
                    ),
                    "pitch": round(
                        float(pose.pitch),
                        2,
                    ),
                    "roll": round(
                        float(pose.roll),
                        2,
                    ),
                }
                if pose is not None
                else None
            ),

            "snapshot": snapshot_url,
        }

        with self._lock:
            self.events.appendleft(event)

    # =========================================================
    # API state
    # =========================================================

    def snapshot(self) -> dict:

        now = time.time()

        with self._lock:

            people = list(
                self.people.values()
            )

            events = list(
                self.events
            )

        suspicious = sum(
            person["state"] == "SUSPICIOUS"
            for person in people
        )

        alert = sum(
            person["state"] == "ALERT"
            for person in people
        )

        normal = sum(
            person["state"] == "NORMAL"
            for person in people
        )

        return {
            "session_id": self.session_id,

            "uptime": round(
                now - self.started_at,
                1,
            ),

            "active_faces": len(
                people
            ),

            "normal": normal,

            "suspicious": suspicious,

            "alert": alert,

            "people": sorted(
                people,
                key=lambda item:
                    item["person_id"],
            ),

            "events": events,
        }

    # =========================================================
    # MJPEG stream
    # =========================================================

    def mjpeg_stream(self):

        last_sequence = -1

        while True:

            with self._condition:

                self._condition.wait_for(
                    lambda:
                        self._frame_seq
                        != last_sequence,
                    timeout=2.0,
                )

                frame = self._frame_jpeg

                current_sequence = (
                    self._frame_seq
                )

            if frame is None:
                continue

            last_sequence = current_sequence

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(
                    len(frame)
                ).encode()
                + b"\r\n\r\n"
                + frame
                + b"\r\n"
            )


# =============================================================
# Flask application
# =============================================================

def create_app(
    state: DashboardState,
    log_dir: str = "logs",
):

    template_dir = (
        Path(__file__).resolve().parent.parent
        / "templates"
    )

    app = Flask(
        __name__,
        template_folder=str(
            template_dir
        ),
    )

    @app.get("/")
    def index():

        return render_template(
            "dashboard.html"
        )

    @app.get("/api/status")
    def api_status():

        return jsonify(
            state.snapshot()
        )

    @app.get("/video_feed")
    def video_feed():

        return Response(
            state.mjpeg_stream(),
            mimetype=(
                "multipart/x-mixed-replace;"
                " boundary=frame"
            ),
        )

    @app.get(
        "/snapshot/<path:relative_path>"
    )
    def snapshot(relative_path):

        return send_from_directory(
            log_dir,
            relative_path,
        )

    @app.get("/health")
    def health():

        return jsonify({
            "status": "ok",
            "session_id": state.session_id,
        })

    return app


class DashboardServer:

    def __init__(
        self,
        state: DashboardState,
        host: str = "127.0.0.1",
        port: int = 5000,
        log_dir: str = "logs",
    ):

        self.state = state

        self.host = host
        self.port = port

        self.app = create_app(
            state,
            log_dir=log_dir,
        )

    def start(self):

        thread = threading.Thread(
            target=self._run,
            name="flask-dashboard",
            daemon=True,
        )

        thread.start()

        return thread

    def _run(self):

        self.app.run(
            host=self.host,
            port=self.port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
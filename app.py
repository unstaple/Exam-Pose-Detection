import argparse

import cv2

from src.tracker import FaceTracker
from src.pose_estimator import HeadPoseEstimator
from src.pipeline import ClassroomPipeline
from src.logger import ClassroomLogger
from src.dashboard import (
    DashboardState,
    DashboardServer,
)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="0",
        help=(
            "Camera index, video path, "
            "or ESP32 stream URL"
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--face-model",
        default="Models/yolov8n-face.pt",
    )

    parser.add_argument(
        "--pose-model",
        default=(
            "Models/"
            "6DRepNet_300W_LP_AFLW2000.pth"
        ),
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--dashboard-host",
        default="127.0.0.1",
    )

    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=5000,
    )

    return parser.parse_args()


def open_source(source):

    if source.isdigit():

        return cv2.VideoCapture(
            int(source)
        )

    return cv2.VideoCapture(
        source
    )


def draw_person(
    frame,
    face,
    person_id,
    pose,
    behavior,
):

    # ---------------------------------------------------------
    # Color based on behavior
    # ---------------------------------------------------------

    color = (
        0,
        255,
        0,
    )

    if behavior is not None:

        if behavior.state == "SUSPICIOUS":

            color = (
                0,
                255,
                255,
            )

        elif behavior.state == "ALERT":

            color = (
                0,
                0,
                255,
            )

    # ---------------------------------------------------------
    # Bounding box
    # ---------------------------------------------------------

    cv2.rectangle(
        frame,

        (
            face.x1,
            face.y1,
        ),

        (
            face.x2,
            face.y2,
        ),

        color,

        2,
    )

    # ---------------------------------------------------------
    # ID + tracker ID
    # ---------------------------------------------------------

    label = (
        f"{person_id} "
        f"(trk:{face.tracker_id})"
    )

    cv2.putText(
        frame,
        label,

        (
            face.x1,
            max(
                20,
                face.y1 - 10,
            ),
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        color,

        2,
    )

    # ---------------------------------------------------------
    # Pose
    # ---------------------------------------------------------

    if pose is not None:

        text = (
            f"Y:{pose.yaw:+.0f} "
            f"P:{pose.pitch:+.0f} "
            f"R:{pose.roll:+.0f}"
        )

        cv2.putText(
            frame,

            text,

            (
                face.x1,
                face.y2 + 20,
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            color,

            1,
        )

    # ---------------------------------------------------------
    # Behavior
    # ---------------------------------------------------------

    if behavior is not None:

        cv2.putText(
            frame,

            behavior.state,

            (
                face.x1,
                face.y2 + 40,
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.45,

            color,

            1,
        )


def main():

    args = parse_args()

    # =========================================================
    # Logger
    # =========================================================

    logger = ClassroomLogger(
        base_dir="logs",
        jpeg_quality=90,
    )

    print(
        f"Session: {logger.session_id}"
    )

    print(
        f"Logs: {logger.session_dir}"
    )

    # =========================================================
    # Dashboard
    # =========================================================

    dashboard_state = DashboardState(
        session_id=logger.session_id,
        max_events=50,
    )

    dashboard_server = DashboardServer(
        state=dashboard_state,
        host=args.dashboard_host,
        port=args.dashboard_port,
        log_dir="logs",
    )

    dashboard_server.start()

    print(
        "Dashboard:"
        f" http://{args.dashboard_host}:"
        f"{args.dashboard_port}"
    )

    # =========================================================
    # Models
    # =========================================================

    tracker = FaceTracker(

        model_path=args.face_model,

        image_size=args.imgsz,

        device=args.device,

        # Static classroom camera.
        tracker=(
            "config/"
            "classroom_botsort.yaml"
        ),
    )

    pose_estimator = (
        HeadPoseEstimator(
            model_path=args.pose_model,
            device=args.device,
        )
    )

    # =========================================================
    # Pipeline
    # =========================================================

    pipeline = ClassroomPipeline(

        tracker=tracker,

        pose_estimator=pose_estimator,

        pose_interval=2,

        min_face_size=64,
    )

    # =========================================================
    # Camera
    # =========================================================

    cap = open_source(
        args.source
    )

    if not cap.isOpened():

        logger.close()

        raise RuntimeError(
            f"Unable to open source:"
            f" {args.source}"
        )

    # =========================================================
    # State tracking
    # =========================================================

    previous_states = {}

    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                print(
                    "Failed to read frame."
                )

                break

            # =================================================
            # Inference
            # =================================================

            results = pipeline.process(
                frame
            )

            # =================================================
            # Dashboard state
            # =================================================

            dashboard_state.update_people(
                results
            )

            # =================================================
            # Per-person processing
            # =================================================

            for (
                face,
                person_id,
                pose,
                behavior,
                pose_updated,
            ) in results:

                draw_person(
                    frame,
                    face,
                    person_id,
                    pose,
                    behavior,
                )

                if not pose_updated:
                    continue

                if behavior is None:
                    continue

                current_state = (
                    behavior.state
                )

                previous_state = (
                    previous_states.get(
                        person_id
                    )
                )

                # =============================================
                # NORMAL → SUSPICIOUS
                # =============================================

                if (
                    current_state
                    == "SUSPICIOUS"

                    and

                    previous_state
                    != "SUSPICIOUS"
                ):

                    (
                        snapshot_path,
                        _record,
                    ) = (
                        logger.log_behavior_event(

                            frame=frame,

                            person_id=person_id,

                            tracker_id=(
                                face.tracker_id
                            ),

                            event_type=(
                                "suspicious_start"
                            ),

                            behavior=behavior,

                            pose=pose,

                            bbox=face,
                        )
                    )

                    dashboard_state.add_event(

                        person_id=person_id,

                        tracker_id=(
                            face.tracker_id
                        ),

                        event_type=(
                            "suspicious_start"
                        ),

                        behavior=behavior,

                        pose=pose,

                        snapshot_url=(
                            snapshot_path
                        ),
                    )

                # =============================================
                # SUSPICIOUS → ALERT
                # =============================================

                elif (
                    current_state
                    == "ALERT"

                    and

                    previous_state
                    != "ALERT"
                ):

                    (
                        snapshot_path,
                        _record,
                    ) = (
                        logger.log_behavior_event(

                            frame=frame,

                            person_id=person_id,

                            tracker_id=(
                                face.tracker_id
                            ),

                            event_type=(
                                "alert_start"
                            ),

                            behavior=behavior,

                            pose=pose,

                            bbox=face,
                        )
                    )

                    dashboard_state.add_event(

                        person_id=person_id,

                        tracker_id=(
                            face.tracker_id
                        ),

                        event_type=(
                            "alert_start"
                        ),

                        behavior=behavior,

                        pose=pose,

                        snapshot_url=(
                            snapshot_path
                        ),
                    )

                previous_states[
                    person_id
                ] = current_state

            # =================================================
            # Dashboard video
            # =================================================

            dashboard_state.update_frame(
                frame,
                max_fps=10,
            )

            # =================================================
            # Local OpenCV window
            # =================================================

            cv2.putText(

                frame,

                (
                    f"Active faces: "
                    f"{len(results)}"
                ),

                (20, 30),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (
                    255,
                    255,
                    255,
                ),

                2,
            )

            cv2.imshow(
                "Classroom Monitoring",
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break

    except KeyboardInterrupt:

        print(
            "Stopping..."
        )

    finally:

        cap.release()

        cv2.destroyAllWindows()

        logger.close()

        print(
            f"Logs saved to:"
            f" {logger.session_dir}"
        )


if __name__ == "__main__":
    main()
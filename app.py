import argparse

import cv2

from src.tracker import FaceTracker
from src.pose_estimator import HeadPoseEstimator
from src.pipeline import ClassroomPipeline
from src.logger import ClassroomLogger


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default="0",
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
        default="Models/6DRepNet_300W_LP_AFLW2000.pth",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
    )

    return parser.parse_args()


def open_source(source):

    if source.isdigit():
        return cv2.VideoCapture(int(source))

    return cv2.VideoCapture(source)


def draw_person(
    frame,
    face,
    person_id,
    pose,
    behavior,
):

    color = (0, 255, 0)

    if behavior is not None:

        if behavior.state == "SUSPICIOUS":
            color = (0, 255, 255)

        elif behavior.state == "ALERT":
            color = (0, 0, 255)

    cv2.rectangle(
        frame,
        (face.x1, face.y1),
        (face.x2, face.y2),
        color,
        2,
    )

    cv2.putText(
        frame,
        person_id,
        (face.x1, face.y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )

    if pose is not None:

        text = (
            f"Y:{pose.yaw:+.0f} "
            f"P:{pose.pitch:+.0f} "
            f"R:{pose.roll:+.0f}"
        )

        cv2.putText(
            frame,
            text,
            (face.x1, face.y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )

    if behavior is not None:

        cv2.putText(
            frame,
            behavior.state,
            (face.x1, face.y2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
        )


def main():

    args = parse_args()

    # -----------------------------------------------------
    # Logger
    # -----------------------------------------------------

    logger = ClassroomLogger(
        base_dir="logs",
    )

    logger.log_session_start(
        source=args.source,
        device=args.device,
        config={
            "image_size": args.imgsz,
            "pose_interval": 2,
            "min_face_size": 64,
        },
    )

    print(
        f"Logging to: {logger.session_dir}"
    )

    tracker = FaceTracker(
        model_path=args.face_model,
        image_size=args.imgsz,
        device=args.device,
        tracker="bytetrack.yaml",
    )

    pose_estimator = HeadPoseEstimator(
        model_path=args.pose_model,
        device=args.device,
    )

    pipeline = ClassroomPipeline(
        tracker=tracker,
        pose_estimator=pose_estimator,
        pose_interval=2,
        min_face_size=64,
    )

    cap = open_source(args.source)

    if not cap.isOpened():
        logger.log_session_end(
            "camera_open_failed"
        )
        logger.close()

        raise RuntimeError(
            f"Unable to open source {args.source}"
        )

    # Stores the previous state for each person.
    previous_states = {}

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                logger.log_session_end(
                    "frame_read_failed"
                )
                break

            results = pipeline.process(frame)

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

                current_state = behavior.state

                previous_state = previous_states.get(
                    person_id
                )

                # ---------------------------------------------------------
                # ONLY LOG WHEN ENTERING SUSPICIOUS
                # OR ENTERING ALERT
                # ---------------------------------------------------------

                if (
                    current_state == "SUSPICIOUS"
                    and previous_state != "SUSPICIOUS"
                ):

                    logger.log_behavior_event(
                        frame=frame,
                        person_id=person_id,
                        tracker_id=face.tracker_id,
                        event_type="suspicious_start",
                        behavior=behavior,
                        pose=pose,
                        bbox=face,
                    )

                elif (
                    current_state == "ALERT"
                    and previous_state != "ALERT"
                ):

                    logger.log_behavior_event(
                        frame=frame,
                        person_id=person_id,
                        tracker_id=face.tracker_id,
                        event_type="alert_start",
                        behavior=behavior,
                        pose=pose,
                        bbox=face,
                    )

                previous_states[person_id] = current_state

            # ---------------------------------------------
            # Classroom info
            # ---------------------------------------------

            active = len(results)

            cv2.putText(
                frame,
                f"Active faces: {active}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "Classroom Monitoring",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                logger.log_session_end(
                    "user_quit"
                )
                break

    except KeyboardInterrupt:

        logger.log_session_end(
            "keyboard_interrupt"
        )

    except Exception as exc:

        logger.log_event(
            person_id="SYSTEM",
            tracker_id=-1,
            event_type="application_error",
            extra={
                "error": repr(exc),
            },
        )

        logger.log_session_end(
            "application_exception"
        )

        raise

    finally:

        cap.release()
        cv2.destroyAllWindows()

        logger.close()

        print(
            f"Logs saved to: {logger.session_dir}"
        )


if __name__ == "__main__":
    main()
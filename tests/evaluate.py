from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tracker import FaceTracker
from src.pose_estimator import HeadPoseEstimator
from src.pipeline import ClassroomPipeline


# ============================================================
# Ground truth
# ============================================================

def load_ground_truth(path: str):

    path = Path(path)

    print(
        f"[INFO] Loading ground truth: {path}",
        flush=True,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth not found: {path}"
        )

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "video",
            "zone",
            "start_sec",
            "end_sec",
            "label",
        }

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise ValueError(
                "Ground truth missing columns: "
                f"{sorted(missing)}"
            )

        for row in reader:

            rows.append({
                "video": row["video"],
                "zone": row["zone"],
                "start": float(row["start_sec"]),
                "end": float(row["end_sec"]),
                "label": row["label"],
            })

    print(
        f"[INFO] Loaded {len(rows)} GT events.",
        flush=True,
    )

    return rows


# ============================================================
# Zones
# ============================================================

def load_zones(
    path: str,
    video_name: str,
):

    path = Path(path)

    print(
        f"[INFO] Loading zones: {path}",
        flush=True,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Zone configuration not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(file)

    videos = config.get(
        "videos",
        {},
    )

    if video_name not in videos:

        raise ValueError(
            f"No zones configured for video "
            f"'{video_name}'. "
            f"Available: {list(videos)}"
        )

    zones = videos[
        video_name
    ]["zones"]

    print(
        f"[INFO] Loaded {len(zones)} zones.",
        flush=True,
    )

    return zones


def bbox_to_zone(
    face,
    zones,
    frame_width,
    frame_height,
):

    cx = (
        (face.x1 + face.x2) / 2
    ) / frame_width

    cy = (
        (face.y1 + face.y2) / 2
    ) / frame_height

    candidates = []

    for zone in zones:

        if (
            zone["x1"] <= cx <= zone["x2"]
            and
            zone["y1"] <= cy <= zone["y2"]
        ):

            zx = (
                zone["x1"]
                + zone["x2"]
            ) / 2

            zy = (
                zone["y1"]
                + zone["y2"]
            ) / 2

            distance = (
                (cx - zx) ** 2
                +
                (cy - zy) ** 2
            )

            candidates.append(
                (
                    distance,
                    zone["id"],
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: x[0]
    )

    return candidates[0][1]


# ============================================================
# Interval utilities
# ============================================================

def merge_intervals(
    intervals,
    max_gap=0.75,
):

    if not intervals:
        return []

    intervals = sorted(
        intervals,
        key=lambda x: x[0],
    )

    merged = [
        list(intervals[0])
    ]

    for start, end in intervals[1:]:

        previous = merged[-1]

        if start <= previous[1] + max_gap:

            previous[1] = max(
                previous[1],
                end,
            )

        else:

            merged.append(
                [start, end]
            )

    return [
        tuple(x)
        for x in merged
    ]


def interval_iou(a, b):

    start = max(
        a[0],
        b[0],
    )

    end = min(
        a[1],
        b[1],
    )

    intersection = max(
        0.0,
        end - start,
    )

    union = (
        max(a[1], b[1])
        -
        min(a[0], b[0])
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# Predicted event extraction
# ============================================================

def extract_predicted_events(
    predictions,
    max_gap=0.75,
):

    output = {}

    for zone, rows in predictions.items():

        intervals = []

        start = None
        previous = None

        for row in rows:

            timestamp = row["time"]

            abnormal = (
                row["state"]
                in {
                    "SUSPICIOUS",
                    "ALERT",
                }
            )

            if abnormal:

                if start is None:
                    start = timestamp

                elif (
                    previous is not None
                    and
                    timestamp - previous
                    > max_gap
                ):

                    intervals.append(
                        (
                            start,
                            previous,
                        )
                    )

                    start = timestamp

            else:

                if start is not None:

                    intervals.append(
                        (
                            start,
                            (
                                previous
                                if previous
                                is not None
                                else timestamp
                            ),
                        )
                    )

                    start = None

            previous = timestamp

        if start is not None:

            intervals.append(
                (
                    start,
                    previous,
                )
            )

        output[zone] = merge_intervals(
            intervals,
            max_gap=max_gap,
        )

    return output


# ============================================================
# Evaluation
# ============================================================

def evaluate_events(
    ground_truth,
    predictions,
    iou_threshold=0.30,
):

    gt_by_zone = defaultdict(list)

    for event in ground_truth:

        gt_by_zone[
            event["zone"]
        ].append(
            (
                event["start"],
                event["end"],
                event["label"],
            )
        )

    pred_by_zone = defaultdict(list)

    for zone, intervals in predictions.items():
        pred_by_zone[zone] = intervals

    candidates = []

    for zone, gt_events in (
        gt_by_zone.items()
    ):

        for gt_index, gt in enumerate(
            gt_events
        ):

            for pred_index, pred in enumerate(
                pred_by_zone.get(
                    zone,
                    [],
                )
            ):

                iou = interval_iou(
                    gt,
                    pred,
                )

                if iou >= iou_threshold:

                    candidates.append(
                        (
                            iou,
                            zone,
                            gt_index,
                            pred_index,
                        )
                    )

    candidates.sort(
        reverse=True,
        key=lambda x: x[0],
    )

    matched_gt = set()
    matched_pred = set()

    matches = []

    for (
        iou,
        zone,
        gt_index,
        pred_index,
    ) in candidates:

        gt_key = (
            zone,
            gt_index,
        )

        pred_key = (
            zone,
            pred_index,
        )

        if gt_key in matched_gt:
            continue

        if pred_key in matched_pred:
            continue

        matched_gt.add(gt_key)
        matched_pred.add(pred_key)

        gt = gt_by_zone[
            zone
        ][gt_index]

        pred = pred_by_zone[
            zone
        ][pred_index]

        matches.append({
            "zone": zone,
            "label": gt[2],
            "gt_start": gt[0],
            "gt_end": gt[1],
            "pred_start": pred[0],
            "pred_end": pred[1],
            "iou": iou,
            "latency": pred[0] - gt[0],
        })

    tp = len(matches)

    total_gt = sum(
        len(x)
        for x in gt_by_zone.values()
    )

    total_pred = sum(
        len(x)
        for x in pred_by_zone.values()
    )

    fn = total_gt - tp
    fp = total_pred - tp

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        /
        (precision + recall)
        if precision + recall
        else 0.0
    )

    valid_latency = [
        x["latency"]
        for x in matches
        if x["latency"] >= 0
    ]

    mean_latency = (
        sum(valid_latency)
        /
        len(valid_latency)
        if valid_latency
        else None
    )

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,

        "precision": precision,
        "recall": recall,
        "f1": f1,

        "mean_latency_sec":
            mean_latency,

        "matches": matches,
    }


# ============================================================
# Evaluation video
# ============================================================

def evaluate_video(
    video_path,
    ground_truth,
    zones,
    pipeline,
    output_dir,
    display=False,
    save_video=False,
    max_frames=None,
):

    video_path = Path(
        video_path
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print(
        f"[INFO] Opening video: {video_path}"
    )
    print("=" * 70, flush=True)

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"OpenCV could not open video: "
            f"{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if not fps or fps <= 0:
        fps = 30.0

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    duration = (
        total_frames / fps
        if total_frames > 0
        else 0
    )

    print(
        f"[INFO] Resolution: "
        f"{width}x{height}",
        flush=True,
    )

    print(
        f"[INFO] FPS: {fps:.2f}",
        flush=True,
    )

    print(
        f"[INFO] Frames: {total_frames}",
        flush=True,
    )

    print(
        f"[INFO] Duration: "
        f"{duration:.2f}s",
        flush=True,
    )

    if max_frames is not None:

        print(
            f"[INFO] Max frames: "
            f"{max_frames}",
            flush=True,
        )

    # --------------------------------------------------------
    # Optional annotated output video
    # --------------------------------------------------------

    writer = None

    if save_video:

        output_video = (
            output_dir
            / "annotated.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(output_video),
            fourcc,
            fps,
            (
                width,
                height,
            ),
        )

        if not writer.isOpened():

            raise RuntimeError(
                "Could not create output video."
            )

        print(
            f"[INFO] Writing preview: "
            f"{output_video}",
            flush=True,
        )

    predictions = defaultdict(list)

    frame_index = 0

    started = time.perf_counter()

    last_report = started

    last_people_count = 0

    try:

        while True:

            ret, frame = cap.read()

            if not ret:

                print(
                    "\n[INFO] End of video.",
                    flush=True,
                )

                break

            timestamp = (
                frame_index / fps
            )

            frame_index += 1

            # ------------------------------------------------
            # Pipeline
            # ------------------------------------------------

            results = pipeline.process(
                frame,
                timestamp=timestamp,
            )

            last_people_count = len(
                results
            )

            # ------------------------------------------------
            # Collect predictions
            # ------------------------------------------------

            for (
                face,
                person_id,
                pose,
                behavior,
                pose_updated,
            ) in results:

                if not pose_updated:
                    continue

                if pose is None:
                    continue

                zone = bbox_to_zone(
                    face,
                    zones,
                    width,
                    height,
                )

                if zone is None:
                    continue

                state = (
                    behavior.state
                    if behavior
                    else "NO_POSE"
                )

                score = (
                    float(
                        behavior.score
                    )
                    if behavior
                    else 0.0
                )

                predictions[
                    zone
                ].append({

                    "time":
                        timestamp,

                    "state":
                        state,

                    "score":
                        score,

                    "yaw":
                        float(
                            pose.yaw
                        ),

                    "pitch":
                        float(
                            pose.pitch
                        ),

                    "roll":
                        float(
                            pose.roll
                        ),

                    "person_id":
                        person_id,

                    "tracker_id":
                        int(
                            face.tracker_id
                        ),
                })

            # ------------------------------------------------
            # Draw boxes
            # ------------------------------------------------

            if display or save_video:

                output_frame = (
                    frame.copy()
                )

                for (
                    face,
                    person_id,
                    pose,
                    behavior,
                    _,
                ) in results:

                    color = (
                        0,
                        255,
                        0,
                    )

                    if behavior:

                        if (
                            behavior.state
                            == "SUSPICIOUS"
                        ):
                            color = (
                                0,
                                255,
                                255,
                            )

                        elif (
                            behavior.state
                            == "ALERT"
                        ):
                            color = (
                                0,
                                0,
                                255,
                            )

                    cv2.rectangle(
                        output_frame,
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

                    cv2.putText(
                        output_frame,
                        (
                            f"{person_id}"
                            f" "
                            f"(trk:{face.tracker_id})"
                        ),
                        (
                            face.x1,
                            max(
                                20,
                                face.y1 - 8,
                            ),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

                    if pose:

                        pose_text = (
                            f"Y:{pose.yaw:+.0f} "
                            f"P:{pose.pitch:+.0f} "
                            f"R:{pose.roll:+.0f}"
                        )

                        cv2.putText(
                            output_frame,
                            pose_text,
                            (
                                face.x1,
                                face.y2 + 18,
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            color,
                            1,
                        )

                    if behavior:

                        cv2.putText(
                            output_frame,
                            behavior.state,
                            (
                                face.x1,
                                face.y2 + 36,
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.42,
                            color,
                            1,
                        )

                cv2.putText(
                    output_frame,

                    (
                        f"Frame: "
                        f"{frame_index}/"
                        f"{total_frames}"
                    ),

                    (15, 25),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (255, 255, 255),

                    2,
                )

                cv2.putText(
                    output_frame,

                    (
                        f"Time: "
                        f"{timestamp:.1f}s "
                        f"Faces: "
                        f"{last_people_count}"
                    ),

                    (15, 50),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.6,

                    (255, 255, 255),

                    2,
                )

                if writer is not None:
                    writer.write(
                        output_frame
                    )

                if display:

                    cv2.imshow(
                        "Evaluation",
                        output_frame,
                    )

                    key = (
                        cv2.waitKey(1)
                        & 0xFF
                    )

                    if key == ord("q"):

                        print(
                            "\n[INFO] "
                            "Stopped by user.",
                            flush=True,
                        )

                        break

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            now = time.perf_counter()

            if (
                now - last_report
                >= 2.0
            ):

                elapsed = (
                    now - started
                )

                processing_fps = (
                    frame_index
                    / elapsed
                    if elapsed > 0
                    else 0
                )

                if total_frames > 0:

                    progress = (
                        frame_index
                        / total_frames
                        * 100
                    )

                    print(
                        (
                            f"[PROGRESS] "
                            f"{frame_index}/"
                            f"{total_frames} "
                            f"({progress:5.1f}%) | "
                            f"video={timestamp:7.1f}s | "
                            f"faces={last_people_count:2d} | "
                            f"speed={processing_fps:5.2f} FPS"
                        ),
                        flush=True,
                    )

                else:

                    print(
                        (
                            f"[PROGRESS] "
                            f"frame={frame_index} | "
                            f"time={timestamp:.1f}s | "
                            f"faces={last_people_count} | "
                            f"speed={processing_fps:.2f} FPS"
                        ),
                        flush=True,
                    )

                last_report = now

            if (
                max_frames is not None
                and frame_index
                >= max_frames
            ):

                print(
                    "\n[INFO] "
                    "Reached max_frames.",
                    flush=True,
                )

                break

    except Exception:

        print(
            "\n[ERROR] Exception occurred "
            "while processing the video.",
            file=sys.stderr,
            flush=True,
        )

        raise

    finally:

        cap.release()

        if writer is not None:
            writer.release()

        if display:
            cv2.destroyAllWindows()

    print(
        f"[INFO] Frames actually processed: "
        f"{frame_index}",
        flush=True,
    )

    predicted_events = (
        extract_predicted_events(
            predictions
        )
    )

    video_name = (
        video_path.name
    )

    video_gt = [
        event
        for event in ground_truth
        if event["video"]
        == video_name
    ]

    metrics = evaluate_events(
        video_gt,
        predicted_events,
        iou_threshold=0.30,
    )

    # --------------------------------------------------------
    # Save raw predictions
    # --------------------------------------------------------

    predictions_path = (
        output_dir
        / "predictions.csv"
    )

    with open(
        predictions_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        writer_csv = csv.writer(
            file
        )

        writer_csv.writerow([
            "zone",
            "time",
            "state",
            "score",
            "yaw",
            "pitch",
            "roll",
            "person_id",
            "tracker_id",
        ])

        for zone, rows in (
            predictions.items()
        ):

            for row in rows:

                writer_csv.writerow([
                    zone,
                    row["time"],
                    row["state"],
                    row["score"],
                    row["yaw"],
                    row["pitch"],
                    row["roll"],
                    row["person_id"],
                    row["tracker_id"],
                ])

    # --------------------------------------------------------
    # Save evaluation results
    # --------------------------------------------------------

    result = {
        "video": video_name,

        "fps": fps,

        "duration_sec": duration,

        "frames_processed":
            frame_index,

        "ground_truth_events":
            len(video_gt),

        "predicted_events":
            {
                k: v
                for k, v
                in predicted_events.items()
            },

        "metrics": metrics,
    }

    evaluation_path = (
        output_dir
        / "evaluation.json"
    )

    with open(
        evaluation_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        f"Video: {video_name}"
    )

    print(
        f"Processed frames: {frame_index}"
    )

    print(
        f"Ground truth events: "
        f"{len(video_gt)}"
    )

    print()

    print(
        f"TP: {metrics['true_positives']}"
    )

    print(
        f"FP: {metrics['false_positives']}"
    )

    print(
        f"FN: {metrics['false_negatives']}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{metrics['f1']:.4f}"
    )

    if (
        metrics[
            "mean_latency_sec"
        ]
        is not None
    ):

        print(
            f"Mean detection latency: "
            f"{metrics['mean_latency_sec']:.3f}s"
        )

    print()

    print(
        f"Predictions: "
        f"{predictions_path}"
    )

    print(
        f"Evaluation: "
        f"{evaluation_path}"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
    )

    parser.add_argument(
        "--ground-truth",
        default="tests/ground_truth.csv",
    )

    parser.add_argument(
        "--zones",
        default="config/test_zones.yaml",
    )

    parser.add_argument(
        "--output",
        default="evaluation_results",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--face-model",
        default=(
            "Models/yolov8n-face.pt"
        ),
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
        "--pose-interval",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Stop after N frames. "
            "Useful for debugging."
        ),
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help=(
            "Show annotated processing window."
        ),
    )

    parser.add_argument(
        "--save-video",
        action="store_true",
        help=(
            "Save annotated evaluation video."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    print(
        "=" * 70,
        flush=True,
    )

    print(
        "CLASSROOM EXAM EVALUATOR",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )

    print(
        f"[INFO] Python executable: "
        f"{sys.executable}",
        flush=True,
    )

    print(
        f"[INFO] Device: {args.device}",
        flush=True,
    )

    print(
        f"[INFO] Video: {args.video}",
        flush=True,
    )

    # --------------------------------------------------------
    # Load GT
    # --------------------------------------------------------

    ground_truth = load_ground_truth(
        args.ground_truth
    )

    # --------------------------------------------------------
    # Video filename
    # --------------------------------------------------------

    video_path = Path(
        args.video
    )

    if not video_path.exists():

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    # --------------------------------------------------------
    # Zones
    # --------------------------------------------------------

    zones = load_zones(
        args.zones,
        video_path.name,
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    print(
        "[INFO] Loading face tracker...",
        flush=True,
    )

    tracker = FaceTracker(
        model_path=args.face_model,
        confidence=0.35,
        image_size=args.imgsz,
        device=args.device,
        tracker=(
            "config/"
            "classroom_botsort.yaml"
        ),
    )

    print(
        "[INFO] Face tracker loaded.",
        flush=True,
    )

    print(
        "[INFO] Loading 6DRepNet...",
        flush=True,
    )

    pose_estimator = (
        HeadPoseEstimator(
            model_path=args.pose_model,
            device=args.device,
        )
    )

    print(
        "[INFO] 6DRepNet loaded.",
        flush=True,
    )

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    pipeline = ClassroomPipeline(
        tracker=tracker,
        pose_estimator=pose_estimator,
        pose_interval=args.pose_interval,
        min_face_size=64,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    output_dir = (
        Path(args.output)
        / video_path.stem
    )

    evaluate_video(
        video_path=video_path,
        ground_truth=ground_truth,
        zones=zones,
        pipeline=pipeline,
        output_dir=output_dir,
        display=args.display,
        save_video=args.save_video,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print(
            "\n[INFO] Interrupted.",
            flush=True,
        )

    except Exception as exc:

        print(
            "\n"
            + "=" * 70,
            file=sys.stderr,
        )

        print(
            "[FATAL ERROR]",
            file=sys.stderr,
        )

        print(
            repr(exc),
            file=sys.stderr,
        )

        print(
            "=" * 70,
            file=sys.stderr,
        )

        raise
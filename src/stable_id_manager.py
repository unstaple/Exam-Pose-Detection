from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Optional

import numpy as np


@dataclass
class StableTrack:
    session_id: str

    tracker_id: int

    x1: int
    y1: int
    x2: int
    y2: int

    last_seen_frame: int

    @property
    def center(self):
        return (
            (self.x1 + self.x2) / 2.0,
            (self.y1 + self.y2) / 2.0,
        )

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return max(1, self.width * self.height)


class StableIDManager:
    """
    Converts unstable/raw tracker IDs into stable session IDs.

    A session ID represents a tracked person during the current
    application session. It is NOT a real-world identity.

    If the underlying tracker changes from tracker ID 4 -> 9
    but the new bbox is sufficiently close to the old bbox,
    the same session ID is retained.
    """

    def __init__(
        self,
        max_lost_frames: int = 90,
        min_iou: float = 0.05,
        max_center_distance: float = 0.08,
        min_size_similarity: float = 0.35,
    ):
        self.max_lost_frames = max_lost_frames
        self.min_iou = min_iou
        self.max_center_distance = max_center_distance
        self.min_size_similarity = min_size_similarity

        self.next_session_number = 1

        self.tracks: dict[str, StableTrack] = {}

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def update(
        self,
        detections,
        frame_index: int,
        frame_width: int,
        frame_height: int,
    ):
        """
        Assign stable faceXX IDs to current detections.

        Returns:
            list of (detection, session_id)
        """

        self._remove_stale(frame_index)

        if not detections:
            return []

        # -----------------------------------------------------
        # Stage 1:
        # Preserve exact tracker-ID matches.
        # -----------------------------------------------------

        assignments = {}

        used_sessions = set()

        tracker_to_session = {
            track.tracker_id: session_id
            for session_id, track in self.tracks.items()
        }

        unmatched_detections = []

        for index, detection in enumerate(detections):

            session_id = tracker_to_session.get(
                detection.tracker_id
            )

            if (
                session_id is not None
                and session_id not in used_sessions
            ):
                assignments[index] = session_id
                used_sessions.add(session_id)

                self._update_track(
                    session_id,
                    detection,
                    frame_index,
                )

            else:
                unmatched_detections.append(index)

        # -----------------------------------------------------
        # Stage 2:
        # If tracker ID changed, try spatial reassociation.
        # -----------------------------------------------------

        for detection_index in unmatched_detections:

            detection = detections[detection_index]

            best_session: Optional[str] = None
            best_score = -1.0

            for session_id, track in self.tracks.items():

                if session_id in used_sessions:
                    continue

                # Only reassociate recently lost tracks.
                age = (
                    frame_index
                    - track.last_seen_frame
                )

                if age > self.max_lost_frames:
                    continue

                score = self._association_score(
                    track,
                    detection,
                    frame_width,
                    frame_height,
                )

                if score > best_score:
                    best_score = score
                    best_session = session_id

            if best_session is not None:
                assignments[detection_index] = (
                    best_session
                )

                used_sessions.add(best_session)

                self._update_track(
                    best_session,
                    detection,
                    frame_index,
                )

            else:
                # ------------------------------------------------
                # Stage 3:
                # Create genuinely new session person.
                # ------------------------------------------------

                session_id = self._create_session(
                    detection,
                    frame_index,
                )

                assignments[detection_index] = session_id
                used_sessions.add(session_id)

        return [
            (
                detections[index],
                assignments[index],
            )
            for index in range(len(detections))
        ]

    # ---------------------------------------------------------
    # Association
    # ---------------------------------------------------------

    def _association_score(
        self,
        track: StableTrack,
        detection,
        frame_width: int,
        frame_height: int,
    ) -> float:

        iou = self._iou(
            (
                track.x1,
                track.y1,
                track.x2,
                track.y2,
            ),
            (
                detection.x1,
                detection.y1,
                detection.x2,
                detection.y2,
            ),
        )

        old_cx, old_cy = track.center

        new_cx = (
            detection.x1 + detection.x2
        ) / 2.0

        new_cy = (
            detection.y1 + detection.y2
        ) / 2.0

        distance = hypot(
            old_cx - new_cx,
            old_cy - new_cy,
        )

        frame_diagonal = hypot(
            frame_width,
            frame_height,
        )

        normalized_distance = (
            distance / frame_diagonal
        )

        center_score = max(
            0.0,
            1.0
            - (
                normalized_distance
                / self.max_center_distance
            ),
        )

        size_similarity = min(
            track.area,
            detection.width * detection.height,
        ) / max(
            track.area,
            detection.width * detection.height,
        )

        # Reject obviously unrelated detections.
        spatial_match = (
            iou >= self.min_iou
            or (
                normalized_distance
                <= self.max_center_distance
            )
        )

        if not spatial_match:
            return -1.0

        if size_similarity < self.min_size_similarity:
            return -1.0

        # Weight overlap more heavily than size.
        score = (
            0.60 * iou
            + 0.30 * center_score
            + 0.10 * size_similarity
        )

        return score

    # ---------------------------------------------------------
    # Track management
    # ---------------------------------------------------------

    def _create_session(
        self,
        detection,
        frame_index: int,
    ) -> str:

        session_id = (
            f"face{self.next_session_number:02d}"
        )

        self.next_session_number += 1

        self.tracks[session_id] = StableTrack(
            session_id=session_id,
            tracker_id=detection.tracker_id,

            x1=detection.x1,
            y1=detection.y1,
            x2=detection.x2,
            y2=detection.y2,

            last_seen_frame=frame_index,
        )

        return session_id

    def _update_track(
        self,
        session_id: str,
        detection,
        frame_index: int,
    ):

        track = self.tracks[session_id]

        track.tracker_id = detection.tracker_id

        track.x1 = detection.x1
        track.y1 = detection.y1
        track.x2 = detection.x2
        track.y2 = detection.y2

        track.last_seen_frame = frame_index

    def _remove_stale(
        self,
        frame_index: int,
    ):

        stale = []

        for session_id, track in self.tracks.items():

            age = (
                frame_index
                - track.last_seen_frame
            )

            if age > self.max_lost_frames:
                stale.append(session_id)

        for session_id in stale:
            del self.tracks[session_id]

    # ---------------------------------------------------------
    # IoU
    # ---------------------------------------------------------

    @staticmethod
    def _iou(box_a, box_b) -> float:

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        intersection_x1 = max(ax1, bx1)
        intersection_y1 = max(ay1, by1)

        intersection_x2 = min(ax2, bx2)
        intersection_y2 = min(ay2, by2)

        intersection_width = max(
            0,
            intersection_x2 - intersection_x1,
        )

        intersection_height = max(
            0,
            intersection_y2 - intersection_y1,
        )

        intersection_area = (
            intersection_width
            * intersection_height
        )

        area_a = max(
            0,
            (ax2 - ax1)
            * (ay2 - ay1),
        )

        area_b = max(
            0,
            (bx2 - bx1)
            * (by2 - by1),
        )

        union = area_a + area_b - intersection_area

        if union <= 0:
            return 0.0

        return intersection_area / union
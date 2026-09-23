import time

from .person_state import PersonState
from .stable_id_manager import StableIDManager


class ClassroomPipeline:

    def __init__(
        self,
        tracker,
        pose_estimator,
        pose_interval: int = 2,
        min_face_size: int = 64,
        frame_width: int = 1280,
        frame_height: int = 720,
    ):

        self.tracker = tracker
        self.pose_estimator = pose_estimator

        self.pose_interval = pose_interval
        self.min_face_size = min_face_size

        self.frame_index = 0

        self.stable_ids = StableIDManager(
            max_lost_frames=90,
        )

        self.people: dict[
            str,
            PersonState,
        ] = {}

        self.frame_width = frame_width
        self.frame_height = frame_height

    def process(self, frame):

        timestamp = time.monotonic()

        self.frame_index += 1

        height, width = frame.shape[:2]

        self.frame_width = width
        self.frame_height = height

        # -----------------------------------------------------
        # Raw detection + tracking
        # -----------------------------------------------------

        tracked_faces = self.tracker.track(
            frame
        )

        # -----------------------------------------------------
        # Convert raw tracker IDs to stable faceXX IDs
        # -----------------------------------------------------

        stable_faces = self.stable_ids.update(
            tracked_faces,
            frame_index=self.frame_index,
            frame_width=width,
            frame_height=height,
        )

        outputs = []

        for face, person_id in stable_faces:

            # -------------------------------------------------
            # Create state for new stable person
            # -------------------------------------------------

            if person_id not in self.people:

                self.people[person_id] = (
                    PersonState(
                        session_id=person_id,
                        tracker_id=face.tracker_id,
                    )
                )

            person = self.people[person_id]

            person.tracker_id = (
                face.tracker_id
            )

            person.last_seen_timestamp = (
                timestamp
            )

            # -------------------------------------------------
            # Face too small
            # -------------------------------------------------

            if (
                face.width < self.min_face_size
                or face.height < self.min_face_size
            ):

                outputs.append(
                    (
                        face,
                        person_id,
                        person.last_pose,
                        person.last_behavior,
                        False,
                    )
                )

                continue

            pose_updated = False

            # -------------------------------------------------
            # Run pose estimation only every N frames
            # -------------------------------------------------

            if (
                self.frame_index
                % self.pose_interval
                == 0
            ):

                crop = self.tracker.crop_face(
                    frame,
                    face,
                    padding=0.25,
                )

                pose = (
                    self.pose_estimator.predict(
                        crop
                    )
                )

                (
                    smoothed_pose,
                    behavior,
                ) = person.update_pose(
                    pose,
                    timestamp,
                )

                pose_updated = True

            else:

                smoothed_pose = (
                    person.last_pose
                )

                behavior = (
                    person.last_behavior
                )

            outputs.append(
                (
                    face,
                    person_id,
                    smoothed_pose,
                    behavior,
                    pose_updated,
                )
            )

        return outputs
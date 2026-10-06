from dataclasses import dataclass, field

from .types import HeadPose, BehaviorResult
from .smoothing import PoseSmoother
from .behavior_analyzer import BehaviorAnalyzer
from .calibration import PoseCalibration


@dataclass
class PersonState:

    session_id: str
    tracker_id: int

    smoother: PoseSmoother = field(
        default_factory=lambda:
            PoseSmoother(alpha=0.30)
    )

    analyzer: BehaviorAnalyzer = field(
        default_factory=BehaviorAnalyzer
    )

    calibration: PoseCalibration = field(
        default_factory=lambda:
            PoseCalibration(
                min_samples=30
            )
    )

    last_pose: HeadPose = None
    last_relative_pose = None

    last_behavior: BehaviorResult = None

    frames_seen: int = 0
    pose_updates: int = 0

    last_seen_timestamp: float = 0.0

    def update_pose(
        self,
        pose,
        timestamp,
    ):

        self.frames_seen += 1

        if pose is None:

            return (
                None,
                None,
            )

        self.pose_updates += 1

        # -----------------------------------------------------
        # Smooth first
        # -----------------------------------------------------

        pose = self.smoother.update(
            pose
        )

        self.last_pose = pose

        # -----------------------------------------------------
        # Calibration
        # -----------------------------------------------------

        if not self.calibration.is_calibrated(
            self.session_id
        ):

            self.calibration.add_sample(
                self.session_id,
                pose,
            )

            # Don't classify until baseline exists.
            if not self.calibration.is_calibrated(
                self.session_id
            ):

                behavior = BehaviorResult(
                    state="NORMAL",
                    score=0.0,
                    reason="Calibrating",
                    duration=0.0,
                )

                self.last_behavior = (
                    behavior
                )

                return pose, behavior

        # -----------------------------------------------------
        # Relative pose
        # -----------------------------------------------------

        relative = (
            self.calibration.relative_pose(
                self.session_id,
                pose,
            )
        )

        self.last_relative_pose = (
            relative
        )

        # -----------------------------------------------------
        # Behavioral analysis
        # -----------------------------------------------------

        behavior = self.analyzer.update(
            pose,
            relative_pose=relative,
            timestamp=timestamp,
        )

        self.last_behavior = (
            behavior
        )

        return (
            pose,
            behavior,
        )
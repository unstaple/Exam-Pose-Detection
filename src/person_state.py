from dataclasses import dataclass, field
from typing import Optional

from .types import HeadPose, BehaviorResult
from .smoothing import PoseSmoother
from .behavior_analyzer import BehaviorAnalyzer


@dataclass
class PersonState:
    """
    State belonging to one tracked face during a session.
    """

    session_id: str
    tracker_id: int

    smoother: PoseSmoother = field(
        default_factory=lambda: PoseSmoother(alpha=0.35)
    )

    analyzer: BehaviorAnalyzer = field(
        default_factory=BehaviorAnalyzer
    )

    last_pose: Optional[HeadPose] = None
    last_behavior: Optional[BehaviorResult] = None

    frames_seen: int = 0
    pose_updates: int = 0

    last_seen_timestamp: float = 0.0

    def update_pose(self, pose: HeadPose, timestamp: float):
        self.frames_seen += 1

        if pose is None:
            return None, None

        self.pose_updates += 1

        smoothed_pose = self.smoother.update(pose)

        behavior = self.analyzer.update(
            smoothed_pose,
            timestamp=timestamp,
        )

        self.last_pose = smoothed_pose
        self.last_behavior = behavior
        self.last_seen_timestamp = timestamp

        return smoothed_pose, behavior
from collections import deque

from .types import HeadPose


class PoseSmoother:
    """
    Exponential moving average for head pose.

    alpha:
        Higher = more responsive
        Lower = smoother
    """

    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self.current = None

    def update(self, pose: HeadPose) -> HeadPose:

        if pose is None:
            return None

        if self.current is None:
            self.current = pose
            return pose

        self.current = HeadPose(
            pitch=(
                self.alpha * pose.pitch
                + (1 - self.alpha) * self.current.pitch
            ),
            yaw=(
                self.alpha * pose.yaw
                + (1 - self.alpha) * self.current.yaw
            ),
            roll=(
                self.alpha * pose.roll
                + (1 - self.alpha) * self.current.roll
            ),
        )

        return self.current

    def reset(self):
        self.current = None
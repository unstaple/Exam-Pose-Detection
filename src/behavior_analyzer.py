import time

from .types import HeadPose, BehaviorResult


class BehaviorAnalyzer:
    """
    Temporal heuristic system for head-pose analysis.

    Important:
        This class does NOT determine whether cheating occurred.
        It identifies suspicious head-pose events.
    """

    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    ALERT = "ALERT"

    def __init__(
        self,
        yaw_threshold: float = 10.0,
        pitch_up_threshold: float = 15.0,
        roll_threshold: float = 10.0,
        min_duration: float = 1.0,
        alert_duration: float = 3.0,
        cooldown: float = 2.0,
    ):
        self.yaw_threshold = yaw_threshold
        self.pitch_up_threshold = pitch_up_threshold
        self.roll_threshold = roll_threshold

        self.min_duration = min_duration
        self.alert_duration = alert_duration
        self.cooldown = cooldown

        self.suspicious_since = None
        self.last_event_time = 0.0

    def _is_suspicious(self, pose: HeadPose) -> bool:

        suspicious_yaw = abs(pose.yaw) >= self.yaw_threshold

        # Depending on the coordinate convention of your model,
        # you may need to invert this condition.
        suspicious_pitch = (
            pose.pitch >= self.pitch_up_threshold
        )

        suspicious_roll = (
            abs(pose.roll) >= self.roll_threshold
        )

        return (
            suspicious_yaw
            or suspicious_pitch
            or suspicious_roll
        )

    def update(self, pose: HeadPose, timestamp=None):
        """
        Process one pose observation.

        Returns:
            BehaviorResult
        """

        if pose is None:
            self.suspicious_since = None

            return BehaviorResult(
                state=self.NORMAL,
                score=0.0,
                reason="No face detected",
                duration=0.0,
            )

        now = timestamp if timestamp is not None else time.monotonic()

        suspicious = self._is_suspicious(pose)

        if not suspicious:
            self.suspicious_since = None

            return BehaviorResult(
                state=self.NORMAL,
                score=0.0,
                reason="Head pose within normal range",
                duration=0.0,
            )

        if self.suspicious_since is None:
            self.suspicious_since = now

        duration = now - self.suspicious_since

        # Calculate a simple score.
        score = self._calculate_score(pose)

        if duration < self.min_duration:
            return BehaviorResult(
                state=self.NORMAL,
                score=score,
                reason="Short abnormal movement",
                duration=duration,
            )

        if duration < self.alert_duration:
            return BehaviorResult(
                state=self.SUSPICIOUS,
                score=score,
                reason=self._get_reason(pose),
                duration=duration,
            )

        # Cooldown prevents constantly generating alerts.
        if now - self.last_event_time >= self.cooldown:
            self.last_event_time = now

        return BehaviorResult(
            state=self.ALERT,
            score=score,
            reason=self._get_reason(pose),
            duration=duration,
        )

    def _calculate_score(self, pose: HeadPose) -> float:

        yaw_score = min(
            abs(pose.yaw) / self.yaw_threshold,
            2.0,
        )

        pitch_score = min(
            max(
                0.0,
                -pose.pitch / abs(self.pitch_up_threshold)
            ),
            2.0,
        )

        roll_score = min(
            abs(pose.roll) / self.roll_threshold,
            2.0,
        )

        score = (
            0.5 * yaw_score
            + 0.35 * pitch_score
            + 0.15 * roll_score
        )

        return min(score / 2.0, 1.0)

    def _get_reason(self, pose: HeadPose) -> str:

        reasons = []

        if abs(pose.yaw) >= self.yaw_threshold:
            direction = "left" if pose.yaw < 0 else "right"

            reasons.append(
                f"Excessive yaw ({direction})"
            )

        if pose.pitch <= self.pitch_up_threshold:
            reasons.append(
                f"Abnormal upward pitch ({pose.pitch:.1f}°)"
            )

        if abs(pose.roll) >= self.roll_threshold:
            reasons.append(
                f"Excessive roll ({pose.roll:.1f}°)"
            )

        return ", ".join(reasons)
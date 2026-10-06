from __future__ import annotations

import time

from .types import HeadPose, BehaviorResult


class BehaviorAnalyzer:

    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    ALERT = "ALERT"

    def __init__(
        self,

        yaw_warning: float = 25.0,
        yaw_alert: float = 40.0,

        pitch_up_warning: float = 15.0,
        pitch_up_alert: float = 25.0,

        min_duration: float = 0.8,
        alert_duration: float = 2.5,

        cooldown: float = 5.0,
    ):

        self.yaw_warning = yaw_warning
        self.yaw_alert = yaw_alert

        self.pitch_up_warning = (
            pitch_up_warning
        )

        self.pitch_up_alert = (
            pitch_up_alert
        )

        self.min_duration = (
            min_duration
        )

        self.alert_duration = (
            alert_duration
        )

        self.cooldown = cooldown

        self.suspicious_since = None

        self.last_alert_time = 0.0

    def update(
        self,
        pose: HeadPose,
        relative_pose=None,
        timestamp=None,
    ):

        if timestamp is None:
            timestamp = time.monotonic()

        if pose is None:

            self.suspicious_since = None

            return BehaviorResult(
                state=self.NORMAL,
                score=0.0,
                reason="No face detected",
                duration=0.0,
            )

        # -----------------------------------------------------
        # IMPORTANT:
        # Use relative pose when available.
        # -----------------------------------------------------

        if relative_pose is not None:

            yaw = relative_pose["yaw"]
            pitch = relative_pose["pitch"]

        else:

            # Fallback for debugging.
            yaw = pose.yaw
            pitch = pose.pitch

        abs_yaw = abs(yaw)

        # -----------------------------------------------------
        # Suspicious conditions
        # -----------------------------------------------------

        yaw_suspicious = (
            abs_yaw
            >= self.yaw_warning
        )

        # Only upward pitch is suspicious.
        #
        # If your 6DRepNet sign convention is opposite,
        # invert this after verifying the axis.
        pitch_up = max(
            0.0,
            pitch,
        )

        pitch_suspicious = (
            pitch_up
            >= self.pitch_up_warning
        )

        suspicious = (
            yaw_suspicious
            or pitch_suspicious
        )

        # -----------------------------------------------------
        # Normal
        # -----------------------------------------------------

        if not suspicious:

            self.suspicious_since = None

            return BehaviorResult(
                state=self.NORMAL,
                score=0.0,
                reason="Normal head movement",
                duration=0.0,
            )

        # -----------------------------------------------------
        # Start abnormal interval
        # -----------------------------------------------------

        if self.suspicious_since is None:

            self.suspicious_since = (
                timestamp
            )

        duration = (
            timestamp
            - self.suspicious_since
        )

        # -----------------------------------------------------
        # Score
        # -----------------------------------------------------

        yaw_score = min(
            abs_yaw / self.yaw_alert,
            1.0,
        )

        pitch_score = min(
            pitch_up / self.pitch_up_alert,
            1.0,
        )

        score = max(
            yaw_score,
            pitch_score,
        )

        # -----------------------------------------------------
        # Short movement
        # -----------------------------------------------------

        if duration < self.min_duration:

            return BehaviorResult(
                state=self.NORMAL,
                score=score,
                reason="Transient movement",
                duration=duration,
            )

        # -----------------------------------------------------
        # Suspicious
        # -----------------------------------------------------

        if duration < self.alert_duration:

            return BehaviorResult(
                state=self.SUSPICIOUS,
                score=score,
                reason=self._reason(
                    yaw,
                    pitch_up,
                ),
                duration=duration,
            )

        # -----------------------------------------------------
        # Alert
        # -----------------------------------------------------

        return BehaviorResult(
            state=self.ALERT,
            score=score,
            reason=self._reason(
                yaw,
                pitch_up,
            ),
            duration=duration,
        )

    def _reason(
        self,
        yaw,
        pitch_up,
    ):

        reasons = []

        if abs(yaw) >= self.yaw_warning:

            if yaw > 0:
                direction = "right"
            else:
                direction = "left"

            reasons.append(
                (
                    f"Head turned {direction} "
                    f"({abs(yaw):.1f}°)"
                )
            )

        if (
            pitch_up
            >= self.pitch_up_warning
        ):

            reasons.append(
                (
                    f"Head raised "
                    f"({pitch_up:.1f}°)"
                )
            )

        return ", ".join(reasons)
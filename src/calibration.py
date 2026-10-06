from __future__ import annotations

from collections import defaultdict


class PoseCalibration:
    """
    Maintains a per-person or per-seat neutral head-pose baseline.

    The baseline is learned while the examinee is behaving normally.
    """

    def __init__(
        self,
        min_samples: int = 30,
    ):
        self.min_samples = min_samples

        self.samples = defaultdict(list)

        self.baselines = {}

    def add_sample(
        self,
        person_id: str,
        pose,
    ):
        if pose is None:
            return

        if person_id in self.baselines:
            return

        samples = self.samples[
            person_id
        ]

        samples.append(
            (
                pose.yaw,
                pose.pitch,
                pose.roll,
            )
        )

        if len(samples) >= self.min_samples:

            self.baselines[
                person_id
            ] = self._calculate_baseline(
                samples
            )

    def _calculate_baseline(
        self,
        samples,
    ):

        yaw = sum(
            x[0] for x in samples
        ) / len(samples)

        pitch = sum(
            x[1] for x in samples
        ) / len(samples)

        roll = sum(
            x[2] for x in samples
        ) / len(samples)

        return {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
        }

    def is_calibrated(
        self,
        person_id: str,
    ) -> bool:

        return (
            person_id
            in self.baselines
        )

    def get_baseline(
        self,
        person_id: str,
    ):
        return self.baselines.get(
            person_id
        )

    def relative_pose(
        self,
        person_id: str,
        pose,
    ):

        if pose is None:
            return None

        baseline = self.baselines.get(
            person_id
        )

        if baseline is None:
            return None

        return {
            "yaw": (
                pose.yaw
                - baseline["yaw"]
            ),

            "pitch": (
                pose.pitch
                - baseline["pitch"]
            ),

            "roll": (
                pose.roll
                - baseline["roll"]
            ),
        }

    def reset(
        self,
        person_id=None,
    ):

        if person_id is None:

            self.samples.clear()
            self.baselines.clear()

        else:

            self.samples.pop(
                person_id,
                None,
            )

            self.baselines.pop(
                person_id,
                None,
            )
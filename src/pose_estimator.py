from pathlib import Path

import torch

from sixdrepnet import SixDRepNet

from .types import HeadPose


class HeadPoseEstimator:
    """
    6DRepNet wrapper.

    Input:
        BGR OpenCV face crop

    Output:
        pitch, yaw, roll in degrees
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
    ):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"6DRepNet model not found: {model_path}"
            )

        if device.startswith("cuda") and torch.cuda.is_available():
            gpu_id = int(device.split(":")[-1])
        else:
            gpu_id = -1

        self.device = device
        self.model = SixDRepNet(
            gpu_id=gpu_id,
            dict_path=str(model_path),
        )

    def predict(self, face):
        """
        Estimate head pose.

        Returns:
            HeadPose
        """

        if face is None or face.size == 0:
            return None

        pitch, yaw, roll = self.model.predict(face)

        return HeadPose(
            pitch=float(pitch),
            yaw=float(yaw),
            roll=float(roll),
        )
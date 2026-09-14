from dataclasses import dataclass
from typing import Optional


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self):
        return (
            (self.x1 + self.x2) // 2,
            (self.y1 + self.y2) // 2,
        )


@dataclass
class HeadPose:
    pitch: float
    yaw: float
    roll: float

    def __repr__(self):
        return (
            f"HeadPose("
            f"pitch={self.pitch:.1f}, "
            f"yaw={self.yaw:.1f}, "
            f"roll={self.roll:.1f})"
        )


@dataclass
class BehaviorResult:
    state: str
    score: float
    reason: str
    duration: float


@dataclass
class DetectionResult:
    bbox: Optional[BoundingBox]
    pose: Optional[HeadPose]
    behavior: Optional[BehaviorResult]
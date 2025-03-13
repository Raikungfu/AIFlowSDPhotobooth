from enum import Enum

class DetectPose(Enum):
    FullBody = "full_body"
    Headshot = "headshot"
    Unknown = "unknown"
import cv2
from enums.detect_pose import DetectPose
import mediapipe as mp
from PIL import Image

def detect_face_and_body(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_detector = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    face_results = face_detector.process(img_rgb)

    pose_detector = mp.solutions.pose.Pose()
    pose_results = pose_detector.process(img_rgb)

    has_face = face_results.detections is not None
    has_body = pose_results.pose_landmarks is not None

    if has_face and has_body:
        return DetectPose.FullBody
    elif has_face:
        return DetectPose.Headshot
    else:
        return DetectPose.Unknown

image_type = detect_face_and_body("user_image.jpg")
print(f"Image type detected: {image_type}")

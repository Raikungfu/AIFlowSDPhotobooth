import numpy as np
import cv2
from PIL import Image
from utils.face_detection import face_detector, landmark_predictor
import traceback


def align_face(image, aspect_ratio=(3, 4), face_ratio=0.3):
    try:
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        faces = face_detector(gray)
        if len(faces) == 0:
            print("No face detected.")
            return image

        face = faces[0]
        landmarks = landmark_predictor(gray, face)
        left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
        right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])

        delta_x = right_eye[0] - left_eye[0]
        delta_y = right_eye[1] - left_eye[1]
        angle = np.arctan2(delta_y, delta_x) * 180.0 / np.pi
        print(f"Left Eye: {left_eye}, Right Eye: {right_eye}, Angle: {angle}")

        center = tuple(map(int, ((left_eye + right_eye) / 2)))
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_image = cv2.warpAffine(image_np, rotation_matrix, (image_np.shape[1], image_np.shape[0]))

        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        target_height = h / face_ratio
        target_width = target_height * (aspect_ratio[0] / aspect_ratio[1])

        center_x, center_y = x + w // 2, y + h // 2
        crop_x1 = max(int(center_x - target_width // 2), 0)
        crop_y1 = max(int(center_y - target_height // 2), 0)
        crop_x2 = min(int(center_x + target_width // 2), image_np.shape[1])
        crop_y2 = min(int(center_y + target_height // 2), image_np.shape[0])

        cropped_image = aligned_image[crop_y1:crop_y2, crop_x1:crop_x2]
        cropped_image_pil = Image.fromarray(cropped_image)

        return cropped_image_pil
    except Exception as e:
        print(f"Error in face alignment: {e}")
        traceback.print_exc()
        return image
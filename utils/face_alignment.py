import numpy as np
import cv2
from PIL import Image
from utils.face_detection import face_detector, landmark_predictor
import traceback



def align_face(image, aspect_ratio=(4, 6), face_ratio=0.3):
    try:
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        faces = face_detector(gray)
        if len(faces) == 0:
            print("No face detected.")
            return image

        face = min(faces, key=lambda f: abs(f.center().x - gray.shape[1] // 2))
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())

        landmarks = landmark_predictor(gray, face)
        left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
        right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])

        delta_x = right_eye[0] - left_eye[0]
        delta_y = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(delta_y, delta_x))

        eye_center = tuple(map(int, ((left_eye + right_eye) / 2)))
        rotation_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        aligned_image = cv2.warpAffine(image_np, rotation_matrix, (image_np.shape[1], image_np.shape[0]))

        rotated_gray = cv2.cvtColor(aligned_image, cv2.COLOR_RGB2GRAY)
        rotated_faces = face_detector(rotated_gray)
        if len(rotated_faces) == 0:
            print("Face lost after rotation.")
            return image

        rotated_face = min(rotated_faces, key=lambda f: abs(f.center().x - rotated_gray.shape[1] // 2))

        rotated_landmarks = landmark_predictor(rotated_gray, rotated_face)
        left_eye_y = rotated_landmarks.part(36).y
        right_eye_y = rotated_landmarks.part(45).y
        eye_center_y = (left_eye_y + right_eye_y) // 2

        img_h, img_w = aligned_image.shape[:2]
        move_x = img_w // 2 - (x + w // 2)
        target_eye_y = int((2/3) * (img_h - eye_center_y))
        move_y = target_eye_y - eye_center_y
        translation_matrix = np.float32([[1, 0, move_x], [0, 1, move_y]])
        aligned_image = cv2.warpAffine(aligned_image, translation_matrix, (img_w, img_h))

        target_eye_y = img_h / 3
        eye_shift_y = target_eye_y - eye_center_y
        translation_matrix = np.float32([[1, 0, 0], [0, 1, eye_shift_y]])
        aligned_image = cv2.warpAffine(aligned_image, translation_matrix, (img_w, img_h))

        target_height = img_h / face_ratio
        target_width = target_height * (aspect_ratio[0] / aspect_ratio[1])
        center_x = img_w // 2
        crop_x1 = max(int(center_x - target_width // 2), 0)
        crop_x2 = min(int(center_x + target_width // 2), img_w)
        crop_y1 = max(int(target_eye_y - target_height / 3), 0)
        crop_y2 = min(int(crop_y1 + target_height), img_h)

        cropped_image = aligned_image[crop_y1:crop_y2, crop_x1:crop_x2]
        return Image.fromarray(cropped_image)
    except Exception as e:
        print(f"Error in face alignment: {e}")
        import traceback
        traceback.print_exc()
        return image
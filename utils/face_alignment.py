
import numpy as np
import cv2
from PIL import Image
from models.face_detection import face_detector, landmark_predictor
from utils.masks import load_overlay_image_width


def align_face(image, aspect_ratio=(4, 6), face_ratio=0.3):
    try:
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        faces = face_detector(gray)
        if len(faces) == 0:
            print("No face detected.")
            return image

        face = min(faces, key=lambda f: abs(f.center().x - gray.shape[1] // 2))
        # x, y, w, h = (face.left(), face.top(), face.width(), face.height())

        landmarks = landmark_predictor(gray, face)
        left_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])
        right_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])

        delta_x = right_eye[0] - left_eye[0]
        delta_y = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(delta_y, delta_x))
        angle = min(max(angle, -10), 10)

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

        eye_center = (
            (rotated_landmarks.part(36).x + rotated_landmarks.part(45).x) // 2,
            (rotated_landmarks.part(36).y + rotated_landmarks.part(45).y) // 2
        )

        face_h = rotated_face.height()
        canvas_h = int(face_h / face_ratio)
        canvas_w = int(canvas_h * aspect_ratio[0] / aspect_ratio[1])

        target_eye_x = canvas_w // 2
        target_eye_y = int((2 / 5) * canvas_h)

        move_x = target_eye_x - eye_center[0]
        move_y = target_eye_y - eye_center[1]

        translation_matrix = np.float32([[1, 0, move_x], [0, 1, move_y]])
        translated_image = cv2.warpAffine(aligned_image, translation_matrix, (canvas_w, canvas_h), borderValue=(255, 255, 255))

        crop_top = int(canvas_h * 2 / 5)
        crop_bottom = int(canvas_h * 3 / 5)
        crop = translated_image[crop_top:crop_bottom, :, :]

        stretch_height = canvas_h - crop_bottom
        stretched = cv2.resize(crop, (canvas_w, stretch_height), interpolation=cv2.INTER_LINEAR)

        mask = 255 * np.ones(stretched.shape, stretched.dtype)

        center_y = crop_bottom + stretch_height // 2
        center = (canvas_w // 2, center_y)

        blended = cv2.seamlessClone(
            stretched, translated_image, mask, center, cv2.NORMAL_CLONE
        )

        return Image.fromarray(blended).convert("RGB")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in face alignment: {e}")
        return image

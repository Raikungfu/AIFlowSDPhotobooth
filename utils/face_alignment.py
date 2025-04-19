
import numpy as np
import cv2
from PIL import Image
from models.face_detection import face_detector, landmark_predictor


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
        aligned_image = cv2.warpAffine(image_np, rotation_matrix, (image_np.shape[1], image_np.shape[0]), borderValue=(255, 255, 255))


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

        translated_image = fill_bottom_by_stretch(translated_image)

        return Image.fromarray(translated_image).convert("RGB")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in face alignment: {e}")
        return image
        

def fill_bottom_by_stretch(image, stretch_height=100, segment_width=10):
    h, w = image.shape[:2]

    for start_x in range(0, w, segment_width):
        end_x = min(start_x + segment_width, w)
        
        last_non_white_row = None
        for y in range(h - 1, -1, -1):
            row = image[y, start_x:end_x]
            white_pixels = np.sum(row > 240)
            total_pixels = len(row)
            
            if white_pixels < total_pixels * 0.02:
                last_non_white_row = y
                break
        
        if last_non_white_row is None:
            continue

        if last_non_white_row > 0:
            prev_row = image[last_non_white_row - 1, start_x:end_x]
            color_diff = np.sum(np.abs(prev_row - image[last_non_white_row, start_x:end_x]))
            stretch_height = max(50, int(color_diff / 100)) + 20

        stretch_start = last_non_white_row + 1
        stretch_end = min(h, stretch_start + stretch_height)

        for i in range(stretch_start, stretch_end):
            image[i, start_x:end_x] = image[last_non_white_row, start_x:end_x]

        if stretch_end < h:
            image[stretch_end:, start_x:end_x] = image[last_non_white_row, start_x:end_x]

    return image

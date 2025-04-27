import numpy as np
import cv2
from PIL import Image
from models.face_detection import landmark_predictor
from utils.clothes_utils.clothes_stretch import fill_bottom_by_stretch
from utils.face_utils.face_calculate import calculate_rotation_angle, calculate_translation, get_eye_centers
from utils.face_utils.face_detection import detect_main_face
from utils.face_utils.face_move import rotate_image, translate_image

def align_face(image, aspect_ratio=(4, 6), face_ratio=0.3):
    try:
        # Step 1: Preprocessing
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Step 2: Detect face
        face = detect_main_face(gray)
        if face is None:
            print("No face detected.")
            return image

        # Step 3: Find eyes and calculate rotation
        landmarks = landmark_predictor(gray, face)
        left_eye, right_eye = get_eye_centers(landmarks)
        angle = calculate_rotation_angle(left_eye, right_eye)
        eye_center = tuple(map(int, ((left_eye + right_eye) / 2)))

        # Step 4: Rotate the image
        aligned_image = rotate_image(image_np, eye_center, angle)

        # Step 5: Detect face again after rotation
        rotated_gray = cv2.cvtColor(aligned_image, cv2.COLOR_RGB2GRAY)
        rotated_face = detect_main_face(rotated_gray)
        if rotated_face is None:
            print("Face lost after rotation.")
            return image

        rotated_landmarks = landmark_predictor(rotated_gray, rotated_face)
        left_eye, right_eye = get_eye_centers(rotated_landmarks)
        eye_center = (
            (left_eye[0] + right_eye[0]) // 2,
            (left_eye[1] + right_eye[1]) // 2
        )

        # Step 6: Calculate new canvas size
        face_h = rotated_face.height()
        canvas_h = int(face_h / face_ratio)
        canvas_w = int(canvas_h * aspect_ratio[0] / aspect_ratio[1])

        # Step 7: Translate the image to reposition eyes
        move_x, move_y = calculate_translation(eye_center, canvas_w, canvas_h)
        translated_image = translate_image(aligned_image, move_x, move_y, canvas_w, canvas_h)

        # Step 8: Stretch bottom if needed
        translated_image = fill_bottom_by_stretch(translated_image)

        return Image.fromarray(translated_image).convert("RGB")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in face alignment: {e}")
        return image

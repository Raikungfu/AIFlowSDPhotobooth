
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
        face_w = rotated_face.width()
        canvas_h = int(face_h / face_ratio)
        canvas_w = int(canvas_h * aspect_ratio[0] / aspect_ratio[1])

        target_eye_x = canvas_w // 2
        target_eye_y = int((2 / 5) * canvas_h)

        move_x = target_eye_x - eye_center[0]
        move_y = target_eye_y - eye_center[1]

        translation_matrix = np.float32([[1, 0, move_x], [0, 1, move_y]])
        translated_image = cv2.warpAffine(aligned_image, translation_matrix, (canvas_w, canvas_h), borderValue=(255, 255, 255))

        chin = rotated_landmarks.part(8)
        chin_x, chin_y = chin.x + move_x, chin.y + move_y

        neck_width = int(face_w * 0.7)
        neck_height = int(face_h * 0.7)

        neck_top = chin_y
        neck_bottom = chin_y + neck_height
        neck_left = max(0, chin_x - neck_width // 2)
        neck_right = min(translated_image.shape[1], chin_x + neck_width // 2)

        head_part = translated_image[:neck_top, :, :]
        neck_part = translated_image[neck_top:neck_bottom, neck_left:neck_right, :]

        neck_part_resized = cv2.resize(neck_part, (head_part.shape[1], neck_part.shape[0]))

        final_image = np.vstack([head_part, neck_part_resized])

        return Image.fromarray(final_image).convert("RGB")

        # new_top = rotated_face.top() + move_y
        # new_height = rotated_face.height()

        # suit_url = "https://firebasestorage.googleapis.com/v0/b/social-networking-8c187.appspot.com/o/%E2%80%94Pngtree%E2%80%94mens%20black%20suit%20with%20white_8991014.png?alt=media&token=3f4aa208-1f5d-4b8f-80c2-d69da7905dc3"

        # image = image.convert("RGBA")
        # suit = load_overlay_image_width(suit_url,  canvas_w)

        # paste_y = int(new_top + new_height)
        # paste_x = 0

        # crop_bottom = int(new_top + new_height)
        # translated_image = translated_image[:crop_bottom, :, :]

        # base_image = Image.fromarray(translated_image).convert("RGBA")

        # base_image.paste(suit, (paste_x, paste_y), suit)

        return base_image.convert("RGB")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in face alignment: {e}")
        return image

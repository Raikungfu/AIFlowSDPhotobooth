import numpy as np
import cv2
from PIL import Image
from model.face_detection import face_detector, landmark_predictor
import traceback


def generate_face_lower_mask(image, image_mask):
    try:
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = face_detector(gray)

        if len(faces) == 0:
            print("No face detected.")
            return None, None, None

        face = faces[0]
        landmarks = landmark_predictor(gray, face)
        upper_lip_y = landmarks.part(51).y
        jaw_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(0, 17)])

        mask_clothes_np = np.zeros_like(image_mask, dtype=np.uint8)
        mask_clothes_np[upper_lip_y:, :] = 255
        cv2.fillPoly(mask_clothes_np, [jaw_points], (0, 0, 0))
        mask_clothes = Image.fromarray(mask_clothes_np)

        mask_face_np = np.array(image_mask, dtype=np.uint8)
        mask_face_np[upper_lip_y:, :] = 0
        cv2.fillPoly(mask_face_np, [jaw_points], (255, 255, 255))
        mask_face = Image.fromarray(mask_face_np)

        mask_combined_np = cv2.bitwise_or(mask_face_np, mask_clothes_np)
        mask_combined = Image.fromarray(mask_combined_np)

        return mask_face, mask_clothes, mask_combined
    except Exception as e:
        print(f"Error to create lower mask: {e}")
        traceback.print_exc()
        return None, None, None



def expand_mask(mask, kernel_size=15, iterations=2, threshold=128):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_np = np.array(mask.convert("L"))
    mask_expanded = cv2.dilate(mask_np, kernel, iterations=iterations)
    mask_expanded = cv2.dilate(mask_np, kernel, iterations=iterations)

    _, mask_binary = cv2.threshold(mask_expanded, threshold, 255, cv2.THRESH_BINARY)
    return Image.fromarray(mask_expanded)

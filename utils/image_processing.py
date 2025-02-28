import random
import string
import os
import numpy as np
from PIL import Image
from rembg import remove
import cv2
from utils.face_detection import face_detector, landmark_predictor
import traceback
import torch
from utils.stable_diffusion import inpainting_pipeline

def remove_background(image):
    try:
        transparent_image = remove(image, detection="soft", model_type="u2net", alpha_matting=True)
        alpha_channel = transparent_image.split()[-1]
        mask = alpha_channel.convert("L")
        return transparent_image, mask
    except Exception as e:
        print(f"Error removing background: {e}")
        return None, None

def place_on_new_background(foreground, new_background):
    new_background = new_background.resize(foreground.size, Image.Resampling.LANCZOS).convert("RGBA")
    return Image.alpha_composite(new_background, foreground)

def align_face(image, aspect_ratio=(4, 6), face_ratio=0.3):
    try:
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        faces = face_detector(gray)
        if len(faces) == 0:
            print("No face detected.")
            return image

        face = min(faces, key=lambda f: abs(f.center().x - gray.shape[1] // 2))

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

def process_image(image, image_mask):
    try:
        mask_face, mask_clothes = generate_face_lower_mask(image, image_mask)

        return mask_face, mask_clothes
    except Exception as e:
        print(f"Error to process image: {e}")
        traceback.print_exc()
        return None, None, None

def generate_random_filename(extension, length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length)) + extension

def inpaint(image, mask, prompt, negative_prompt, seek):
    try:
        image = image.convert("RGB")
        mask = mask.convert("L")

        with torch.no_grad():
            result = inpainting_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                strength=0.5,
                guidance_scale=10,
                num_inference_steps=50,
                height=(image.height // 8) * 8,
                width=(image.width // 8) * 8,
                generator=torch.Generator("cpu").manual_seed(int(seek))
            ).images[0]

        return result
    except Exception as e:
        print(f"Error during inpainting: {e}")
        return None

def generate_inpainting_prompt(IsGenFace=True):
    if IsGenFace:
        base_prompt = "Professional passport-style portrait, even lighting, flawless and smooth skin, sharp and well-defined facial features, natural skin tone, high detail, well-groomed hair, forehead fully visible, ears slightly visible, neutral expression, straight posture, clean background, high resolution, studio lighting, formal appearance."
        negative_prompt = "Messy hair, forehead covered, hair over face, shadows, acne, blemishes, wrinkles, scars, distorted features, poor details, unnatural skin tone, dark patches, extreme makeup, blurry image, bad quality, low resolution, exaggerated expressions, artistic filters."
    else:
        base_prompt = (
            "Professional passport-style portrait, formal attire, professional suit, white or light-colored collared shirt, neutral tie, clean and polished appearance, well-fitted clothing, natural fabric texture, studio lighting, high detail, 8k resolution, realistic folds and shadows, professional portrait style"
        )
        negative_prompt = (
            "casual wear, t-shirts, logos, patterns, bright colors, excessive accessories, low resolution, distorted, unrealistic clothing texture, blurry, pixelated, fantasy elements, overly dramatic lighting, unnatural colors, unprofessional look"
        )

    return base_prompt, negative_prompt

def generate_face_lower_mask(image, image_mask):
    try:
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = face_detector(gray)

        if len(faces) == 0:
            print("No face detected.")
            return None, None

        face = faces[0]
        landmarks = landmark_predictor(gray, face)
        lower_lip_y = landmarks.part(57).y
        jaw_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(0, 17)])

        mask_clothes_np = np.zeros_like(image_mask)
        mask_clothes_np[lower_lip_y:, :] = 255
        cv2.fillPoly(mask_clothes_np, [jaw_points], (0, 0, 0))
        mask_clothes = Image.fromarray(mask_clothes_np)

        mask_face_np = np.array(image_mask)
        mask_face_np[lower_lip_y:, :] = 0
        cv2.fillPoly(mask_face_np, [jaw_points], (255, 255, 255))
        mask_face = Image.fromarray(mask_face_np)

        return mask_face, mask_clothes
    except Exception as e:
        print(f"Error to create lower mask: {e}")
        traceback.print_exc()
        return None, None
import numpy as np
import cv2
from PIL import Image
from models.face_detection import detect_faces, face_detector
import traceback
import requests
from io import BytesIO
from utils.face_utils.face_detection import get_face_landmarks
from utils.face_utils.image_utils import convert_image_to_numpy, ensure_uint8
from torchvision import transforms


def generate_face_lower_mask(image, image_mask, get_list_face):
    try:
        if image is None:
            raise ValueError("Input images cannot be None!")

        image_np = convert_image_to_numpy(image)
        image_np = ensure_uint8(image_np)

        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = detect_faces(gray)

        if len(faces) == 0:
            print("No face detected.")
            return None, None, None
        
        list_face = []
        
        if get_list_face:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                face_crop = image_np[y:y+h, x:x+w]
                pil_face = Image.fromarray(face_crop)
                face_tensor = transform(pil_face).unsqueeze(0)
                list_face.append(face_tensor) 
        else:
            face = faces[0]
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_crop = image_np[y:y+h, x:x+w]
            list_face.append(Image.fromarray(face_crop))            

        mask_clothes = mask_face = mask_combined = None

        if image_mask is not None:
            landmarks = get_face_landmarks(gray, faces[0])
            upper_lip_y = landmarks.part(51).y
            jaw_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(0, 17)])
            mask_clothes = create_clothes_mask(image_mask, upper_lip_y, jaw_points)
            mask_face = create_face_mask(image_mask, upper_lip_y, jaw_points)
            mask_combined = create_combined_mask(np.array(mask_face), np.array(mask_clothes))

        return mask_clothes, mask_combined, list_face
    except Exception as e:
        print(f"Error to create lower mask: {e}")
        traceback.print_exc()
        return None, None, None



def create_clothes_mask(image_mask, upper_lip_y, jaw_points):
    # Creates a clothes mask: white below the upper lip, black around the jaw.
    mask_clothes_np = np.zeros_like(image_mask, dtype=np.uint8)
    mask_clothes_np[upper_lip_y:, :] = 255
    cv2.fillPoly(mask_clothes_np, [jaw_points], (0, 0, 0))
    return Image.fromarray(mask_clothes_np)


def create_face_mask(image_mask, upper_lip_y, jaw_points):
    # Creates a face mask: black below the upper lip, white around the jaw.
    mask_face_np = np.array(image_mask, dtype=np.uint8)
    mask_face_np[upper_lip_y:, :] = 0
    cv2.fillPoly(mask_face_np, [jaw_points], (255, 255, 255))
    return Image.fromarray(mask_face_np)


def create_combined_mask(mask_face_np, mask_clothes_np):
    # Combines face and clothes masks using bitwise OR.
    mask_combined_np = cv2.bitwise_or(mask_face_np, mask_clothes_np)
    return Image.fromarray(mask_combined_np)


def generate_only_face(image):
    try:
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        faces = face_detector(gray)

        if len(faces) == 0:
            print("No face detected.")
            return None, None, None
        
        face = faces[0]
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_crop = image_np[y:y+h, x:x+w]
        
        face_image = Image.fromarray(face_crop)

        return face_image
    except Exception as e:
        print(f"Error to create lower mask: {e}")
        traceback.print_exc()
        return None


def expand_mask(mask, kernel_size=15, iterations=2, threshold=128):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_np = np.array(mask.convert("L"))
    mask_expanded = cv2.dilate(mask_np, kernel, iterations=iterations)
    mask_expanded = cv2.dilate(mask_np, kernel, iterations=iterations)

    _, mask_binary = cv2.threshold(mask_expanded, threshold, 255, cv2.THRESH_BINARY)
    return Image.fromarray(mask_expanded)


def load_overlay_image(url, target_size):
    response = requests.get(url)
    overlay = Image.open(BytesIO(response.content)).convert("RGBA")
    return overlay.resize(target_size, Image.Resampling.LANCZOS)

def load_overlay_image_width(url, canvas_w):
    response = requests.get(url)
    overlay = Image.open(BytesIO(response.content)).convert("RGBA")
    suit_cropped = overlay.crop((0, 1110, overlay.width, overlay.height))
    scale_ratio = canvas_w / suit_cropped.width
    new_suit_size = (canvas_w, int(suit_cropped.height * scale_ratio))
    return suit_cropped.resize(new_suit_size, Image.LANCZOS)


def apply_overlay_to_image(base_image, overlay_image):
    if base_image.mode != "RGBA":
        base_image = base_image.convert("RGBA")
    return Image.alpha_composite(base_image, overlay_image)

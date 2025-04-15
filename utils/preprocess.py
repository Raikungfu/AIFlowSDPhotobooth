import torch
import numpy as np
from PIL import Image

from configs.enums.inpaint_mode import InpaintMode
from utils.background_prcessing import place_on_new_background, remove_background
from utils.controlnet_images import create_controlnet_images
from utils.face_alignment import align_face
from utils.masks import generate_face_lower_mask
from utils.optimize_image import optimize_image

def make_inpaint_condition(image, image_mask=None):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    
    if image_mask is None:
        image_mask = Image.new("L", image.shape[1::-1], 255)
    else:
        image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    
    image[image_mask > 0.5] = -1.0
    
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    
    return image


def align_and_optimize_image(image, should_align: bool = False):
    image = optimize_image(image)
    if(should_align):
        return align_face(image)
    return image
    

def remove_bg_and_place_on_white(optimized_image, background_remover):
    if not background_remover:
        return optimized_image, None

    image_no_bg, image_mask = remove_background(optimized_image)
    if image_no_bg is None or image_mask is None:
        raise ValueError("Failed to remove background or generate mask")

    white_bg = Image.new("RGBA", optimized_image.size, (255, 255, 255, 255))
    image_with_bg = place_on_new_background(image_no_bg, white_bg).convert("RGB")
    return image_with_bg, image_mask


def generate_masks(mode, optimized_image, image_mask):
    mask = mask_combined = face_image = None

    if mode == InpaintMode.KEEP_STRUCTURE:
        mask = optimized_image.convert("L")

    elif mode == InpaintMode.KEEP_FACE:
        mask_face, mask_clothes, mask_combined, face_image = generate_face_lower_mask(optimized_image, image_mask)
        if not all([mask_face, mask_clothes, mask_combined]):
            raise ValueError("Failed to generate face or clothes mask")

    return mask, face_image, mask_combined
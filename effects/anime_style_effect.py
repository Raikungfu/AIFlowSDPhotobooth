import traceback

import numpy as np
from configs.prompt import generate_anime_prompt
from pipelines.inpainting.inpaint_pipeline import inpaint_with_ip_sd_controlnet
from utils.background_prcessing import remove_background
from utils.controlnet_images import create_canny_image, create_openpose_image
from utils.optimize_image import optimize_image
from utils.masks import generate_face_lower_mask
from PIL import Image

def apply_anime_filter(image):
    try:
        optimized_image = optimize_image(image)

        if isinstance(optimized_image, np.ndarray):
            optimized_image = Image.fromarray(optimized_image)

        mask = Image.new("L", optimized_image.size, 255)

        image_no_bg, image_mask = remove_background(optimized_image)
        
        openpose_image = create_openpose_image(optimized_image)
        canny_image = create_canny_image(optimized_image)

        mask_face, mask_other, mask_combined, face_image = generate_face_lower_mask(image_no_bg, image_mask)
        if face_image is None:
            raise ValueError("Cannot detect face!")

        base_prompt, negative_prompt = generate_anime_prompt()

        inpainted_images = inpaint_with_ip_sd_controlnet(
            number=6,
            image=image_no_bg,
            mask=mask,
            face_image=face_image,
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            strength=0.8,
            scale=9.0,
            control_images=[openpose_image, canny_image],
            controlnets=["openpose", "canny"]
        )

        if not inpainted_images:
            raise Exception("Inpainting fail!")

        return inpainted_images
    except Exception as e:
        traceback.print_exc()
        print(f"Error applying anime effect: {e}")
        return None

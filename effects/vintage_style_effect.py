import traceback
from PIL import Image
import numpy as np
from configs.prompt import generate_vintage_prompt
from pipelines.inpainting.inpaint_pipeline import inpaint_with_ip_sd_controlnet
from utils.background_prcessing import remove_background
from utils.controlnet_images import create_controlnet_image
from utils.optimize_image import optimize_image
from utils.masks import generate_face_lower_mask

def apply_vintage_effect(image):
    try:
        optimized_image = optimize_image(image)
        image_no_bg, image_mask = remove_background(optimized_image)

        depth_image, openpose_image, canny_image = create_controlnet_image(optimized_image)

        mask_face, mask_clothes, mask_combined, face_image = generate_face_lower_mask(image_no_bg, image_mask)
        if face_image is None:
            raise ValueError("Cannot detect face!")

        base_prompt, negative_prompt = generate_vintage_prompt()
        
        print(type(image_no_bg), type(mask_combined), type(face_image))

        inpainted_images = inpaint_with_ip_sd_controlnet(
            number=6,
            image=image_no_bg,
            mask=mask_combined,
            face_image=face_image,
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            strength=0.6,
            scale=7.5,
            control_images=[openpose_image, canny_image, depth_image]
        )

        if not inpainted_images:
            raise Exception("Inpainting fail!")

        return inpainted_images

    except Exception as e:
        traceback.print_exc()
        print(f"Error  applying vintage effect: {e}")
        return None

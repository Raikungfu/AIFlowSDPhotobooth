import traceback
from PIL import Image
import numpy as np
from configs.prompt import generate_artistic_prompt
from pipelines.inpainting.inpaint_pipeline import inpaint_with_ip_sd_controlnet
from utils.background_prcessing import remove_background
from utils.controlnet_images import create_depth_image, create_openpose_image
from utils.optimize_image import optimize_image
from utils.masks import generate_face_lower_mask

def apply_artistic_effects(image):
    try:
        optimized_image = optimize_image(image)

        image_no_bg, image_mask = remove_background(optimized_image)
        
        openpose_image = create_openpose_image(optimized_image)
        depth_image = create_depth_image(optimized_image)

        mask_face, mask_clothes, mask_combined, face_image = generate_face_lower_mask(image_no_bg, image_mask)
        if face_image is None:
            raise ValueError("Cannot detect face!")

        base_prompt, negative_prompt = generate_artistic_prompt()

        inpainted_images = inpaint_with_ip_sd_controlnet(
            number=6,
            image=image_no_bg,
            mask=mask_combined,
            face_image=face_image,
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            strength=0.5,
            scale=7.0,
            control_images=[openpose_image, depth_image],
            controlnets=["openpose", "depth"]
        )

        if not inpainted_images:
            raise Exception("Inpainting fail!")
        
        return inpainted_images

    except Exception as e:
        traceback.print_exc()
        print(f"Error applying artistic effect: {e}")
        return None

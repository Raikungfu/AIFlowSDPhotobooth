import traceback

import numpy as np
from configs.prompt import generate_modern_prompt
from pipelines.inpainting.inpaint_pipeline import inpaint_with_ip_sd_controlnet
from configs.enums.inpaint_mode import InpaintMode
from utils.controlnet_images import create_canny_image, create_depth_image
from utils.optimize_image import optimize_image
from PIL import Image

def apply_modern_effect(image):
    try:
        optimized_image = optimize_image(image)
        
        if isinstance(optimized_image, np.ndarray):
            optimized_image = Image.fromarray(optimized_image)

        canny_image = create_canny_image(optimized_image)
        depth_image = create_depth_image(optimized_image)
        
        base_prompt, negative_prompt = generate_modern_prompt()
        print(type(optimized_image))

        inpainted_images = inpaint_with_ip_sd_controlnet(
            number=6,
            image=optimized_image,
            mask=None,
            face_image=None,
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            strength=0.5,
            scale=7.5,
            control_images=[canny_image, depth_image],
            controlnets=["canny", "depth"],
            mode=InpaintMode.KEEP_STRUCTURE
        )

        return inpainted_images
    except Exception as e:
        traceback.print_exc()
        print(f"Error applying modern effect: {e}")
        return None

import traceback

import numpy as np
from configs.enums.inpaint_mode import InpaintMode
from configs.prompt import generate_nature_prompt
from utils.canny import create_canny_image
from utils.optimize_image import optimize_image
from pipelines.inpainting.inpaint_pipeline import inpaint_with_ip_sd_controlnet
from PIL import Image

def apply_nature_effect(image, controlnets):
    try:
        optimized_image = optimize_image(image)
        
        if isinstance(optimized_image, np.ndarray):
            optimized_image = Image.fromarray(optimized_image)

        depth_image = create_canny_image(optimized_image)

        if isinstance(depth_image, np.ndarray):
            depth_image = Image.fromarray(depth_image)

        base_prompt, negative_prompt = generate_nature_prompt()
        print(type(optimized_image))

        inpainted_images = inpaint_with_ip_sd_controlnet(
            number=6,
            image=optimized_image,
            mask=None,
            face_image=None,
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            strength=0.6,
            scale=8.0,
            controlnets=controlnets,
            control_image=[depth_image],
            controlnet_type="depth",
        )

        return inpainted_images

    except Exception as e:
        traceback.print_exc()
        print(f"Error applying nature effect: {e}")
        return None

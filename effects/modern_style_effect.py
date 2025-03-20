import traceback

import numpy as np
from configs.prompt import generate_modern_prompt
from pipelines.inpainting.inpaint_pipeline import inpaint_with_ip_sd_controlnet
from configs.enums.inpaint_mode import InpaintMode
from utils.controlnet_images import create_canny_image, create_depth_image, create_softedge_image
from utils.optimize_image import optimize_image
from PIL import Image

def apply_modern_effect(image):
    try:
        optimized_image = optimize_image(image)
        
        if isinstance(optimized_image, np.ndarray):
            optimized_image = Image.fromarray(optimized_image)

        canny_image = create_canny_image(optimized_image)
        control_mode_canny = 0
        control_conditioning_scale_mode_canny = 0.9

        depth_image = create_depth_image(optimized_image)
        control_mode_depth = 2
        control_conditioning_scale_mode_depth = 0.7

        softedge_image = create_softedge_image(optimized_image)
        control_mode_softedge = 0
        control_conditioning_scale_mode_softedge = 1.0

        base_prompt, negative_prompt = generate_modern_prompt()
        print(type(optimized_image))

        inpainted_images = inpaint_with_ip_sd_controlnet(
            number=6,
            image=optimized_image,
            mask=None,
            face_image=None,
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25,
            guidance_scale=6.8,
            strength=0.7,
            control_image=[canny_image, depth_image, softedge_image],
            control_mode=[control_mode_canny, control_mode_depth, control_mode_softedge],
            controlnet_conditioning_scale=[control_conditioning_scale_mode_canny, control_conditioning_scale_mode_depth, control_conditioning_scale_mode_softedge],
            mode=InpaintMode.KEEP_STRUCTURE
        )

        return inpainted_images
    except Exception as e:
        traceback.print_exc()
        print(f"Error applying modern effect: {e}")
        return None

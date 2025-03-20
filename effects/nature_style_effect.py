import traceback
import numpy as np
from configs.prompt import generate_nature_prompt
from utils.controlnet_images import create_depth_image, create_softedge_image
from utils.optimize_image import optimize_image
from pipelines.inpainting.inpaint_pipeline import inpaint_with_ip_sd_controlnet
from PIL import Image

def apply_nature_effect(image):
    try:
        optimized_image = optimize_image(image)
        
        if isinstance(optimized_image, np.ndarray):
            optimized_image = Image.fromarray(optimized_image)

        depth_image = create_depth_image(optimized_image)
        control_mode_depth = 2
        control_conditioning_scale_mode_depth = 0.7

        softedge_image = create_softedge_image(optimized_image)
        control_mode_softedge = 0
        control_conditioning_scale_mode_softedge = 1.0

        base_prompt, negative_prompt = generate_nature_prompt()
        print(type(optimized_image))

        inpainted_images = inpaint_with_ip_sd_controlnet(
            number=6,
            image=optimized_image,
            mask=None,
            face_image=None,
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.2,
            strength=0.75,
            control_image=[softedge_image, depth_image],
            control_mode=[control_mode_softedge, control_mode_depth],
            controlnet_conditioning_scale=[control_conditioning_scale_mode_softedge, control_conditioning_scale_mode_depth],
        )

        return inpainted_images

    except Exception as e:
        traceback.print_exc()
        print(f"Error applying nature effect: {e}")
        return None

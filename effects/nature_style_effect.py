import traceback
import numpy as np
from configs.prompt import generate_nature_prompt
from utils.canny import create_canny_image
from utils.controlnet_images import create_controlnet_image
from utils.optimize_image import optimize_image
from pipelines.inpainting.inpaint_pipeline import inpaint_with_ip_sd_controlnet
from PIL import Image

def apply_nature_effect(image):
    try:
        optimized_image = optimize_image(image)
        
        if isinstance(optimized_image, np.ndarray):
            optimized_image = Image.fromarray(optimized_image)

        depth_image, openpose_image, canny_image = create_controlnet_image(optimized_image)

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
            control_images=[openpose_image, canny_image, depth_image],
        )

        return inpainted_images

    except Exception as e:
        traceback.print_exc()
        print(f"Error applying nature effect: {e}")
        return None

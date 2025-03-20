import traceback

import numpy as np
from configs.prompt import generate_anime_prompt
from pipelines.inpainting.inpaint_pipeline import inpaint_with_ip_sd_controlnet
from utils.background_prcessing import remove_background
from utils.controlnet_images import create_scribble_image, create_softedge_image
from utils.optimize_image import optimize_image
from PIL import Image

def apply_anime_filter(image):
    try:
        optimized_image = optimize_image(image)

        if isinstance(optimized_image, np.ndarray):
            optimized_image = Image.fromarray(optimized_image)

        mask = Image.new("L", optimized_image.size, 255)

        image_no_bg, image_mask = remove_background(optimized_image)

        scribble_image = create_scribble_image(optimized_image)
        control_mode_scribble = 0
        control_conditioning_scale_mode_scribble = 0.8
        
        softedge_image = create_softedge_image(optimized_image)
        control_mode_softedge = 0
        control_conditioning_scale_mode_softedge = 1.0

        # mask_face, mask_other, mask_combined, face_image = generate_face_lower_mask(image_no_bg, image_mask)
        # if face_image is None:
        #     raise ValueError("Cannot detect face!")

        base_prompt, negative_prompt = generate_anime_prompt()

        inpainted_images = inpaint_with_ip_sd_controlnet(
            number=6,
            image=image_no_bg,
            mask=mask,
            face_image=None,
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            strength=0.8,
            control_image=[scribble_image, softedge_image],
            control_mode=[control_mode_scribble, control_mode_softedge],
            controlnet_conditioning_scale=[control_conditioning_scale_mode_scribble, control_conditioning_scale_mode_softedge],
        )

        if not inpainted_images:
            raise Exception("Inpainting fail!")

        return inpainted_images
    except Exception as e:
        traceback.print_exc()
        print(f"Error applying anime effect: {e}")
        return None

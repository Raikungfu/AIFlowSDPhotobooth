import traceback
import numpy as np
from configs.prompt import generate_cartoon_prompt
from pipelines.inpainting.inpaint_pipeline import inpaint_with_ip_sd_controlnet
from utils.background_prcessing import remove_background
from utils.controlnet_images import create_canny_image, create_scribble_image
from utils.optimize_image import optimize_image
from PIL import Image

def apply_cartoon_filter(image):
    try:
        optimized_image = optimize_image(image)

        if isinstance(optimized_image, np.ndarray):
            optimized_image = Image.fromarray(optimized_image)

        mask = Image.new("L", optimized_image.size, 255)

        image_no_bg, image_mask = remove_background(optimized_image)

        canny_image = create_canny_image(optimized_image)
        control_mode_canny = 0
        control_conditioning_scale_mode_canny = 0.9
        
        scribble_image = create_scribble_image(optimized_image)
        control_mode_scribble = 0
        control_conditioning_scale_mode_scribble = 1.0

        # mask_face, mask_other, mask_combined, face_image = generate_face_lower_mask(image_no_bg, image_mask)

        # if face_image is None:
        #     raise ValueError("Cannot detect face!")

        base_prompt, negative_prompt = generate_cartoon_prompt()

        inpainted_images = inpaint_with_ip_sd_controlnet(
            number=6,
            image=image_no_bg,
            mask=mask,
            face_image=None,
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=32,
            guidance_scale=8,
            strength=0.85,
            control_image=[canny_image, scribble_image],
            control_mode=[control_mode_canny, control_mode_scribble],
            controlnet_conditioning_scale=[control_conditioning_scale_mode_canny, control_conditioning_scale_mode_scribble]
        )

        if not inpainted_images:
            raise Exception("Inpainting fail!")

        return inpainted_images
    except Exception as e:
        traceback.print_exc()
        print(f"Error applying cartoon effect: {e}")
        return None

import traceback
from PIL import Image
import numpy as np
from configs.enums.inpaint_mode import InpaintMode
from configs.prompt import generate_fashion_style_prompt
from pipelines.inpainting.inpaint_pipeline import inpaint_with_ip_sd, inpaint_with_ip_sd_controlnet
from utils.background_prcessing import remove_background
from utils.canny import create_canny_image
from utils.masks import generate_face_lower_mask
from utils.open_pose import create_openpose_image
from utils.optimize_image import optimize_image

def apply_fashion_effect(image):
    try:
        optimized_image = optimize_image(image)

        mask = optimized_image.convert("L")

        image_no_bg, image_mask = remove_background(optimized_image)        
        
        canny_image = create_canny_image(image)

        if isinstance(canny_image, np.ndarray):
            canny_image = Image.fromarray(canny_image)

        openpose_image = create_openpose_image(image)
        if isinstance(openpose_image, np.ndarray):
            openpose_image = Image.fromarray(openpose_image)

        face_mask, clothes_mask, combined_mask, face_image = generate_face_lower_mask(optimized_image, image_mask)
        if face_image is None:
            raise ValueError("Cannot detect face!")

        base_prompt, negative_prompt = generate_fashion_style_prompt()

        inpainted_images = inpaint_with_ip_sd_controlnet(
            number=6,
            image=image_no_bg,
            mask=combined_mask,
            face_image=face_image,
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            strength=0.8,
            scale=9.0,
            control_images=[openpose_image, canny_image, None],
            controlnet_type="canny,openpose",
        )


        return inpainted_images

    except Exception as e:
        traceback.print_exc()
        print(f"Error applying fashion effect: {e}")
        return None
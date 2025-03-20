import traceback
from configs.prompt import generate_artistic_prompt
from pipelines.inpainting.inpaint_pipeline import inpaint_with_ip_sd_controlnet
from utils.background_prcessing import remove_background
from utils.controlnet_images import create_depth_image, create_scribble_image
from utils.optimize_image import optimize_image
from utils.masks import generate_face_lower_mask

def apply_artistic_effects(image):
    try:
        optimized_image = optimize_image(image)

        image_no_bg, image_mask = remove_background(optimized_image)
        
        scribble_image = create_scribble_image(optimized_image)
        control_mode_scribble = 0
        control_conditioning_scale_mode_scribble = 1.2

        depth_image = create_depth_image(optimized_image)
        control_mode_depth = 2
        control_conditioning_scale_mode_depth = 0.7

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
            num_inference_steps=28,
            guidance_scale=7.0,
            strength=0.75,
            control_image=[scribble_image, depth_image],
            control_mode=[control_mode_scribble, control_mode_depth],
            controlnet_conditioning_scale=[control_conditioning_scale_mode_scribble, control_conditioning_scale_mode_depth],
        )

        if not inpainted_images:
            raise Exception("Inpainting fail!")
        
        return inpainted_images

    except Exception as e:
        traceback.print_exc()
        print(f"Error applying artistic effect: {e}")
        return None

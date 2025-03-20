import traceback
from PIL import Image
from configs.prompt import generate_fantasy_prompt
from configs.enums.inpaint_mode import InpaintMode
from pipelines.inpainting.inpaint_pipeline import inpaint_with_ip_sd_controlnet
from utils.background_prcessing import remove_background
from utils.controlnet_images import create_depth_image, create_softedge_image
from utils.masks import generate_face_lower_mask
from utils.optimize_image import optimize_image

def apply_fantasy_effect(image):
    try:
        optimized_image = optimize_image(image)

        mask = Image.new("L", optimized_image.size, 255)
        
        image_no_bg, image_mask = remove_background(optimized_image)
        mask_face, mask_clothes, mask_combined, face_image = generate_face_lower_mask(image_no_bg, image_mask)

        softedge_image = create_softedge_image(optimized_image)
        control_mode_softedge = 0
        control_conditioning_scale_mode_softedge = 0.7
        
        depth_image = create_depth_image(optimized_image)
        control_mode_depth = 2
        control_conditioning_scale_mode_depth = 1.0

        base_prompt, negative_prompt = generate_fantasy_prompt()
        print(type(optimized_image))
        
        inpainted_images = inpaint_with_ip_sd_controlnet(
            number=6,
            image=image_no_bg,
            mask=mask,
            face_image=face_image,
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=35,
            guidance_scale=7.8,
            strength=0.8,
            control_image=[softedge_image, depth_image],
            control_mode=[control_mode_softedge, control_mode_depth],
            controlnet_conditioning_scale=[control_conditioning_scale_mode_softedge, control_conditioning_scale_mode_depth],
            mode=InpaintMode.KEEP_STRUCTURE
        )

        if not inpainted_images:
            raise Exception("Inpainting fail!")

        return inpainted_images
    except Exception as e:
        traceback.print_exc()
        print(f"Error applying fantasy effect: {e}")
        return None

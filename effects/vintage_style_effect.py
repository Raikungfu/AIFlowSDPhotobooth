import traceback
from configs.enums.inpaint_mode import InpaintMode
from configs.prompt import generate_vintage_prompt
from pipelines.inpainting.inpaint_pipeline import inpaint_with_ip_sd_controlnet
from utils.background_prcessing import remove_background
from utils.controlnet_images import create_scribble_image, create_softedge_image
from utils.optimize_image import optimize_image
from utils.masks import generate_face_lower_mask

def apply_vintage_effect(image):
    try:
        optimized_image = optimize_image(image)
        image_no_bg, image_mask = remove_background(optimized_image)

        softedge_image = create_softedge_image(optimized_image)
        control_mode_softedge = 0
        control_conditioning_scale_mode_softedge = 0.6

        scribble_image = create_scribble_image(optimized_image)
        control_mode_scribble = 0
        control_conditioning_scale_mode_scribble = 0.8

        mask_face, mask_clothes, mask_combined, face_image = generate_face_lower_mask(image_no_bg, image_mask)
        if face_image is None:
            raise ValueError("Cannot detect face!")

        base_prompt, negative_prompt = generate_vintage_prompt()
        
        print(type(image_no_bg), type(mask_combined), type(face_image))

        inpainted_images = inpaint_with_ip_sd_controlnet(
            number=6,
            image=image_no_bg,
            mask=mask_combined,
            face_image=face_image,
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=25,
            guidance_scale=7.0,
            strength=0.8,
            control_image=[softedge_image, scribble_image],
            control_mode=[control_mode_softedge, control_mode_scribble],
            controlnet_conditioning_scale=[control_conditioning_scale_mode_softedge, control_conditioning_scale_mode_scribble],
            mode=InpaintMode.KEEP_FACE
        )

        if not inpainted_images:
            raise Exception("Inpainting fail!")

        return inpainted_images

    except Exception as e:
        traceback.print_exc()
        print(f"Error  applying vintage effect: {e}")
        return None

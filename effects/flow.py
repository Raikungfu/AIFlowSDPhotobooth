import traceback
from PIL import Image
from configs.enums.inpaint_mode import InpaintMode
from pipelines.inpainting.inpaint_pipeline import inpaint_with_ip_sd_controlnet
from utils.controlnet_images import create_controlnet_images
from utils.preprocess import align_and_optimize_image, generate_masks, remove_bg_and_place_on_white
from utils.process_input import get_flux_controlnet_mode

def process_image(
    image,
    photo_style,
    prompt,
    negative_prompt,
    controlnets,
    numImages,
    numInferenceSteps,
    guidance_scale,
    strength,
    face_image,
    background_remover,
    height,
    width,
    mode=InpaintMode.KEEP_STRUCTURE
):
    try:
        optimized_image = align_and_optimize_image(image, "ID Photo" in photo_style)
        image_with_bg, image_mask = remove_bg_and_place_on_white(optimized_image, background_remover)

        mask, face_image, mask_combined = generate_masks(mode, optimized_image, image_mask)

        control_images = create_controlnet_images(controlnets, optimized_image)

        control_modes = [get_flux_controlnet_mode(cn["type"]) for cn in controlnets]
        conditioning_scales = [cn["weight"] for cn in controlnets]

        inpainted_images = inpaint_with_ip_sd_controlnet(
            number=numImages,
            image=image_with_bg,
            mask=mask_combined or mask,
            face_image=face_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            num_inference_steps=numInferenceSteps,
            guidance_scale=guidance_scale,
            control_images=control_images if control_images else None,
            control_modes=control_modes,
            controlnet_conditioning_scale=conditioning_scales,
            mode=mode,
            height=height,
            width=width
        )
        
        return inpainted_images if inpainted_images else []

    except Exception as e:
        traceback.print_exc()
        print(f"Error processing photo style {photo_style}: {e}")
        return None
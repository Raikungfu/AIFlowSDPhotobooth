import traceback
from PIL import Image
from configs.enums.inpaint_mode import InpaintMode
from pipelines.inpainting.inpaint_pipeline import inpaint_with_ip_sd_controlnet, inpaint_without_controlnet
from utils.controlnet_images import create_controlnet_images
from utils.preprocess import align_and_optimize_image, closestNumber, generate_masks, remove_bg_and_place_on_white
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

        inpainted_images = []        

        if height == None or width == None:
            height = closestNumber(optimized_image.height, 16)
            width =  closestNumber(optimized_image.width, 16)
        else:
            height = closestNumber(height, 16)
            width = closestNumber(width, 16)

        if photo_style == "ID Photo":
            print("📸 Style: ID Photo → inpaint WITHOUT ControlNet")
            image_with_bg, image_mask = remove_bg_and_place_on_white(optimized_image, background_remover)
            mask, face_image, mask_combined = generate_masks(mode, optimized_image, image_mask)
            inpainted_images = inpaint_without_controlnet(
                number=numImages,
                image=image_with_bg,
                mask=mask_combined or mask,
                face_image=face_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                num_inference_steps=numInferenceSteps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                mode=mode,
            )
        else:
            print("🎨 Style: Other → inpaint WITH ControlNet")
            control_images = create_controlnet_images(controlnets, optimized_image)
            image_with_bg, image_mask = remove_bg_and_place_on_white(optimized_image, background_remover)
            mask, face_image, mask_combined = generate_masks(mode, optimized_image, image_mask)
            inpainted_images = inpaint_with_ip_sd_controlnet(
                number=numImages,
                image=image_with_bg,
                mask=mask,
                face_image=face_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                num_inference_steps=numInferenceSteps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                control_images=control_images if control_images else None,
            )
        
        return inpainted_images

    except Exception as e:
        traceback.print_exc()
        print(f"Error processing photo style {photo_style}: {e}")
        return None
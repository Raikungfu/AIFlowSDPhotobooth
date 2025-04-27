import traceback
from pipelines.inpaint_pipeline import inpaint_with_ip_sd_controlnet, inpaint_without_controlnet
from image_input_controlnets.controlnet_images import create_controlnet_images
from utils.inpaint_pipeline_utils.preprocess import align_and_optimize_image, closestNumber, generate_masks, remove_bg_and_place_on_white

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
    mode
):
    try:
        optimized_image = align_and_optimize_image(image, "ID Photo" in photo_style)

        inpainted_images = []

        if height == None or width == None:
            height = closestNumber(optimized_image.height, 8)
            width =  closestNumber(optimized_image.width, 8)
        else:
            height = closestNumber(height, 8)
            width = closestNumber(width, 8)
            
        image_with_bg, image_mask = remove_bg_and_place_on_white(optimized_image, background_remover)
        mask, face_image, mask_clothes, mask_combined = generate_masks(mode, optimized_image, image_mask)

        if photo_style.lower() == "id photo":
            print("Inpaint WITHOUT ControlNet")
            inpainted_images = inpaint_without_controlnet(
                number=numImages,
                image=image_with_bg,
                mask=mask_combined or mask,
                mask_clothes=mask_clothes,
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
            print("Inpaint WITH ControlNet")
            controlnet_models = [cn["type"] for cn in controlnets]
            control_images = create_controlnet_images(controlnet_models, optimized_image)
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
                controlnets=controlnet_models,
                mode=mode,
            )
        
        return inpainted_images

    except Exception as e:
        traceback.print_exc()
        print(f"Error processing photo style {photo_style}: {e}")
        return None
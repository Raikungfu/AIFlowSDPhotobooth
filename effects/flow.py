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
    background_remover,
    height,
    width,
    mode, 
    ip_adapter_scale
):
    try:
        optimized_image = align_and_optimize_image(image, "ID Photo" in photo_style)

        inpainted_images = []

        if height == None or height == 0 or width == None or width == 0:
            height = closestNumber(optimized_image.height, 8)
            width =  closestNumber(optimized_image.width, 8)
        else:
            height = closestNumber(height, 8)
            width = closestNumber(width, 8)
            
        image_with_bg, image_mask = remove_bg_and_place_on_white(optimized_image, background_remover)

        if photo_style.lower() == "id photo":
            print("Inpaint WITHOUT ControlNet")
            face_images, mask_clothes, mask_combined = generate_masks(mode, optimized_image, image_mask)
            inpainted_images = inpaint_without_controlnet(
                number=numImages,
                image=image_with_bg,
                mask=mask_combined,
                mask_clothes=mask_clothes,
                face_image=face_images[0],
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
            face_images, mask_clothes, mask_combined = generate_masks(mode, optimized_image, image_mask, True)
            controlnet_models = [cn["type"] for cn in controlnets]
            control_images = create_controlnet_images(controlnet_models, optimized_image)
            inpainted_images = inpaint_with_ip_sd_controlnet(
                number=numImages,
                image=image_with_bg,
                mask=None,
                face_images=face_images,
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
                ip_adapter_scale=ip_adapter_scale,
            )
        
        return inpainted_images

    except Exception as e:
        traceback.print_exc()
        print(f"Error processing photo style {photo_style}: {e}")
        return None
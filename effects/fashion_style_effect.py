import traceback
from configs.prompt import generate_fashion_style_prompt
from pipelines.inpainting.inpaint_pipeline import inpaint_with_ip_sd_controlnet
from utils.background_prcessing import remove_background
from utils.controlnet_images import create_canny_image, create_depth_image, create_openpose_image
from utils.masks import generate_face_lower_mask
from utils.optimize_image import optimize_image

def apply_fashion_effect(image):
    try:
        optimized_image = optimize_image(image)

        mask = optimized_image.convert("L")

        image_no_bg, image_mask = remove_background(optimized_image)        
        
        canny_image = create_canny_image(optimized_image)
        control_mode_canny = 0
        control_conditioning_scale_mode_canny = 0.8

        depth_image = create_depth_image(optimized_image)
        control_mode_depth = 2
        control_conditioning_scale_mode_depth = 1.0
        
        pose_image = create_openpose_image(optimized_image)
        control_mode_pose = 4
        control_conditioning_scale_mode_pose = 0.5

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
            num_inference_steps=26,
            guidance_scale=6.5,
            strength=0.7,
            control_image=[canny_image, depth_image, pose_image],
            control_mode=[control_mode_canny, control_mode_depth, control_mode_pose],
            controlnet_conditioning_scale=[control_conditioning_scale_mode_canny, control_conditioning_scale_mode_depth, control_conditioning_scale_mode_pose],
        )

        return inpainted_images

    except Exception as e:
        traceback.print_exc()
        print(f"Error applying fashion effect: {e}")
        return None
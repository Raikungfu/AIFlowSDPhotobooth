import traceback
from sd.prompt import generate_vintage_prompt
from sd.sd_inpainting import inpaint_with_ip_sd_controlnet
from utils.background_prcessing import remove_background
from utils.optimize_image import optimize_image
from utils.masks import generate_face_lower_mask

def apply_vintage_effect(image, bucket):
    try:
        optimized_image = optimize_image(image).convert("RGBA")
        image_no_bg, image_mask = remove_background(optimized_image)

        mask_face, mask_clothes, mask_combined, face_image = generate_face_lower_mask(image_no_bg, image_mask)
        if face_image is None:
            raise ValueError("Không thể nhận diện khuôn mặt!")

        base_prompt, negative_prompt = generate_vintage_prompt()

        inpainted_images = inpaint_with_ip_sd_controlnet(
            number=1,
            image=image_no_bg,
            mask=mask_combined,
            face_image=face_image,
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            strength=0.6,
            scale=7.5,
            controlnet_type="depth",
        )

        if not inpainted_images:
            raise Exception("Inpainting thất bại!")

        return inpainted_images

    except Exception as e:
        traceback.print_exc()
        print(f"Lỗi khi áp dụng hiệu ứng vintage: {e}")
        return None

from enums.inpaint_mode import InpaintMode
from sd.prompt import generate_fashion_style_prompt
from sd.sd_inpainting import inpaint_with_ip_sd
from utils.background_prcessing import remove_background
from utils.masks import generate_face_lower_mask
from utils.optimize_image import optimize_image

def apply_fashion_effect(image):
    try:
        optimized_image = optimize_image(image).convert("RGBA")

        mask = optimized_image.convert("L")

        image_no_bg, image_mask = remove_background(optimized_image)

        face_mask, clothes_mask, combined_mask, face_image = generate_face_lower_mask(optimized_image, image_mask)
        if face_image is None:
            raise ValueError("Không thể nhận diện khuôn mặt!")

        base_prompt, negative_prompt = generate_fashion_style_prompt()

        inpainted_images = inpaint_with_ip_sd(
            number=6,
            image=optimized_image,
            mask=mask,
            face_image=face_image,
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            strength=0.5,
            scale=7.0,
            mode=InpaintMode.KEEP_FACE
        )

        return inpainted_images

    except Exception as e:
        print(f"Lỗi khi áp dụng hiệu ứng thời trang: {e}")
        return None
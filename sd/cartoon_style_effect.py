import traceback
from sd.prompt import generate_cartoon_prompt
from sd.sd_inpainting import inpaint_with_ip_sd_controlnet
from utils.background_prcessing import remove_background
from utils.optimize_image import optimize_image
from utils.masks import generate_face_lower_mask

def apply_cartoon_filter(image, bucket):
    try:
        optimized_image = optimize_image(image).convert("RGBA")

        image_no_bg, image_mask = remove_background(optimized_image)

        mask_face, mask_other, mask_combined, face_image = generate_face_lower_mask(image_no_bg, image_mask)
        if face_image is None:
            raise ValueError("Không thể nhận diện khuôn mặt!")
        print(type(image_no_bg), type(mask_combined), type(face_image))

        base_prompt, negative_prompt = generate_cartoon_prompt()

        inpainted_images = inpaint_with_ip_sd_controlnet(
            number=6,
            image=image_no_bg,
            mask=mask_combined,
            face_image=face_image,
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            strength=0.8,
            scale=9.0,
            controlnet_type="softedge",
        )

        if not inpainted_images:
            raise Exception("Inpainting thất bại!")

        return inpainted_images
    except Exception as e:
        traceback.print_exc()
        print(f"Lỗi khi áp dụng hiệu ứng hoạt hình: {e}")
        return None

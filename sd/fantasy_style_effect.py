import traceback
from sd.prompt import generate_fantasy_prompt
from sd.sd_inpainting import inpaint_with_ip_sd
from enums.inpaint_mode import InpaintMode
from utils.optimize_image import optimize_image

def apply_fantasy_effect(image, bucket):
    try:
        optimized_image = optimize_image(image).convert("RGBA")

        base_prompt, negative_prompt = generate_fantasy_prompt()

        inpainted_images = inpaint_with_ip_sd(
            number=6,
            image=optimized_image,
            mask=None,
            face_image=None,
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            strength=0.6,
            scale=8.0,
            mode=InpaintMode.KEEP_STRUCTURE
        )

        if not inpainted_images:
            raise Exception("Inpainting thất bại!")

        return inpainted_images
    except Exception as e:
        traceback.print_exc()
        print(f"Lỗi khi áp dụng hiệu ứng giả tưởng: {e}")
        return None

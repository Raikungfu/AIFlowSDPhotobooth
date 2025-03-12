import traceback
from sd.prompt import generate_modern_prompt
from sd.sd_inpainting import inpaint_with_ip_sd_controlnet
from enums.inpaint_mode import InpaintMode
from utils.optimize_image import optimize_image

def apply_modern_effect(image):
    try:
        optimized_image = optimize_image(image).convert("RGBA")
        
        base_prompt, negative_prompt = generate_modern_prompt()
        print(type(optimized_image))

        inpainted_images = inpaint_with_ip_sd_controlnet(
            number=6,
            image=optimized_image,
            mask=None,
            face_image=None,
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            strength=0.5,
            scale=7.5,
            controlnet_type="softedge",
            mode=InpaintMode.KEEP_STRUCTURE
        )

        return inpainted_images
    except Exception as e:
        traceback.print_exc()
        print(f"Lỗi khi áp dụng hiệu ứng hiện đại với ControlNet: {e}")
        return None

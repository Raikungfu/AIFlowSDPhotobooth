import traceback
from enums.inpaint_mode import InpaintMode
from sd.prompt import generate_nature_prompt
from utils.optimize_image import optimize_image
from utils.image_upload import upload_image
from sd.sd_inpainting import inpaint_with_ip_sd_controlnet

def apply_nature_effect(image, bucket):
    try:
        optimized_image = optimize_image(image).convert("RGBA")

        base_prompt, negative_prompt = generate_nature_prompt()
        print(type(optimized_image))

        inpainted_images = inpaint_with_ip_sd_controlnet(
            number=1,
            image=optimized_image,
            mask=None,
            face_image=None,
            prompt=base_prompt,
            negative_prompt=negative_prompt,
            strength=0.6,
            scale=8.0,
            controlnet_type="softedge"
        )

        return [upload_image(img, bucket) for img in inpainted_images] if inpainted_images else None

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Error applying nature effect: {e}")
        return None

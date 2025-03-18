import gc
import traceback
import torch
import random
from configs.enums.inpaint_mode import InpaintMode
from utils.masks import expand_mask
from utils.preprocess import make_inpaint_condition
from pipelines.inpainting.inpainting import inpainting_pipeline
from PIL import Image

def inpaint_with_ip_sd_controlnet(
    number, image, mask, face_image, prompt, negative_prompt, strength, scale,control_images,controlnets,mode=InpaintMode.KEEP_FACE
):
    try:
        image = image.convert("RGB")

        if mask is not None and isinstance(mask, Image.Image):
            mask = mask.convert("L")

        if mask is None:
            print("Warning: mask is None. Creating a default white mask.")
            mask = Image.new("L", image.size, 255)
            
        if mode == InpaintMode.KEEP_FACE:
            mask = expand_mask(mask, kernel_size=30, iterations=5)
        elif mode == InpaintMode.KEEP_STRUCTURE:
            mask = Image.new("L", image.size, 255)

        if mask is not None and control_images is None:
            control_images = [make_inpaint_condition(image, mask)]

        generator = [torch.Generator("cpu").manual_seed(random.randint(0, 2**32 - 1)) for _ in range(number)]
            
        pipeline_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": [image],
            "strength": strength,
            "guidance_scale": scale,
            "num_inference_steps": 50,
            "height": (image.height // 8) * 8,
            "width": (image.width // 8) * 8,
            "num_images_per_prompt": number,
            "generator": generator
        }

        if mask is not None:
            pipeline_args["mask_image"] = mask

        if control_images is not None:
            pipeline_args["control_image"] = control_images

        if face_image is not None:
            pipeline_args["reference_image"] = face_image if mode == InpaintMode.KEEP_FACE else None

        with torch.no_grad():
            results = inpainting_pipeline(**pipeline_args).images

            if results is None:
                raise ValueError("Error: 'inpainting_pipeline()' return null!")

        return results
    except Exception as e:
        traceback.print_exc()
        print(f"Error during inpainting: {e}")
        return None

def inpaint_with_ip_sd(number, image, mask, face_image, prompt, negative_prompt, strength, scale, mode=InpaintMode.KEEP_FACE):
    return inpaint_with_ip_sd_controlnet(
        number, image, mask, face_image, prompt, negative_prompt, strength, scale, controlnets=None, control_images=None,mode=mode
    )

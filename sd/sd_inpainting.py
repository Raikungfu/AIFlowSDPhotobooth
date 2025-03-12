import torch
import random
import numpy as np
from enums.inpaint_mode import InpaintMode
from utils.masks import expand_mask
from lib.stable_diffusion import inpainting_pipeline
from model.ip_adapter import apply_ip_adapter
from model.controlnet import apply_controlnet

def inpaint_with_ip_sd_controlnet(
    number, image, mask, face_image, prompt, negative_prompt, strength, scale,
    controlnet_type=None, mode=InpaintMode.KEEP_FACE
):
    try:
        image = image.convert("RGB")
        mask = mask.convert("L")

        if mode == InpaintMode.KEEP_FACE:
            mask = expand_mask(mask, kernel_size=30, iterations=5)
        elif mode == InpaintMode.KEEP_STRUCTURE:
            mask = None

        image = apply_ip_adapter(image, face_image)

        controlnet_image = None
        if controlnet_type:
            controlnet_image = apply_controlnet(image, method=controlnet_type)

        pipeline_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image,
            "strength": strength,
            "guidance_scale": scale,
            "num_inference_steps": 50,
            "controlnet_image": controlnet_image,
            "height": (image.height // 8) * 8,
            "width": (image.width // 8) * 8,
            "num_images_per_prompt": number,
            "generator": [torch.Generator("cpu").manual_seed(random.randint(0, 2**32 - 1)) for _ in range(number)]
        }

        if mask is not None:
            pipeline_args["mask_image"] = mask

        if face_image is not None:
            pipeline_args["reference_image"] = face_image if mode == InpaintMode.KEEP_FACE else None

        with torch.no_grad():
            results = inpainting_pipeline(**pipeline_args).images

            if results is None or not hasattr(results, "images"):
                raise ValueError("❌ Lỗi: 'inpainting_pipeline()' không trả về ảnh hợp lệ!")

        return results
    except Exception as e:
        print(f"Error during inpainting (IP-Adapter, ControlNet={controlnet_type}): {e}")
        return None

def inpaint_with_ip_sd(number, image, mask, face_image, prompt, negative_prompt, strength, scale, mode=InpaintMode.KEEP_FACE):
    return inpaint_with_ip_sd_controlnet(
        number, image, mask, face_image, prompt, negative_prompt, strength, scale,
        controlnet_type=None, mode=mode
    )

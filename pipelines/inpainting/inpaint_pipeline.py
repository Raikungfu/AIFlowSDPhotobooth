import traceback
import torch
import random
from configs.enums.inpaint_mode import InpaintMode
from utils.masks import expand_mask
from utils.preprocess import make_inpaint_condition
from PIL import Image


def inpaint_with_ip_sd_controlnet(
    number, image, mask, face_image, prompt, negative_prompt, strength, num_inference_steps, guidance_scale, height, width,
    control_images=None,control_modes=None,controlnet_conditioning_scale=None,mode=InpaintMode.KEEP_FACE
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

        if height == None or width == None:
            height = (image.height // 8) * 8
            width = (image.width // 8) * 8
        else:
            height = (height // 8) * 8
            width = (width // 8) * 8
            
        pipeline_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": [image],
            "strength": strength,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "height": height,
            "width": width,
            "num_images_per_prompt": number,
            "generator": generator
        }

        if mask is not None:
            pipeline_args["mask_image"] = mask

        if control_images is not None and control_modes is not None and controlnet_conditioning_scale is not None:
            pipeline_args["control_mode"] = control_modes
            pipeline_args["control_image"] = control_images
            pipeline_args["controlnet_conditioning_scale"] = controlnet_conditioning_scale

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

import traceback
import torch
import random
from configs.enums.inpaint_mode import InpaintMode
from utils.controlnet_images import create_controlnet_images
from utils.masks import apply_overlay_to_image, expand_mask, load_overlay_image
from PIL import Image
from pipelines.inpainting import pipeline_instance


def inpaint_with_ip_sd_controlnet(
    number, image, mask, face_image, prompt, negative_prompt, strength, guidance_scale,control_images, controlnets, num_inference_steps, height, width,mode=InpaintMode.KEEP_FACE
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
            
        generator = [torch.Generator("cpu").manual_seed(random.randint(0, 2**32 - 1)) for _ in range(number)]
            
        pipeline_args = {
            "image": [image],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "height": height,
            "width": width,
            "num_images_per_prompt": number,
            "generator": generator,
            "control_image": control_images,
        }

        if mask is not None:
            pipeline_args["mask_image"] = mask
        
        if face_image is not None:
            pipeline_args["reference_image"] = face_image

        results = None

        with torch.inference_mode():
            try:    
                if controlnets == ["canny", "pose"]:
                    results = pipeline_instance.controlnet_canny_pose_pipeline(**pipeline_args).images
                elif controlnets in [["pose", "softedge"], ["pose", "scribble"]]:
                    pipeline_args["control_image"] = control_images.reverse()
                    results = pipeline_instance.controlnet_canny_pose_pipeline(**pipeline_args).images
                elif controlnets == ["depth", "pose"]:
                    results = pipeline_instance.controlnet_depth_pose_pipeline(**pipeline_args).images
                elif controlnets in [["canny", "depth"], ["depth", "softedge"], ["depth", "scribble"]]:
                    results = pipeline_instance.controlnet_canny_depth_pipeline(**pipeline_args).images
                else:
                    raise ValueError(f"Unsupported controlnets combination: {controlnets}")
            except Exception as e:
                print(f"Error during inpainting with ControlNet: {e}")
                raise e

            if results is None:
                raise ValueError("Error: 'inpainting_pipeline()' return null!")

        return results
    except Exception as e:
        traceback.print_exc()
        print(f"Error during inpainting: {e}")
        return None
    

def inpaint_without_controlnet(
    number, image, mask, face_image, prompt, negative_prompt, strength, num_inference_steps, guidance_scale, height, width,mode=InpaintMode.KEEP_FACE
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
        
        # suit_url = "https://firebasestorage.googleapis.com/v0/b/social-networking-8c187.appspot.com/o/%E2%80%94Pngtree%E2%80%94mens%20black%20suit%20with%20white_8991014.png?alt=media&token=3f4aa208-1f5d-4b8f-80c2-d69da7905dc3"

        # image = image.convert("RGBA")
        # suit = load_overlay_image(suit_url, image.size)
        # image = apply_overlay_to_image(image, suit)
        # image = image.convert("RGB")
            
        generator = [torch.Generator("cpu").manual_seed(random.randint(0, 2**32 - 1)) for _ in range(number)]
            
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
            "generator": generator,
            "control_image": create_controlnet_images(controlnet=["pose"], image=image),
        }

        if mask is not None:
            pipeline_args["mask_image"] = mask

        if face_image is not None:
            pipeline_args["reference_image"] = face_image if mode == InpaintMode.KEEP_FACE else None

        results = None

        with torch.inference_mode():
            results = pipeline_instance.inpainting_pipeline(**pipeline_args).images

            if results is None:
                raise ValueError("Error: 'inpainting_pipeline()' return null!")

        return results
    except Exception as e:
        traceback.print_exc()
        print(f"Error during inpainting: {e}")
        return None

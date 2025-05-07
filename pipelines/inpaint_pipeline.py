import traceback
import torch
import random
from PIL import Image
from configs.enums.inpaint_mode import InpaintMode
from pipelines import pipeline_instance
from pipelines.inpainting import load_ip_adapter
from utils.inpaint_pipeline_utils.masks import expand_mask
from configs.const.prompt_clothes import prompts
from diffusers.image_processor import IPAdapterMaskProcessor


def prepare_mask(image, mask, mode: InpaintMode):
    if mask is not None and isinstance(mask, Image.Image):
        mask = mask.convert("L")
    elif mask is None:
        print("Warning: mask is None. Creating a default white mask.")
        mask = Image.new("L", image.size, 255)

    if mode == InpaintMode.KEEP_FACE:
        mask = expand_mask(mask, kernel_size=30, iterations=5)
    elif mode == InpaintMode.KEEP_STRUCTURE:
        mask = Image.new("L", image.size, 255)

    return mask


def get_generator_list(number):
    return [torch.Generator("cpu").manual_seed(random.randint(0, 2**32 - 1)) for _ in range(number)]


def inpaint_with_ip_sd_controlnet(
    number, image, mask, face_images, prompt, negative_prompt,
    strength, guidance_scale, control_images, controlnets,
    num_inference_steps, height, width,
    mode=InpaintMode.KEEP_FACE
):
    try:
        mask = prepare_mask(image, mask, mode)
        generator = get_generator_list(number)

        output_width, output_height = image.size

        processor = IPAdapterMaskProcessor()
        ipadapter_masks = processor.preprocess(face_images, height=output_height, width=output_width)
        ipadapter_masks = [ipadapter_masks.reshape(1, ipadapter_masks.shape[0], ipadapter_masks.shape[2], ipadapter_masks.shape[3])]

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
            "mask_image": mask,
            "ip_adapter_image": [image],
            "cross_attention_kwargs":{"ip_adapter_masks": ipadapter_masks}
        }

        with torch.inference_mode():
            if controlnets == ["canny", "pose"]:
                load_ip_adapter(pipeline_instance.controlnet_canny_pose_pipeline, len(face_images))
                results = pipeline_instance.controlnet_canny_pose_pipeline(**pipeline_args).images
                pipeline_instance.controlnet_canny_pose_pipeline.unload_ip_adapter()

            elif controlnets in [["pose", "softedge"], ["pose", "scribble"]]:
                load_ip_adapter(pipeline_instance.controlnet_canny_pose_pipeline, len(face_images))
                pipeline_args["control_image"] = list(reversed(control_images))
                results = pipeline_instance.controlnet_canny_pose_pipeline(**pipeline_args).images
                pipeline_instance.controlnet_canny_pose_pipeline.unload_ip_adapter()

            elif controlnets == ["depth", "pose"]:
                load_ip_adapter(pipeline_instance.controlnet_depth_pose_pipeline, len(face_images))
                results = pipeline_instance.controlnet_depth_pose_pipeline(**pipeline_args).images
                pipeline_instance.controlnet_depth_pose_pipeline.unload_ip_adapter()

            elif controlnets in [["canny", "depth"], ["depth", "softedge"], ["depth", "scribble"]]:
                load_ip_adapter(pipeline_instance.controlnet_canny_depth_pipeline, len(face_images))
                results = pipeline_instance.controlnet_canny_depth_pipeline(**pipeline_args).images
                pipeline_instance.controlnet_canny_depth_pipeline.unload_ip_adapter()

            else:
                raise ValueError(f"Unsupported controlnets combination: {controlnets}")
        

        if results is None:
            raise ValueError("Error: 'inpainting_pipeline()' returned null!")

        return results

    except Exception as e:
        traceback.print_exc()
        print(f"Error during inpainting with ControlNet: {e}")
        return None


def inpaint_without_controlnet(
    number, image, mask, mask_clothes, face_image,
    prompt, negative_prompt, strength, num_inference_steps,
    guidance_scale, height, width,
    mode=InpaintMode.KEEP_FACE
):
    try:
        mask = prepare_mask(image, mask, mode)
        generator = get_generator_list(number)
        selected_prompt = random.choice(prompts)

        init_pipeline_args = {
            "prompt": selected_prompt["prompt"],
            "negative_prompt": selected_prompt["negative_prompt"],
            "image": [image],
            "strength": 0.7,
            "guidance_scale": 15,
            "num_inference_steps": num_inference_steps,
            "height": height,
            "width": width,
            "num_images_per_prompt": number,
            "generator": generator,
        }

        if mask_clothes is not None:
            init_pipeline_args["mask_image"] = mask_clothes

        with torch.inference_mode():
            initial_images = pipeline_instance.inpainting_pipeline(**init_pipeline_args).images

        results = []
        for i in range(number):
            refine_args = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": initial_images[i],
                "strength": strength,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "height": height,
                "width": width,
                "num_images_per_prompt": 1,
                "generator": [torch.Generator("cpu").manual_seed(0)],
                "reference_image": face_image,
                "mask_image": mask,
            }

            with torch.inference_mode():
                refined_image = pipeline_instance.inpainting_pipeline(**refine_args).images[0]
                results.append(refined_image)

        return results

    except Exception as e:
        traceback.print_exc()
        print(f"Error during inpainting without ControlNet: {e}")
        return None
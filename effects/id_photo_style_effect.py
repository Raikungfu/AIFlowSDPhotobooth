import traceback
from configs.enums.inpaint_mode import InpaintMode
from configs.enums.prompt_type import PromptType
from configs.prompt import generate_id_photo_prompt, generate_id_photo_prompt_with_controlnet
from pipelines.inpainting.inpaint_pipeline import inpaint_with_ip_sd
from utils.masks import generate_face_lower_mask
from utils.optimize_image import optimize_image
from utils.background_prcessing import place_on_new_background, remove_background
from PIL import Image
from utils.face_alignment import align_face

def apply_id_photo_effect(image):
    try:
        aligned_image = align_face(image)
        optimized_image = optimize_image(aligned_image)

        image_no_bg, image_mask = remove_background(optimized_image)
        if image_no_bg is None or image_mask is None:
            raise ValueError("Failed to remove background or generate mask")

        new_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image_with_bg = place_on_new_background(image_no_bg, new_background).convert("RGB")

        mask_face, mask_clothes, mask_combined, face_image = generate_face_lower_mask(image, image_mask)
        if mask_face is None or mask_clothes is None or mask_combined is None:
            raise ValueError("Failed to generate face or clothes mask")

        results = []

        for use_control in [False, True]:
            prompt_func = (
                generate_id_photo_prompt_with_controlnet if use_control else generate_id_photo_prompt
            )
            
            base_prompt, negative_prompt = prompt_func()
            print(f"Testing {'with' if use_control else 'without'} ControlNet")

            inpainted_images = inpaint_with_ip_sd(
                number=6,
                image=image_with_bg,
                mask=mask_combined,
                face_image=face_image,
                prompt=base_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=20,
                guidance_scale=4.0,
                strength=0.3,
                mode=InpaintMode.KEEP_FACE
            )

            if inpainted_images:
                results.extend(inpainted_images)

        return results
    except Exception as e:
        traceback.print_exc()
        print(f"Error applying id photo effect: {e}")
        return None
    

    


# def apply_id_photo_effect(image):
#     try:
#         aligned_image = align_face(image)

#         optimized_image = optimize_image(aligned_image)

#         image_no_bg, image_mask = remove_background(optimized_image)
#         if image_no_bg is None or image_mask is None:
#             raise ValueError("Failed to remove background or generate mask")

#         new_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
#         image_with_bg = place_on_new_background(image_no_bg, new_background).convert("RGB")

#         mask_face, mask_clothes, mask_combined, face_image = generate_face_lower_mask(image, image_mask)
#         if mask_face is None or mask_clothes is None or mask_combined is None:
#             raise ValueError("Failed to generate face or clothes mask")

#         edit_types = {
#             # PromptType.FACE: (1, mask_face, 0.3, 4.0),
#             # PromptType.CLOTHES: (6, mask_clothes, 0.5, 7.5),
#             PromptType.COMBINE: (6, mask_combined, 0.3, 4.0, )
#         }
#         print(type(image_with_bg), type(mask_face), type(mask_clothes), type(mask_combined), type(face_image))

#         results = []

#         for edit_type, (number, mask, strength, scale) in edit_types.items():
#             base_prompt, negative_prompt = generate_id_photo_prompt(edit_type)
#             inpainted_images = inpaint_with_ip_sd(
#                 number,
#                 image=image_with_bg,
#                 mask=mask,
#                 face_image=face_image,
#                 prompt=base_prompt,
#                 negative_prompt=negative_prompt,
#                 num_inference_steps=20,
#                 guidance_scale=scale,
#                 strength=strength,
#                 mode=InpaintMode.KEEP_FACE
#             )
#             if inpainted_images:
#                 results.extend(inpainted_images)

#         return results
#     except Exception as e:
#         traceback.print_exc()
#         print(f"Error applying id photo effect: {e}")
#         return None


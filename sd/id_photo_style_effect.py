import traceback
from enums.inpaint_mode import InpaintMode
from enums.prompt_type import PromptType
from sd.prompt import generate_id_photo_prompt
from sd.sd_inpainting import inpaint_with_ip_sd_controlnet
from utils.masks import generate_face_lower_mask
from utils.optimize_image import optimize_image
from utils.background_prcessing import place_on_new_background, remove_background
from utils.image_upload import upload_image
from PIL import Image
from utils.face_alignment import align_face


def apply_id_photo_effect(image, bucket):
    try:
        aligned_image = align_face(image)
        image_outputs = [upload_image(aligned_image, bucket)]

        optimized_image = optimize_image(aligned_image).convert("RGBA")
        image_outputs.append(upload_image(optimized_image, bucket))

        image, image_mask = remove_background(optimized_image)
        if image is None or image_mask is None:
            raise ValueError("Failed to remove background or generate mask")

        new_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image_with_bg = place_on_new_background(image, new_background).convert("RGB")

        mask_face, mask_clothes, mask_combined, face_image = generate_face_lower_mask(image, image_mask)
        if mask_face is None or mask_clothes is None or mask_combined is None:
            raise ValueError("Failed to generate face or clothes mask")

        image_outputs.extend([
            upload_image(mask_clothes, bucket),
            upload_image(mask_face, bucket)
        ])

        edit_types = {
            PromptType.FACE: (1, mask_face, 0.3, 4.0),
            PromptType.CLOTHES: (6, mask_clothes, 0.5, 7.5),
            PromptType.COMBINE: (6, mask_combined, 0.3, 4.0)
        }

        results = []
        print(type(image_with_bg), type(mask), type(face_image))

        for edit_type, (number, mask, strength, scale) in edit_types.items():
            base_prompt, negative_prompt = generate_id_photo_prompt(edit_type)
            inpainted_images = inpaint_with_ip_sd_controlnet(
                number,
                image=image_with_bg,
                mask=mask,
                face_image=face_image,
                prompt=base_prompt,
                negative_prompt=negative_prompt,
                strength=strength,
                scale=scale,
                controlnet_type="softedge",
                mode=InpaintMode.KEEP_FACE
            )
            if inpainted_images:
                results.extend(inpainted_images)

        return results
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")
        return None

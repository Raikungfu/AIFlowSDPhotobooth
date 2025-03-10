from enums.prompt_type import PromptType
from enums.prompt_type import PromptType
from sd.prompt import generate_id_photo_prompt
from sd.sd_inpainting import inpaint
from utils.masks import generate_face_lower_mask
from utils.optimize_image import optimize_image
from utils.background_prcessing import place_on_new_background, remove_background
from utils.image_upload import upload_image
from PIL import Image
from enums.prompt_type import PromptType
from utils.face_alignment import align_face


def apply_id_photo_effect(image, bucket):
    try:
        aligned_image = align_face(image)
        image_outputs = [upload_image(aligned_image, bucket)]

        optimized_image = optimize_image(aligned_image).convert("RGBA")
        image_outputs.append(upload_image(optimized_image, bucket))
        
        image, image_mask = remove_background(optimized_image)
        if image_mask:
            new_background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = place_on_new_background(image, new_background).convert("RGB")

        mask_face, mask_clothes, mask_combined = generate_face_lower_mask(image, image_mask)

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

        for edit_type, (number, mask, strength, scale) in edit_types.items():
            base_prompt, negative_prompt = generate_id_photo_prompt(edit_type)
            inpainted_images = inpaint(number, image, mask, base_prompt, negative_prompt, strength, scale)
            results.extend(inpainted_images)

        return results
    except Exception as e:
        print(f"Error: {e}")
        return None
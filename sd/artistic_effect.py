import traceback
from sd.prompt import generate_artistic_prompt
from sd.sd_inpainting import inpaint
from utils.image_upload import upload_image
from utils.background_prcessing import remove_background
from utils.optimize_image import optimize_image
from utils.masks import generate_face_lower_mask


def apply_artistic_effects(image, bucket):
    try:
        optimized_image = optimize_image(image).convert("RGBA")
        image_outputs = [upload_image(optimized_image, bucket)]
        
        image, image_mask = remove_background(optimized_image)
        image_outputs.append(upload_image(image_mask, bucket))
        image_outputs.append(upload_image(image, bucket))

        mask_face, mask_clothes, mask_combined = generate_face_lower_mask(image, image_mask)
        if not mask_face or not mask_clothes:
            raise Exception("No face detected, mask creation failed")

        image_outputs.extend([
            upload_image(mask_clothes, bucket),
            upload_image(mask_face, bucket)
        ])

        base_prompt, negative_prompt = generate_artistic_prompt()
        
        results = inpaint(1, image, mask_combined, base_prompt, negative_prompt, 0.5, 7)

        if not results:
            raise Exception("Inpainting failed")
        return results
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")

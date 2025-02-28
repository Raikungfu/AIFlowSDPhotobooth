from utils.image_processing import generate_random_filename
from utils.firebase_utils import upload_to_firebase_storage
from utils.sd_inpainting import generate_id_photo_prompt, inpaint, refine_with_depth2img


def upload_image(image, bucket):
    output_filename = generate_random_filename('.png')
    output_path = f'/tmp/{output_filename}'
    image.save(output_path)
    return upload_to_firebase_storage(output_path, 'processed_images', bucket)


def inpaint_image(image, mask_clothes, seek, prompt_type, gender):
    base_prompt, negative_prompt = generate_id_photo_prompt(prompt_type, gender)
    return inpaint(image, mask_clothes, base_prompt, negative_prompt, seek)


def refine_with_depth2img_image(image, seek, prompt_type, gender):
    base_prompt, negative_prompt = generate_id_photo_prompt(prompt_type, gender)
    return refine_with_depth2img(image, base_prompt, negative_prompt, seek)
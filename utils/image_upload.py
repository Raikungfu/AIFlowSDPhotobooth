from sd.prompt import generate_id_photo_prompt
from utils.firebase_utils import generate_random_filename, upload_to_firebase_storage
from utils.sd_inpainting import inpaint


def upload_image(image, bucket):
    output_filename = generate_random_filename('.jpeg')
    output_path = f'/tmp/{output_filename}'
    image.save(output_path)
    return upload_to_firebase_storage(output_path, 'processed_images', bucket)


def inpaint_image(image, mask_clothes, prompt_type, strength, scale):
    base_prompt, negative_prompt = generate_id_photo_prompt(prompt_type)
    return inpaint(image, mask_clothes, base_prompt, negative_prompt, strength, scale)
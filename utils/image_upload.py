from utils.image_processing import generate_random_filename, inpaint, generate_inpainting_prompt
from utils.firebase_utils import upload_to_firebase_storage

def upload_image(image):
    output_filename = generate_random_filename('.png')
    output_path = f'/tmp/{output_filename}'
    image.save(output_path)
    return upload_to_firebase_storage(output_path, 'processed_images')

def inpaint_image(image, mask_clothes, seek, is_face):
    base_prompt, negative_prompt = generate_inpainting_prompt(is_face)
    return inpaint(image, mask_clothes, base_prompt, negative_prompt, seek)
import torch
from enums.prompt_type import PromptType
from utils.masks import expand_mask
from utils.stable_diffusion import inpainting_pipeline, depth_pipeline


def inpaint(image, mask, prompt, negative_prompt, seek):
    try:
        image = image.convert("RGB")
        mask = mask.convert("L")

        mask = expand_mask(mask, kernel_size=10, iterations=2)

        with torch.no_grad():
            result = inpainting_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                strength=0.75,
                guidance_scale=7.5,
                num_inference_steps=50,
                height=(image.height // 8) * 8,
                width=(image.width // 8) * 8,
                generator=torch.Generator("cpu").manual_seed(int(seek))
            ).images[0]

        return result
    except Exception as e:
        print(f"Error during inpainting: {e}")
        return None



def refine_with_depth2img(image, prompt, negative_prompt, seek):
    try:
        with torch.no_grad():
            result = depth_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                strength=0.5,
                guidance_scale=7.5,
                num_inference_steps=50,
                generator=torch.Generator("cpu").manual_seed(int(seek))
            ).images[0]
        return result
    except Exception as e:
        print(f"Error during depth-based refinement: {e}")
        return image



def generate_id_photo_prompt(prompt_type=PromptType.COMBINE, gender="male"):
    base_prompt = ""
    
    if prompt_type == PromptType.FACE or prompt_type == PromptType.COMBINE:
        base_prompt += (
            "Professional ID photo, studio-quality lighting, neutral white background, high-resolution, "
            "symmetrical facial features, smooth skin, well-groomed hair, clean and sharp details, "
            "natural skin tone, balanced lighting, no harsh shadows, straight posture, neutral facial expression, "
            "ears slightly visible, forehead fully visible, no hair covering the face, high-detail 8K resolution."
        )

    if prompt_type == PromptType.CLOTHES or prompt_type == PromptType.COMBINE:
        if base_prompt:
            base_prompt += " "
        base_prompt += "Professional ID photo, clean and professional look, realistic fabric texture, "
        
        if gender.lower() == "male":
            base_prompt += "well-fitted suit, light-colored collared shirt, neutral tie, polished and elegant appearance."
        elif gender.lower() == "female":
            base_prompt += "formal blouse or professional suit, subtle and natural makeup, minimal accessories, polished and elegant appearance."
    
    negative_prompt = (
        "Casual wear, bright colors, patterns, excessive accessories, messy hair, artistic filters, "
        "exaggerated makeup, unnatural skin tone, harsh shadows, wrinkles, scars, acne, unrealistic clothing texture, "
        "low resolution, distorted features, blurry, pixelated, fantasy elements, overly dramatic lighting, unprofessional look."
    )

    return base_prompt, negative_prompt

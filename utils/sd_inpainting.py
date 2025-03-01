import torch
from enums.prompt_type import PromptType
from enums.user_info import Gender
from utils.masks import expand_mask
from utils.stable_diffusion import inpainting_pipeline

def inpaint(image, mask, prompt, negative_prompt, seek, strength, scale):
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
                strength=strength,
                guidance_scale=scale,
                num_inference_steps=50,
                height=(image.height // 8) * 8,
                width=(image.width // 8) * 8,
                generator=torch.Generator("cpu").manual_seed(int(seek))
            ).images[0]

        return result
    except Exception as e:
        print(f"Error during inpainting: {e}")
        return None

def generate_id_photo_prompt(prompt_type=PromptType.COMBINE, gender=Gender.Male):
    base_prompt = (
        "ID photo, studio lighting, clear skin, well-groomed, bright look, natural expression, straight posture, "
        "visible forehead, no hair covering face, sharp details, high resolution."
    )

    if prompt_type in {PromptType.CLOTHES, PromptType.COMBINE}:
        base_prompt += " Professional attire, clean, elegant, realistic fabric."
        if gender == Gender.Male:
            base_prompt += " Suit, light shirt, neutral tie."
        else:
            base_prompt += " Blouse or suit, neat hair, minimal makeup."

    negative_prompt = (
        "Casual wear, bright colors, patterns, excessive accessories, messy hair, artistic filters, exaggerated makeup, "
        "unnatural skin tone, harsh shadows, wrinkles, acne, low resolution, distorted features, blurry, fantasy elements, "
        "frowning, sad, angry, tired, closed eyes, asymmetry, greasy hair, bangs, uneven posture, overexposed, underexposed, "
        "soft focus, low contrast, cartoon, anime, CGI, unrealistic lighting."
    )

    return base_prompt, negative_prompt

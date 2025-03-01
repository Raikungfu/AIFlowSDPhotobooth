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
    base_prompt = "ID photo, studio lighting, natural expression, clear skin, straight posture, high resolution. "
    
    if prompt_type in {PromptType.FACE, PromptType.COMBINE}:
        base_prompt += "Keep original face features, no alteration, no extra hair, no makeup change. "

    if prompt_type in {PromptType.CLOTHES, PromptType.COMBINE}:
        base_prompt += "Professional attire, clean, elegant. "
        base_prompt += "Suit, light shirt, neutral tie. " if gender == Gender.Male else "Blouse or suit, minimal makeup. "

    negative_prompt = (
        "Distorted face, extra hair, altered facial structure, exaggerated makeup, casual wear, messy hair, blurry, cartoon, CGI, unrealistic."
    )

    return base_prompt.strip(), negative_prompt.strip()



import torch
import random
from utils.masks import expand_mask
from model.stable_diffusion import inpainting_pipeline

def inpaint(number, image, mask, prompt, negative_prompt, strength, scale):
    try:
        image = image.convert("RGB")
        mask = mask.convert("L")

        mask = expand_mask(mask, kernel_size=10, iterations=2)

        results = []
        with torch.no_grad():
            for _ in range(number):
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
                    generator=torch.Generator("cpu").manual_seed(random.randint(0, 2**32 - 1))
                ).images[0]
                results.append(result)
        return results
    except Exception as e:
        print(f"Error during inpainting: {e}")
        return None


import torch
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
    



def generate_inpainting_prompt(IsGenFace=True):
    if IsGenFace:
        base_prompt = "Professional passport-style portrait, even lighting, flawless and smooth skin, sharp and well-defined facial features, natural skin tone, high detail, well-groomed hair, forehead fully visible, ears slightly visible, neutral expression, straight posture, clean background, high resolution, studio lighting, formal appearance."
        negative_prompt = "Messy hair, forehead covered, hair over face, shadows, acne, blemishes, wrinkles, scars, distorted features, poor details, unnatural skin tone, dark patches, extreme makeup, blurry image, bad quality, low resolution, exaggerated expressions, artistic filters."
    else:
        base_prompt = (
            "Professional passport-style portrait, formal attire, professional suit, white or light-colored collared shirt, neutral tie, clean and polished appearance, well-fitted clothing, natural fabric texture, studio lighting, high detail, 8k resolution, realistic folds and shadows, professional portrait style"
        )
        negative_prompt = (
            "casual wear, t-shirts, logos, patterns, bright colors, excessive accessories, low resolution, distorted, unrealistic clothing texture, blurry, pixelated, fantasy elements, overly dramatic lighting, unnatural colors, unprofessional look"
        )

    return base_prompt, negative_prompt


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

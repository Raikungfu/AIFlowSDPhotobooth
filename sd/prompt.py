
from enums.prompt_type import PromptType


def generate_id_photo_prompt(prompt_type=PromptType.COMBINE):
    base_prompt = "ID photo, studio lighting, natural expression, clear skin, straight posture, high resolution. "
    
    if prompt_type in {PromptType.FACE, PromptType.COMBINE}:
        base_prompt += "Keep original face features, no alteration, no extra hair, no makeup change. "

    if prompt_type in {PromptType.CLOTHES, PromptType.COMBINE}:
        base_prompt += "Professional attire, clean, elegant. "

    negative_prompt = (
        "Distorted face, extra hair, altered facial structure, exaggerated makeup, casual wear, messy hair, blurry, cartoon, CGI, unrealistic."
    )

    return base_prompt.strip(), negative_prompt.strip()


def generate_fashion_style_prompt():
    base_prompt = "Fashion style, trendy, stylish, elegant, glamorous, chic, sophisticated, high fashion, luxury, designer clothes, runway, catwalk, model, fashion show. "
    negative_prompt = "Casual style, everyday wear, street style, sporty, athletic, loungewear, pajamas, home clothes, unbranded, fast fashion, low quality, cheap, outdated, unfashionable, old-fashioned, vintage, retro, second-hand"

    return base_prompt.strip(), negative_prompt.strip()



def generate_cartoon_prompt():
    base_prompt = "Cartoon effect, colorful, vibrant, high saturation, high contrast, sharp lines, clear details. "
    negative_prompt = "Realistic, natural colors, soft tones, low saturation, low contrast, smooth lines, soft focus, blurred details."

    return base_prompt.strip(), negative_prompt.strip()



def generate_artistic_prompt():
    base_prompt = "Artistic effect, creative, imaginative, expressive, abstract, artistic style, unique, original, artistic vision, artistic interpretation. "
    negative_prompt = "Realistic, natural, true to life, accurate, faithful, precise, detailed, lifelike, authentic, genuine, unembellished, unadorned, unvarnished, unexaggerated, unidealized"

    return base_prompt.strip(), negative_prompt.strip()



def generate_vintage_prompt():
    base_prompt = "Vintage photo, old style, sepia tone, grainy texture, faded colors, classic look, retro, nostalgic, black and white. "
    negative_prompt = "Modern style, vibrant colors, sharp details, high contrast, digital look, futuristic, glossy, saturated colors, high resolution."

    return base_prompt.strip(), negative_prompt.strip()

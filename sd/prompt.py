from enums.prompt_type import PromptType

def generate_id_photo_prompt(prompt_type=PromptType.COMBINE):
    base_prompt = "ID photo, studio lighting, neutral expression, clear skin, straight posture, high resolution, realistic details."
    
    if prompt_type in {PromptType.FACE, PromptType.COMBINE}:
        base_prompt += " Maintain original facial features, no modification, no extra hair, no makeup change, no unnatural smoothing."

    if prompt_type in {PromptType.CLOTHES, PromptType.COMBINE}:
        base_prompt += " Professional attire, clean, well-fitted, neutral colors, no distracting accessories."

    negative_prompt = (
        "Altered facial features, extra hair, exaggerated makeup, unnatural skin texture, blur, distortion, messy hair, jewelry, reflections, unnatural shadows, cartoon effect, CGI, unrealistic enhancements."
    )

    return base_prompt.strip(), negative_prompt.strip()


def generate_fashion_style_prompt():
    base_prompt = "High-fashion photography, elegant, stylish, luxury designer wear, well-lit, high resolution, runway model pose."
    negative_prompt = (
        "Casual outfit, streetwear, unbranded clothes, low quality fabric, excessive wrinkles, poorly fitted clothes, messy hair, distracting accessories, blurry details, oversaturated colors, unnatural textures."
    )

    return base_prompt.strip(), negative_prompt.strip()


def generate_cartoon_prompt():
    base_prompt = "Stylized cartoon effect, vibrant colors, bold outlines, sharp details, expressive facial features."
    negative_prompt = (
        "Photorealistic textures, lifelike skin details, low contrast, blurry lines, distorted anatomy, excessive noise, unwanted visual effects."
    )

    return base_prompt.strip(), negative_prompt.strip()


def generate_artistic_prompt():
    base_prompt = "Artistic composition, unique creative expression, bold strokes, stylized details, textured shading, expressive lighting."
    negative_prompt = (
        "Photorealistic rendering, overly smooth textures, artificial effects, unwanted distortions, color noise, unnatural blurs."
    )

    return base_prompt.strip(), negative_prompt.strip()


def generate_vintage_prompt():
    base_prompt = "Classic vintage photography, aged film texture, sepia tone, soft contrast, nostalgic atmosphere, natural facial expression."
    negative_prompt = (
        "Modern high-resolution look, unnatural lighting, sharp digital textures, unwanted reflections, excessive grain, unrealistic blur."
    )

    return base_prompt.strip(), negative_prompt.strip()


def generate_fantasy_prompt():
    base_prompt = "Fantasy world, magical atmosphere, ethereal lighting, glowing accents, fairytale aesthetic, dreamlike setting."
    negative_prompt = (
        "Modern urban elements, industrial design, photorealistic human textures, excessive shadows, noisy backgrounds, distorted anatomy."
    )

    return base_prompt.strip(), negative_prompt.strip()


def generate_modern_prompt():
    base_prompt = "Futuristic photography, sleek design, well-balanced lighting, clear facial details, contemporary fashion."
    negative_prompt = (
        "Rustic elements, vintage aesthetics, distracting patterns, excessive reflections, color distortion, unwanted shadows, unnatural textures."
    )

    return base_prompt.strip(), negative_prompt.strip()


def generate_nature_prompt():
    base_prompt = "Natural outdoor photography, well-lit environment, realistic skin tones, soft natural light, vibrant colors."
    negative_prompt = (
        "Urban backgrounds, artificial structures, oversaturated colors, washed-out tones, distracting elements, excessive blur."
    )

    return base_prompt.strip(), negative_prompt.strip()


def generate_anime_prompt():
    base_prompt = (
        "High-quality anime illustration, soft gradients, large expressive eyes, smooth shading, cel-shaded details, vibrant colors."
    )
    
    negative_prompt = (
        "Realistic proportions, overly detailed facial features, Western comic style, harsh lighting, blurry outlines, CGI effects, unwanted shadows, low-resolution textures."
    )

    return base_prompt.strip(), negative_prompt.strip()

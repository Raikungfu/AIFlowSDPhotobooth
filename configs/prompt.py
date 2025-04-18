def generate_id_photo_prompt():
    base_prompt = (
        "Super Portrait, professional ID photo, studio lighting, sharp focus, neutral background, straight-on angle, natural skin texture, neutral expression, straight posture, ultra-high resolution."
    )

    negative_prompt = (
        "blur, distortion, extra hair, heavy makeup, artificial skin smoothing, exaggerated features, harsh shadows, unrealistic retouching, incorrect posture, head tilt."
    )

    return base_prompt.strip(), negative_prompt.strip()


def generate_id_photo_prompt_with_controlnet():
    base_prompt = (
        "Super Portrait, professional ID photo, studio lighting, sharp focus, neutral background, ultra-high resolution, natural skin texture, neutral expression."
    )

    negative_prompt = (
        "blur, distortion, extra hair, heavy makeup, artificial skin smoothing, exaggerated features, harsh shadows, unrealistic retouching."
    )

    return base_prompt.strip(), negative_prompt.strip()



def generate_fashion_style_prompt():
    base_prompt = (
        "Editorial fashion portrait, striking pose, high-definition details, elegant outfit, perfect fit, confident expression, soft studio lighting. The background is clean and minimalistic, enhancing the subject."
    )
    negative_prompt = (
        "bad lighting, ill-fitting clothes, low-quality textures, unnatural expression, distorted proportions, cluttered background."
    )

    return base_prompt.strip(), negative_prompt.strip()



def generate_cartoon_prompt():
    base_prompt = (
        "Vibrant cartoon character, smooth shading, expressive eyes, bold outlines, dynamic pose. The background is colorful and playful, enhancing the lively personality."
    )
    negative_prompt = (
        "blurred lines, dull colors, photorealistic details, excessive shading, missing body parts, distorted proportions."
    )

    return base_prompt.strip(), negative_prompt.strip()



def generate_artistic_prompt():
    base_prompt = (
        "Fine art portrait, soft and balanced lighting, rich textures, warm color harmony. The subject exudes elegance and emotion, resembling classical masterpieces."
    )
    negative_prompt = (
        "oversaturated, artificial blur, harsh contrast, distorted proportions, lifeless expression, missing elements."
    )

    return base_prompt.strip(), negative_prompt.strip()



def generate_vintage_prompt():
    base_prompt = (
        "Classic vintage portrait, warm sepia tones, soft film-like contrast, subtle grain texture. The subject appears timeless with natural lighting and elegant styling."
    )
    negative_prompt = (
        "modern sharpness, digital artifacts, unnatural lighting, reflections, artificial blur, excessive grain."
    )

    return base_prompt.strip(), negative_prompt.strip()



def generate_fantasy_prompt():
    base_prompt = (
        "Ethereal fantasy portrait, glowing atmosphere, intricate costume details, vibrant magical background. The subject radiates an enchanting aura with a dreamlike aesthetic."
    )
    negative_prompt = (
        "lifeless expression, dull tones, distorted anatomy, harsh shadows, CGI artifacts, unnatural blur."
    )

    return base_prompt.strip(), negative_prompt.strip()



def generate_modern_prompt():
    base_prompt = (
        "Sleek modern portrait, sharp lighting, high-definition textures, dynamic composition, stylish outfit. The background is minimalistic with a futuristic touch."
    )
    negative_prompt = (
        "low resolution, washed-out colors, bad lighting, asymmetry, distorted shapes, excessive blur."
    )

    return base_prompt.strip(), negative_prompt.strip()



def generate_nature_prompt():
    base_prompt = (
        "Natural outdoor portrait, warm sunlight, soft breeze, fresh atmosphere, glowing skin. The background features lush greenery or scenic landscapes, complementing the subject."
    )
    negative_prompt = (
        "harsh lighting, artificial colors, dull tones, cluttered background, distortion, missing details."
    )

    return base_prompt.strip(), negative_prompt.strip()



def generate_anime_prompt():
    base_prompt = (
        "High-quality anime illustration, smooth shading, expressive eyes, perfect symmetry, vibrant color palette. The background is detailed yet balanced, complementing the character's design."
    )
    negative_prompt = (
        "blurred outlines, muted colors, photorealistic textures, distorted anatomy, excessive shading."
    )

    return base_prompt.strip(), negative_prompt.strip()

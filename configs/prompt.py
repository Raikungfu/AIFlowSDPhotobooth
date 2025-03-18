from configs.enums.prompt_type import PromptType



def generate_id_photo_prompt(prompt_type=PromptType.COMBINE):
    base_prompt = "ID photo, studio lighting, neutral expression, clear skin, straight posture, high resolution, realistic details."
    
    if prompt_type in {PromptType.FACE, PromptType.COMBINE}:
        base_prompt += " Maintain original facial features, no modification, no extra hair, no makeup change, no unnatural smoothing."

    if prompt_type in {PromptType.CLOTHES, PromptType.COMBINE}:
        base_prompt += " Professional attire, clean, well-fitted, neutral colors, no distracting accessories."

    negative_prompt = (
        "altered facial features, extra hair, exaggerated makeup, unnatural skin texture, blur, distortion, messy hair, jewelry, reflections, unnatural shadows, cartoon effect, CGI, unrealistic enhancements."
    )

    return base_prompt.strip(), negative_prompt.strip()



def generate_fashion_style_prompt():
    base_prompt = (
        "Generate a high-fashion editorial image showcasing a model with radiant skin and a trendy outfit. The outfit should be perfectly fitted and elegant, with a striking pose that conveys confidence. Use soft studio lighting to enhance the model's features, and ensure the background is minimalistic to keep the focus on the fashion."
    )
    negative_prompt = (
        "ugly, extra fingers, mutated hands, extra limbs, long neck, watermark, text, logo, messy hair, cheap fabrics, bad fit, wrinkles, dull colors, or distracting elements in the background."
    )

    return base_prompt.strip(), negative_prompt.strip()



def generate_cartoon_prompt():
    base_prompt = (
        "Create a cartoon character with smooth features, big eyes, sharp outlines, and bright colors. The background should be playful and colorful. The character should have a lively expression and dynamic pose, capturing the essence of a classic cartoon style."
    )
    negative_prompt = (
        "ugly, extra fingers, mutated hands, extra limbs, long neck, watermark, text, logo, rough textures, distorted, realistic anatomy, muted colors, blurred lines, missing body parts, unnatural proportions."
    )

    return base_prompt.strip(), negative_prompt.strip()



def generate_artistic_prompt():
    base_prompt = (
        "Design a masterpiece portrait with elegant features and soft lighting. The colors should be balanced and glowing, creating a dreamy aesthetic. The subject should be posed gracefully, and the overall composition should evoke a sense of tranquility and beauty, reminiscent of classical art."
    )
    negative_prompt = (
        "ugly, extra fingers, mutated hands, extra limbs, long neck, watermark, text, logo, harsh shadows, rough textures, dull colors, distortions, missing parts, or excessive blurring."
    )

    return base_prompt.strip(), negative_prompt.strip()


def generate_vintage_prompt():
    base_prompt = "Create a classic vintage photo with a sepia tone and soft contrast. The image should evoke a nostalgic feel, featuring smooth skin and a natural expression. Use high-quality textures to enhance the vintage look, and ensure the background complements the overall aesthetic without being distracting."
    negative_prompt = "ugly, extra fingers, mutated hands, extra limbs, long neck, watermark, text, logo, modern sharpness, digital artifacts, unnatural lighting, reflections, excessive grain, or aged skin."

    return base_prompt.strip(), negative_prompt.strip()


def generate_fantasy_prompt():
    base_prompt = (
        "Generate an enchanting fantasy scene featuring a radiant character with an ethereal glow. Use soft lighting to create a magical aura, and focus on delicate and elegant details in the character's attire. The background should be a whimsical landscape, filled with vibrant colors and fantastical elements."
    )
    negative_prompt = (
        "ugly, extra fingers, mutated hands, extra limbs, long neck, watermark, text, logo, harsh shadows, dull colors, distortions, missing parts, or lifeless expressions."
    )

    return base_prompt.strip(), negative_prompt.strip()


def generate_modern_prompt():
    base_prompt = (
        "Create a futuristic style image featuring a sleek model in a stylish outfit. Use sharp lighting to highlight the model's features and ensure perfect symmetry in the composition. The background should be modern and minimalistic, with vibrant colors that enhance the overall editorial feel of the shot."
    )
    negative_prompt = (
        "ugly, extra fingers, mutated hands, extra limbs, long neck, watermark, text, logo, old-fashioned elements, dull colors, messy hair, asymmetry, wrinkles, or noisy backgrounds.")

    return base_prompt.strip(), negative_prompt.strip()



def generate_nature_prompt():
    base_prompt = (
        "Design an outdoor photography scene with a model showcasing glowing skin and wind-blown hair. Use soft sunlight to create a fresh atmosphere, and incorporate vibrant colors in the natural surroundings. The setting should be peaceful and serene, capturing the beauty of nature."
    )
    negative_prompt = (
        "ugly, extra fingers, mutated hands, extra limbs, long neck, watermark, text, logo, dry skin, dull tones, harsh shadows, distortions, cluttered backgrounds, or artificial lighting.")

    return base_prompt.strip(), negative_prompt.strip()



def generate_anime_prompt():
    base_prompt = (
        "Create a high-quality anime image featuring a youthful character with bright eyes and soft gradients. Use vibrant colors and perfect symmetry to enhance the character's appeal. The background should be detailed and colorful, complementing the character's design while maintaining a cohesive anime aesthetic."
    )
    negative_prompt = (
        "ugly, extra fingers, mutated hands, extra limbs, long neck, watermark, text, logo, Rough lines, muted colors, photorealistic details, missing parts, extra limbs, or blurry outlines."
    )

    return base_prompt.strip(), negative_prompt.strip()
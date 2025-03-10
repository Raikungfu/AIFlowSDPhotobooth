
    
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

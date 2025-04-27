from configs.enums.inpaint_mode import InpaintMode
from utils.process_input import parse_controlnets


def extract_message_data(message):
    mode_map = [InpaintMode.NONE, InpaintMode.KEEP_FACE, InpaintMode.KEEP_STRUCTURE]
    return {
        "image_url": message.get("imageUrl"),
        "correlation_id": message.get("correlationId"),
        "photo_style_id": message.get("photoStyleId"),
        "session_code": message.get("sessionCode"),
        "photo_style": message.get("photoStyle"),
        "prompt": message.get("prompt"),
        "negative_prompt": message.get("negativePrompt"),
        "controlnets": parse_controlnets(message.get("controlnets")) if message.get("controlnets") else None,
        "num_images": message.get("numImagesPerGen", 1),
        "guidance_scale": message.get("guidanceScale", 7.5),
        "strength": message.get("strength", 0.5),
        "face_image": message.get("faceImage"),
        "background_remover": message.get("backgroundRemover", False),
        "height": message.get("height"),
        "width": message.get("width"),
        "mode": mode_map[message.get("mode", 0)],
        "num_inference_steps": message.get("numInferenceSteps", 30)
    }

def validate_message(data):
    if not data["image_url"] or not data["correlation_id"] or not data["photo_style"]:
        raise ValueError("Missing required fields in message")

from PIL import Image
from rembg import remove


def remove_background(image):
    try:
        transparent_image = remove(image, detection="soft", model_type="u2net", alpha_matting=True)
        alpha_channel = transparent_image.split()[-1]
        mask = alpha_channel.convert("L")
        return transparent_image, mask
    except Exception as e:
        print(f"Error removing background: {e}")
        return None, None



def place_on_new_background(foreground, new_background):
    new_background = new_background.resize(foreground.size, Image.Resampling.LANCZOS).convert("RGBA")
    return Image.alpha_composite(new_background, foreground)
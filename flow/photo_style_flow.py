from sd.anime_style_effect import apply_anime_filter
from sd.artistic_effect import apply_artistic_effects
from sd.cartoon_style_effect import apply_cartoon_filter
from sd.fashion_style_effect import apply_fashion_effect
from sd.id_photo_style_effect import apply_id_photo_effect
from sd.modern_style_effect import apply_modern_effect
from sd.nature_style_effect import apply_nature_effect
from sd.vintage_style_effect import apply_vintage_effect


def process_id_photo(image):
    return apply_id_photo_effect(image)


def process_artistic_portrait(image):
    return apply_artistic_effects(image)


def process_cartoon(image):
    return apply_cartoon_filter(image)


def process_vintage(image):
    return apply_vintage_effect(image)


def process_fashion_style(image):
    return apply_fashion_effect(image)


def process_anime_style(image):
    return apply_anime_filter(image)


def process_nature_style(image):
    return apply_nature_effect(image)


def process_modern_style(image):
    return apply_modern_effect(image)
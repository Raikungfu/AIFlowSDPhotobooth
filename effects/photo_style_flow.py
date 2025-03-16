from effects.anime_style_effect import apply_anime_filter
from effects.artistic_effect import apply_artistic_effects
from effects.cartoon_style_effect import apply_cartoon_filter
from effects.fantasy_style_effect import apply_fantasy_effect
from effects.fashion_style_effect import apply_fashion_effect
from effects.id_photo_style_effect import apply_id_photo_effect
from effects.modern_style_effect import apply_modern_effect
from effects.nature_style_effect import apply_nature_effect
from effects.vintage_style_effect import apply_vintage_effect


def process_anime(image):
    return apply_anime_filter(image)


def process_artistic(image):
    return apply_artistic_effects(image)


def process_cartoon(image):
    return apply_cartoon_filter(image)


def process_fantasy(image):
    return apply_fantasy_effect(image)


def process_fashion_style(image):
    return apply_fashion_effect(image)


def process_id_photo(image):
    return apply_id_photo_effect(image)


def process_modern_style(image):
    return apply_modern_effect(image)


def process_nature_style(image):
    return apply_nature_effect(image)


def process_vintage(image):
    return apply_vintage_effect(image)
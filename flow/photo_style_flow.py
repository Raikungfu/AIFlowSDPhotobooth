from sd.artistic_effect import apply_artistic_effects
from sd.cartoon_style_effect import apply_cartoon_filter
from sd.fashion_style_effect import apply_fashion_effect
from sd.id_photo_style_effect import apply_id_photo_effect
from sd.vintage_style_effect import apply_vintage_effect


def process_id_photo(image, bucket):
    return apply_id_photo_effect(image, bucket)


def process_artistic_portrait(image, bucket):
    return apply_artistic_effects(image, bucket)


def process_cartoon(image, bucket):
    return apply_cartoon_filter(image, bucket)


def process_vintage(image, bucket):
    return apply_vintage_effect(image, bucket)


def process_fashion_style(image, bucket):
    return apply_fashion_effect(image, bucket)
from effects.anime_style_effect import apply_anime_filter
from effects.artistic_effect import apply_artistic_effects
from effects.cartoon_style_effect import apply_cartoon_filter
from effects.fantasy_style_effect import apply_fantasy_effect
from effects.fashion_style_effect import apply_fashion_effect
from effects.id_photo_style_effect import apply_id_photo_effect
from effects.modern_style_effect import apply_modern_effect
from effects.nature_style_effect import apply_nature_effect
from effects.vintage_style_effect import apply_vintage_effect
from models.controlnets import controlnet_openpose, controlnet_canny, controlnet_depth


def process_anime(image):
    controlnets_anime = [controlnet_openpose, controlnet_canny]
    return apply_anime_filter(image, controlnets_anime)


def process_artistic(image):
    controlnets_artistic = [controlnet_openpose, controlnet_depth]
    return apply_artistic_effects(image, controlnets_artistic)


def process_cartoon(image):
    controlnets_cartoon = [controlnet_openpose, controlnet_canny]
    return apply_cartoon_filter(image, controlnets_cartoon)


def process_fantasy(image):
    controlnets_fantasy = [controlnet_openpose, controlnet_depth]
    return apply_fantasy_effect(image, controlnets_fantasy)


def process_fashion_style(image):
    controlnets_fashion = [controlnet_openpose, controlnet_canny]
    return apply_fashion_effect(image, controlnets_fashion)


def process_id_photo(image):
    #controlnets_id_photo = [controlnet_openpose, controlnet_canny]
    return apply_id_photo_effect(image)


def process_modern_style(image):
    controlnets_modern = [controlnet_depth, controlnet_canny]
    return apply_modern_effect(image, controlnets_modern)


def process_nature_style(image):
    controlnets_nature = [controlnet_depth]
    return apply_nature_effect(image, controlnets_nature)


def process_vintage(image):
    controlnets_vintage = [controlnet_openpose, controlnet_canny]
    return apply_vintage_effect(image, controlnets_vintage)
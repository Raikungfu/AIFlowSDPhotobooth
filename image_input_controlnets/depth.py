from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation
import torch
import numpy as np
from PIL import Image
import cv2

from configs.model_paths import MODEL_PATHS
from pipelines import pipeline_instance


def ensure_pil_image(image):
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    elif isinstance(image, str):
        return Image.open(image)
    elif isinstance(image, Image.Image):
        return image
    else:
        raise ValueError("Unsupported image format for depth processing")


def load_zoe_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    image_processor = AutoImageProcessor.from_pretrained(MODEL_PATHS["zoe"], cache_dir="/content/pretrained")
    zoe_model = ZoeDepthForDepthEstimation.from_pretrained(MODEL_PATHS["zoe"], cache_dir="/content/pretrained")
    zoe_model.to(device).eval()

    print("Zoe model loaded.")
    return image_processor, zoe_model, device


def depth_image(image):
    width, height = image.size
    image = ensure_pil_image(image)

    inputs = pipeline_instance.zoe_image_processor(images=image, return_tensors="pt").to(pipeline_instance.device)

    with torch.inference_mode():
        outputs = pipeline_instance.zoe_model(**inputs)

    post_processed_output = pipeline_instance.zoe_image_processor.post_process_depth_estimation(
        outputs,
        source_sizes=[(height, width)],
    )

    predicted_depth = post_processed_output[0]["predicted_depth"]

    depth_map = predicted_depth * 255 / predicted_depth.max()
    depth_map = depth_map.detach().cpu().numpy()

    depth_colored = cv2.applyColorMap(depth_map.astype(np.uint8), cv2.COLORMAP_INFERNO)
    return Image.fromarray(depth_colored)

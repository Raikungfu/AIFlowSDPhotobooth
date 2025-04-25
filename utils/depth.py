import torch
import numpy as np
import cv2
from PIL import Image

from configs.model_paths import MODEL_PATHS
from pipelines.inpainting import pipeline_instance


def load_midas_model(model_type="DPT_Large", device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"Loading MiDaS model ({model_type}) on {device}...")
    midas_model = torch.hub.load(MODEL_PATHS["midas"], model_type).to(device)
    midas_model.eval()

    midas_transforms = torch.hub.load(MODEL_PATHS["midas"], "transforms")
    midas_transform = midas_transforms.dpt_transform if "DPT" in model_type else midas_transforms.small_transform
    print("MiDaS model loaded.")

    return midas_model, midas_transform, device


def depth_image(image):
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    input_tensor = pipeline_instance.midas_transform(image).to(pipeline_instance.device)

    with torch.inference_mode():
        prediction = pipeline_instance.midas_model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
    depth_map = (depth_map * 255).astype(np.uint8)

    depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
    return Image.fromarray(depth_colored)
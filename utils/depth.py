import torch
import numpy as np
import cv2
from PIL import Image

def create_depth_image(image, model_type="DPT_Large"):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform if "DPT" in model_type else midas_transforms.small_transform

    print("Image shape after preprocessing:", image.shape)

    input_batch = transform(image).to(device)
    print("Input batch shape:", input_batch.shape)

    with torch.no_grad():
        depth_map = midas(input_batch)

    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_map = (depth_map * 255).astype(np.uint8)

    depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)

    depth_pil = Image.fromarray(depth_colored)
    
    return depth_pil

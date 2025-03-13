import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from midas.model_loader import load_model

def create_depth_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("DPT_Large", device)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])
    
    image_tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)

    with torch.no_grad():
        depth_map = model(image_tensor)

    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_map = (depth_map * 255).astype(np.uint8)

    depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
    depth_pil = Image.fromarray(depth_colored)
    
    depth_pil.save("depth_map.png")
    return depth_pil

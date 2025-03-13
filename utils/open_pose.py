from PIL import Image, ImageDraw
import numpy as np
import torch
from models.controlnets import controlnet_openpose

def create_openpose_image(image):
    image = image.convert("RGB")
    image_tensor = torch.tensor(np.array(image)).unsqueeze(0).float()

    with torch.no_grad():
        output = controlnet_openpose(image_tensor)
    
    openpose_image = draw_openpose(output)
    
    return openpose_image


def draw_openpose(output):
    openpose_image = Image.new("RGB", (output.size(2), output.size(1)), (255, 255, 255))
    draw = ImageDraw.Draw(openpose_image)

    for i in range(output.shape[1]):
        x, y = output[0, i, 0], output[0, i, 1]
        draw.point((x, y), fill="red")
    
    return openpose_image
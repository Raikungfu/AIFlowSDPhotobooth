import traceback
from lib.stable_diffusion import controlnet_pipeline

def apply_controlnet(image, controlnet_type="canny"):
    try:
        return controlnet_pipeline(
            image=image,
            control_type=controlnet_type,
            guidance_scale=1.0,
            num_inference_steps=50
        ).images[0]
    except Exception as e:
        traceback.print_exc()
        print(f"Error during ControlNet: {e}")

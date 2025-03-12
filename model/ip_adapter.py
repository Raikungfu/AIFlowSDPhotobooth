import traceback
from lib.stable_diffusion import inpainting_pipeline

def apply_ip_adapter(image, face_image, strength=0.8):
    try:
        return inpainting_pipeline(
        prompt="",
        negative_prompt="",
        image=image,
        strength=strength,
        guidance_scale=1.0,
        reference_image=face_image,
        num_inference_steps=50
        ).images[0]
    except Exception as e:
        traceback.print_exc()
        print(f"Error during IP-Adapter: {e}")


    

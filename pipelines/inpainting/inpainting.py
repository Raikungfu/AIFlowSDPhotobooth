import torch
from configs.model_paths import MODEL_PATHS
from diffusers import StableDiffusionControlNetInpaintPipeline, UniPCMultistepScheduler
from models.controlnets import controlnet_openpose, controlnet_canny, controlnet_depth

def load_inpainting_pipeline(device="cuda", controlnets=None):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

    def dummy_checker(images, **kwargs):
        return images, [False] * len(images)

    controlnet_map = {
        "openpose": controlnet_openpose,
        "canny": controlnet_canny,
        "depth": controlnet_depth
    }
    selected_controlnets = [controlnet_map[name] for name in controlnets] if controlnets else None

    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "botp/stable-diffusion-v1-5-inpainting",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        controlnet=selected_controlnets
    )

    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

    ip_adapter = torch.load(MODEL_PATHS["ip_adapter"], map_location=device)
    pipeline.unet.load_state_dict(ip_adapter, strict=False)

    pipeline.enable_attention_slicing()
    pipeline.safety_checker = dummy_checker
    pipeline.to(device)

    return pipeline

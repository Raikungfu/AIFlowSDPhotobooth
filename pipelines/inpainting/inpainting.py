import torch
from configs.model_paths import MODEL_PATHS
from diffusers import StableDiffusionControlNetInpaintPipeline, UniPCMultistepScheduler
from models.controlnets import controlnet_openpose, controlnet_canny, controlnet_depth

inpainting_pipeline = None

def load_inpainting_pipeline(controlnets=None):
    global inpainting_pipeline
    try:
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)

        def dummy_checker(images, **kwargs):
            return images, [False] * len(images)
        
        if inpainting_pipeline is not None:
            if controlnets:
                controlnet_map = {
                    "openpose": controlnet_openpose,
                    "canny": controlnet_canny,
                    "depth": controlnet_depth
                }
                selected_controlnets = [controlnet_map[name].to("cuda") for name in controlnets]
                inpainting_pipeline.controlnet = selected_controlnets
            return inpainting_pipeline

        controlnet_map = {
            "openpose": controlnet_openpose,
            "canny": controlnet_canny,
            "depth": controlnet_depth
        }
        selected_controlnets = [controlnet_map[name].to("cuda") for name in controlnets] if controlnets else None

        inpainting_pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "botp/stable-diffusion-v1-5-inpainting",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            controlnet=selected_controlnets
        ).to("cuda")

        inpainting_pipeline.scheduler = UniPCMultistepScheduler.from_config(inpainting_pipeline.scheduler.config)

        ip_adapter = torch.load(MODEL_PATHS["ip_adapter"], map_location="cuda")
        inpainting_pipeline.unet.load_state_dict(ip_adapter, strict=False)

        inpainting_pipeline.enable_attention_slicing()
        inpainting_pipeline.safety_checker = dummy_checker
        inpainting_pipeline.to("cuda")
    except Exception as e:
        print(f"Error during loading inpainting pipeline: {e}")
        return None

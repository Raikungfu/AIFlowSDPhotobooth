import traceback
import torch
from configs.model_paths import MODEL_PATHS
from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler

torch.set_num_threads(4)
torch.set_num_interop_threads(4)


def load_inpainting_pipeline(controlnets=None):
    try:
        def dummy_checker(images, **kwargs):
            return images, [False] * len(images)

        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            "botp/stable-diffusion-v1-5-inpainting",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to("cuda")

        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

        ip_adapter = torch.load(MODEL_PATHS["ip_adapter"], map_location="cuda")
        pipeline.unet.load_state_dict(ip_adapter, strict=False)

        pipeline.enable_attention_slicing()
        pipeline.safety_checker = dummy_checker
        pipeline.to("cuda")
        
        print(f"✅ Loaded pipeline: {type(pipeline)}")

        return pipeline
    except Exception as e:
        traceback.print_exc()
        print(f"Error during loading inpainting pipeline: {e}")
        return None


inpainting_pipeline = load_inpainting_pipeline()
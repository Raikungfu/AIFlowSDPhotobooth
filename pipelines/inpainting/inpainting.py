import gc, torch, traceback
from configs.model_paths import MODEL_PATHS
from diffusers import (
    FluxControlNetModel, FluxMultiControlNetModel,
    FluxControlNetPipeline, UniPCMultistepScheduler, StableDiffusionInpaintPipeline
)


def load_controlnet_pipeline():
    try:
        gc.collect()
        torch.cuda.empty_cache()

        def dummy_checker(images, **kwargs):
            return images, [False] * len(images)

        controlnet_union = FluxControlNetModel.from_pretrained(
            MODEL_PATHS["controlnet_union"],
            torch_dtype=torch.float16,
        )
        
        if controlnet_union is None:
            print("❌ controlnet_union not loaded properly")
        else:
            print("✅ controlnet_union loaded successfully")

        print("Loading controlnet...")
        controlnet = FluxMultiControlNetModel([controlnet_union])
        
        if controlnet is None:
            print("❌ controlnet not loaded properly")
        else:
            print("✅ controlnet loaded successfully")

        print("Loading inpainting pipeline...")
        inpainting_pipeline = FluxControlNetPipeline.from_pretrained(
            MODEL_PATHS["base_model"],
            controlnet=controlnet,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to("cuda")
        
        if inpainting_pipeline is None:
            print("❌ inpainting_pipeline not loaded properly")
        else:
            print("✅ inpainting_pipeline loaded successfully")
        
        inpainting_pipeline.scheduler = UniPCMultistepScheduler.from_config(inpainting_pipeline.scheduler.config)
        inpainting_pipeline.safety_checker = dummy_checker

        inpainting_pipeline.enable_attention_slicing()
        inpainting_pipeline.enable_xformers_memory_efficient_attention()
        inpainting_pipeline.enable_model_cpu_offload()

        print("✅ Loaded FLUX.1-dev with Union ControlNet successfully")
        return inpainting_pipeline

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Error loading pipeline: {e}")
        return None


def load_inpainting_pipeline():
    try:
        def dummy_checker(images, **kwargs):
            return images, [False] * len(images)

        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            MODEL_PATHS["inpainting_model"],
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to("cuda")

        if pipeline is None:
            print("❌ inpainting_pipeline not loaded properly")
        else:
            print("✅ inpainting_pipeline loaded successfully")

        ip_adapter = torch.load(MODEL_PATHS["ip_adapter"])
        pipeline.unet.load_state_dict(ip_adapter, strict=False)

        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)


        pipeline.enable_attention_slicing()
        pipeline.enable_xformers_memory_efficient_attention()
        pipeline.safety_checker = dummy_checker
        
        pipeline.enable_model_cpu_offload()
        
        print("✅ Loaded Inpainting with Stable Diffusion 1.5 successfully")

        return pipeline
    except Exception as e:
        traceback.print_exc()
        print(f"Error during loading inpainting pipeline: {e}")
        return None
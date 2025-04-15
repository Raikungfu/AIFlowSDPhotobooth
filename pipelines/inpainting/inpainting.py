import gc, torch, traceback
from configs.model_paths import MODEL_PATHS
from diffusers.hooks import apply_group_offloading
from transformers import BitsAndBytesConfig as BitsAndBytesConfig, T5EncoderModel
from diffusers import (
    FluxControlNetPipeline,
    FluxMultiControlNetModel,
    FluxControlNetModel,
    UniPCMultistepScheduler,
    StableDiffusionInpaintPipeline,
    FluxTransformer2DModel,
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig
)

def load_flux_pipeline():
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True

        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        text_encoder_8bit = T5EncoderModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="text_encoder_2",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
        )

        quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
        transformer_8bit = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="transformer",
            quantization_config=quant_config,
            torch_dtype=torch.float16,
        )

        controlnet_union = FluxControlNetModel.from_pretrained(
            MODEL_PATHS["controlnet_union"],
            torch_dtype=torch.float16,
        )

        controlnet = FluxMultiControlNetModel([controlnet_union])
        
        pipe = FluxControlNetPipeline.from_pretrained(
            MODEL_PATHS["base_model"],
            text_encoder_2=text_encoder_8bit,
            transformer=transformer_8bit,
            torch_dtype=torch.float16,
            device_map="balanced",
        ).to("cuda")
        
        if pipe is None:
            print("❌ Pipeline not loaded properly")
            return None
        else:
            print("✅ Pipeline loaded successfully")

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        pipe.safety_checker = None
        pipe.feature_extractor = None

        pipe.enable_attention_slicing()
        pipe.enable_xformers_memory_efficient_attention()

        apply_group_offloading(pipe.transformer, offload_type="leaf_level", offload_device=torch.device("cpu"), onload_device=torch.device("cuda"), use_stream=True)
        apply_group_offloading(pipe.text_encoder, offload_type="leaf_level", offload_device=torch.device("cpu"), onload_device=torch.device("cuda"), use_stream=True)
        apply_group_offloading(pipe.text_encoder_2, offload_type="leaf_level", offload_device=torch.device("cpu"), onload_device=torch.device("cuda"), use_stream=True)
        apply_group_offloading(pipe.vae, offload_type="leaf_level", offload_device=torch.device("cpu"), onload_device=torch.device("cuda"), use_stream=True)

        pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        pipe.to(torch.float16)

        print("✅ Loaded FLUX.1-dev with full optimization")
        return pipe

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
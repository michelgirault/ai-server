import torch
import os
import base64
import io
from huggingface_hub import hf_hub_download
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler, BitsAndBytesConfig
import runpod

# Global variables for model paths
model_name = os.environ.get('EMBED_MODEL_NAME')



def init():
    """Initialize the model and pipeline"""
    global pipe

    print("Downloading the model locally...")
    hf_hub_download(repo_id=repo_id, cache_dir=local_dir, local_dir=local_dir, filename=sd_model_name)
    print("Model download complete")

    # Configure quantization settings
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the model
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16
    )

    # Initialize the pipeline
    print("Starting the SD3 pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16,
        transformer=model_nf4,
        text_encoder_3=None,
        tokenizer_3=None,
    )

    # Setup the scheduler
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift=1.0)
    pipe = pipe.to("cuda")

    # Additional optimization
    pipe.enable_xformers_memory_efficient_attention()
    
    print("Pipeline initialization complete")
    return pipe

def handler(event):
    """
    This function is called when the serverless function is invoked.
    Args:
        event: RunPod event object containing the request data
    Returns:
        JSON response with the generated image as a base64 string
    """
    global pipe
    
    # Initialize the pipeline if it's not already done
    if pipe is None:
        pipe = init()
    
    # Extract parameters from the event
    job_input = event.get("input", {})
    prompt = job_input.get("prompt", "a beautiful landscape")
    negative_prompt = job_input.get("negative_prompt", "")
    height = job_input.get("height", 768)
    width = job_input.get("width", 768)
    num_inference_steps = job_input.get("num_inference_steps", 30)
    guidance_scale = job_input.get("guidance_scale", 7.5)
    
    try:
        # Generate the image
        print(f"Generating image with prompt: {prompt}")
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images
        end_time.record()
        
        torch.cuda.synchronize()
        generation_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
        
        # Convert the first image to base64
        image_store = io.BytesIO()
        images[0].save(image_store, "PNG")
        image_store.seek(0)
        base64_image = base64.b64encode(image_store.getvalue()).decode('utf-8')
        
        return {
            "status": "success",
            "data": {
                "image": base64_image,
                "generation_time_seconds": generation_time
            }
        }
    
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

# Start the serverless function
runpod.serverless.start({
    "handler": handler
})
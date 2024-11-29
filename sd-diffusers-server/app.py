import torch
import os
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler, BitsAndBytesConfig
import transformers
import io
from fastapi import FastAPI, Response

#declare global variable and repos
repo_id=os.environ['SD_REPO_ID']
repo_id_loras=os.environ['SD_REPO_LORA_ID'] 
local_dir=os.environ['SD_LOCAL_MODEL_REP']
sd_model_name=os.environ['SD_MODEL_NAME']
lora_model_name=os.environ['LORA_MODEL_NAME']

sd3_file =sd_model_name
lora_file =lora_model_name
#download base model
hf_hub_download(repo_id=repo_id,  cache_dir=local_dir, local_dir=local_dir, filename=sd_model_name)
hf_hub_download(repo_id=repo_id_loras, cache_dir=local_dir, local_dir=local_dir, filename=lora_model_name)
#settings for quant
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
#initiate the app
app = FastAPI()

#start pipline with single file as model
pipe = StableDiffusion3Pipeline.from_single_file(
    sd3_file,
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    text_encoder_3 = None,
    tokenizer_3 = None
)

#load the lora weight
pipe.load_lora_weights(lora_file)

#setup the scheduler
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift=1.0)
#fuse the lora with the base model
pipe.fuse_lora(lora_scale=0.125)

#offload to cpu
pipe.enable_model_cpu_offload()

#faster
#pipe.enable_sequential_cpu_offload()

#pipe.to("cuda", dtype=torch.float16)
#image=pipe(prompt="fantasy medieval village", height=512, width=512, num_inference_steps=12, guidance_scale=5.0).images[0]

#save image in current folder
#image.save("output.png")

#call the api and create the options
@app.get("/generate")
def generate(prompt: str):
     image_store = io.BytesIO()
     images = pipe(prompt).images
     images[0].save(image_store, "PNG")

     return Response(content=image_store.getvalue(), media_type="image/png")
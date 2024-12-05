import torch
import os
import base64
import transformers
import io
from huggingface_hub import hf_hub_download
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler, BitsAndBytesConfig
from fastapi import FastAPI, Depends, Query, Body, Response
from base64 import b64encode 


#declare global variable and repos
repo_id=os.environ['SD_REPO_ID']
repo_id_loras=os.environ['SD_REPO_LORA_ID'] 
local_dir=os.environ['SD_LOCAL_MODEL_REP']
sd_model_name=os.environ['SD_MODEL_NAME']
lora_model_name=os.environ['LORA_MODEL_NAME']

#single file loader
sd3_file = os.path.join(local_dir, sd_model_name)
lora_file = os.path.join(local_dir, lora_model_name)

#download base model
print("Download the model locally ...")
hf_hub_download(repo_id=repo_id,  cache_dir=local_dir, local_dir=local_dir, filename=sd_model_name)
hf_hub_download(repo_id=repo_id_loras, cache_dir=local_dir, local_dir=local_dir, filename=lora_model_name)
print("done")
#Declare settings for quant
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)  


#load single file
#transformer = SD3Transformer2DModel.from_single_file(
#    sd3_file,
#    quantization_config=quantization_config,
#    torch_dtype=torch.bfloat16
#)

model_nf4 = SD3Transformer2DModel.from_pretrained(
    repo_id,
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16
)

#start pipline with single file as model
print("starting the pipeline sd3")
pipe = StableDiffusion3Pipeline.from_pretrained(
    repo_id,
    torch_dtype=torch.bfloat16,
    transformer=model_nf4,
    text_encoder_3 = None,
    tokenizer_3 = None
)

#load the lora weight
#pipe.load_lora_weights(lora_file)


#fuse the lora with the base model
#pipe.fuse_lora(lora_scale=0.125)
#setup the scheduler
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config, shift=1.0)

#offload to cpu
pipe.enable_model_cpu_offload()

#alternative solution
#pipe.enable_sequential_cpu_offload()

#pipe.to("cuda", dtype=torch.float16)
#image=pipe(prompt="fantasy medieval village", height=512, width=512, num_inference_steps=12, guidance_scale=5.0).images[0]

#save image in current folder
#image.save("output.png")

#initiate the app
app = FastAPI()

#define class for the body request
#call the api and create the options
@app.post("/generate")
def generate(
    prompt: str = Body(),
    negative_prompt: str = Body(),
    height: int = Body(),
    width: int = Body(), 
    num_inference_steps: int = Body(),
    guidance_scale: float = Body(),
    ):
    #get the image and convert into base64
    image_store = io.BytesIO()
    images = pipe(prompt=prompt, negative_prompt=negative_prompt, height=height, width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images
    #images = pipe(prompt).images[0]
    
    images[0].save(image_store, "PNG")
    print(images)
    return b64encode(image_store.getvalue())
#   return Response(content=image_store.getvalue(), media_type="image/png")
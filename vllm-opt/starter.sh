#!/bin/bash

#aws configure set aws_access_key_id $aws_access_key_id
#aws configure set aws_secret_access_key $aws_secret_access_key

#download model
wget -nc $LLM_MODEL_URL -P $LLM_LOCAL_MODEL_REP


#start server
vllm serve $LLM_MODEL_ID --dtype auto --api-key $LLAMA_API_KEY \
    --host 0.0.0.0 \
    --port 4000 \
    --served-model-name $LLM_MODEL_NAME \
    --quantization $QUANT \
    --cpu-offload-gb $OFFLOAD_CPU
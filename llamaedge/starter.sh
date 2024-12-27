#!/bin/bash

#aws configure set aws_access_key_id $aws_access_key_id
#aws configure set aws_secret_access_key $aws_secret_access_key

#download models for sd
wget -nc $LLM_MODEL_URL -P $LLM_LOCAL_MODEL_REP


#start server
server/bin/wasmedge --dir .:. --env API_KEY=$LLAMA_API_KEY --nn-preload default:GGML:AUTO:$LLM_LOCAL_MODEL_REP$LLM_MODEL_NAME \
    /app/server/llama-api-server.wasm --prompt-template $LLM_TEMPLATE \
    --port 4000 \
    --model-name $LLM_MODEL_NAME \
    --n-gpu-layers $GPU_LAYER \
    --ctx-size $CTX \
    --main-gpu $CUDA
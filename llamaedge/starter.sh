#!/bin/bash

#aws configure set aws_access_key_id $aws_access_key_id
#aws configure set aws_secret_access_key $aws_secret_access_key

#download model
if ! [ -f $LLM_LOCAL_MODEL_REP$LLM_MODEL_NAME ]; then
  echo "File does not exist. start download" && huggingface-cli download bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Meta-Llama-3.1-8B-Instruct-Q4_K_L.gguf --local-dir $LLM_LOCAL_MODEL_REP
fi

#start server
server/bin/wasmedge --dir .:. --nn-preload default:GGML:AUTO:$LLM_LOCAL_MODEL_REP$LLM_MODEL_NAME \
    /app/server/llama-api-server.wasm --prompt-template $LLM_TEMPLATE \
    --port 4000 \
    --model-name $LLM_MODEL_NAME_SIMPLE \
    --n-gpu-layers $GPU_LAYER \
    --ctx-size $CTX &

#wait before starting the runpod
sleep 20
#start the endpoint runpod
python3 -u llama_runpod.py
#!/bin/bash

#aws configure set aws_access_key_id $aws_access_key_id
#aws configure set aws_secret_access_key $aws_secret_access_key

#download models for llama
wget -nc $LLM_MODEL_URL -P $LLM_LOCAL_MODEL_REP

#start server
python3 -m llama_cpp.server --model $LLM_LOCAL_MODEL_REP$LLM_MODEL_NAME --n_gpu_layers $GPU_LAYER \
    --chat_format $LLM_TEMPLATE \
    --embedding $EMBED \
    --port 4000 \
    --host 0.0.0.0 \
    --n_ctx $CTX \
    --$EXTRA_ARGS
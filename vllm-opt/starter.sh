#!/bin/bash

#start server
vllm serve $LLM_MODEL_ID --dtype $DTYPE --api-key $LLAMA_API_KEY \
    --host 0.0.0.0 \
    --port 4000 \
    --enforce-eager $EAGER \
    --served-model-name $LLM_MODEL_NAME \
    --quantization $QUANT \
    --cpu-offload-gb $OFFLOAD_CPU \
    --max-model-len $MODEL_LEN \
    --gpu_memory_utilization $MEM_USAGE
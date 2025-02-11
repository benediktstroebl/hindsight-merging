#!/bin/bash
NODE_NAME=della-l07g3
CLUSTER_NUMBER=2

# Start deepseek server in a detached screen session
screen -dmS ${CLUSTER_NUMBER}_v2 ssh -t $NODE_NAME "cd /scratch/gpfs/bs6865/messi-thinking && \
export HF_HUB_OFFLINE=1 && \
export HF_HOME='/scratch/gpfs/vv7118/models/mixed_models' && \
conda activate vllm && \
vllm serve /scratch/gpfs/vv7118/models/mixed_models/llama-3.1-8b-mixed-slerp-0.70 --gpu_memory_utilization 0.8 --tensor_parallel_size 4 --port 8000"

# wait
sleep 60
# Example lerp 0.7 model from mergekit (on two GPUs)
evalplus.evaluate --model "mergekit/ll31_70b_lerp_0.7" --dataset mbpp --backend vllm --n_samples 11 --temperature 0.7 --tp 2

# Llama-3.1-70B-Instruct (on two GPUs)
evalplus.evaluate --model "meta-llama/Llama-3.1-70B-Instruct" --dataset mbpp --backend vllm --n_samples 11 --temperature 0.7 --tp 2

# Llama-3.1-8B (on one GPU)
evalplus.evaluate --model "meta-llama/Llama-3.1-8B" --dataset mbpp --backend vllm --n_samples 11 --temperature 0.7 --force_base_prompt

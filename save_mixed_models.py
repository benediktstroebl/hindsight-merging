import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
import torch.nn.functional as F

# Get HF_HOME directory
hf_home = os.getenv('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
mixed_models_dir = os.path.join(hf_home, 'mixed_models')
os.makedirs(mixed_models_dir, exist_ok=True)

# Check if GPU is available and set the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths
base_name = "meta-llama/Llama-3.1-8B"
instruct_name = "meta-llama/Llama-3.1-8B-Instruct"

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(base_name)
base_model = AutoModelForCausalLM.from_pretrained(base_name, device_map="auto")
instruct_model = AutoModelForCausalLM.from_pretrained(instruct_name, device_map="auto")

# Create mixed models at different interpolation points
alphas = [0.3, 0.5, 0.9]  # Example interpolation points

for alpha in alphas:
    print(f"Creating mixed model with alpha={alpha}")
    
    # Create a new model by interpolating weights
    mixed_model = deepcopy(base_model)
    
    for param_base, param_instruct, param_mixed in zip(
        base_model.parameters(), 
        instruct_model.parameters(), 
        mixed_model.parameters()
    ):
        param_mixed.data = (1 - alpha) * param_base.data + alpha * param_instruct.data
    
    # Create directory for this mixed model
    model_dir = os.path.join(mixed_models_dir, f'llama-3.1-8b-mixed-{alpha:.2f}')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the mixed model and tokenizer
    mixed_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    
    # Save config file with interpolation information
    config = mixed_model.config
    config.interpolation = {
        'base_model': base_name,
        'instruct_model': instruct_name,
        'alpha': alpha
    }
    config.save_pretrained(model_dir)
    
    print(f"Saved mixed model to {model_dir}")
    
    # Clear CUDA cache and delete the mixed model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del mixed_model
    import gc
    gc.collect()  # Run garbage collector

# Optional: Test generation with one of the mixed models
def test_generation(model_path):
    print(f"\nTesting generation with model from {model_path}")
    
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
    try:
        story_query = "Tell a 100-word story about a cat that finds a lost treasure in an abandoned lighthouse."
        inputs = tokenizer(story_query, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
        
        story = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Generated Story:\n", story)
    
    finally:
        # Clean up after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del model
        gc.collect()

# Test the 0.5 interpolation model
test_model_path = os.path.join(mixed_models_dir, 'llama-3.2-1b-mixed-0.40')
if os.path.exists(test_model_path):
    test_generation(test_model_path)
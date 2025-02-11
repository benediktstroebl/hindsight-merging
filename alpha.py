import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
import torch.nn.functional as F

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths
base_name = "meta-llama/Llama-3.2-1B"
instruct_name = "meta-llama/Llama-3.2-1B-Instruct"

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(base_name)
base_model = AutoModelForCausalLM.from_pretrained(base_name).to(device)
instruct_model = AutoModelForCausalLM.from_pretrained(instruct_name).to(device)

# Interpolation factor (alpha): 0.5 means equal weighting
alphas = np.arange(0, 1, 0.01)

# Maximum length of generated story
max_length = 200
model = deepcopy(base_model).to(device)
# Storytelling query
story_query = "Tell a 100-word story about a cat that finds a lost treasure in an abandoned lighthouse."

# Tokenize the input
inputs = tokenizer(story_query, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
for i in range(max_length):

    # Interpolate weights
    if i % 10 == 0:
        alpha = np.random.uniform(0, 1)
        print(alpha)
        for param_base, param_instruct, param_model in zip(base_model.parameters(), instruct_model.parameters(), model.parameters()):
            param_model.data = ((1 - alpha) * param_base.data + alpha * param_instruct.data).to(device)

    # Forward pass to get logits for the next token
    with torch.no_grad():
        outputs = model(input_ids=input_ids)

    logits = outputs.logits[:, -1, :]  # Next-token logits

    # Calculate next-word entropy
    probs = F.softmax(logits, dim=-1)

    # Sample the next token (you can use different sampling methods here)
    next_token_id = torch.multinomial(probs, num_samples=1).item()

    # Append the generated token to the sequence
    input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)

    # Stop if the token is a special token like <eos>
    if next_token_id == tokenizer.eos_token_id:
        break

# Decode the generated story
story = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print("Generated Story:\n", story)
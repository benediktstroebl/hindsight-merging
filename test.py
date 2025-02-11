import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model pathsd
base_name = "meta-llama/Llama-3.2-1B"
instruct_name = "meta-llama/Llama-3.2-1B-Instruct"

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(base_name)
base_model = AutoModelForCausalLM.from_pretrained(base_name, device_map=device)
instruct_model = AutoModelForCausalLM.from_pretrained(instruct_name, device_map=device)

# Maximum length of generated story
max_length = 200

# Top-k parameter
top_k = 50

# Select models for alternating use
models = [base_model, instruct_model]


# Storytelling query
story_query = "Tell a 100-word story about a cat that finds a lost treasure in an abandoned lighthouse."

for s in range(100):
    # Tokenize the input
    inputs = tokenizer(story_query, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    for i in range(max_length):
        # Randomly choose between the two models
        k = i % 2

        # Forward pass to get logits for the next token
        with torch.no_grad():
            outputs = models[k](input_ids=input_ids)

        logits = outputs.logits[:, -1, :]  # Next-token logits

        # Apply top-k filtering
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)

        # Sample from the top-k tokens
        next_token_index = torch.multinomial(top_k_probs, num_samples=1).item()
        next_token_id = top_k_indices[0, next_token_index].item()

        # Append the generated token to the sequence
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)

        # Stop if the token is a special token like <eos>
        if next_token_id == tokenizer.eos_token_id:
            break

    # Decode the generated story
    prompt_length = inputs["input_ids"].shape[1]
    generated_story_ids = input_ids[0][prompt_length:]
    story = tokenizer.decode(generated_story_ids, skip_special_tokens=True)
    words = story.split()
    unique_words = set(words)
    unique_ratio = len(unique_words) / len(words) if words else 0
    print(unique_ratio)


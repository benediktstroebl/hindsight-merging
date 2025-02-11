import os
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

MODELS = [
    "ll8_distill_linear_0.7_instruct",
    "ll8_distill_linear_0.9_instruct",
    "ll8_distill_linear_0.5_instruct",
    "ll8_distill_linear_0.3_instruct",
]


def compute_model_entropy(model, tokenizer, texts):
    total_entropy = 0.0
    total_tokens = 0
    for sample in texts:
        text = sample['text']
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        logits = outputs.logits  # shape (1, seq_len, vocab_size)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropies = -(probs * log_probs).sum(dim=-1)  # shape (1, seq_len)
        total_entropy += entropies.sum().item()
        total_tokens += entropies.numel()
    return total_entropy / total_tokens if total_tokens > 0 else float('nan')

def compute_model_entropies(model, tokenizer, texts):
    results = []
    for sample in tqdm(texts, desc="Computing entropies"):
        text = sample
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
        input_ids = inputs["input_ids"]
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropies_tensor = -(probs * log_probs).sum(dim=-1)
        token_entropies = entropies_tensor.squeeze(0).tolist()
        sum_entropy = sum(token_entropies)
        avg_entropy = sum_entropy / len(token_entropies) if token_entropies else float('nan')
        results.append({
            "avg_entropy": avg_entropy,
            "sum_entropy": sum_entropy,
            "token_entropies": token_entropies,
            "num_tokens": len(token_entropies)
        })
    return results

def compute_position_averages(token_lists):
    max_len = max(len(tokens) for tokens in token_lists)
    pos_sums = [0.0] * max_len
    pos_counts = [0] * max_len
    for tokens in token_lists:
        for i, t in enumerate(tokens):
            pos_sums[i] += t
            pos_counts[i] += 1
    return [pos_sums[i] / pos_counts[i] if pos_counts[i] > 0 else None for i in range(max_len)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, required=True, help='Directory where models are stored')
    args = parser.parse_args()

    dataset = load_dataset("simplescaling/s1K")
    texts = dataset['train']
    texts = [text[0] for text in texts['thinking_trajectories']]
    
    # random sample 50 texts
    texts = random.sample(texts, 50)

    results_dir = "entropies"
    os.makedirs(results_dir, exist_ok=True)
    all_avg_entropies = {}
    all_token_entropies_flat = {}
    all_token_entropies_raw = {}

    for model_name in MODELS:
        model_path = os.path.join(args.target_dir, model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        model.eval()
        model_results = compute_model_entropies(model, tokenizer, texts)

        # Save entropies for each model to a JSON file
        with open(os.path.join(results_dir, f"{model_name}_entropies.json"), "w") as f:
            json.dump(model_results, f)

        all_avg_entropies[model_name] = [r["avg_entropy"] for r in model_results]
        all_token_entropies_flat[model_name] = [t for r in model_results for t in r["token_entropies"]]
        all_token_entropies_raw[model_name] = [r["token_entropies"] for r in model_results]

        overall_avg = sum(all_avg_entropies[model_name]) / len(all_avg_entropies[model_name])
        print(f"Model: {model_name} - Overall Average Entropy: {overall_avg:.4f}")

    # Boxplot: Distribution of average entropies per text across models
    fig, ax = plt.subplots()
    model_labels = list(all_avg_entropies.keys())
    data = [all_avg_entropies[m] for m in model_labels]
    ax.boxplot(data, labels=model_labels)
    ax.set_title("Distribution of Average Entropy per Text")
    ax.set_ylabel("Average Entropy")
    plt.savefig(os.path.join(results_dir, "average_entropy_boxplot.png"))
    plt.close(fig)

    # Histogram: Token-level entropy distribution across models
    fig, ax = plt.subplots()
    for model, tokens in all_token_entropies_flat.items():
        ax.hist(tokens, bins=50, alpha=0.5, label=model)
    ax.set_title("Histogram of Token-Level Entropies")
    ax.set_xlabel("Token Entropy")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.savefig(os.path.join(results_dir, "token_entropy_histogram.png"))
    plt.close(fig)

    # Line Plot: Average token entropy by position for each model
    fig, ax = plt.subplots()
    for model, token_lists in all_token_entropies_raw.items():
        avg_per_position = compute_position_averages(token_lists)
        ax.plot(range(len(avg_per_position)), avg_per_position, label=model)
    ax.set_title("Average Token Entropy by Position")
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Average Token Entropy")
    ax.legend()
    plt.savefig(os.path.join(results_dir, "token_entropy_by_position.png"))
    plt.close(fig)

if __name__ == '__main__':
    main() 
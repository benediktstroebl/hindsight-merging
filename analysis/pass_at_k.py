import os
import glob
import json
import re
import numpy as np
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt

def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""
    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)
    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def process_directory(root_path):
    pattern = os.path.join(root_path, "**", "converted_*.json")
    files = glob.glob(pattern, recursive=True)
    questions = {}
    filename_regex = r"converted_question_(\d+)_sample_(\d+)(_with_embeddings)?\.json"
    for file in tqdm(files, desc=f"Processing {root_path}"):
        
        match = re.search(filename_regex, os.path.basename(file))
        
        if not match:
            continue
        qid = int(match.group(1))
        sample_idx = int(match.group(2))
        
        # skip questions that are not in the first 1000
        if qid >= 1000:
            print(f"Skipping {file} because question index is greater than 1000")
            continue
        # skip samples that are not in the first 10
        if sample_idx >= 10:
            print(f"Skipping {file} because sample index is greater than 10")
            continue
        
        with open(file, "r") as f:
            data = json.load(f)
        correct = data.get("correctness", False)
        questions.setdefault(qid, []).append((sample_idx, correct))
    total_questions = len(questions)
    n_list = []
    c_list = []
    for qid, samples in questions.items():
        n = len(samples)
        c = sum(1 for _, correct in samples if correct)
        n_list.append(n)
        c_list.append(c)
    average_pass_by_k = []
    for k in range(1, 11):
        pass_at_k_vals = estimate_pass_at_k(n_list, c_list, k)
        average_pass = np.mean(pass_at_k_vals) if total_questions > 0 else 0
        average_pass_by_k.append(average_pass)
    return average_pass_by_k

def main():
    directories = [
        "taco_instruct_llama_8b_single_slerp_0.7",
        "taco_instruct_llama_8b_single",
        "/scratch/gpfs/vv7118/projects/messi-thinking/llama-3.1-8b",
        # "/scratch/gpfs/vv7118/projects/messi-thinking/embeddings/taco_instruct_llama_8b_single",
        # "taco_instruct_llama_8b_single_slerp_0.5",
        "/scratch/gpfs/vv7118/projects/messi-thinking/embeddings/taco_instruct_llama_8b_single_slerp_0.90"
    ]
    results = {}
    for directory in directories:
        results[directory] = process_directory(directory)
    ks = list(range(1, 11))
    plt.figure(figsize=(10, 6))
    for directory, pass_rates in results.items():
        plt.plot(ks, pass_rates, marker="o", label=directory.split("/")[-1])
    plt.xlabel("k")
    plt.ylabel("pass@k")
    plt.ylim(0, 0.5)
    plt.title("Pass@k for each directory")
    plt.legend()
    plt.grid(True)
    plt.savefig("pass_at_k.png")
    plt.show()

if __name__ == "__main__":
    main()

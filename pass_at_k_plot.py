import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')



def estimator(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def calculate_pass_rates(directory):
    pattern = os.path.join(directory, "question_*_sample_*.json")
    files = glob.glob(pattern)
    questions = {}
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
        qid = data["task_id"]
        if qid not in questions:
            questions[qid] = []
        questions[qid].append(data)
    num_samples_list = [len(samples) for samples in questions.values()]
    num_correct_list = [sum(1 for s in samples if s["correctness"]) for samples in questions.values()]
    overall_pass_rates = []
    for k in range(1, 11):
        rates = []
        for n, c in zip(num_samples_list, num_correct_list):
            rates.append(estimator(n, c, k))
        overall_pass = np.mean(rates) if rates else 0
        overall_pass_rates.append(overall_pass)
    return overall_pass_rates

def main():
    directories = [
        "SkyThought/samples/taco-ll3-8B-llama-8b-big-lora-256-epochs-2-merged",
        "SkyThought/samples/taco-ll3-8B-llama-8b-slerp-0.7-big-lora-256-epochs-2-merged",
        "SkyThought/samples/taco-ll3-8B-llama-8b-big-slerp-0.5-lora-256-epochs-2-merged"
    ]

    k_values = list(range(1, 11))
    plt.figure(figsize=(8, 6))
    
    names = {
        "SkyThought/samples/taco-ll3-8B-llama-8b-big-lora-256-epochs-2-merged": r"$\alpha=1$",
        "SkyThought/samples/taco-ll3-8B-llama-8b-slerp-0.7-big-lora-256-epochs-2-merged": r"$\alpha=0.7$",
        "SkyThought/samples/taco-ll3-8B-llama-8b-big-slerp-0.5-lora-256-epochs-2-merged": r"$\alpha=0.5$"
    }
    
    for directory in directories:
        pass_rates = calculate_pass_rates(directory)
        label = names.get(directory, os.path.basename(os.path.normpath(directory)))
        # print it in a nice format
        print(f"{label}: {pass_rates}")
        sns.lineplot(x=k_values, y=pass_rates, marker="o", label=label, linewidth=3, markersize=10)
    
    plt.xlabel("k", fontsize=22)
    plt.ylabel("Pass@k", fontsize=22)
    plt.xticks(k_values, fontsize=22)
    plt.yticks(np.arange(0, 0.41, 0.1), fontsize=22)
    plt.ylim(0, 0.4)
    plt.grid(False)
    plt.legend(fontsize=20, frameon=False, loc="lower right")
    plt.tight_layout()
    sns.despine()
    plt.savefig("finetuning_pass_at_k.pdf")
    plt.show()

if __name__ == "__main__":
    main() 
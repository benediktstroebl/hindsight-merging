import os
import re
import glob
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from tqdm import tqdm
def truncate_text(text, max_tokens=8000):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = encoding.decode(tokens)
    return text

def load_samples(directory):
    samples_by_question = {}
    pattern = re.compile(r"question_(\d+)_sample_\d+\.json")
    json_files = glob.glob(os.path.join(directory, "*.json"))
    for file_path in json_files:
        basename = os.path.basename(file_path)
        match = pattern.search(basename)
        if not match:
            continue
        question_id = match.group(1)
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
        # Use the generated_text as the sample text and truncate to 8000 tokens
        sample_text = data.get("generated_text", "")
        if not sample_text:
            continue
        sample_text = truncate_text(sample_text, 8000)
        samples_by_question.setdefault(question_id, []).append(sample_text)
    return samples_by_question

def get_pairwise_cosine_similarities(embeddings):
    if len(embeddings) < 2:
        return []
    sims = cosine_similarity(embeddings)
    n = sims.shape[0]
    values = []
    for i in range(n):
        for j in range(i + 1, n):
            values.append(sims[i, j])
    return values

def get_openai_embeddings(texts):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    embeddings = [data.embedding for data in response.data]
    return np.array(embeddings)

def compute_similarity_distribution(samples_by_question, num_questions=None, global_pairwise=False):
    question_ids = list(samples_by_question.keys())
    if num_questions is not None and num_questions < len(question_ids):
        import random
        question_ids = random.sample(question_ids, num_questions)
    if global_pairwise:
        texts_global = []
        for qid in tqdm(question_ids, desc="Computing global pairwise similarities"):
            texts_global.extend(samples_by_question[qid])
        if len(texts_global) < 2:
            return []
        embeddings = get_openai_embeddings(texts_global)
        return get_pairwise_cosine_similarities(embeddings)
    else:
        all_similarities = []
        for question_id in tqdm(question_ids, desc="Computing similarity distribution"):
            texts = samples_by_question[question_id]
            if len(texts) < 2:
                continue
            embeddings = get_openai_embeddings(texts)
            sims = get_pairwise_cosine_similarities(embeddings)
            all_similarities.extend(sims)
        return all_similarities

def plot_distributions(sim1, sim2, label1="Dataset 1", label2="Dataset 2", global_pairwise=False):
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 50)
    plt.hist(sim1, bins=bins, alpha=0.5, label=label1, density=True)
    plt.hist(sim2, bins=bins, alpha=0.5, label=label2, density=True)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.legend(loc="upper left")
    plt.title("Distribution of Pairwise Cosine Similarities")
    plt.tight_layout()
    plt.savefig(f"similarity_distribution_{'global' if global_pairwise else 'local'}.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir1", help="Directory for first dataset")
    parser.add_argument("--dir2", help="Directory for second dataset")
    parser.add_argument("--num_questions", type=int, default=50, help="Number of randomly sampled questions for analysis (-1 for all)")
    parser.add_argument("--global_pairwise", action="store_true", help="Compute pairwise similarities globally across all selected questions for each dataset")
    args = parser.parse_args()

    samples1 = load_samples(args.dir1)
    samples2 = load_samples(args.dir2)
    
    num_questions = args.num_questions if args.num_questions > 0 else None
    sims1 = compute_similarity_distribution(samples1, num_questions, global_pairwise=args.global_pairwise)
    sims2 = compute_similarity_distribution(samples2, num_questions, global_pairwise=args.global_pairwise)
    
    plot_distributions(sims1, sims2, global_pairwise=args.global_pairwise)

if __name__ == "__main__":
    main() 
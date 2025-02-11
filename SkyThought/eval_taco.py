import asyncio
import json
from tqdm import tqdm
from openai import AsyncOpenAI
from datasets import load_dataset
from skythought_evals.tasks.taco.taco_handler import TACOTaskHandler
from skythought_evals.tasks.base import TaskConfig
import re
import argparse
import os
import glob
import numpy as np
import itertools

from prompt import SKY_T1_FIXED, BASE_MODEL_SYSTEM_PROMPT, generate_prompt, convert_prompt, convert_prompt_example

task_config = TaskConfig.from_yaml('skythought/skythought_evals/tasks/taco/taco.yaml')
handler = TACOTaskHandler(task_config)

# Initialize evaluation dataset 
# difficulties = ['ALL']
# difficulties = ["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD"] 
diff = "EASY"
# skills = ['ALL']
# skills = ["Data structures", "Sorting", "Range queries", "Complete search", "Amortized analysis", "Dynamic programming", "Bit manipulation", "Greedy algorithms"]

from datasets import load_dataset
taco = load_dataset("BAAI/TACO", trust_remote_code=True)["test"].filter(lambda x: x["difficulty"] == diff)
# taco = load_dataset('BAAI/TACO', split='test', skills=skills)
from prompt import SKY_T1_FIXED, BASE_MODEL_SYSTEM_PROMPT, generate_prompt

def get_prompt(sample):
    test_case = json.loads(sample["input_output"])
    starter_code = sample["starter_code"]
    prompt_text = generate_prompt(test_case, sample["question"], starter_code)
    return [{"role": "system", "content": SKY_T1_FIXED}, {"role": "user", "content": prompt_text}]

# setting up times of run
# n_samples = 1  # Removed; now set via command-line argument in main
temperature = 0.7
top_p = 0.7
max_new_tokens = 8192
n_samples = 10

instruct_model = "/scratch/gpfs/bs6865/messi-thinking/SkyThought/skythought/train/LLaMA-Factory/saves/taco-ll3-8B-llama-8b-big-lora-256-epochs-2-merged"
output_file = f'generations_{diff}_{n_samples}_{temperature}_{top_p}_{instruct_model.split("/")[-1]}.json'

OUTPUT_DIR = f"samples/{instruct_model.split('/')[-1]}"

instruct_client = AsyncOpenAI(api_key="token-abc123", base_url="http://localhost:8000/v1")
formatting_client = AsyncOpenAI()

async def perform_instruct_inference(conversation):
    chat_completion = await instruct_client.chat.completions.create(
        model=instruct_model,
        messages=conversation,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )
    generated = chat_completion.choices[0].message.content.replace("</think>", "[end_of_thought]").replace("<think>", "[begin_of_thought]")
    return generated

async def perform_format_conversion(content):
    conv_prompt = convert_prompt.format(example=convert_prompt_example, content=content)
    conv = [
        {"role": "system", "content": "You are a solution format convertor."},
        {"role": "user", "content": conv_prompt}
    ]
    conversion = await formatting_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conv,
        max_tokens=16384,
        temperature=0.7
    )
    return conversion.choices[0].message.content

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

# New function to process a single generation for a task sample
async def process_generation(idx, sample, conversation, gen_idx, semaphore, format_output):
    async with semaphore:
        gen = await perform_instruct_inference(conversation)
        if format_output:
            gen_reformatted = await perform_format_conversion(gen)
        else:
            gen_reformatted = None
        res = handler.update_results(sample, gen)
        return {
            "task_id": idx,
            "sample_idx": gen_idx,
            "prompt": conversation,
            "output": gen,
            "output_reformatted": gen_reformatted,
            "correctness": res["correctness"],
            "reason": res["reason"]
        }

# Updated function to process reevaluation for individual generation
async def process_reevaluation_sample(entry, semaphore, format_output):
    async with semaphore:
        idx = entry["task_id"]
        sample = taco[idx]
        gen = entry["output"]
        if format_output:
            gen_reformatted = await perform_format_conversion(gen)
            entry["output_reformatted"] = gen_reformatted
        res = handler.update_results(sample, gen)
        entry["correctness"] = res["correctness"]
        entry["reason"] = res["reason"]
        return entry

async def main(format_output, input_file=None, n_samples_arg=1):
    global n_samples
    n_samples = n_samples_arg
    output_file = f'generations_{diff}_{n_samples}_{temperature}_{top_p}_{instruct_model.split("/")[-1]}.json'
    if input_file:
        # Reevaluation mode: load existing generation file and reprocess outputs with semaphore
        with open(input_file, 'r') as f:
            data = json.load(f)
        semaphore = asyncio.Semaphore(100)  # Added semaphore for reevaluation mode
        tasks = []
        for entry in data:
            tasks.append(asyncio.create_task(process_reevaluation_sample(entry, semaphore, format_output)))
        reevaluated_file = "reevaluated_" + input_file.split("/")[-1]
        total_tasks = len(tasks)
        count = 0
        with open(reevaluated_file, 'w') as f:
            f.write("[\n")
            for future in tqdm(asyncio.as_completed(tasks), total=total_tasks, desc="Reevaluating samples"):
                result = await future
                count += 1
                if count < total_tasks:
                    f.write(json.dumps(result) + ",\n")
                else:
                    f.write(json.dumps(result) + "\n")
            f.write("]\n")
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        semaphore = asyncio.Semaphore(100)
        tasks = []
        for idx, sample in enumerate(taco):
            conversation = get_prompt(sample)
            for j in range(n_samples):
                if os.path.exists(os.path.join(OUTPUT_DIR, f"question_{idx}_sample_{n_samples-1}.json")):
                    print(f"Skipping sample {idx} because it has already been processed")
                    continue
                tasks.append(asyncio.create_task(process_generation(idx, sample, conversation, j, semaphore, format_output)))
        total_tasks = len(tasks)
        for future in tqdm(asyncio.as_completed(tasks), total=total_tasks, desc="Processing samples"):
            result = await future
            filename = os.path.join(OUTPUT_DIR, f"question_{result['task_id']}_sample_{result['sample_idx']}.json")
            with open(filename, 'w') as f:
                json.dump(result, f, indent=4)
        sample_files = glob.glob(os.path.join(OUTPUT_DIR, "question_*_sample_*.json"))
        questions = {}
        for file in sample_files:
            with open(file, 'r') as f:
                data = json.load(f)
            qid = data["task_id"]
            if qid not in questions:
                questions[qid] = []
            questions[qid].append(data)

        num_samples_list = [len(samples) for samples in questions.values()]
        num_correct_list = [sum(1 for s in samples if s["correctness"]) for samples in questions.values()]

        for k in range(1, 11):
            pass_rates = estimate_pass_at_k(num_samples_list, num_correct_list, k)
            overall_pass = pass_rates.mean()
            print(f"Overall Pass@{k}: {overall_pass:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run taco evaluation with optional output formatting and reevaluation of existing generation file.")
    parser.add_argument("--format_output", action="store_true", help="Format output before evaluation.")
    parser.add_argument("--input_file", type=str, default=None, help="Path to an existing generation file for reevaluation.")
    args = parser.parse_args()
    asyncio.run(main(args.format_output, args.input_file, n_samples))



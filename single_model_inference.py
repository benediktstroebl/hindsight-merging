import os
import json
import asyncio
import logging
import argparse

from datasets import load_dataset
from tqdm import tqdm
from dynamic_scaling.prompt import SKY_T1_FIXED, BASE_MODEL_SYSTEM_PROMPT, generate_prompt
from openai import AsyncOpenAI
from vllm import SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parameters
MAX_TOKENS = 8192
NUM_SAMPLES = 10  # samples per dataset entry

# OUTPUT_DIR = "llama-3.1-8b"
OUTPUT_DIR = "messi-thinking/taco_instruct_llama_8b_single_slerp_0.5/"
# OUTPUT_DIR = 'taco_instruct_llama_8b_single_slerp_0.90'
use_slerp = True
# Models
merge_frac = "0.50"
base_model = "models/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
if use_slerp:
    instruct_model = f"models/mixed_models/llama-3.1-8b-mixed-slerp-{merge_frac}"
else:
    instruct_model = f"models/mixed_models/llama-3.1-8b-mixed-{merge_frac}"

# Use async inference with OpenAI client
USE_OPENAI = True


base_client = AsyncOpenAI(api_key="token-abc123", base_url="http://localhost:8000/v1")
instruct_client = AsyncOpenAI(api_key="token-abc123", base_url="http://localhost:8000/v1")

def get_prompt(sample):
    """Parse test cases and starter code from problem to create a prompt for the LLM."""
    test_case = json.loads(sample["input_output"])
    starter_code = sample["starter_code"]
    prompt_text = generate_prompt(test_case, sample["question"], starter_code)
    return [{"role": "system", "content": SKY_T1_FIXED}, {"role": "user", "content": prompt_text}]

def get_prompt_instruct(sample):
    return get_prompt(sample)
    
def get_prompt_base(sample):
    prompt = get_prompt(sample)
    return BASE_MODEL_SYSTEM_PROMPT + prompt[1]["content"] + "\nAssistant:\n[begin_of_thought]"

async def perform_base_inference(input_text):
    completion = await base_client.completions.create(
        model=base_model,
        prompt=input_text,
        max_tokens=MAX_TOKENS,
        temperature=0.8,
        top_p=0.95)
    generated = completion.choices[0].text.replace("</think>", "[end_of_thought]").replace("<think>", "[begin_of_thought]")
    return generated

async def perform_instruct_inference(conversation):
    chat_completion = await instruct_client.chat.completions.create(
        model=instruct_model,
        messages=conversation,
        max_tokens=MAX_TOKENS,
        temperature=0.7,
        top_p=0.7)
    generated = chat_completion.choices[0].message.content.replace("</think>", "[end_of_thought]").replace("<think>", "[begin_of_thought]")
    return generated

async def process_sample(idx, sample, sample_num, prompt, output_filename, model_flag):
    if model_flag == "base":
        input_text = get_prompt_base(sample)
        generated = await perform_base_inference(input_text)
    else:
        conversation = get_prompt_instruct(sample)
        generated = await perform_instruct_inference(conversation)
    output_data = {
        "prompt": prompt,
        "generated_text": generated,
        "metadata": sample,
        "question_id": idx,
        "sample_index": sample_num,
        "generation_config": {
            "base_model": base_model,
            "instruct_model": instruct_model,
            "base_temperature": 0.8,
            "base_top_p": 0.95,
            "base_top_k": 0,
            "base_max_tokens": MAX_TOKENS,
            "instruct_temperature": 0.7,
            "instruct_top_p": 0.7,
            "instruct_top_k": 0,
            "instruct_max_tokens": MAX_TOKENS
        }
    }
    with open(output_filename, "w") as f:
        json.dump(output_data, f, indent=2)

async def limited_process_sample(semaphore, idx, sample, sample_num, prompt, output_filename, model_flag):
    async with semaphore:
        await process_sample(idx, sample, sample_num, prompt, output_filename, model_flag)

async def main(model_flag):
    semaphore = asyncio.Semaphore(200)
    tasks = []
    ds = load_dataset("BAAI/TACO", trust_remote_code=True)["train"].filter(lambda x: x["difficulty"] == "MEDIUM")
    for idx, sample in tqdm(enumerate(ds), desc="Processing samples"):
        if idx < 900:
            continue
        prompt = get_prompt_base(sample) if model_flag == "base" else get_prompt_instruct(sample)
        if not prompt:
            continue
        for sample_num in range(NUM_SAMPLES):
            output_filename = os.path.join(OUTPUT_DIR, f"question_{idx}_sample_{sample_num}.json")
            if os.path.exists(output_filename):
                logger.info(f"Output file {output_filename} exists, skipping sample {sample_num} for question_{idx}")
                continue
            tasks.append(asyncio.create_task(limited_process_sample(semaphore, idx, sample, sample_num, prompt, output_filename, model_flag)))
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Waiting for tasks"):
        await task

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "instruct"], required=True, help="Select model for inference (base or instruct)")
    args = parser.parse_args()
    MODEL_FLAG = args.model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    asyncio.run(main(MODEL_FLAG)) 
from transformers import AutoModelForCausalLM, AutoTokenizer
from dynamic_decoder import DynamicTemperatureDecoder
from config import DynamicDecodingConfig, DecayType, SimilarityType
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import json
import os
from prompts import REASONING_SYSTEM_PROMPT
import argparse
from prompt import generate_prompt, SKY_T1_SYSTEM_PROMPT, SKY_T1_FIXED
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def prompt(sample):
    """Parse test cases and starter code from problem to create a prompt for the LLM."""
    test_case = json.loads(sample["input_output"])
    starter_code = sample["starter_code"]
    # Generate prompt text using test case, question and starter code
    prompt_text = generate_prompt(test_case, sample["question"], starter_code)
    return [{"role": "system", "content": SKY_T1_FIXED}, {"role": "user", "content": prompt_text}]

# Load dataset
# dataset_name = "sky_1k_sample"
# ds = load_from_disk(f"data/{dataset_name}")
dataset_name = "taco_train"
ds = load_dataset("BAAI/TACO", trust_remote_code=True)["train"].filter(lambda x: x["difficulty"] == "MEDIUM")
print("Number of samples: ", len(ds))
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--constant_temp", action="store_true", help="Use constant temperature sampling at 0.7")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for dynamic sampling")
    args = parser.parse_args()
    
    if args.constant_temp:
        from openai import OpenAI
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="token-abc123",
        )

        OUTPUT_DIR = f"samples/{dataset_name}/{model_name.replace('/', '_')}_constant_temp"
    else:
        # Initialize model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
        # Create config
        config = DynamicDecodingConfig(
            similarity_threshold=0.98,
            verbose_generation=True,
            cache_embeddings=False,
        )

        # Initialize decoder
        decoder = DynamicTemperatureDecoder(
            model=model,
            tokenizer=tokenizer,
            config=config,
        )
        OUTPUT_DIR = f"samples/{dataset_name}/{model_name.replace('/', '_')}_4096"
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for idx, sample in enumerate(tqdm(ds, desc="Processing inputs")):
        
        # Create question directory
        question_dir = os.path.join(OUTPUT_DIR, f"question_{idx}")
        os.makedirs(question_dir, exist_ok=True)
        
        # Clear reference thoughts for new question
        if not args.constant_temp:
            decoder.reference_thoughts = []
            decoder.reference_embeddings = None
        
        if args.constant_temp:
            prompt_list = prompt(sample)
            
            completion = client.chat.completions.create(
                model=model_name,
                messages=prompt_list,
                temperature=args.temperature,
                top_p=0.7,
                top_k=50,
                max_tokens=12000,
                n=8,
            )
            # prompt_text = tokenizer.apply_chat_template(prompt_list, tokenize=False, add_generation_prompt=True)
            # encoded = tokenizer(prompt_text, return_tensors="pt")
            # input_ids = encoded["input_ids"].to(model.device)
            # attention_mask = encoded["attention_mask"].to(model.device)
            # outputs = model.generate(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     do_sample=True,
            #     temperature=args.temperature,
            #     max_new_tokens=12000,
            #     pad_token_id=tokenizer.eos_token_id,
            #     num_return_sequences=8,
            #     return_dict_in_generate=True
            # )
            # prompt_len = input_ids.shape[1]
            for i, seq in enumerate(completion.choices):
                sample_path = os.path.join(question_dir, f"sample_{i}.json")
                if os.path.exists(sample_path):
                    continue
                # generated_text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                generated_text = seq.message.content
                sample_data = {
                    "generation": {
                        "text": generated_text,
                        "temperature_history": [],
                        "similarity_history": [],
                        "reference_thoughts": []
                    },
                    "metadata": sample,
                    
                }
                with open(sample_path, "w") as f:
                    json.dump(sample_data, f, indent=2)
        else:
            question = prompt(sample)
            
            print("QUESTION: ", question)
            # Generate 8 samples
            for i in range(8):
                sample_path = os.path.join(question_dir, f"sample_{i}.json")
                if os.path.exists(sample_path):
                    with open(sample_path, "r") as f:
                        existing_sample = json.load(f)
                    decoder.add_reference_thoughts([existing_sample["generation"]["text"]])
                    continue
                result = decoder.generate(system_prompt=question[0]["content"], prompt=question[1]["content"], max_new_tokens=12000, max_length=12000)
                
                # Save sample with metadata
                sample_data = {
                    "generation": {
                        "text": result["generated_text"],
                        "temperature_history": result["temperature_history"],
                        "similarity_history": result["similarity_history"],
                        "reference_thoughts": decoder.reference_thoughts.copy(),
                    },
                    "metadata": sample,
                    
                }
                
                # Save to file
                with open(sample_path, "w") as f:
                    json.dump(sample_data, f, indent=2)
                
                # Add to reference thoughts for next generation
                decoder.add_reference_thoughts([result["generated_text"]])
            
    
    
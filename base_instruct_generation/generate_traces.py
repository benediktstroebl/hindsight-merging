from datasets import load_from_disk, load_dataset
import json 
import os 
from mm_generation_batch import MessiModels
import torch
import gc
from tqdm.auto import tqdm
import hashlib

from prompt import SKY_T1_FIXED

def get_question_id(question):
    """Generate a short, unique ID for a question"""
    return hashlib.md5(question.encode()).hexdigest()[:8]

def save_sample(output_dir, question, sample_data, sample_num):
    """Save a single sample to its question directory"""
    question_id = get_question_id(question)
    question_dir = os.path.join(output_dir, f"question_{question_id}")
    os.makedirs(question_dir, exist_ok=True)
    
    sample_file = os.path.join(question_dir, f"sample_{sample_num}.json")
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

def generate_traces(dataset, output_dir, system_prompt, batch_size=32, max_tokens_total=2000, max_base_tokens=5, max_it_tokens=30, temperature=0.7, repetitions=10):
    mm = MessiModels(temperature=temperature, system_prompt=system_prompt)
    
    # Track samples per question
    samples_count = {}
    
    # Create progress bars for repetitions and batches
    pbar_rep = tqdm(range(repetitions), desc="Repetitions", position=0)
    
    for rep in pbar_rep:
        n_batches = (len(dataset) + batch_size - 1) // batch_size  # Ceiling division
        pbar_batch = tqdm(range(0, len(dataset), batch_size), 
                         desc=f"Repetition {rep + 1}/{repetitions}", 
                         total=n_batches,
                         position=1, 
                         leave=False)
        
        for i in pbar_batch:
            print(i)
            batch = dataset.select(range(i, min(i+batch_size, len(dataset))))
            prompts = [example['question'] for example in batch]
            
            try:
                print('generating base')
                # base_story = mm.generate_from_base(prompts, max_tokens=max_tokens_total)
                print('generating it')
                # it_story = mm.generate_from_it(prompts, max_tokens=max_tokens_total)
                print('generating both')
                both_story = mm.generate_from_both(prompts, max_tokens_total=max_tokens_total, max_base_tokens=max_base_tokens, max_it_tokens=max_it_tokens)
                
                # Save each sample individually
                for j, question in enumerate(prompts):
                    # Initialize counter for new questions
                    if question not in samples_count:
                        samples_count[question] = 0
                    
                    sample_data = {
                        "question": question,
                        "base_story": "",# base_story[j],
                        "it_story": "",# it_story[j],
                        "both_story": both_story[j],
                        "sample_num": samples_count[question],
                        "repetition": rep,
                        "batch": i
                    }
                    print('saving sample')
                    save_sample(output_dir, question, sample_data, samples_count[question])
                    samples_count[question] += 1
                
                pbar_batch.set_postfix({"Batch Size": len(batch), "Questions Processed": len(samples_count)})
                
            except Exception as e:
                tqdm.write(f"Error processing batch: {e}")
                continue
            finally:
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        pbar_batch.close()

    # Save metadata about the run
    metadata = {
        "total_questions": len(samples_count),
        "samples_per_question": {get_question_id(q): count for q, count in samples_count.items()},
        "parameters": {
            "max_tokens_total": max_tokens_total,
            "max_base_tokens": max_base_tokens,
            "max_it_tokens": max_it_tokens,
            "temperature": temperature,
            "repetitions": repetitions,
            "batch_size": batch_size
        }
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

def main(output_dir: str = 'data/traces'):
    # max_tokens_total = 4096
    max_tokens_total = 8192
    max_base_tokens = 40
    max_it_tokens = 60  
    temperature = 0.7
    repetitions = 8
    name_of_setting = f"{max_tokens_total}_{max_base_tokens}_{max_it_tokens}_{temperature}_{repetitions}"
    os.makedirs(os.path.join(output_dir, name_of_setting), exist_ok=True)
    
    data = load_dataset("BAAI/TACO", trust_remote_code=True)["train"].filter(lambda x: x["difficulty"] == "MEDIUM")

    # data = load_from_disk("data/1k_taco_train/")

    generate_traces(data, 
                    os.path.join(output_dir, name_of_setting), 
                    batch_size=32, 
                    max_tokens_total=max_tokens_total, 
                    max_base_tokens=max_base_tokens, 
                    max_it_tokens=max_it_tokens, 
                    temperature=temperature,
                    repetitions=repetitions,
                    system_prompt=SKY_T1_FIXED)

if __name__ == "__main__":
    main()
import argparse
import json
import multiprocessing as mp
import os
import time
from itertools import cycle

import openai
from tqdm import tqdm

from dotenv import load_dotenv

from dynamic_scaling.prompt import convert_prompt, convert_prompt_example

load_dotenv()

def set_openai_key(api_key):
    openai.api_key = api_key


def process_content(content, api_key):
    set_openai_key(api_key)
    prompt = convert_prompt.format(example=convert_prompt_example, content=content)
    retries = 3
    while retries > 0:
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a solution format convertor.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=16384,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except openai.RateLimitError:
            retries -= 1
            if retries == 0:
                return "Error: Rate limit reached and retries exhausted."
            print("Sleep for 5 seconds for API limit.")
            time.sleep(5)
        except Exception as e:
            return f"Error processing content: {e}"


def process_file(file_path, api_key_cycle):
    if "converted_" in file_path:
        print(f"Skipping {file_path} because it is already converted")
        return file_path, file_path
    elif "converted_" + os.path.basename(file_path) in os.listdir(os.path.dirname(file_path)):
        print(f"Skipping {file_path} because it is already converted")
        return file_path, file_path
    else:
        print(f"Processing {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)
    correctness = data.get("correctness", None)
    print(f"Processing {file_path} with correctness {correctness}")
    # if correctness is None:
    #     print(f"Skipping {file_path} because it is not scored")
    #     return file_path, file_path
    # elif not correctness:
    #     print(f"Skipping {file_path} because it is not correct")
    #     return file_path, file_path
    content = data.get("generated_text", "").replace("[end_of_thought][begin_of_thought]", "[begin_of_thought]")
    api_key = next(api_key_cycle)
    processed = process_content(content, api_key)
    data["converted_text"] = processed
    output_file = os.path.join(os.path.dirname(file_path), f"converted_{os.path.basename(file_path)}")
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)
    return file_path, output_file


def process_file_wrapper(args):
    return process_file(*args)

def extract_question_num(file_path):
    question_num = file_path.split('question_')[1].split('_')[0]
    return question_num

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process sample files from switching_inference output and convert their format.")
    parser.add_argument("--input_dir", type=str, help="Input directory containing sample JSON files.")
    # parser.add_argument("--keys", type=str, help="File containing OpenAI API keys (one per line).")
    args = parser.parse_args()
    
    api_keys = [os.getenv("OPENAI_API_KEY")]

    api_key_cycle = cycle(api_keys)

    file_paths = [os.path.join(args.input_dir, filename) for filename in os.listdir(args.input_dir) if filename.endswith(".json")]

    # Filter file_paths based on question numbers < 500
    filtered_file_paths = []
    for file_path in file_paths:
        question_num = extract_question_num(file_path)
        if question_num is not None and int(question_num) < 1000:
            filtered_file_paths.append(file_path)
    
    file_paths = filtered_file_paths
    
    results = []
    with mp.Pool(os.cpu_count()) as pool:
        tasks = [(file_path, api_key_cycle) for file_path in file_paths]
        for result in tqdm(pool.imap(process_file_wrapper, tasks), total=len(file_paths)):
            results.append(result)
            
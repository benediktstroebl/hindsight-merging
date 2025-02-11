import argparse
import json
import os
import random
from tqdm import tqdm
import re
def main():
    parser = argparse.ArgumentParser(description="Convert JSON data for processing.")
    parser.add_argument(
        "--input_dir", type=str, help="Directory containing input JSON files."
    )
    parser.add_argument("--output", type=str, help="Output JSON file.")
    parser.add_argument("--max_samples", type=int, default=0, help="Maximum number of correct samples per question (0 for no limit)")
    args = parser.parse_args()

    data_by_question = {}
    for filename in tqdm(os.listdir(args.input_dir), desc="Processing files"):
        if filename.endswith(".json") and filename.startswith("converted_"):
            regex = re.compile(r"converted_question_(?P<question_id>\d+)_sample_(?P<sample_id>\d+)(?P<suffix>.*)\.json")
            match = regex.match(filename)
            question_id = int(match.group("question_id"))
            sample_id = int(match.group("sample_id"))
            suffix = match.group("suffix")
            
            if suffix == "_wrong_model":
                continue
        

            # skip questions with id 500 or higher
            
            if question_id >= 1000:
                print(f"Skipping {filename} because it has id {question_id}")
                continue

            # skip samples 10 or higher
            
            if sample_id >= 10:
                print(f"Skipping {filename} because it has sample {sample_id}")
                continue
            
            filepath = os.path.join(args.input_dir, filename)
            with open(filepath, "r") as f:
                sample = json.load(f)

            system_prompt = sample["prompt"][0]["content"]
            user_prompt = sample["prompt"][1]["content"]
            assistant_response = sample["converted_text"]

            # Accept this data
            if sample["correctness"]:
                # Create the conversation format
                conversations = [
                    {"from": "user", "value": user_prompt},
                    {"from": "assistant", "value": assistant_response},
                ]

                # Prepare the final structure
                cur_data = {
                    "system": system_prompt,
                    "conversations": conversations,
                }
                if question_id not in data_by_question:
                    data_by_question[question_id] = []
                data_by_question[question_id].append(cur_data)

    all_data = []
    # Randomly select up to k correct samples per question
    for samples in data_by_question.values():
        if args.max_samples > 0 and len(samples) > args.max_samples:
            selected = random.sample(samples, args.max_samples)
        else:
            selected = samples
        all_data.extend(selected)

    # Save the converted data to the output file
    with open(args.output, "w") as f:
        json.dump(all_data, f, indent=4)

    # Print number of unique questions based on unique user prompts
    unique_questions = len({data["conversations"][0]["value"] for data in all_data})
    print(f"Number of unique questions: {unique_questions}")

    print(
        f"Conversion completed. The data has been saved to {args.output} with {len(all_data)} data."
    )

if __name__ == "__main__":
    main()
import os
import json
import argparse

from skythought_evals.tasks.taco.taco_handler import TACOTaskHandler

def main():
    parser = argparse.ArgumentParser(
        description="Check correctness of converted samples using TacoTaskHandler."
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory containing raw and converted sample JSON files.",
    )
    args = parser.parse_args()
    handler = TACOTaskHandler()

    for filename in os.listdir(args.dir):
        if filename.endswith(".json") and filename.startswith("converted_"):
            filepath = os.path.join(args.dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)
            problem = data.get("metadata", {})
            response = data.get("converted_text", "")
            result = handler.update_results(problem, response)
            data["correctness"] = result["correctness"]
            data["correctness_reason"] = result["reason"]
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Processed {filename}: correctness={result['correctness']}")

if __name__ == "__main__":
    main() 
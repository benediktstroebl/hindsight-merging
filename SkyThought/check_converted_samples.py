import os
import json
import argparse
import asyncio

from skythought_evals.tasks.taco.taco_handler import TACOTaskHandler
from skythought_evals.tasks.base import TaskConfig

async def process_file(filepath, handler):
    def sync_process():
        with open(filepath, "r") as f:
            data = json.load(f)
        if "correctness" in data:
            print(f"Skipping {os.path.basename(filepath)} because it is already scored")
            return
        problem = data.get("metadata", {})
        response = data.get("converted_text", "")
        result = handler.update_results(problem, response)
        data["correctness"] = result["correctness"]
        data["correctness_reason"] = result["reason"]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Processed {os.path.basename(filepath)}: correctness={result['correctness']}")
    await asyncio.to_thread(sync_process)

async def async_main(args):
    task_config = TaskConfig.from_yaml(args.task_config)
    handler = TACOTaskHandler(task_config)
    tasks = []
    for filename in os.listdir(args.dir):
        if filename.endswith(".json") and filename.startswith("converted_"):
            filepath = os.path.join(args.dir, filename)
            tasks.append(process_file(filepath, handler))
    if tasks:
        await asyncio.gather(*tasks)

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
    parser.add_argument(
        "--task_config",
        type=str,
        help="Path to the task configuration file.",
        default="skythought/skythought_evals/tasks/taco/taco.yaml"
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))

if __name__ == "__main__":
    main() 
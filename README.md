## Hindsight Merging: Diverse Data Generation with Language Models

This repository contains the code for the paper "Hindsight Merging: Diverse Data Generation with Language Models". It provides the code for the experiments in the paper.
### Code File Breakdown:

*   **`check_converted_samples.py`**:  Checks the correctness of converted samples.  It uses the `TACOTaskHandler` to evaluate the correctness of converted responses and adds this information to the JSON files.  Used after `format_converter.py`.
*   **`convert_to_data.py`**: Converts the JSON files (generated by switching inference scripts) to a format suitable for fine-tuning, such as that used by Llama-Factory, and randomly selects the most correct samples to a limit to generate a high-quality dataset.
*   **`format_converter.py`**: Converts the raw JSON outputs to a uniform format by prompting a reformatting LLM to take the raw text and place it in the format <begin_of_slow_thought> *text* <end_of_slow_thought> <begin_of_solution> *text* <end_of_solution>
*   **`pass_at_k_plot.py`**: Calculates and plots the Pass@k metric for code generation tasks, demonstrating the performance of different models and merging strategies.
*   **`single_model_inference.py`**: Uses a *single* model for inference (base or instruct). Generates reasoning traces and stores them to JSON files.
*   **`analysis/` directory:**
    *   **`correctness.ipynb`**: Probably for calculating correctness
    *   **`embed.py`**: Python script for embedding `generated_text` and `converted_text` entries in JSON files generated by one of the inference scripts (`single_model_inference.py` or `switching_inference_*.py`). It uses the OpenAI API.
    *   **`explore_embeddings.ipynb`**: A Jupyter Notebook for exploring the generated embeddings, likely for visualization or analysis.
    *   **`finetuning_analysis.ipynb`**: Jupyter Notebook for performing finetuning analysis.
    *   **`model_exploration.ipynb`**: Jupyter Notebook for exploration model outputs.
    

### Reproduction Steps (General Outline):

1.  **Data Generation (`single_model_inference.py`):**
    *   Set environment variables such as `OPENAI_API_KEY` (if using OpenAI) and model paths.
    *   Adjust generation parameters like `MAX_TOKENS`, `NUM_SAMPLES`, `K`, and `P`.
    *   Modify and run `single_model_inference.py` to generate synthetic data.

2.  **Format Conversion (`format_converter.py`):**
    *   After running `single_model_inference.py` to generate the data, process those `converted_*.json` files using `format_converter.py` to reformat the output.
    *   Set `OPENAI_API_KEY`.

3.  **Correctness and Data Curation (`SkyThought/check_converted_samples.py` and `convert_to_data.py`):**
    *   Use `SkyThought/check_converted_samples.py` to calculate correctness
    *   Then use the converted format and the correctness information and create the final dataset via  `convert_to_data.py`. This includes setting `max_samples` to the dataset size.

4.  **Finetuning:**

    *   The exact steps for finetuning are external to this repository. We use Llama-Factory for finetuning. Code is provided in `SkyThought/`

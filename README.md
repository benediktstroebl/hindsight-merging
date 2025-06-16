## Hindsight Merging: Diverse Data Generation with Language Models

This repository contains the code for the paper "Hindsight Merging: Diverse Data Generation with Language Models". It provides the code for the experiments in the paper.

### Reproduction Instructions:

0. **Model Merging:**
    *   Follow the instructions in [mergekit](https://github.com/arcee-ai/mergekit) to merge models with LERP.

1.  **Data Generation:**
    *   Follow the instructions in [evalplus](https://github.com/evalplus/evalplus).
    *   We include some example commands we used in `example_commands.sh`

2. **SorryBench:**
    * We include a copy of the [SorryBench](https://sorry-bench.github.io/) repository in the sorrybench folder. 
    * You have to request the questions.jsonl file from the original sorry-bench creators and place it in `sorry-bench/data/sorryibench`. 
    * Run `run_all_models.py` to calculate the performance on sorry-bench.
    * Visualization is done with `visualize_results.ipynb`.   

3.  **Analysis (`analysis/`):**
    *   For plotting the pass@k plots, unzip `mbpp_data.zip` and use the `pass_at_k.py`
    *   For calculating embeddings and reproducing the diversity distribution plot, use `embed.py` and `explore_embeddings.ipynb`
 
    

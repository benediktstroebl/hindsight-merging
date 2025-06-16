## Hindsight Merging: Diverse Data Generation with Language Models

This repository contains the code for the paper "Hindsight Merging: Diverse Data Generation with Language Models". It provides the code for the experiments in the paper.

### Reproduction Instructions:

0. **Model Merging:**
    *   Follow the instructions in [MergeKit](https://github.com/arcee-ai/mergekit) to merge models with SLERP.

1.  **Data Generation:**
    *   Follow the instructions in [evalplus](https://github.com/evalplus/evalplus).
    *   We include some example commands we used in `example_commands.sh`

2.  **Analysis (`analysis/`):**
    *   For plotting the pass@k plots, unzip `mbpp_data.zip` and use the `pass_at_k.py`
    *   For calculating embeddings and reproducing the diversity distribution plot, use `embed.py` and `explore_embeddings.ipynb`
    
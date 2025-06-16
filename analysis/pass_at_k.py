#!/usr/bin/env python3
"""
Analyze eval results and create pass@k plots for different model families.
"""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re

def parse_model_name(filename):
    """Extract model family, type, and lerp info from filename."""
    # Extract the filename without path and extension
    basename = os.path.basename(filename).replace('.eval_results.json', '')
    
    # Identify model families
    if 'vicuna-7b' in basename:
        family = 'Vicuna-7B v1.5'
    elif 'vicuna-13b' in basename:
        family = 'Vicuna-13B v1.5'  
    elif 'Llama-3.1-8B' in basename:
        family = 'Llama-3.1-8B'
    elif 'Llama-3.1-70B' in basename:
        family = 'Llama-3.1-70B'
    else:
        return None, None, None
    
    # Identify model type
    if 'Instruct' in basename:
        model_type = 'Instruct'
    elif 'lerp_0.7' in basename:
        model_type = 'α = 0.7'
    elif 'lerp_0.9' in basename:
        model_type = 'α = 0.9'
    elif 'lmsys--vicuna' in basename or 'meta-llama--Llama-3.1-8B_' in basename or 'meta-llama--Llama-3.1-70B_' in basename:
        model_type = 'Pretrained'
    else:
        model_type = 'Unknown'
    
    return family, model_type, basename

def estimate_pass_at_k(num_samples, num_correct, k):
    """
    Estimates pass@k of each problem and returns them in an array.
    """
    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        import itertools
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )

def calculate_pass_at_k(results, k_values):
    """Calculate unbiased pass@k for given k values using the proper estimator."""
    pass_at_k = {}
    
    # Group results by task_id and count passes based on plus_status
    task_results = defaultdict(list)
    for task_id, solutions in results.items():
        for solution in solutions:
            # Count as pass if plus_status is pass (as specified by user)
            passed = (solution.get('plus_status') == 'pass')
            task_results[task_id].append(passed)
    
    # Prepare data for pass@k calculation
    num_samples = []
    num_correct = []
    
    for task_id, passes in task_results.items():
        n = len(passes)  # total number of solutions for this task
        c = sum(passes)  # number of correct solutions
        if n > 0:
            num_samples.append(n)
            num_correct.append(c)
    
    # Calculate pass@k for each k
    for k in k_values:
        if len(num_correct) > 0:
            pass_at_k_scores = estimate_pass_at_k(num_samples, num_correct, k)
            pass_at_k[k] = np.mean(pass_at_k_scores)
        else:
            pass_at_k[k] = 0.0
    
    return pass_at_k

def load_results():
    """Load all eval results files."""
    results_dir = "../evalplus_results/mbpp"
    
    # Hardcode the specific files to include
    file_configs = [
        # Vicuna 7B models
        ("meta-llama--Llama-2-7b-hf_vllm_temp_0.7.eval_results.json", "Vicuna-7B v1.5", "Pretrained"),
        ("home--benediktstroebl--mergekit--vicuna_7b_lerp_0.7_vllm_temp_0.7.eval_results.json", "Vicuna-7B v1.5", "α = 0.7"),
        ("home--benediktstroebl--mergekit--vicuna_7b_lerp_0.9_vllm_temp_0.7.eval_results.json", "Vicuna-7B v1.5", "α = 0.9"),
        ("lmsys--vicuna-7b-v1.5_vllm_temp_0.7.eval_results.json", "Vicuna-7B v1.5", "Instruct"),
        
        # Vicuna 13B models
        ("meta-llama--Llama-2-13b-hf_vllm_temp_0.7.eval_results.json", "Vicuna-13B v1.5", "Pretrained"),
        ("home--benediktstroebl--mergekit--vicuna_13b_lerp_0.7_vllm_temp_0.7.eval_results.json", "Vicuna-13B v1.5", "α = 0.7"),
        ("home--benediktstroebl--mergekit--vicuna_13b_lerp_0.9_vllm_temp_0.7.eval_results.json", "Vicuna-13B v1.5", "α = 0.9"),
        ("lmsys--vicuna-13b-v1.5_vllm_temp_0.7.eval_results.json", "Vicuna-13B v1.5", "Instruct"),
        
        # Llama 3.1 8B models  
        ("meta-llama--Llama-3.1-8B_vllm_temp_0.7.eval_results.json", "Llama-3.1-8B", "Pretrained"),
        ("home--benediktstroebl--mergekit--ll31_8b_lerp_0.7_vllm_temp_0.7.eval_results.json", "Llama-3.1-8B", "α = 0.7"),
        ("home--benediktstroebl--mergekit--ll31_8b_lerp_0.9_vllm_temp_0.7.eval_results.json", "Llama-3.1-8B", "α = 0.9"),
        ("meta-llama--Llama-3.1-8B-Instruct_vllm_temp_0.7.eval_results.json", "Llama-3.1-8B", "Instruct"),
        
        # Llama 3.1 70B models
        ("meta-llama--Llama-3.1-70B_vllm_temp_0.7.eval_results.json", "Llama-3.1-70B", "Pretrained"),
        ("home--benediktstroebl--mergekit--ll31_70b_lerp_0.7_vllm_temp_0.7.eval_results.json", "Llama-3.1-70B", "α = 0.7"),
        ("home--benediktstroebl--mergekit--ll31_70b_lerp_0.9_vllm_temp_0.7.eval_results.json", "Llama-3.1-70B", "α = 0.9"),
        ("meta-llama--Llama-3.1-70B-Instruct_vllm_temp_0.7.eval_results.json", "Llama-3.1-70B", "Instruct"),
    ]
    
    model_results = defaultdict(lambda: defaultdict(dict))
    
    for filename, family, model_type in file_configs:
        file_path = os.path.join(results_dir, filename)
        if not os.path.exists(file_path):
            print(f"File not found: {filename}")
            continue
            
        print(f"Loading {filename}...")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                results = data.get('eval', {})
                
                # Calculate pass@k for k = 1 to 10
                k_values = list(range(1, 11))
                pass_at_k = calculate_pass_at_k(results, k_values)
                
                model_results[family][model_type] = {
                    'pass_at_k': pass_at_k,
                    'filename': filename
                }
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return model_results

def create_plots(model_results):
    """Create 2x2 subplot with pass@k plots for each model family."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # fig.suptitle('Pass@k Performance on MBPP', fontsize=16)
    
    # Define colors and line styles for different model types
    colors = {
        'Pretrained': 'blue',
        'Instruct': 'red', 
        'α = 0.7': 'green',
        'α = 0.9': 'orange'
    }
    
    line_styles = {
        'Pretrained': '-',
        'Instruct': '--',
        'α = 0.7': '-.',
        'α = 0.9': ':'
    }
    
    # Families to plot (4 families)
    families = ['Vicuna-7B v1.5', 'Vicuna-13B v1.5', 'Llama-3.1-8B', 'Llama-3.1-70B']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    k_values = list(range(1, 11))
    
    for i, family in enumerate(families):
        if family not in model_results:
            print(f"Family {family} not found in model_results")
            continue
            
        print(f"Plotting {family} at position {positions[i]}")
        row, col = positions[i]
        ax = axes[row, col]
        
        for model_type, data in model_results[family].items():
            if 'pass_at_k' not in data:
                continue
                
            pass_at_k = data['pass_at_k']
            k_vals = []
            pass_vals = []
            
            for k in k_values:
                if k in pass_at_k:
                    k_vals.append(k)
                    pass_vals.append(pass_at_k[k])
            
            if k_vals:
                ax.plot(k_vals, pass_vals, 
                       color=colors.get(model_type, 'black'),
                       linestyle=line_styles.get(model_type, '-'),
                       marker='o', 
                       label=model_type,
                       linewidth=2)
        
        ax.set_title(f'{family}', fontweight='bold', fontsize=16)
        ax.set_xlabel('k', fontsize=16)
        ax.set_ylabel('Pass@k', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(np.arange(0, 11, 1))
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    # All subplots are now used
    
    # Create a single legend above the plots
    handles, labels = [], []
    for family in families:
        if family in model_results:
            for model_type in model_results[family]:
                if model_type not in labels:
                    handles.append(plt.Line2D([0], [0], 
                                            color=colors.get(model_type, 'black'),
                                            linestyle=line_styles.get(model_type, '-'),
                                            marker='o', linewidth=2))
                    labels.append(model_type)
    
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=4, frameon=False, fontsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig('pass_at_k_plots.pdf', 
                bbox_inches='tight')
    plt.show()

def main():
    print("Loading eval results...")
    model_results = load_results()
    
    print("\nModel families found:")
    for family in model_results:
        print(f"  {family}:")
        for model_type in model_results[family]:
            print(f"    - {model_type}")
    
    print("\nCreating plots...")
    create_plots(model_results)
    print("Plots saved to pass_at_k_plots.png")

if __name__ == "__main__":
    main()
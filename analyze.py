import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

# Define directory containing generated stories
output_dir = "./generated_stories"

# Initialize data structures
alphas = []
diversity_scores_list = []

# Loop through each alpha directory and calculate diversity scores
for alpha_dir in sorted(os.listdir(output_dir)):
    if alpha_dir.startswith("alpha_"):
        # Extract the alpha value from the directory name
        alpha_value = float(alpha_dir.split("_")[1])
        alphas.append(alpha_value)

        # Initialize word set and word counts
        total_words_list = []
        unique_word_ratios = []

        # Read all stories for this alpha and calculate diversity
        alpha_path = os.path.join(output_dir, alpha_dir)
        for story_file in os.listdir(alpha_path):
            with open(os.path.join(alpha_path, story_file), "r") as f:
                story = f.read()
                words = story.split()
                if len(words) > 0:
                    total_words_list.append(len(words))
                    unique_word_ratios.append(len(set(words)) / len(words))

        # Calculate average diversity score for this alpha
        if unique_word_ratios:
            diversity_scores_list.append(unique_word_ratios)
        else:
            diversity_scores_list.append([0])

# Calculate means and confidence intervals
mean_scores = [np.mean(scores) for scores in diversity_scores_list]
confidence_intervals = [1.96 * sem(scores) for scores in diversity_scores_list]

# Plot the results with confidence intervals
plt.figure(figsize=(10, 6))
plt.errorbar(alphas, mean_scores, yerr=confidence_intervals, fmt='o-', capsize=5, label='Diversity Score')
plt.title("Diversity Score vs. Alpha with Confidence Intervals")
plt.xlabel("Alpha")
plt.ylabel("Diversity Score (Unique Words / Total Words)")
plt.grid(True)
plt.legend()
plt.show()

import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from datasets import load_dataset

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths
base_name = "meta-llama/Llama-3.2-1B"
instruct_name = "meta-llama/Llama-3.2-1B-Instruct"

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained(base_name)
base_model = AutoModelForCausalLM.from_pretrained(base_name).to(device)
instruct_model = AutoModelForCausalLM.from_pretrained(instruct_name).to(device)

text = """
Artificial neural networks have been used to explain how aspects of human knowledge that were once thought to be unlearnable -- such as elements of language -- might be learned from data \cite<e.g.,>[]{rumelhartm86}. This approach offers a new perspective on central questions in cognitive science, such as what information we need to assume is ``innate'' to human learners \cite{elman1996rethinking}. In machine learning, a parallel set of questions focuses on the {\em inductive biases} of neural networks -- defined as those factors other than the data that influence the solutions that they find \cite{mitchell97}. The convergence of these literatures offers an opportunity to explore different ways in which innate knowledge might be implicitly expressed in artificial neural networks. 

Different neural network architectures display different inductive biases. For instance, one clear signature of inductive bias is the amount of data needed to learn a task, and convolutional neural networks can learn image classification tasks from less data than multi-layer perceptrons \cite{chen2021review}. In addition to network architecture, however, recent work has highlighted the importance of a network’s initial weights as a source of inductive bias \cite{finn2017model}. Specifically, techniques based on meta-learning can optimize the initial weights of a neural network (leaving the architecture unchanged) in ways that enable the network to learn new tasks from far less data than it would require using standard, randomly-selected initial weights. For instance, a network with meta-learned initial weights can learn new linguistic rules from just 100 examples, compared to the roughly 20,000 examples needed by the same architecture with non-meta-learned initial weights \cite{mccoy2020universal}. Such meta-learning results show that a given neural network architecture can realize very different inductive biases thanks to the flexibility afforded by the initial weights.

Here we consider this flexibility from the opposite direction: can a given inductive bias be realized equally well in very different network architectures? This question directly engages with the issue of whether architecture or initial weights provide a better focus for understanding the innate constraints on learning implicitly instantiated in a neural network. Prior work using meta-learning typically makes comparisons within a fixed architecture, comparing a version of that architecture with meta-learned initial weights to a version with randomly-selected initial weights. These comparisons make it clear that the initial weights afford a substantial degree of flexibility, but they leave open the question of whether that flexibility is extensive enough to override the influence of architecture such that a given inductive bias could be realized equally well in different architectures.

To address this, we explore several inductive biases, investigating how compatible each inductive bias is with different types of network architectures and data representations. We consider four widely-used, general-purpose neural architectures—multilayer perceptrons (MLPs; \citeNP{rosenblatt1958perceptron}), convolutional neural networks (CNNs; \citeNP{lecun1998gradient}), long short-term memory networks (LSTMs; \citeNP{hochreiter1997long}), and Transformers \cite{vaswani2017attention}—with variations in depth and width, meta-training a total of 430 models. To establish baselines where differences across architectures and data representations should be more pronounced—free from task-specific biases introduced by meta-learning—we compare these meta-trained models to the same architectures trained under typical regimes, starting from random initialization and optimizing along that trajectory. This design enables us to isolate how much of the performance variation can be attributed to architectural and data representation choices, as opposed to the learning processes that are agnostic to those choices.

Across both data representation and architecture, we observe substantial performance differences when models are trained using the usual random initial weights. However, introducing meta-learned inductive biases reduces, and in some cases completely eliminates, these differences, demonstrating that a given inductive bias can be instantiated in multiple, disparate architectures. Interestingly, architectures and data representations that perform well under random initialization also tend to meta-train more effectively, suggesting that some residual biases remain important for certain tasks. In few-shot learning, for example, models that excel without meta-learning are less sensitive to shifts in the training task distribution. Despite this, when models are required to generalize far outside their training distribution, all architectures—regardless of inductive bias—fail catastrophically. This highlights that these general-purpose architectures may require stronger inductive biases for more robust forms of generalization but remain general enough to realize a wide range of biases via the training algorithm.
"""

# Function to compute token-level entropy for a given text sample
def compute_token_entropy(model, tokenizer, text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        # Obtain the logits from the model
        outputs = model(input_ids)
        logits = outputs.logits  # shape: [batch, seq_len, vocab_size]

    # Compute log probabilities and probabilities over the vocabulary
    log_probs = F.log_softmax(logits, dim=-1)  # shape: [batch, seq_len, vocab_size]
    probs = torch.exp(log_probs)               # shape: [batch, seq_len, vocab_size]

    print(probs.shape)

    # Compute entropy for each token position:
    # For each token, entropy = -sum(p * log p) over the vocabulary dimension.
    token_entropies = -(probs * log_probs).sum(dim=-1)  # shape: [batch, seq_len]

    # Return the average token entropy for this text sample
    return token_entropies.squeeze()

# Calculate entropy for both models
base_entropies = []
instruct_entropies = []


base_entropy = compute_token_entropy(base_model, tokenizer, text)
instruct_entropy = compute_token_entropy(instruct_model, tokenizer, text)

print(base_entropy.mean())
print(instruct_entropy.mean())

# Print the results
#print(f"Base model entropy: {base_entropy:.4f}")
#print(f"Instruct model entropy: {instruct_entropy:.4f}")
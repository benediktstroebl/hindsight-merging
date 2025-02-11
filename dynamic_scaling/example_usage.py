from transformers import AutoModelForCausalLM, AutoTokenizer
from dynamic_decoder import DynamicTemperatureDecoder
from config import DynamicDecodingConfig, DecayType
from datasets import load_dataset, load_from_disk
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import json
from prompts import REASONING_SYSTEM_PROMPT
load_dotenv(override=True)

def plot_temperature_history(temperature_history):
    plt.figure(figsize=(10, 5))
    plt.plot(temperature_history)
    plt.title('Temperature Evolution During Generation')
    plt.xlabel('Generation Step')
    plt.ylabel('Temperature')
    plt.grid(True)
    plt.show()

# Load dataset and get reference thoughts
print("Loading dataset")
ds = load_from_disk("data/1k_taco_train")


# Initialize model and tokenizer
print("Loading model and tokenizer")
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # or your preferred model
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

# Create config
config = DynamicDecodingConfig(
    decay_type=DecayType.EXPONENTIAL,
    similarity_threshold=0.8,
    top_k=5
)

# Initialize decoder
decoder = DynamicTemperatureDecoder(
    model=model,
    tokenizer=tokenizer,
    config=config,
)


# Add new reference thoughts
print("\nAdding reference thoughts")
new_thoughts = [sample["solutions"] for sample in ds][1:3]
decoder.add_reference_thoughts(new_thoughts)

# Generate with updated reference thoughts
print("\nGenerating with updated reference thoughts:")
prompt = ds[0]["question"]

print("input prompt: ", prompt)
result = decoder.generate(system_prompt=REASONING_SYSTEM_PROMPT, prompt=prompt, max_length=4096, max_new_tokens=16384)
print(result["generated_text"])

print("\nTemperature History:", result["temperature_history"])
plot_temperature_history(result["temperature_history"]) 
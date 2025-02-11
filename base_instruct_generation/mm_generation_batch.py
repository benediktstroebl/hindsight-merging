from copy import deepcopy
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import List, Union, Dict
import torch 

import warnings
warnings.filterwarnings("ignore")


from model_utils import model_paths

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def continue_generation_prompt(prompt: str):
    # replace the final occurance of <|im_end|>\n<|im_start|>assistant\n 
    # Find the last occurrence and only replace that one
    # last_idx = prompt.rindex("<|im_end|>\n<|im_start|>assistant\n")
    last_idx = prompt.rindex("<｜end▁of▁sentence｜><｜Assistant｜>")
    text = prompt[:last_idx] + prompt[last_idx:].replace("<｜end▁of▁sentence｜><｜Assistant｜>", "", 1)
    return text

@dataclass
class MessiModels:
    base_model_path: str = field(default=model_paths["Qwen/Qwen2-1.5B"])
    it_model_path: str = field(default=model_paths["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"])
    # it_model_path: str = field(default=model_paths["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"])
    device: str = "cuda"
    temperature: float = 0.4

    system_prompt: str = field(default="Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <begin_of_thought> {thought with steps separated with ' '} <end_of_thought> Each step should include detailed considerations such as analyzing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|response|> {final formatted, precise, and clear solution} </response> Now, try to solve the following question through the above guidelines:")
    # system_prompt: str = field(default="You are a helpful assistant.")

    # TODO make sure to change this depending on the dataset
    base_suffix: str = field(default="Solution:")

    verbose: bool = field(default=False)

    it_history: List[str] = field(default_factory=list)
    base_history: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_model = self.load_model(self.base_model_path)
        self.it_model = self.load_model(self.it_model_path)
        self.base_tokenizer = self.load_tokenizer(self.base_model_path)
        self.it_tokenizer = self.load_tokenizer(self.it_model_path)
        # self.logger.info("Models and tokenizers loaded successfully")

    
    def generate_from_base(self, prompt: Union[str, List[str]], max_tokens: int = 1, generate_solo: bool = True):
        """
        Generate tokens from the base model. The prompt may be a single string or a list of strings.
        generate_solo: if True, we only generate base, if false we generate both base and it
        Returns a string for a single prompt or a list of strings for batched prompts.
        """
        self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        # If a single string was passed, wrap it in a list.
        single_prompt = False
        if isinstance(prompt, str):
            prompt = [prompt]
            single_prompt = True

        if generate_solo:
            prompt2 = deepcopy([self.system_prompt + ' ' + p + ' ' + self.base_suffix for p in prompt])
        else:
            prompt2 = deepcopy(prompt)

        self.logger.debug(f"Generating from base model for batch size {len(prompt)} with max tokens: {max_tokens}")
        
        # print("\033[94mprompt2:\033[0m", prompt2)  # Print in blue color
        # Tokenize with padding
        base_inputs = self.base_tokenizer(prompt2, return_tensors="pt", padding=True) #.to(self.device)
        base_input_ids = base_inputs.input_ids
        len_base_output = base_input_ids.shape[1]

        # Generate new tokens
        base_output = self.base_model.generate(
            base_input_ids,
            attention_mask=base_inputs.attention_mask,
            max_new_tokens=max_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.base_tokenizer.eos_token_id
        )

        # Determine where actual content starts (for left padding)
        if "attention_mask" in base_inputs:
            input_lengths = base_inputs.attention_mask.sum(dim=1)
        else:
            input_lengths = (base_input_ids != self.base_tokenizer.pad_token_id).sum(dim=1)
        # print("input lengths", input_lengths)
        # Extract generated text
        results = []
        for i, end_idx in enumerate(input_lengths):
            # Get the full output sequence
            full_output = base_output[i]
            
            # Only get tokens generated after the prompt
            out_tokens_tensor = full_output[len_base_output:]
            out_string = self.base_tokenizer.decode(out_tokens_tensor, skip_special_tokens=True)
            # print("\033[94mout_string:\033[0m", out_string)  # Print in blue color
            generated_token_count = out_tokens_tensor.size(0)
            out_tokens = "[BASE]" * generated_token_count
            # if self.verbose:
            #     print(out_string)
            results.append((out_string, generated_token_count, out_tokens))

        return results[0] if single_prompt else results
    

    def generate_from_it(self, prompt: Union[str, List[str]], chat_templates: List[List[Dict[str, str]]]=None, max_tokens: int = 10, generate_solo: bool = True):
        """
        Generate tokens from the instruct (IT) model. The prompt may be a single string or a list.
        generate_solo: if True, we only generate it, if false we generate both base and it
        It applies a chat template to each prompt and then decodes the newly generated tokens.
        """
        self.it_tokenizer.pad_token = self.it_tokenizer.eos_token
        single_prompt = False
        if isinstance(prompt, str):
            prompt = [prompt]
            single_prompt = True

        if generate_solo:
            chat_templates = [[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": p}, {"role": "assistant", "content": ""}] for p in prompt]
            chat_templates = deepcopy(chat_templates)
        else:
            chat_templates = deepcopy(chat_templates)
            for i, p in enumerate(prompt):
                chat_templates[i].append({"role": "assistant", "content": p})

        self.logger.debug(f"Generating from IT model for batch size {len(prompt)} with max tokens: {max_tokens}")
        # First get all tokenized prompts to find max length
        tokenized_prompts = []
        # apply the chat template 
        for p in chat_templates:
            formatted_prompt = self.it_tokenizer.apply_chat_template(
                p,
                tokenize=False,
                add_generation_prompt=True,
                continue_final_message=False,
            )
            formatted_prompt = continue_generation_prompt(formatted_prompt)
            tokenized_prompts.append(formatted_prompt)
        # print("\033[93mtokenized_prompts:\033[0m", tokenized_prompts)  # Print in yellow color
        tokens = self.it_tokenizer(tokenized_prompts, return_tensors="pt", padding=True) #.to(self.device)
        
        # Generate new tokens.
        it_output = self.it_model.generate(
            tokens.input_ids,
            max_new_tokens=max_tokens,
            attention_mask=tokens.attention_mask,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.it_tokenizer.eos_token_id
        )

        # Determine where actual content starts (for left padding)
        if "attention_mask" in tokens:
            input_lengths = tokens.attention_mask.sum(dim=1)
        else:
            input_lengths = (tokens != self.base_tokenizer.pad_token_id).sum(dim=1)
        
        # Extract generated text
        results = []
        for i, end_idx in enumerate(input_lengths):
            out_tokens_tensor = it_output[i, end_idx:]  # Generated tokens
            out_string = self.it_tokenizer.decode(out_tokens_tensor, skip_special_tokens=True)
            # print in yellow 
            # print("\033[93mout_string:\033[0m", out_string)  # Print in yellow color
            generated_token_count = out_tokens_tensor.size(0)
            out_tokens = "[INSTRUCT]" * generated_token_count

            if self.verbose:
                print(out_string)
            results.append((out_string, generated_token_count, out_tokens))

        return results[0] if single_prompt else results

    def generate_from_both(
        self,
        prompt: Union[str, List[str]],
        max_tokens_total: int = 100,
        max_base_tokens: int = 10,
        max_it_tokens: int = 10
    ):
        """
        Alternates generation between the IT and base models until a total of max_tokens_total
        (per prompt) have been generated. Uses batched inference for efficiency.
        """
        # Ensure prompt is a list.
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)
        prompts = prompt.copy()  # Create a copy to avoid modifying the input
        base_prompts = [self.system_prompt + ' ' + p + ' ' + self.base_suffix + ' 'for p in prompt]
        chat_templates = [[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": p}] for p in prompt]
        prompts = ["" for _ in prompt]

        LLM_source = [""] * batch_size
        generated_tokens = [0] * batch_size

        # self.logger.info(
        #     f"Starting alternating batched generation for batch of size {batch_size} with max_tokens_total={max_tokens_total}"
        # )

        while any(gt < max_tokens_total for gt in generated_tokens):
            # --- IT model generation step ---
            indices = [i for i, gt in enumerate(generated_tokens) if gt < max_tokens_total]
            if indices:  # Only proceed if there are prompts that need more tokens
                current_prompts = deepcopy([prompts[i] for i in indices])
                current_templates = deepcopy([chat_templates[i] for i in indices])
                # Calculate max tokens for each prompt individually
                max_tokens = [min(max_it_tokens, max_tokens_total - generated_tokens[i]) for i in indices]
                self.logger.debug(f"IT generation step for indices {indices} with max tokens: {max_tokens}")
                
                # Batch process all prompts at once
                it_results = self.generate_from_it(current_prompts, 
                                                   current_templates, max_tokens=max(max_tokens), generate_solo=False)
                # Update results for each prompt
                for batch_idx, idx in enumerate(indices):
                    out_string, token_count, out_tokens = it_results[batch_idx]
                    prompts[idx] += out_string
                    base_prompts[idx] += out_string
                    LLM_source[idx] += out_tokens
                    generated_tokens[idx] += token_count

            if all(gt >= max_tokens_total for gt in generated_tokens):
                break

            # --- Base model generation step ---
            indices = [i for i, gt in enumerate(generated_tokens) if gt < max_tokens_total]
            if indices:  # Only proceed if there are prompts that need more tokens
                current_prompts = deepcopy([base_prompts[i] for i in indices])
                # Calculate max tokens for each prompt individually
                max_tokens = [min(max_base_tokens, max_tokens_total - generated_tokens[i]) for i in indices]
                self.logger.debug(f"Base generation step for indices {indices} with max tokens: {max_tokens}")
                # Batch process all prompts at once
                base_results = self.generate_from_base(current_prompts, max_tokens=max(max_tokens), generate_solo=False)
                # Update results for each prompt
                for batch_idx, idx in enumerate(indices):
                    out_string, token_count, out_tokens = base_results[batch_idx]
                    prompts[idx] += out_string
                    base_prompts[idx] += out_string
                    LLM_source[idx] += out_tokens
                    generated_tokens[idx] += token_count

        # self.logger.info(f"Completed alternating generation for batch. Generated tokens per prompt: {generated_tokens}")
        # Return a single tuple if only one prompt was processed; otherwise return lists.
        return (prompts, LLM_source) if batch_size == 1 else (prompts, LLM_source)

    def load_model(self, model_path: str):
        self.logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.float16, trust_remote_code=True)
        self.logger.info(f"Model loaded successfully to {self.device}")
        return model

    def load_tokenizer(self, model_path: str):
        self.logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, device_map='auto')
        # Set the pad token to be the EOS token if not already set.
        tokenizer.pad_token = tokenizer.eos_token
        # Change padding side to left for decoder-only models
        tokenizer.padding_side = "left"  # Changed from "right" to "left"
        self.logger.info("Tokenizer loaded successfully")
        return tokenizer
    

if __name__ == "__main__":
    mmg = MessiModels()
    # prompt = """You have a deck of $n$ cards, and you'd like to reorder it to a new one.\n\nEach card has a value between $1$ and $n$ equal to $p_i$."""
    # # Single-prompt examples
    # both_story = mmg.generate_from_both("Write me a hundred word story about a dog.")
    # base_story = mmg.generate_from_base("Write me a hundred word story about a dog.", max_tokens=400)
    # it_story = mmg.generate_from_it("Write me a hundred word story about a dog.", max_tokens=400)
    # print("Single prompt results:")
    # print("Both:", both_story)
    # print("Base:", base_story)
    # print("IT:", it_story)
    
    # Batched examples with multiple prompts
    prompts = [
        "Write me a hundred word story about a cat.",
        # "Write me a hundred word story about a dog."
    ]
    both_stories = mmg.generate_from_both(prompts, max_tokens_total=1000)
    # base_stories = mmg.generate_from_base(prompts, max_tokens=100)
    # it_stories = mmg.generate_from_it(prompts, max_tokens=100)
    # print("\nBatched results:")
    print("Both:", both_stories)
    # print("Base:", base_stories)
    # print("IT:", it_stories)
# class to generate from two models 
# load in the base and it model
# define some different generation templates
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_utils import model_paths
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# TODO: think about batched generation...

@dataclass
class MessiModels:
    base_model_path: str = field(default=model_paths["meta-llama/llama-3.2-3B"])
    it_model_path: str = field(default=model_paths["meta-llama/llama-3.2-3B-Instruct"])

    device: str = "cuda"

    temperature: float = 0.7


    def __post_init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.base_model = self.load_model(self.base_model_path)
        self.it_model = self.load_model(self.it_model_path)

        self.base_tokenizer = self.load_tokenizer(self.base_model_path)
        self.it_tokenizer = self.load_tokenizer(self.it_model_path)
        self.logger.info("Models and tokenizers loaded successfully")
    
    def generate_from_base(self, prompt: str, max_tokens: int = 1):
        self.logger.debug(f"Generating from base model with prompt: {prompt[:50]}... (max tokens: {max_tokens})")
        base_input_ids = self.base_tokenizer(prompt, return_tensors="pt").to(self.device).input_ids
        base_output = self.base_model.generate(base_input_ids, max_new_tokens=max_tokens, temperature=self.temperature)
        self.logger.debug(f"Base model output: {base_output}")
        
        # Get only the new tokens by creating a new tensor
        input_length = base_input_ids.size(1)
        new_tokens = base_output[:, input_length:]
        
        generated_tokens = new_tokens.size(1)  # Get number of new tokens
        out_string = self.base_tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        out_tokens = "[BASE]" * generated_tokens
        self.logger.debug(f"Base model generated {generated_tokens} tokens")
        return out_string, generated_tokens, out_tokens
    
    def generate_from_it(self, prompt: str, max_tokens: int = 10):
        self.logger.debug(f"Generating from IT model with prompt: {prompt[:50]}... (max tokens: {max_tokens})")
        prompt_in_chat_template = [{"role": "assistant", "content": prompt}]
        it_input_ids = self.it_tokenizer.apply_chat_template(prompt_in_chat_template, tokenize=True, continue_final_message=True, return_tensors="pt").to(self.device)
        self.logger.debug(f"IT model input: {it_input_ids.shape}")
        self.logger.debug(f"IT model input: {it_input_ids}")
        it_output = self.it_model.generate(it_input_ids, max_new_tokens=max_tokens, temperature=self.temperature)
        self.logger.debug(f"IT model output: {it_output}")

        # Get only the new tokens
        input_length = it_input_ids.size(1)
        new_tokens = it_output[:, input_length:]
        
        generated_tokens = new_tokens.size(1)
        out_string = self.it_tokenizer.decode(new_tokens[0], skip_special_tokens=True)
        out_tokens = "[INSTRUCT]" * generated_tokens
        self.logger.debug(f"IT model generated {generated_tokens} tokens")
        return out_string, generated_tokens, out_tokens
    
    def generate_from_both(self, prompt: str, max_tokens_total: int = 100, max_base_tokens: int = 10, max_it_tokens: int = 10):
        self.logger.info(f"Starting alternating generation with max_tokens_total={max_tokens_total}")
        # for the max number of tokens
        # rotate between the two models 
        # if the number of tokens is less than the max, then generate from the other model
        # if the number of tokens is greater than the max, then stop
        generated_tokens = 0
        LLM_source = ""
        while generated_tokens < max_tokens_total:
            self.logger.debug(f"Current total tokens: {generated_tokens}/{max_tokens_total}")
            base_output, base_generated_tokens, base_out_tokens = self._generate_tokens(prompt, self.generate_from_base, generated_tokens, max_base_tokens, max_tokens_total)
            generated_tokens += base_generated_tokens
            prompt, LLM_source = self._update_prompt_and_llm_source(prompt, LLM_source, base_output, base_out_tokens)
            if generated_tokens == max_tokens_total:
                break

            it_output, it_generated_tokens, it_out_tokens = self._generate_tokens(prompt, self.generate_from_it, generated_tokens, max_it_tokens, max_tokens_total)
            generated_tokens += it_generated_tokens
            prompt, LLM_source = self._update_prompt_and_llm_source(prompt, LLM_source, it_output, it_out_tokens)
        self.logger.info(f"Completed generation with {generated_tokens} total tokens")
        return prompt, LLM_source
    
    def _update_prompt_and_llm_source(self, prompt, LLM_source, output, out_tokens):
        prompt += output
        LLM_source += out_tokens
        return prompt, LLM_source

    @staticmethod
    def _generate_tokens(prompt, generation_method, generated_tokens, max_tokens_to_generate, max_tokens_total):
        new_tokens_total = generated_tokens + max_tokens_to_generate
        new_tokens_to_generate = max_tokens_to_generate if new_tokens_total < max_tokens_total else max_tokens_total - generated_tokens
        if new_tokens_to_generate == 0:
            return "", 0, ""
        output, generated_tokens, out_tokens = generation_method(prompt, max_tokens=new_tokens_to_generate)
        return output, generated_tokens, out_tokens

    def load_model(self, model_path: str):
        self.logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map=self.device)
        self.logger.info(f"Model loaded successfully to {self.device}")
        return model

    def load_tokenizer(self, model_path: str):
        self.logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        self.logger.info(f"Tokenizer loaded successfully")
        return tokenizer
    

if __name__ == "__main__":
    mmg = MessiModels()
    both_story = mmg.generate_from_both("Write me a hundred word story about a cat.")

    base_story = mmg.generate_from_base("Write me a hundred word story about a cat.", max_tokens=100)
    it_story = mmg.generate_from_it("Write me a hundred word story about a cat.", max_tokens=100)
    print(both_story)
    print(base_story)
    print(it_story)
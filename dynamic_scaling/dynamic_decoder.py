from typing import List, Dict, Any, Set
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
import numpy as np
from config import DynamicDecodingConfig, DecayType, SimilarityType
import logging
from tqdm import tqdm
import os
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class DynamicTemperatureDecoder:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: DynamicDecodingConfig,
    ):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        self.config = config
        self.current_temperature = config.initial_temperature
        self.reference_embeddings = None
        self.reference_thoughts = []
        
        # Create cache directory if it doesn't exist
        if self.config.cache_dir:
            os.makedirs(self.config.cache_dir, exist_ok=True)
            self._load_cached_data()

    def _get_cache_paths(self):
        if not self.config.cache_dir:
            return None, None
        return (
            os.path.join(self.config.cache_dir, "embeddings.pt"),
            os.path.join(self.config.cache_dir, "thoughts.json")
        )

    def _load_cached_data(self):
        emb_path, thoughts_path = self._get_cache_paths()
        if not emb_path or not thoughts_path:
            return
            
        if os.path.exists(emb_path) and os.path.exists(thoughts_path):
            logger.info("Loading cached embeddings and thoughts")
            self.reference_embeddings = torch.load(emb_path)
            with open(thoughts_path, 'r') as f:
                self.reference_thoughts = json.load(f)
            logger.info(f"Loaded {len(self.reference_thoughts)} cached thoughts")

    def _save_cached_data(self):
        emb_path, thoughts_path = self._get_cache_paths()
        if not emb_path or not thoughts_path:
            return
            
        logger.info("Saving embeddings and thoughts to cache")
        torch.save(self.reference_embeddings, emb_path)
        with open(thoughts_path, 'w') as f:
            json.dump(list(self.reference_thoughts), f)

    def compute_embeddings(self, texts: List[str]) -> torch.Tensor:
        computed_embeddings = self._compute_embeddings(texts)
        if self.reference_embeddings is not None:
            self.reference_embeddings = torch.cat((self.reference_embeddings, computed_embeddings), dim=0)
        else:
            self.reference_embeddings = computed_embeddings
        return computed_embeddings

    def add_reference_thoughts(self, new_thoughts: List[str]) -> None:
        """Add new thoughts to the existing reference thoughts."""
        
        # Filter out thoughts we already have
        unique_thoughts = [t for t in new_thoughts if t not in set(self.reference_thoughts)]
        if not unique_thoughts:
            logger.info("All thoughts already in reference set")
            return
        
        self.reference_thoughts.extend(unique_thoughts)
        self.compute_embeddings(unique_thoughts)
        logger.info(f"Added {len(unique_thoughts)} new reference thoughts. Total reference thoughts: {len(self.reference_thoughts)}")
        if self.config.cache_dir:
            self._save_cached_data()

    def _compute_embeddings(self, texts: List[str]) -> torch.Tensor:
        embeddings = []
        for text in tqdm(texts, desc="Computing embeddings"):
            tokenizer_output = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**tokenizer_output, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # shape: [batch, sequence_length, hidden_dim]
                seq_len = hidden_states.shape[1]
                weights = torch.arange(1, seq_len + 1, dtype=torch.float, device=hidden_states.device)
                weights = weights / weights.sum()
                weights = weights.unsqueeze(0).unsqueeze(-1)
                embedding = (hidden_states * weights).sum(dim=1)
                embeddings.append(embedding)
        return torch.cat(embeddings, dim=0)

    def _compute_similarity(self, current_thought: str) -> tuple[float, str]:
        if not self.reference_thoughts or self.reference_embeddings is None:
            return 0.0, ""
            
        tokenizer_output = self.tokenizer(current_thought, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**tokenizer_output, output_hidden_states=True)
            current_embedding = outputs.hidden_states[-1].mean(dim=1)  
        ref_embeddings = self.reference_embeddings.to(self.device)
        repeated_embedding = current_embedding.expand(ref_embeddings.size(0), -1)
        
        if self.config.similarity_type == SimilarityType.COSINE:
            similarities = torch.cosine_similarity(repeated_embedding, ref_embeddings, dim=1)
        else:  # DOT_PRODUCT similarity
            similarities = torch.sum(repeated_embedding * ref_embeddings, dim=1)
            
        k = min(self.config.top_k, similarities.size(0))
        top_k_similarities, top_k_indices = torch.topk(similarities, k)
        most_similar_idx = top_k_indices[0].item()
        most_similar_thought = self.reference_thoughts[most_similar_idx]
        return top_k_similarities.mean().item(), most_similar_thought

    def _adjust_temperature(self, similarity: float, most_similar_thought: str, current_output: str) -> float:
        old_temp = self.current_temperature
        if similarity > self.config.similarity_threshold:
            # Increase temperature when too similar
            self.current_temperature = min(
                self.config.max_temperature,
                self.current_temperature + self.config.linear_step
            )
            if self.config.show_similarity_matches:
                logger.info(f"\n{'='*80}\nHigh similarity detected ({similarity:.3f})!\nCurrent generation:\n{current_output}\n\nMost similar reference:\n{most_similar_thought}\n{'='*80}\n")
            logger.info(f"Max similarity: {similarity:.3f}, Temperature increased: {old_temp:.3f} -> {self.current_temperature:.3f}")
        else:
            # Decay temperature
            if self.config.decay_type == DecayType.LINEAR:
                self.current_temperature = max(
                    self.config.min_temperature,
                    self.current_temperature - self.config.linear_step
                )
            else:  # Exponential decay
                self.current_temperature = max(
                    self.config.min_temperature,
                    self.current_temperature * (1 - self.config.decay_rate)
                )
            logger.info(f"Max similarity: {similarity:.3f}. Temperature decayed: {old_temp:.3f} -> {self.current_temperature:.3f}")
        
        return self.current_temperature

    def generate(
        self,
        system_prompt: str,
        prompt: str,
        max_length: int = 512,
        **kwargs
    ) -> Dict[str, Any]:


        # use apply_chat_template to add system prompt
        if system_prompt:
            prompt_dict = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                return_tensors="pt"
            )
        else:
            prompt_dict = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                return_tensors="pt"
            )

        prompt_dict = self.tokenizer(prompt_dict, return_tensors="pt")
        prompt_dict = {k: v.to(self.device) for k, v in prompt_dict.items()}
        # Initialize context with the prompt's input_ids and attention_mask
        context_ids = prompt_dict["input_ids"]
        attention_mask = prompt_dict["attention_mask"]
        # This list will store only the generated tokens (excluding the original prompt)
        generated_tokens = []
        
        temperature_history = []
        similarity_history = []
        current_temperature = self.config.initial_temperature
        
        # Set up generation config
        generation_config = {
            "do_sample": True,
            "temperature": current_temperature,
            "top_p": 0.7,
            "top_k": 50,
            "return_dict_in_generate": True,
            "output_scores": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }
        
        # Generate tokens in chunks based on similarity check frequency
        while len(generated_tokens) < max_length:
            # Determine how many tokens to generate in this chunk
            tokens_to_next_check = self.config.similarity_check_frequency
            if len(generated_tokens) < self.config.initial_skip_tokens:
                tokens_to_next_check = self.config.initial_skip_tokens - len(generated_tokens)
            
            # Update generation config with current temperature
            generation_config["temperature"] = current_temperature
            generation_config["max_new_tokens"] = tokens_to_next_check
            
            outputs = self.model.generate(input_ids=context_ids, attention_mask=attention_mask, **generation_config)
            # Extract new tokens: slice from the end of the current context
            new_tokens = outputs.sequences[0, context_ids.shape[1]:]
            if new_tokens.numel() == 0:
                break
            
            # Concatenate the new tokens to the generated tokens list
            generated_tokens.extend(new_tokens.tolist())
                
            # Update the context_ids by appending the new tokens (unsqueeze to match dimensions)
            context_ids = torch.cat([context_ids, new_tokens.unsqueeze(0)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((1, new_tokens.shape[0]), dtype=torch.long, device=self.device)], dim=1)
            
            # Update temperature history for each newly generated token
            temperature_history.extend([current_temperature] * new_tokens.shape[0])
            
            current_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            if self.config.verbose_generation and len(generated_tokens) % 10 == 0:
                
                logger.info(f"\nGenerated text at {len(generated_tokens)} tokens:\n{current_output}\n")
                # logger.info(f"\nNew tokens:'{self.tokenizer.decode(new_tokens, skip_special_tokens=True)}'")
                
            
            # Check if we need to adjust temperature
            if (len(generated_tokens) >= self.config.initial_skip_tokens and 
                len(generated_tokens) % self.config.similarity_check_frequency == 0):
                similarity, most_similar_thought = self._compute_similarity(current_output)
                current_temperature = self._adjust_temperature(similarity, most_similar_thought, current_output)
                similarity_history.append(similarity)
            
            # Check if generation is complete (end-of-sequence token encountered)
            print("NEW TOKENS: ", new_tokens[-1].item(), "EOS TOKEN: ", self.tokenizer.eos_token_id)
            if new_tokens[-1].item() == self.tokenizer.eos_token_id:
                logger.info("End of sequence token generated. Stopping generation.")
                break
        
        final_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        logger.info(f"Generation completed. Output length: {len(final_output)}")
        
        return {
            "generated_text": final_output,
            "temperature_history": temperature_history,
            "similarity_history": similarity_history
        } 
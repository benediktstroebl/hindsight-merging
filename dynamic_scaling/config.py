from dataclasses import dataclass
from enum import Enum
from typing import Optional

class DecayType(Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"

class SimilarityType(Enum):
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"

@dataclass
class DynamicDecodingConfig:
    # Temperature settings
    min_temperature: float = 0.01
    max_temperature: float = 1.3
    initial_temperature: float = 0.5
    
    # Decay settings
    decay_type: DecayType = DecayType.LINEAR
    decay_rate: float = 0.3  # For exponential decay
    linear_step: float = 0.3  # For linear decay
    
    # Similarity settings
    similarity_threshold: float = 0.8
    similarity_type: SimilarityType = SimilarityType.COSINE
    top_k: int = 5
    similarity_check_frequency: int = 50  # Check similarity every n tokens
    initial_skip_tokens: int = 200  # Skip first k tokens before checking similarity
    show_similarity_matches: bool = False  # Whether to print similar examples during generation
    
    # Processing settings
    batch_size: int = 1
    verbose_generation: bool = False
    
    # Cache settings
    cache_embeddings: bool = True
    cache_dir: Optional[str] = ".cache" 
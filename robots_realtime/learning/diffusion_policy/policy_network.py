from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    """Configuration for model input/output keys."""
    
    action_keys: Tuple[str, ...]
    mlp_keys: Tuple[str, ...]
    image_keys: Tuple[str, ...]


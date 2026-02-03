"""
Utility functions for prompt loading and sampling.
"""

import json
import random
from typing import Dict, List


def load_prompts(prompts_path: str) -> Dict[str, List[str]]:
    """
    Load prompts from JSON file.
    
    Args:
        prompts_path: Path to prompts.json file
    
    Returns:
        Dictionary mapping class names to lists of prompt strings
    """
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)
    return prompts


def sample_prompt(class_name: str, prompts: Dict[str, List[str]], is_training: bool = True) -> str:
    """
    Sample a prompt for the given class name.
    
    Args:
        class_name: Name of the class (e.g., "crack")
        prompts: Dictionary of prompts loaded from prompts.json
        is_training: If True, randomly sample. If False, use default prompt.
    
    Returns:
        Prompt string
    """
    if not is_training:
        # During validation, always use "segment crack"
        return "segment crack"
    
    # During training, randomly sample from prompts[class_name]
    if class_name not in prompts:
        raise ValueError(f"Class '{class_name}' not found in prompts")
    
    prompt_list = prompts[class_name]
    return random.choice(prompt_list)


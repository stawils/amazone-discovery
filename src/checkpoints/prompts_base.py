#!/usr/bin/env python3
"""
Base class for SAAM-enhanced checkpoint prompts
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class BaseCheckpointPrompts(ABC):
    """Base class for checkpoint-specific SAAM-enhanced prompts"""
    
    def __init__(self, checkpoint: int, name: str):
        self.checkpoint = checkpoint
        self.name = name
        logger.debug(f"Initialized {self.__class__.__name__} for checkpoint {checkpoint}")
    
    def get_saam_context_prefix(self) -> str:
        """Get standard SAAM context prefix for all prompts"""
        return f"checkpoint{self.checkpoint}_enhanced + saam_integration + archaeological_analysis"
    
    def get_cultural_considerations_suffix(self) -> str:
        """Get standard cultural considerations for all prompts"""
        return """
CULTURAL CONSIDERATIONS:
- Respect indigenous knowledge systems and territorial rights
- Ensure collaborative approaches with local communities  
- Prioritize cultural sensitivity in all interpretations
- Support heritage protection and community-led conservation
- Acknowledge traditional archaeological knowledge
"""
    
    def validate_prompt_structure(self, prompt_dict: Dict[str, str]) -> bool:
        """Validate that prompt dictionary has required structure"""
        required_keys = ["base_prompt", "specialized_context", "specialized_instructions"]
        return all(key in prompt_dict for key in required_keys)
    
    @abstractmethod
    def get_primary_prompt(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Get the primary prompt for this checkpoint (must be implemented by subclasses)"""
        pass
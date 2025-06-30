#!/usr/bin/env python3
"""
SAAM-Enhanced Checkpoint Prompts
Modular prompt system for archaeological analysis
"""

from .checkpoint1_prompts import Checkpoint1Prompts
from .checkpoint2_prompts import Checkpoint2Prompts  
from .checkpoint3_prompts import Checkpoint3Prompts
from .checkpoint4_prompts import Checkpoint4Prompts
from .checkpoint5_prompts import Checkpoint5Prompts

__all__ = [
    'Checkpoint1Prompts',
    'Checkpoint2Prompts', 
    'Checkpoint3Prompts',
    'Checkpoint4Prompts',
    'Checkpoint5Prompts'
]
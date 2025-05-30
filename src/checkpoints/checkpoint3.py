# src/checkpoints/checkpoint3.py
"""
Checkpoint 3: New Site Discovery
- Pick single best site discovery and back it up with evidence
- Detect feature algorithmically (Hough transform, segmentation)
- Show historical-text cross-reference via GPT extraction
- Compare discovery to known archaeological feature
"""

from .base_checkpoint import BaseCheckpoint
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class Checkpoint3Discovery(BaseCheckpoint):
    """Checkpoint 3: New Site Discovery with evidence"""
    
    def get_requirements(self) -> Dict[str, Any]:
        """Requirements for checkpoint 3"""
        return {
            'best_discovery': {
                'type': 'exists',
                'path': 'best_discovery',
                'description': 'Must select single best site discovery'
            },
            'algorithmic_detection': {
                'type': 'exists',
                'path': 'algorithmic_detection',
                'description': 'Must detect feature algorithmically (Hough, segmentation, etc.)'
            },
            'historical_crossreference': {
                'type': 'exists',
                'path': 'historical_crossreference',
                'description': 'Must show historical text cross-reference via GPT'
            },
            'comparison_to_known': {
                'type': 'exists',
                'path': 'comparison_to_known_sites',
                'description': 'Must compare to known archaeological features'
            },
            'evidence_package': {
                'type': 'exists',
                'path': 'evidence_package',
                'description': 'Must provide comprehensive evidence backing'
            },
            'coordinates': {
                'type': 'exists',
                'path': 'best_discovery.coordinates',
                'description': 'Must provide discovery coordinates'
            },
            'confidence_score': {
                'type': 'min_value',
                'path': 'best_discovery.confidence',
                'min_value': 0.1,
                'description': 'Must have measurable confidence score'
            }
        }
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute checkpoint 3 - implemented in openai_checkpoints.py"""
        
        return {
            'title': 'New Site Discovery with Evidence',
            'status': 'implemented_in_runner',
            'requirements_summary': 'Best site + algorithmic detection + historical reference + comparison'
        }

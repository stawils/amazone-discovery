# src/checkpoints/checkpoint1.py
"""
Checkpoint 1: Familiarize yourself with the challenge and data
- Download one OpenTopography LiDAR tile or one Sentinel-2 scene ID
- Run a single OpenAI prompt on that data
- Print model version and dataset ID
"""

from .base_checkpoint import BaseCheckpoint
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class Checkpoint1Familiarize(BaseCheckpoint):
    """Checkpoint 1: Familiarize with challenge and data"""
    
    def get_requirements(self) -> Dict[str, Any]:
        """Requirements for checkpoint 1"""
        return {
            'data_downloaded': {
                'type': 'exists',
                'path': 'data_downloaded',
                'description': 'Must download at least one satellite scene'
            },
            'scene_id': {
                'type': 'not_empty',
                'path': 'data_downloaded.scene_id',
                'description': 'Must have valid scene ID'
            },
            'openai_analysis': {
                'type': 'exists', 
                'path': 'openai_analysis',
                'description': 'Must run OpenAI analysis on the data'
            },
            'model_version': {
                'type': 'not_empty',
                'path': 'openai_analysis.model',
                'description': 'Must print OpenAI model version'
            },
            'provider_used': {
                'type': 'not_empty',
                'path': 'data_downloaded.provider',
                'description': 'Must specify data provider'
            }
        }
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute checkpoint 1 - implemented in openai_checkpoints.py"""
        
        # This is a placeholder - actual implementation is in CheckpointRunner
        return {
            'title': 'Familiarize with Challenge and Data',
            'status': 'implemented_in_runner',
            'requirements_summary': 'Download 1 scene + OpenAI analysis + print model/dataset ID'
        }

# src/checkpoints/checkpoint4.py
"""
Checkpoint 4: Story & impact draft
- Craft narrative for livestream presentation
- Create two-page PDF explaining cultural context, hypotheses for function/age
- Proposed survey effort with local partners
"""

from .base_checkpoint import BaseCheckpoint
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class Checkpoint4Story(BaseCheckpoint):
    """Checkpoint 4: Story & impact draft"""
    
    def get_requirements(self) -> Dict[str, Any]:
        """Requirements for checkpoint 4"""
        return {
            'story_document': {
                'type': 'exists',
                'path': 'story_document',
                'description': 'Must create comprehensive story document'
            },
            'cultural_context': {
                'type': 'not_empty',
                'path': 'story_document.main_narrative',
                'description': 'Must explain cultural context'
            },
            'function_age_hypotheses': {
                'type': 'not_empty',
                'path': 'story_document.function_age_hypotheses',
                'description': 'Must provide hypotheses for function and age'
            },
            'survey_proposal': {
                'type': 'not_empty',
                'path': 'story_document.survey_proposal',
                'description': 'Must propose survey effort with local partners'
            },
            'presentation_outline': {
                'type': 'min_count',
                'path': 'presentation_outline',
                'min_count': 5,
                'description': 'Must prepare livestream presentation outline'
            },
            'document_path': {
                'type': 'exists',
                'path': 'document_path',
                'description': 'Must save document to file'
            }
        }
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute checkpoint 4 - implemented in openai_checkpoints.py"""
        
        return {
            'title': 'Story & Impact Draft',
            'status': 'implemented_in_runner',
            'requirements_summary': 'Narrative + cultural context + hypotheses + survey proposal'
        }

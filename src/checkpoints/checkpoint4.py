# src/checkpoints/checkpoint4.py
"""
Checkpoint 4: Story & impact draft
- Craft the narrative you'll present on the livestream
- Create a two-page PDF explaining cultural context, hypotheses for function/age
- Proposed survey effort with local partners
"""

from .base_checkpoint import BaseCheckpoint
from typing import Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class Checkpoint4StoryImpact(BaseCheckpoint):
    """Checkpoint 4: Story & impact draft"""
    
    def get_requirements(self) -> Dict[str, Any]:
        """Requirements for checkpoint 4"""
        return {
            'narrative_analysis': {
                'type': 'exists',
                'path': 'narrative_analysis'
            },
            'cultural_context': {
                'type': 'exists',
                'path': 'narrative_analysis.response'
            },
            'livestream_narrative': {
                'type': 'not_empty',
                'path': 'narrative_analysis.response'
            },
            'target_zone': {
                'type': 'exists',
                'path': 'target_zone'
            },
            'tokens_used': {
                'type': 'min_value',
                'path': 'narrative_analysis.tokens_used',
                'min_value': 0
            }
        }
    
    def execute(self, zone: str = None, openai_integration=None, **kwargs) -> Dict[str, Any]:
        """Execute checkpoint 4: Story & impact draft"""
        
        if not openai_integration:
            raise ValueError("OpenAI integration required for checkpoint 4")
            
        from src.core.config import TARGET_ZONES
        
        logger.info("📖 Checkpoint 4: Story & Impact")

        # Require explicit zone specification
        if zone is None:
            raise ValueError("Zone must be specified for checkpoint analysis. Use --zone parameter.")
        else:
            zone_id = zone  # Use exactly what was passed

        # Get zone configuration
        zone_info = TARGET_ZONES.get(zone_id)
        if not zone_info:
            raise ValueError(f"Zone '{zone}' not found in TARGET_ZONES")

        # Create comprehensive narrative using OpenAI
        narrative_prompt = f"""
        Create a compelling 2-page narrative for the OpenAI to Z Challenge livestream presentation.
        
        CONTEXT:
        - Target Zone: {zone_info.name}
        - Coordinates: {zone_info.center}
        - Historical Evidence: {zone_info.historical_evidence}
        - Expected Features: {zone_info.expected_features}
        
        REQUIREMENTS:
        1. Cultural Context: Explain the indigenous peoples and historical significance
        2. Hypotheses: Provide scientific hypotheses for function and age of discoveries
        3. Survey Proposal: Outline proposed field survey with local community partners
        4. Impact Assessment: Describe potential archaeological and cultural impact
        
        Make this engaging for a live audience while maintaining scientific rigor.
        Focus on respectful collaboration with indigenous communities and heritage protection.
        """

        narrative_analysis = openai_integration.analyze_with_openai(
            narrative_prompt, f"Livestream narrative for {zone_info.name}"
        )

        print(f"\n🎯 CHECKPOINT 4 COMPLETE:")
        print(f"Narrative created for: {zone_info.name}")
        print(f"Tokens used: {narrative_analysis.get('tokens_used', 0)}")

        return {
            "title": "Story & impact draft",
            "target_zone": zone_id,
            "narrative_analysis": narrative_analysis,
            "summary": f"Created livestream narrative and impact assessment for {zone_info.name}"
        }

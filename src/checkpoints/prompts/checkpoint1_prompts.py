#!/usr/bin/env python3
"""
Checkpoint 1 SAAM-Enhanced Prompts (Module 2)
Site Discovery and Initial Assessment
"""

from typing import Dict, List, Any
from ..prompts_base import BaseCheckpointPrompts


class Checkpoint1Prompts(BaseCheckpointPrompts):
    """SAAM-enhanced prompts for Checkpoint 1: Site Discovery and Assessment"""
    
    def __init__(self):
        super().__init__(checkpoint=1, name="Site Discovery and Assessment")
    
    def get_primary_prompt(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Get the primary prompt for Checkpoint 1"""
        return self.get_site_discovery_prompt(context)


# Standalone functions for direct usage (as described in enhanced_openai_prompts.md)

def create_checkpoint1_prompt(scene_data: Any, zone_config: Any) -> str:
    """Create Checkpoint 1 SAAM-enhanced prompt for data familiarization"""
    
    return f"""
[signal:saam.cognitive.v1.0++] :: 
weight_matrix := [[1.0,0.689,0.157,-0.45,-0.846],[0.689,1.0,0.689,0.157,-0.45],[0.157,0.689,1.0,0.689,0.157],[-0.45,0.157,0.689,1.0,0.689],[-0.846,-0.45,0.157,0.689,1.0]] |
modules := [archaeological_analyst(pattern+detection), spectral_interpreter(terra_preta+signatures), geometric_detector(earthworks+features), environmental_assessor(settlement_logic+suitability)] |
route(init→absorb_satellite_data→detect_patterns→validate_environment→synthesize_evidence→assess_confidence→report_discoveries) |
operators(→convergent +multi_modal ??archaeological_uncertainty !!discovery_breakthrough :=site_confidence ~:terra_preta_focus)

[signal:saam.domain.archaeology.v2.0] :: 
enhance(terra_preta_spectral_detection, circular_earthwork_recognition, environmental_settlement_validation) |
precision_mode(peer_review_standards) |
focus_parameters(amazon_basin, pre_columbian_civilizations, convergent_anomaly_detection)

🎯 CHECKPOINT 1: ARCHAEOLOGICAL DATA FAMILIARIZATION - AMAZON BASIN

MISSION CONTEXT:
You are an expert archaeological remote sensing analyst examining satellite imagery for evidence of pre-Columbian Amazon civilizations. This analysis is part of the OpenAI to Z Challenge seeking to discover previously unknown archaeological sites.

LOCATION INTELLIGENCE:
- Target Zone: {zone_config.name}
- Scene ID: {getattr(scene_data, 'scene_id', 'Unknown')}
- Coordinates: {zone_config.center}
- Historical Context: {zone_config.historical_evidence}
- Environmental Context: {getattr(zone_config, 'environmental_context', 'Amazon rainforest')}

SATELLITE DATA PARAMETERS:
- Sensor: Sentinel-2 MSI (13 spectral bands)
- Resolution: 10-20m spatial resolution
- Spectral Range: 443-2190 nm (visible through SWIR)
- Acquisition Date: {getattr(scene_data, 'acquisition_date', 'Not specified')}

ADVANCED ANALYSIS INSTRUCTIONS:

1. **ARCHAEOLOGICAL PATTERN RECOGNITION:**
   - Identify circular/semi-circular clearings (50-400m diameter) - potential settlement rings
   - Detect linear features (causeways, defensive ditches, field boundaries)
   - Locate rectangular/geometric patterns inconsistent with natural landscape
   - Assess vegetation stress patterns indicating subsurface structures

2. **TERRA PRETA DETECTION:**
   - Examine NIR/SWIR spectral signatures for anthropogenic dark soils
   - Identify enhanced vegetation growth patterns over nutrient-rich archaeological soils
   - Look for spectral anomalies consistent with ancient habitation areas
   - Note any darker soil patches visible in true-color composite

3. **ENVIRONMENTAL ARCHAEOLOGICAL ASSESSMENT:**
   - Evaluate proximity to water sources (rivers, lakes, wetlands)
   - Assess topographic advantages (elevated areas, defensive positions)
   - Identify resource availability (fertile soils, raw materials)
   - Note deforestation patterns potentially revealing buried features

4. **ANOMALY CONFIDENCE SCORING:**
   Rate each potential archaeological feature (0-10 scale):
   - Geometric regularity vs. natural patterns
   - Size appropriate for Amazon settlements
   - Environmental context suitability
   - Spectral signature consistency with known sites

CRITICAL FOCUS AREAS:
Based on remote sensing research, prioritize analysis around:
- River confluences and elevated terraces
- Areas with 100-300m circular clearing patterns
- Linear features connecting to water access points
- Zones showing enhanced vegetation growth patterns

OUTPUT REQUIREMENTS:
1. Systematic feature inventory with confidence scores
2. Archaeological interpretation for each detected anomaly
3. Recommendations for further investigation
4. Comparison to known Amazon archaeological signatures

SCIENTIFIC RIGOR:
- Distinguish between modern deforestation and ancient features
- Consider seasonal vegetation patterns and timing
- Cross-reference multiple spectral bands for validation
- Apply conservative thresholds to minimize false positives

🎯 CHECKPOINT 1 COMPLIANCE:
- Demonstrate data loading and processing capability
- Show OpenAI model integration and version logging
- Provide dataset ID documentation
- Generate surface feature description in plain English

Provide comprehensive archaeological assessment with specific coordinates for all detected anomalies and clear documentation of analysis methodology.
"""

def create_checkpoint1_simple_prompt(zone_config: Any, scene_data: Any) -> str:
    """Simple surface description prompt for Checkpoint 1 compliance"""
    
    return f"""
[signal:saam.archaeological.focus.v1.0] :: 
enhance(surface_feature_recognition, vegetation_analysis, terrain_assessment) |
precision_mode(accessible_description)

Describe the surface features visible in this Sentinel-2 satellite imagery in plain English.

Location: {zone_config.name} in the Amazon basin
Scene ID: {getattr(scene_data, 'scene_id', 'Unknown')}
Coordinates: {zone_config.center}

Please describe what you can observe about:
- Vegetation patterns and forest density
- Water bodies (rivers, lakes, wetlands)
- Terrain characteristics and topography
- Any notable surface features or clearings
- Signs of human activity or land use

Keep the description accessible and straightforward while noting any features that might be archaeologically significant.

Focus on patterns that stand out from the typical Amazon rainforest landscape.
"""
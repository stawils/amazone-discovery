#!/usr/bin/env python3
"""
SAAM Prompt Router System (Module 7)
Intelligent prompt management and routing for archaeological analysis
"""

import logging
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptComplexity(Enum):
    """Prompt complexity levels for different analysis requirements"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate" 
    ADVANCED = "advanced"
    EXPERT = "expert"


class AnalysisType(Enum):
    """Types of archaeological analysis"""
    SITE_IDENTIFICATION = "site_identification"
    PATTERN_RECOGNITION = "pattern_recognition"
    CULTURAL_INTERPRETATION = "cultural_interpretation"
    CONSERVATION_ASSESSMENT = "conservation_assessment"
    MULTI_MODAL_SYNTHESIS = "multi_modal_synthesis"


@dataclass
class PromptTemplate:
    """Template for SAAM-enhanced prompts"""
    name: str
    checkpoint: int
    analysis_type: AnalysisType
    complexity: PromptComplexity
    base_prompt: str
    specialized_context: str = ""
    specialized_instructions: str = ""
    required_data_types: List[str] = None
    cultural_considerations: str = ""
    
    def __post_init__(self):
        if self.required_data_types is None:
            self.required_data_types = []


class SAAMPromptRouter:
    """
    Intelligent prompt routing system for SAAM-enhanced archaeological analysis
    
    Features:
    - Dynamic prompt selection based on analysis requirements
    - Complexity scaling for different discovery scenarios
    - Cultural context integration
    - Competition-optimized prompt templates
    """
    
    def __init__(self):
        """Initialize the SAAM prompt router"""
        self.prompt_registry = {}
        self.checkpoint_prompts = {}
        self._initialize_core_prompts()
        logger.info("🧭 SAAM Prompt Router initialized")
    
    def _initialize_core_prompts(self):
        """Initialize core prompt templates for all checkpoints"""
        
        # Checkpoint 1: Site Discovery and Assessment
        self.register_prompt(PromptTemplate(
            name="checkpoint1_site_discovery",
            checkpoint=1,
            analysis_type=AnalysisType.SITE_IDENTIFICATION,
            complexity=PromptComplexity.INTERMEDIATE,
            base_prompt="""
Analyze the provided satellite and LiDAR data to identify potential archaeological sites in the Amazon rainforest. 
Focus on detecting anomalies that could indicate human settlements, earthworks, or terra preta formations.

Apply the convergent anomaly detection framework to score potential sites on a 0-16 scale based on:
- Historical evidence (0-2 points)
- Geometric patterns (0-6 points) 
- Spectral signatures (0-2 points)
- Environmental context (0-1 points)
- Convergence factors (0-3 points)
- AI enhancement bonus (0-2 points)

Provide specific coordinates and confidence assessments for each potential site discovered.
""",
            specialized_context="archaeological_site_discovery + convergent_anomaly_detection + openai_competition",
            specialized_instructions="""
Focus on systematic site discovery methodology. Document all OpenAI model interactions for competition validation.
Emphasize innovative approaches that differentiate from traditional remote sensing methods.
""",
            required_data_types=["sentinel2", "gedi_lidar"],
            cultural_considerations="Coordinate with indigenous communities and respect traditional knowledge systems."
        ))
        
        # Checkpoint 2: Pattern Recognition and Analysis
        self.register_prompt(PromptTemplate(
            name="checkpoint2_pattern_analysis",
            checkpoint=2,
            analysis_type=AnalysisType.PATTERN_RECOGNITION,
            complexity=PromptComplexity.ADVANCED,
            base_prompt="""
Conduct advanced pattern recognition analysis on the identified potential sites. 
Examine spatial relationships, geometric configurations, and environmental patterns that suggest organized human activity.

Analyze:
1. Settlement patterns and spatial organization
2. Geometric earthwork configurations
3. Vegetation anomalies indicating past human modification
4. Hydrological relationships and water management systems
5. Inter-site connectivity and regional networks

Provide detailed pattern analysis with archaeological interpretation and cultural significance assessment.
""",
            specialized_context="advanced_pattern_recognition + spatial_archaeology + cultural_landscape_analysis",
            specialized_instructions="""
Apply sophisticated pattern recognition algorithms. Use AI models to identify subtle patterns invisible to traditional analysis.
Document methodological innovations for competition differentiation.
""",
            required_data_types=["sentinel2", "gedi_lidar", "dem"],
            cultural_considerations="Consider indigenous spatial concepts and traditional settlement patterns."
        ))
        
        # Checkpoint 3: Cultural and Historical Integration
        self.register_prompt(PromptTemplate(
            name="checkpoint3_cultural_integration",
            checkpoint=3,
            analysis_type=AnalysisType.CULTURAL_INTERPRETATION,
            complexity=PromptComplexity.EXPERT,
            base_prompt="""
Integrate cultural and historical evidence with remote sensing discoveries to build comprehensive site interpretations.
Synthesize archaeological and ethnographic data to understand the cultural significance of discovered sites.

Examine:
1. Archaeological context and cultural periods
2. Ethnographic parallels and indigenous oral histories
3. Historical documentation and early explorer accounts
4. Environmental reconstruction and human-landscape interactions
5. Cultural continuity and change over time

Provide culturally-informed interpretations that respect indigenous knowledge and highlight archaeological significance.
""",
            specialized_context="cultural_archaeology + ethnohistory + indigenous_collaboration + heritage_interpretation",
            specialized_instructions="""
Emphasize cultural sensitivity and collaborative approaches. Integrate multiple knowledge systems respectfully.
Highlight unique cultural insights enabled by AI-enhanced analysis methods.
""",
            required_data_types=["ethnographic_data", "oral_histories"],
            cultural_considerations="Prioritize indigenous perspectives and ensure collaborative interpretation approaches."
        ))
        
        # Checkpoint 4: Conservation and Impact Assessment
        self.register_prompt(PromptTemplate(
            name="checkpoint4_conservation_assessment",
            checkpoint=4,
            analysis_type=AnalysisType.CONSERVATION_ASSESSMENT,
            complexity=PromptComplexity.ADVANCED,
            base_prompt="""
Assess conservation threats and develop protection strategies for discovered archaeological sites.
Analyze current and projected threats including deforestation, mining, agriculture, and climate change impacts.

Evaluate:
1. Current site condition and integrity
2. Immediate and long-term threats
3. Conservation priority ranking
4. Protection strategy recommendations
5. Community engagement opportunities
6. Legal and policy considerations

Provide actionable conservation recommendations with urgency assessments and implementation strategies.
""",
            specialized_context="heritage_conservation + threat_assessment + community_engagement + policy_development",
            specialized_instructions="""
Focus on practical, implementable conservation strategies. Emphasize community-based protection approaches.
Document innovative conservation technologies enabled by AI analysis.
""",
            required_data_types=["threat_assessment", "deforestation_data", "community_maps"],
            cultural_considerations="Engage indigenous communities as primary stakeholders in conservation planning."
        ))
        
        # Checkpoint 5: Final Synthesis and Presentation
        self.register_prompt(PromptTemplate(
            name="checkpoint5_final_synthesis",
            checkpoint=5,
            analysis_type=AnalysisType.MULTI_MODAL_SYNTHESIS,
            complexity=PromptComplexity.EXPERT,
            base_prompt="""
Create comprehensive final synthesis integrating all discovery, analysis, and conservation findings.
Prepare competition-ready presentation highlighting methodological innovations and archaeological discoveries.

Synthesize:
1. Complete site discovery inventory with confidence rankings
2. Pattern analysis and cultural interpretations
3. Conservation assessments and protection strategies
4. Methodological innovations and AI integration
5. Community collaboration outcomes
6. Future research recommendations

Prepare presentation materials suitable for scientific peer review and public engagement.
""",
            specialized_context="final_synthesis + competition_presentation + scientific_communication + public_engagement",
            specialized_instructions="""
Emphasize methodological innovations and breakthrough discoveries. Prepare for live presentation and Q&A.
Document complete AI model usage for competition validation and transparency.
""",
            required_data_types=["all_previous_analyses", "site_documentation", "conservation_plans"],
            cultural_considerations="Ensure indigenous communities are credited as collaborative partners and knowledge holders."
        ))
    
    def register_prompt(self, prompt_template: PromptTemplate):
        """Register a new prompt template"""
        key = f"{prompt_template.checkpoint}_{prompt_template.name}"
        self.prompt_registry[key] = prompt_template
        
        if prompt_template.checkpoint not in self.checkpoint_prompts:
            self.checkpoint_prompts[prompt_template.checkpoint] = []
        self.checkpoint_prompts[prompt_template.checkpoint].append(prompt_template)
        
        logger.debug(f"Registered prompt: {prompt_template.name} for checkpoint {prompt_template.checkpoint}")
    
    def get_prompt_for_checkpoint(self, 
                                  checkpoint: int, 
                                  analysis_type: Optional[AnalysisType] = None,
                                  complexity: Optional[PromptComplexity] = None) -> Optional[PromptTemplate]:
        """
        Get appropriate prompt template for checkpoint and requirements
        
        Args:
            checkpoint: Checkpoint number (1-5)
            analysis_type: Optional specific analysis type
            complexity: Optional complexity requirement
            
        Returns:
            Best matching prompt template or None
        """
        
        if checkpoint not in self.checkpoint_prompts:
            logger.warning(f"No prompts registered for checkpoint {checkpoint}")
            return None
        
        available_prompts = self.checkpoint_prompts[checkpoint]
        
        # Filter by analysis type if specified
        if analysis_type:
            available_prompts = [p for p in available_prompts if p.analysis_type == analysis_type]
        
        # Filter by complexity if specified
        if complexity:
            available_prompts = [p for p in available_prompts if p.complexity == complexity]
        
        if not available_prompts:
            # Fallback to any prompt for the checkpoint
            available_prompts = self.checkpoint_prompts[checkpoint]
        
        # Return the first matching prompt (could be enhanced with scoring logic)
        return available_prompts[0] if available_prompts else None
    
    def get_enhanced_prompt_data(self, 
                                 checkpoint: int,
                                 zone_name: str,
                                 available_data_types: List[str],
                                 analysis_type: Optional[AnalysisType] = None,
                                 complexity: Optional[PromptComplexity] = None) -> Tuple[str, str, str]:
        """
        Get enhanced prompt data for SAAM integration
        
        Args:
            checkpoint: Checkpoint number
            zone_name: Target archaeological zone
            available_data_types: List of available data types
            analysis_type: Optional analysis type
            complexity: Optional complexity level
            
        Returns:
            Tuple of (base_prompt, specialized_context, specialized_instructions)
        """
        
        template = self.get_prompt_for_checkpoint(checkpoint, analysis_type, complexity)
        
        if not template:
            logger.error(f"No suitable prompt found for checkpoint {checkpoint}")
            return ("", "", "")
        
        # Check data requirements
        missing_data = [dt for dt in template.required_data_types if dt not in available_data_types]
        if missing_data:
            logger.warning(f"Missing required data types for optimal analysis: {missing_data}")
        
        # Enhance context with zone-specific information
        enhanced_context = template.specialized_context
        if zone_name:
            enhanced_context += f" + target_zone_{zone_name.lower()}"
        
        # Add data availability context
        enhanced_context += f" + available_data_{'+'.join(available_data_types)}"
        
        # Enhance instructions with data-specific guidance
        enhanced_instructions = template.specialized_instructions
        if missing_data:
            enhanced_instructions += f"\n\nNOTE: Optimal analysis requires {missing_data} data. Adapt analysis approach accordingly."
        
        if template.cultural_considerations:
            enhanced_instructions += f"\n\nCultural Considerations: {template.cultural_considerations}"
        
        logger.info(f"🎯 Selected prompt: {template.name} (complexity: {template.complexity.value})")
        
        return (template.base_prompt, enhanced_context, enhanced_instructions)
    
    def list_available_prompts(self, checkpoint: Optional[int] = None) -> Dict[str, Any]:
        """List all available prompts, optionally filtered by checkpoint"""
        
        if checkpoint:
            prompts = self.checkpoint_prompts.get(checkpoint, [])
            return {
                f"checkpoint_{checkpoint}": [
                    {
                        "name": p.name,
                        "analysis_type": p.analysis_type.value,
                        "complexity": p.complexity.value,
                        "required_data": p.required_data_types
                    }
                    for p in prompts
                ]
            }
        
        return {
            f"checkpoint_{cp}": [
                {
                    "name": p.name,
                    "analysis_type": p.analysis_type.value,
                    "complexity": p.complexity.value,
                    "required_data": p.required_data_types
                }
                for p in prompts
            ]
            for cp, prompts in self.checkpoint_prompts.items()
        }


# Global router instance for easy access
_router_instance = None

def get_saam_router() -> SAAMPromptRouter:
    """Get global SAAM prompt router instance"""
    global _router_instance
    if _router_instance is None:
        _router_instance = SAAMPromptRouter()
    return _router_instance
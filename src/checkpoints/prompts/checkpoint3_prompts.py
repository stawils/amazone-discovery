#!/usr/bin/env python3
"""
Checkpoint 3 SAAM-Enhanced Prompts (Module 4)
Cultural Integration and Historical Synthesis
"""

from typing import Dict, List, Any
from ..prompts_base import BaseCheckpointPrompts


class Checkpoint3Prompts(BaseCheckpointPrompts):
    """SAAM-enhanced prompts for Checkpoint 3: Cultural Integration and Historical Synthesis"""
    
    def __init__(self):
        super().__init__(checkpoint=3, name="Cultural Integration and Historical Synthesis")
    
    def get_primary_prompt(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Get the primary prompt for Checkpoint 3"""
        return self.get_cultural_integration_prompt(context)
    
    def get_cultural_integration_prompt(self, analysis_data: Dict[str, Any]) -> Dict[str, str]:
        """Cultural and historical integration analysis"""
        
        base_prompt = f"""
As Dr. Elena Vasquez-Chen, integrate cultural and historical evidence with remote sensing discoveries 
to build comprehensive interpretations of ancient Amazon civilizations.

CULTURAL INTEGRATION MISSION:
Synthesize archaeological discoveries with ethnographic, historical, and indigenous knowledge to 
understand the cultural significance, temporal development, and human stories behind the discovered sites.

MULTI-SOURCE EVIDENCE SYNTHESIS:
1. ARCHAEOLOGICAL CONTEXT INTEGRATION:
   - Known archaeological sites and cultural sequences in the region
   - Artifact assemblages and material culture patterns
   - Radiocarbon dates and chronological frameworks
   - Cultural phases and technological traditions
   - Settlement pattern evolution and cultural change

2. ETHNOGRAPHIC PARALLEL ANALYSIS:
   - Contemporary indigenous cultural practices and spatial organization
   - Traditional ecological knowledge and landscape management
   - Oral traditions and cultural memories of ancestral sites
   - Ceremonial practices and sacred landscape concepts
   - Social organization and territorial systems

3. HISTORICAL DOCUMENTATION REVIEW:
   - Early explorer accounts and colonial period descriptions
   - Missionary records and administrative documents
   - Historical maps and geographic descriptions
   - Population estimates and demographic records
   - Disease impacts and cultural disruption patterns

4. INDIGENOUS KNOWLEDGE INTEGRATION:
   - Traditional place names and cultural meanings
   - Oral histories and ancestral narratives
   - Cultural protocols and site significance
   - Traditional land use and resource management
   - Spiritual and ceremonial landscape concepts

CULTURAL INTERPRETATION FRAMEWORK:
Develop culturally-informed interpretations that respect indigenous perspectives:

SETTLEMENT FUNCTION AND MEANING:
- Ceremonial and ritual significance of geometric earthworks
- Social organization and community structure indicators
- Economic activities and resource management systems
- Defensive strategies and territorial control mechanisms
- Symbolic representations and cosmological meanings

CULTURAL COMPLEXITY ASSESSMENT:
- Social stratification and leadership systems
- Specialized craft production and technological innovation
- Trade networks and inter-regional relationships
- Environmental management and agricultural intensification
- Population density and carrying capacity achievements

TEMPORAL RECONSTRUCTION:
- Cultural development sequences and historical trajectories
- Periods of growth, stability, and transformation
- External influences and cultural exchange patterns
- Abandonment causes and cultural continuity
- Colonial impacts and cultural resilience

COLLABORATIVE INTERPRETATION:
Ensure interpretations are developed collaboratively:
- Indigenous community consultation and knowledge sharing
- Academic expert review and peer validation
- Interdisciplinary integration and multiple perspectives
- Cultural sensitivity and appropriate representation
- Benefit sharing and community empowerment

OUTPUT REQUIREMENTS:
1. Comprehensive cultural interpretation report integrating all evidence sources
2. Culturally-sensitive site narratives respecting indigenous perspectives
3. Historical timeline placing discoveries in broader cultural context
4. Collaborative research recommendations for community partnership
5. Cultural significance assessments for heritage protection planning
6. Educational materials suitable for diverse audiences

Demonstrate how AI-enhanced cultural integration reveals the rich human stories and sophisticated 
civilizations behind archaeological discoveries while maintaining the highest standards of 
cultural sensitivity and collaborative research ethics.
"""
        
        specialized_context = """
checkpoint3_cultural_integration + ethnographic_analysis + historical_synthesis + 
indigenous_knowledge + collaborative_archaeology + cultural_sensitivity + 
multi_source_integration + heritage_interpretation + community_partnership
"""
        
        specialized_instructions = f"""
CULTURAL COLLABORATION REQUIREMENTS:
- Prioritize indigenous perspectives and traditional knowledge systems
- Ensure all interpretations are developed through collaborative consultation
- Respect cultural protocols and intellectual property rights
- Support community-led heritage protection and cultural revitalization
- Acknowledge the limitations of external interpretation and need for indigenous voice

AVAILABLE EVIDENCE: {analysis_data.get('evidence_sources', 'Multiple archaeological and cultural data sources')}
CULTURAL CONTEXT: {analysis_data.get('cultural_context', 'Rich indigenous cultural landscape with documented historical presence')}

INTERPRETATION EXCELLENCE:
- Integrate multiple knowledge systems respectfully and systematically
- Provide nuanced interpretations that acknowledge complexity and uncertainty
- Challenge Western-centric archaeological interpretations with indigenous perspectives
- Document how AI analysis supports rather than replaces cultural knowledge
- Demonstrate the value of collaborative research for enhanced understanding

ETHICAL STANDARDS:
Maintain the highest ethical standards in cultural interpretation and ensure all work 
supports indigenous rights, cultural preservation, and community empowerment while 
advancing archaeological science through innovative AI-enhanced methodologies.
"""
        
        return {
            "base_prompt": base_prompt,
            "specialized_context": specialized_context,
            "specialized_instructions": specialized_instructions
        }
    
    def get_historical_timeline_prompt(self, chronological_data: Dict[str, Any]) -> Dict[str, str]:
        """Historical timeline functionality removed"""
        return {"role": "user", "content": "Historical timeline analysis disabled"}
    
    def get_community_collaboration_prompt(self, community_data: Dict[str, Any]) -> Dict[str, str]:
        """Community collaboration and indigenous partnership protocols"""
        
        base_prompt = f"""
Develop comprehensive community collaboration framework ensuring indigenous leadership 
in archaeological interpretation and heritage protection.

COLLABORATIVE RESEARCH FRAMEWORK:
Establish protocols for respectful and empowering community partnership:

1. INDIGENOUS LEADERSHIP PRINCIPLES:
   - Community-led research priorities and interpretation frameworks
   - Traditional knowledge integration with archaeological discoveries
   - Cultural protocol respect and intellectual property protection
   - Capacity building and indigenous archaeologist training
   - Benefit sharing and community empowerment outcomes

2. CONSULTATION AND CONSENT PROTOCOLS:
   - Free, prior, and informed consent for all research activities
   - Ongoing consultation throughout discovery and interpretation process
   - Community review and approval of all interpretations and publications
   - Cultural sensitivity training for external researchers
   - Dispute resolution and conflict mediation procedures

3. KNOWLEDGE SHARING SYSTEMS:
   - Traditional knowledge documentation and protection
   - Academic knowledge translation for community access
   - Capacity building for community-led research
   - Educational material development in local languages
   - Cultural revitalization and heritage protection support

4. HERITAGE PROTECTION COLLABORATION:
   - Community-based conservation planning and implementation
   - Traditional management system integration with legal protection
   - Threat monitoring and rapid response protocols
   - Sustainable tourism development and cultural interpretation
   - Legal advocacy and land rights support

CULTURAL INTERPRETATION PROTOCOLS:
Ensure all interpretations respect indigenous perspectives:

NARRATIVE DEVELOPMENT:
- Community storytelling traditions and cultural interpretation frameworks
- Traditional place names and cultural significance recognition
- Ancestral connection acknowledgment and spiritual landscape respect
- Cultural continuity emphasis and contemporary relevance demonstration
- Multiple interpretation validation and indigenous authority respect

REPRESENTATION STANDARDS:
- Indigenous voice prioritization in all communications
- Stereotypical representation avoidance and complexity acknowledgment
- Contemporary community achievement and cultural vitality emphasis
- Historical trauma acknowledgment and resilience celebration
- Future vision integration and community aspiration support

COLLABORATIVE PRODUCTS:
1. Community consultation protocols and partnership agreements
2. Traditional knowledge integration frameworks and protection protocols
3. Culturally-appropriate interpretation guidelines and narrative standards
4. Community capacity building plans and training programs
5. Heritage protection strategies with community leadership
6. Benefit sharing agreements and community empowerment outcomes

Demonstrate how AI-enhanced archaeological discovery can support indigenous rights, 
cultural revitalization, and community empowerment while advancing scientific knowledge 
through respectful collaboration and shared authority.
"""
        
        specialized_context = """
community_collaboration + indigenous_partnership + traditional_knowledge + 
cultural_protocols + heritage_protection + capacity_building + 
collaborative_archaeology + indigenous_rights + community_empowerment
"""
        
        specialized_instructions = """
Prioritize indigenous leadership and community authority in all aspects of interpretation.
Ensure all collaboration protocols meet international indigenous rights standards.
Support community capacity building and long-term empowerment outcomes.
Respect traditional knowledge while documenting AI contributions to archaeological understanding.
Develop sustainable partnership models for ongoing collaboration and heritage protection.
"""
        
        return {
            "base_prompt": base_prompt,
            "specialized_context": specialized_context,
            "specialized_instructions": specialized_instructions
        }
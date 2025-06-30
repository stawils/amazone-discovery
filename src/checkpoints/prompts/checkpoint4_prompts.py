#!/usr/bin/env python3
"""
Checkpoint 4 SAAM-Enhanced Prompts (Module 5)
Conservation Assessment and Impact Analysis
"""

from typing import Dict, List, Any
from ..prompts_base import BaseCheckpointPrompts


class Checkpoint4Prompts(BaseCheckpointPrompts):
    """SAAM-enhanced prompts for Checkpoint 4: Conservation Assessment and Impact Analysis"""
    
    def __init__(self):
        super().__init__(checkpoint=4, name="Conservation Assessment and Impact Analysis")
    
    def get_primary_prompt(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Get the primary prompt for Checkpoint 4"""
        return self.get_conservation_assessment_prompt(context)
    
    def get_conservation_assessment_prompt(self, discovery_data: Dict[str, Any]) -> Dict[str, str]:
        """Comprehensive conservation assessment and protection planning"""
        
        base_prompt = f"""
As Dr. Elena Vasquez-Chen, conduct urgent conservation assessment of discovered archaeological sites 
and develop comprehensive protection strategies for Amazon cultural heritage.

CONSERVATION CRISIS ASSESSMENT:
Evaluate immediate and long-term threats to discovered archaeological sites in the context 
of accelerating Amazon destruction and develop actionable protection strategies.

THREAT ANALYSIS FRAMEWORK:
1. IMMEDIATE THREATS (0-2 years):
   - Active deforestation and forest clearing
   - Illegal mining and mineral extraction
   - Agricultural expansion and cattle ranching
   - Infrastructure development and road construction
   - Illegal occupation and land grabbing

2. MEDIUM-TERM THREATS (2-10 years):
   - Climate change impacts and extreme weather events
   - Hydrological changes and water table fluctuations
   - Ecosystem degradation and biodiversity loss
   - Increased access and uncontrolled visitation
   - Political instability and policy changes

3. LONG-TERM THREATS (10+ years):
   - Sea level rise and regional climate shifts
   - Ecosystem collapse and tipping points
   - Cultural disruption and knowledge loss
   - Economic pressure and resource exploitation
   - Global market forces and development pressure

SITE VULNERABILITY ASSESSMENT:
Evaluate site-specific conservation needs:

PHYSICAL VULNERABILITY:
- Site integrity and current condition assessment
- Erosion risk and environmental stability
- Accessibility and exposure to human impact
- Natural disaster susceptibility and resilience
- Restoration potential and intervention requirements

CULTURAL VULNERABILITY:
- Indigenous community capacity for protection
- Traditional management system strength
- Cultural knowledge transmission status
- Community connection and ancestral ties
- Legal recognition and protection status

CONSERVATION PRIORITY RANKING:
Develop scientific framework for protection prioritization:

HIGH PRIORITY (Immediate Action Required):
- Sites under immediate threat with high cultural significance
- Unique archaeological features with no known parallels
- Sites with strong community connections and protection capacity
- Sites with high research potential and educational value
- Gateway sites for broader landscape protection

MEDIUM PRIORITY (Action within 2 years):
- Sites with moderate threats and good protection potential
- Sites requiring community capacity building for protection
- Sites needing legal recognition and formal protection status
- Sites with restoration potential and intervention needs
- Sites important for regional landscape connectivity

PROTECTION STRATEGY DEVELOPMENT:
Create comprehensive protection strategies:

LEGAL PROTECTION MECHANISMS:
- Archaeological site designation and legal recognition
- Indigenous territory demarcation and land rights support
- Protected area expansion and corridor creation
- Environmental law enforcement and penalty enhancement
- International heritage designation and protection

COMMUNITY-BASED CONSERVATION:
- Indigenous protection system strengthening and capacity building
- Traditional management practice revitalization and support
- Community monitoring and rapid response system development
- Cultural protocol development and visitor management
- Economic incentive creation and sustainable livelihood support

TECHNOLOGICAL PROTECTION TOOLS:
- Satellite monitoring and early warning systems
- Drone surveillance and regular condition assessment
- GIS mapping and spatial analysis for protection planning
- AI-enhanced threat detection and response coordination
- Digital documentation and virtual preservation

CONSERVATION OUTCOMES:
1. Comprehensive threat assessment with urgency rankings
2. Site-specific conservation plans with implementation timelines
3. Community-based protection protocols and capacity building programs
4. Legal protection strategy recommendations and advocacy priorities
5. Technology integration plans for enhanced monitoring and protection
6. Funding strategies and partnership development for long-term sustainability

Demonstrate how AI-enhanced archaeological discovery enables rapid conservation response 
and innovative protection strategies for Amazon cultural heritage preservation.
"""
        
        specialized_context = """
checkpoint4_conservation + threat_assessment + heritage_protection + 
community_conservation + legal_protection + technology_monitoring + 
amazon_conservation + cultural_heritage + urgent_response + sustainable_protection
"""
        
        specialized_instructions = f"""
CONSERVATION URGENCY REQUIREMENTS:
- Prioritize immediate action for sites under active threat
- Develop practical, implementable protection strategies with clear timelines
- Integrate community leadership with technical conservation approaches
- Balance scientific documentation with rapid protection response
- Support indigenous land rights and traditional protection systems

DISCOVERED SITES: {discovery_data.get('total_sites', 'Multiple significant')} archaeological sites requiring protection assessment
THREAT LEVEL: {discovery_data.get('threat_level', 'High - multiple active threats identified')}
PROTECTION CAPACITY: {discovery_data.get('protection_capacity', 'Variable - community partnerships essential')}

INNOVATIVE CONSERVATION:
- Apply AI monitoring and predictive threat assessment technologies
- Develop rapid response protocols for emerging threats
- Create scalable protection models applicable to broader Amazon region
- Document conservation innovations for global heritage protection
- Integrate cultural heritage with biodiversity conservation strategies

ETHICAL IMPERATIVES:
Recognize that archaeological discovery creates moral obligation for heritage protection 
and community support. Ensure all conservation strategies prioritize indigenous rights, 
community empowerment, and cultural sovereignty while leveraging advanced technologies 
for enhanced protection effectiveness.
"""
        
        return {
            "base_prompt": base_prompt,
            "specialized_context": specialized_context,
            "specialized_instructions": specialized_instructions
        }
    
    def get_threat_monitoring_prompt(self, monitoring_data: Dict[str, Any]) -> Dict[str, str]:
        """AI-enhanced threat monitoring and early warning systems"""
        
        base_prompt = f"""
Develop advanced threat monitoring and early warning systems for archaeological site protection.

THREAT MONITORING FRAMEWORK:
Create comprehensive monitoring system using AI-enhanced satellite analysis:

1. DEFORESTATION MONITORING:
   - Real-time forest cover change detection
   - Illegal clearing alert systems with rapid response protocols
   - Deforestation front tracking and prediction modeling
   - Agricultural expansion monitoring and intervention strategies
   - Mining activity detection and impact assessment

2. INFRASTRUCTURE THREAT ASSESSMENT:
   - Road construction monitoring and access point tracking
   - Development project identification and impact prediction
   - Transportation corridor analysis and site exposure assessment
   - Utility line expansion and infrastructure encroachment
   - Settlement expansion monitoring and population pressure analysis

3. ENVIRONMENTAL CHANGE MONITORING:
   - Climate impact assessment and ecosystem health tracking
   - Hydrological change monitoring and water table fluctuations
   - Erosion and soil stability assessment with intervention triggers
   - Vegetation health analysis and ecosystem degradation indicators
   - Natural disaster impact assessment and recovery monitoring

4. HUMAN ACTIVITY SURVEILLANCE:
   - Unauthorized access detection and visitor impact monitoring
   - Looting activity identification and archaeological theft prevention
   - Illegal occupation tracking and land grabbing response
   - Tourism impact assessment and carrying capacity management
   - Community activity support and traditional use facilitation

AI-ENHANCED MONITORING CAPABILITIES:
Leverage advanced AI technologies for enhanced protection:

PREDICTIVE THREAT MODELING:
- Machine learning algorithms for threat pattern recognition
- Predictive modeling for deforestation front advancement
- Risk assessment algorithms for site vulnerability evaluation
- Early warning systems with automated alert generation
- Intervention timing optimization for maximum protection effectiveness

SATELLITE ANALYSIS INTEGRATION:
- Multi-sensor satellite data fusion for comprehensive monitoring
- Change detection algorithms for rapid threat identification
- Spectral analysis for subtle environmental change detection
- Temporal analysis for trend identification and pattern recognition
- High-resolution imagery analysis for detailed threat assessment

AUTOMATED RESPONSE SYSTEMS:
- Real-time alert generation for immediate threat response
- Automated notification to protection teams and authorities
- Rapid response protocol activation and coordination
- Documentation and evidence gathering for legal action
- Performance monitoring and system effectiveness assessment

MONITORING OUTPUTS:
1. Real-time threat monitoring dashboard with alert systems
2. Predictive threat models with intervention recommendations
3. Automated reporting systems for stakeholders and authorities
4. Evidence documentation protocols for legal protection action
5. Performance metrics and system effectiveness evaluation
6. Technology transfer protocols for broader application

Demonstrate how AI-enhanced monitoring creates unprecedented capabilities for 
archaeological site protection and heritage conservation in threatened landscapes.
"""
        
        specialized_context = """
threat_monitoring + ai_surveillance + early_warning + satellite_monitoring + 
predictive_modeling + automated_response + heritage_protection + technology_innovation
"""
        
        specialized_instructions = """
Focus on practical, implementable monitoring systems with clear response protocols.
Integrate advanced AI capabilities with community-based protection approaches.
Develop scalable technologies applicable to broader heritage protection challenges.
Ensure monitoring systems respect privacy and community protocols.
Document technological innovations for global heritage protection advancement.
"""
        
        return {
            "base_prompt": base_prompt,
            "specialized_context": specialized_context,
            "specialized_instructions": specialized_instructions
        }
    
    def get_impact_storytelling_prompt(self, impact_data: Dict[str, Any]) -> Dict[str, str]:
        """Compelling impact storytelling for conservation advocacy"""
        
        base_prompt = f"""
Develop compelling impact narratives that communicate the urgency and importance of 
Amazon archaeological heritage protection for diverse audiences.

IMPACT STORYTELLING FRAMEWORK:
Create powerful narratives that inspire action and support for heritage protection:

1. DISCOVERY IMPACT NARRATIVE:
   - Revolutionary archaeological discoveries challenging assumptions about Amazon civilizations
   - AI-enhanced methodology enabling previously impossible discoveries
   - Cultural complexity and sophisticated achievements of ancient Amazon peoples
   - Contemporary relevance and lessons for sustainable development
   - Indigenous community empowerment through archaeological collaboration

2. CONSERVATION URGENCY STORY:
   - Immediate threats to irreplaceable cultural heritage
   - Race against time to protect discoveries before destruction
   - Community leadership in heritage protection and cultural revitalization
   - Technology innovation enabling enhanced protection capabilities
   - Global implications for heritage protection and cultural preservation

3. HUMAN STORIES AND CONNECTIONS:
   - Indigenous community connections to ancestral landscapes and cultural heritage
   - Researcher dedication and collaborative partnerships
   - Community leaders fighting for heritage protection and land rights
   - Youth engagement and cultural knowledge transmission
   - Global citizen concern and support for heritage protection

4. FUTURE VISION AND HOPE:
   - Successful protection enabling ongoing research and cultural revitalization
   - Model for collaborative archaeology and community empowerment
   - Technology innovation supporting global heritage protection
   - Cultural tourism supporting community development and conservation
   - Educational impact and global awareness of Amazon cultural heritage

AUDIENCE-SPECIFIC MESSAGING:
Tailor narratives for different stakeholder groups:

SCIENTIFIC COMMUNITY:
- Methodological innovations and archaeological significance
- Research potential and scientific collaboration opportunities
- Technology advancement and global application potential
- Publication and career development opportunities
- Peer review and scientific validation requirements

POLICY MAKERS AND GOVERNMENTS:
- Legal obligations and international heritage protection standards
- Economic benefits of heritage tourism and cultural industries
- Environmental conservation and climate change mitigation connections
- International reputation and diplomatic relationship impacts
- Law enforcement and legal protection strategy requirements

INDIGENOUS COMMUNITIES:
- Cultural sovereignty and self-determination support
- Land rights and territorial protection advancement
- Cultural revitalization and knowledge transmission opportunities
- Economic development and sustainable livelihood creation
- Capacity building and leadership development support

GENERAL PUBLIC AND MEDIA:
- Amazing discoveries and revolutionary archaeological insights
- David vs. Goliath protection stories and community heroism
- Technology innovation and AI advancement applications
- Environmental destruction and heritage loss urgency
- Individual action opportunities and support mechanisms

FUNDING AND DONOR COMMUNITIES:
- Impact measurement and success demonstration
- Cost-effectiveness and leverage potential
- Sustainability and long-term outcome achievement
- Partnership and collaboration enhancement
- Recognition and visibility benefits

STORYTELLING PRODUCTS:
1. Compelling impact narratives for different audiences and platforms
2. Visual storytelling materials including maps, images, and infographics
3. Media outreach strategies and key message development
4. Social media campaigns and digital engagement strategies
5. Presentation materials for diverse stakeholder meetings
6. Funding proposals and grant application narratives

Demonstrate how powerful storytelling amplifies archaeological discovery impact and 
mobilizes support for heritage protection and community empowerment.
"""
        
        specialized_context = """
impact_storytelling + conservation_advocacy + audience_engagement + 
narrative_development + media_outreach + stakeholder_communication + 
heritage_protection + community_empowerment + global_awareness
"""
        
        specialized_instructions = """
Balance scientific accuracy with compelling narrative development for broad audience appeal.
Prioritize indigenous voices and community leadership in all storytelling approaches.
Develop culturally-sensitive narratives that respect traditional knowledge and protocols.
Create actionable engagement opportunities for diverse audiences and support levels.
Document storytelling innovations for broader heritage protection and conservation advocacy.
"""
        
        return {
            "base_prompt": base_prompt,
            "specialized_context": specialized_context,
            "specialized_instructions": specialized_instructions
        }
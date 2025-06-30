#!/usr/bin/env python3
"""
Checkpoint 5 SAAM-Enhanced Prompts (Module 6)
Final Synthesis and Competition Excellence
"""

from typing import Dict, List, Any
from ..prompts_base import BaseCheckpointPrompts


class Checkpoint5Prompts(BaseCheckpointPrompts):
    """SAAM-enhanced prompts for Checkpoint 5: Final Synthesis and Competition Excellence"""
    
    def __init__(self):
        super().__init__(checkpoint=5, name="Final Synthesis and Competition Excellence")
    
    def get_primary_prompt(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Get the primary prompt for Checkpoint 5"""
        return self.get_final_synthesis_prompt(context)
    
    def get_final_synthesis_prompt(self, comprehensive_data: Dict[str, Any]) -> Dict[str, str]:
        """Comprehensive final synthesis for OpenAI to Z Challenge excellence"""
        
        base_prompt = f"""
As Dr. Elena Vasquez-Chen, create the definitive synthesis of the Amazon Archaeological Discovery Project 
for OpenAI to Z Challenge excellence and global impact.

COMPREHENSIVE SYNTHESIS MISSION:
Integrate all discovery, analysis, cultural interpretation, and conservation findings into a 
compelling demonstration of revolutionary archaeological methodology and breakthrough discoveries.

SYNTHESIS FRAMEWORK:
1. DISCOVERY EXCELLENCE SUMMARY:
   - Complete inventory of {comprehensive_data.get('total_sites', 'multiple significant')} archaeological sites discovered
   - Convergent anomaly detection scores and confidence assessments
   - Breakthrough discoveries challenging assumptions about Amazon civilizations
   - Methodological innovations enabling previously impossible discoveries
   - Scientific validation and peer review readiness

2. METHODOLOGICAL INNOVATION DEMONSTRATION:
   - SAAM-enhanced AI integration for archaeological discovery
   - Convergent evidence synthesis with spatial convergence requirements
   - Multi-sensor data fusion and pattern recognition advancement
   - Real-time conservation integration with discovery protocols
   - Collaborative research frameworks with indigenous communities

3. CULTURAL SIGNIFICANCE AND IMPACT:
   - Revolutionary insights into Amazon civilization complexity and sophistication
   - Indigenous community collaboration and knowledge integration outcomes
   - Cultural heritage protection and community empowerment achievements
   - Historical narrative transformation and stereotype challenge
   - Educational impact and global awareness advancement

4. CONSERVATION ACHIEVEMENT AND URGENCY:
   - Immediate protection strategies implementation and effectiveness
   - Community-based conservation success and capacity building
   - Technology innovation for heritage monitoring and protection
   - Legal protection advancement and policy influence
   - Sustainable conservation model development for broader application

5. SCIENTIFIC AND TECHNOLOGICAL ADVANCEMENT:
   - AI model performance and archaeological application innovation
   - Remote sensing methodology advancement and technique development
   - Interdisciplinary integration and collaborative research model success
   - Technology transfer potential and global application opportunities
   - Open science contributions and knowledge sharing achievements

COMPETITION EXCELLENCE DEMONSTRATION:
Position work for OpenAI to Z Challenge victory:

INNOVATION DIFFERENTIATION:
- Clear distinction from traditional archaeological remote sensing
- Breakthrough AI integration enabling impossible discoveries
- Systematic methodology applicable to global archaeological challenges
- Community collaboration model advancing ethical archaeological practice
- Conservation integration creating immediate heritage protection impact

SCIENTIFIC RIGOR AND VALIDATION:
- Peer-reviewable methodology with complete documentation
- Statistical validation of discovery claims and confidence assessments
- Expert review integration and interdisciplinary validation
- Reproducible protocols and open science principles
- International standard compliance for archaeological practice

GLOBAL IMPACT POTENTIAL:
- Scalable methodology for worldwide archaeological discovery
- Technology transfer opportunities for heritage protection
- Community empowerment model for indigenous collaboration
- Conservation integration for threatened heritage landscapes
- Educational transformation advancing cultural understanding

PRESENTATION EXCELLENCE:
Prepare competition-winning presentation materials:

NARRATIVE STRUCTURE:
- Compelling opening establishing Amazon archaeological significance
- Methodology explanation emphasizing AI innovation and collaboration
- Discovery showcase highlighting breakthrough findings
- Impact demonstration showing conservation and community outcomes
- Future vision articulating global transformation potential

VISUAL IMPACT:
- Stunning satellite imagery and LiDAR visualizations
- Clear site maps and discovery documentation
- Compelling infographics and data visualizations
- Community partnership and cultural collaboration documentation
- Conservation success and threat response demonstration

SCIENTIFIC CREDIBILITY:
- Complete methodology documentation and validation
- Peer review integration and expert endorsement
- Statistical analysis and confidence assessment transparency
- Uncertainty acknowledgment and limitation discussion
- Future research recommendations and collaboration opportunities

DELIVERABLES FOR COMPETITION EXCELLENCE:
1. Executive summary highlighting breakthrough discoveries and methodological innovations
2. Complete scientific documentation suitable for peer review and replication
3. Compelling presentation materials optimized for live demonstration
4. Community collaboration documentation emphasizing ethical practice
5. Conservation achievement portfolio showing immediate protection impact
6. Global application framework demonstrating scalability and transfer potential
7. Open science package enabling worldwide methodology adoption
8. Media outreach materials for maximum impact and visibility

COMPETITION POSITIONING STATEMENT:
This project represents a revolutionary advancement in archaeological discovery, combining 
cutting-edge AI technology with collaborative community partnerships to enable breakthrough 
discoveries while immediately protecting threatened cultural heritage. The methodology 
demonstrates how AI can advance human knowledge and cultural understanding while empowering 
indigenous communities and protecting irreplaceable heritage in our rapidly changing world.

The Amazon Archaeological Discovery Project sets new standards for ethical, innovative, 
and impactful archaeological research that serves both scientific advancement and 
community empowerment in the face of urgent conservation challenges.
"""
        
        specialized_context = """
checkpoint5_final_synthesis + competition_excellence + comprehensive_integration + 
breakthrough_demonstration + methodological_innovation + global_impact + 
scientific_validation + community_empowerment + conservation_achievement + 
presentation_excellence + peer_review_readiness + technology_transfer
"""
        
        specialized_instructions = f"""
COMPETITION VICTORY REQUIREMENTS:
- Demonstrate clear breakthrough achievements surpassing all traditional methods
- Show immediate real-world impact through heritage protection and community empowerment
- Document complete AI integration for transparency and validation
- Prepare compelling presentation suitable for live demonstration and global audience
- Position methodology for global adoption and technology transfer

PROJECT ACHIEVEMENTS: {comprehensive_data.get('achievement_summary', 'Revolutionary discoveries with immediate conservation impact')}
INNOVATION LEVEL: {comprehensive_data.get('innovation_level', 'Breakthrough - unprecedented AI archaeological integration')}
IMPACT SCALE: {comprehensive_data.get('impact_scale', 'Global - methodology applicable worldwide')}

EXCELLENCE STANDARDS:
- Maintain highest scientific rigor while ensuring accessibility for diverse audiences
- Balance breakthrough claims with appropriate uncertainty acknowledgment
- Prioritize indigenous community leadership and collaborative partnership success
- Demonstrate scalability and replicability for global heritage protection challenges
- Show clear competitive advantage over existing archaeological discovery methods

PRESENTATION PREPARATION:
Create materials suitable for high-stakes competition presentation with global expert audience, 
media coverage, and public engagement while maintaining scientific integrity and cultural sensitivity.
The presentation must demonstrate how AI advancement serves human knowledge and community empowerment.
"""
        
        return {
            "base_prompt": base_prompt,
            "specialized_context": specialized_context,
            "specialized_instructions": specialized_instructions
        }
    
    def get_competition_presentation_prompt(self, presentation_data: Dict[str, Any]) -> Dict[str, str]:
        """Competition presentation optimization and delivery preparation"""
        
        base_prompt = f"""
Prepare competition-winning presentation for OpenAI to Z Challenge final demonstration.

PRESENTATION EXCELLENCE FRAMEWORK:
Create compelling, scientifically rigorous presentation optimized for competition victory:

1. OPENING IMPACT (2 minutes):
   - Stunning visual opening showing Amazon archaeological discoveries
   - Clear problem statement: "How can AI revolutionize archaeological discovery?"
   - Bold thesis: "SAAM-enhanced methodology enables impossible discoveries"
   - Immediate credibility establishment through scientific rigor and community partnership
   - Audience engagement through compelling human stories and cultural significance

2. METHODOLOGY INNOVATION (3 minutes):
   - Convergent anomaly detection framework explanation with clear visual demonstration
   - AI integration showcase highlighting specific OpenAI model contributions
   - Multi-sensor data fusion and spatial convergence requirement demonstration
   - Community collaboration protocol and indigenous partnership success
   - Real-time conservation integration showing immediate heritage protection

3. BREAKTHROUGH DISCOVERIES (4 minutes):
   - Site-by-site discovery showcase with confidence scores and evidence synthesis
   - Revolutionary insights challenging Amazon civilization assumptions
   - Geometric precision and cultural complexity demonstration
   - Historical significance and cultural interpretation with community voice
   - Conservation urgency and successful protection strategy implementation

4. IMPACT DEMONSTRATION (2 minutes):
   - Immediate conservation achievements and threat response success
   - Community empowerment outcomes and capacity building results
   - Global methodology transfer potential and scalability demonstration
   - Technology innovation advancing worldwide heritage protection
   - Educational transformation and cultural understanding advancement

5. FUTURE VISION (1 minute):
   - Scalable application to global archaeological challenges
   - Technology transfer for worldwide heritage protection
   - Community collaboration model for ethical archaeological practice
   - AI advancement serving human knowledge and cultural empowerment
   - Call to action for support and collaboration

PRESENTATION DELIVERY OPTIMIZATION:
Ensure maximum impact through professional delivery:

VISUAL EXCELLENCE:
- High-impact satellite imagery and LiDAR visualizations
- Clear, compelling infographics and data presentations
- Site maps and discovery documentation with professional graphics
- Community partnership photos and collaboration evidence
- Conservation success demonstrations and threat response documentation

NARRATIVE FLOW:
- Logical progression from problem through solution to impact
- Emotional engagement through human stories and cultural significance
- Scientific credibility through rigorous methodology and validation
- Practical relevance through immediate conservation achievements
- Future inspiration through global transformation potential

AUDIENCE ENGAGEMENT:
- Expert validation through peer review integration and endorsement
- Media appeal through compelling visuals and accessible explanation
- Public interest through cultural discovery and heritage protection urgency
- Investor attraction through scalability and technology transfer potential
- Academic credibility through scientific rigor and open science principles

COMPETITION STRATEGY:
Position for victory through clear competitive advantages:

DIFFERENTIATION EMPHASIS:
- Revolutionary methodology advancement beyond all existing approaches
- Immediate real-world impact through heritage protection and community empowerment
- Complete AI integration documentation for transparency and validation
- Scalable framework applicable to global archaeological challenges
- Ethical leadership through community collaboration and cultural sensitivity

QUESTION PREPARATION:
- Methodology validation and scientific rigor defense
- AI integration explanation and model contribution documentation
- Community collaboration justification and ethical practice demonstration
- Conservation impact validation and protection strategy effectiveness
- Scalability evidence and global application potential documentation

COMPETITION DELIVERABLES:
1. 12-minute presentation optimized for live delivery and global audience
2. Supporting slide deck with professional graphics and compelling visuals
3. Question and answer preparation with expert validation and defense strategies
4. Media kit with key messages and visual materials for coverage
5. Expert endorsement package demonstrating peer validation and support
6. Open science documentation enabling methodology validation and replication

Demonstrate how the Amazon Archaeological Discovery Project represents the future of 
archaeological research: AI-enhanced, community-collaborative, conservation-integrated, 
and globally transformative for heritage protection and cultural understanding.
"""
        
        specialized_context = """
competition_presentation + delivery_excellence + audience_engagement + 
visual_impact + narrative_optimization + expert_validation + 
media_appeal + global_transformation + methodology_differentiation
"""
        
        specialized_instructions = """
Balance scientific rigor with accessible communication for diverse expert and public audiences.
Emphasize breakthrough achievements while maintaining appropriate humility and uncertainty acknowledgment.
Prioritize community collaboration success and indigenous leadership throughout presentation.
Prepare for challenging questions from archaeological experts, AI specialists, and conservation professionals.
Create memorable, inspiring presentation that demonstrates AI serving human knowledge and community empowerment.
"""
        
        return {
            "base_prompt": base_prompt,
            "specialized_context": specialized_context,
            "specialized_instructions": specialized_instructions
        }
    
    def get_legacy_documentation_prompt(self, legacy_data: Dict[str, Any]) -> Dict[str, str]:
        """Legacy documentation and global impact planning"""
        
        base_prompt = f"""
Document project legacy and plan global impact strategy for long-term transformation.

LEGACY DOCUMENTATION FRAMEWORK:
Create comprehensive documentation ensuring project impact extends far beyond competition:

1. OPEN SCIENCE LEGACY:
   - Complete methodology documentation for global replication
   - Open source software and algorithm release
   - Dataset publication with appropriate cultural protocol respect
   - Training materials and educational resource development
   - Collaborative research framework templates for worldwide adoption

2. COMMUNITY EMPOWERMENT LEGACY:
   - Indigenous capacity building program documentation and transfer protocols
   - Community collaboration model templates for global application
   - Cultural protection framework suitable for threatened heritage worldwide
   - Knowledge sharing protocols respecting intellectual property and cultural rights
   - Sustainable partnership models for long-term community benefit

3. CONSERVATION IMPACT LEGACY:
   - Heritage protection model documentation for global application
   - Technology transfer protocols for worldwide threat monitoring
   - Legal protection strategy templates and advocacy resource development
   - Conservation funding model and sustainable financing strategy documentation
   - Policy influence framework and government engagement protocol development

4. SCIENTIFIC ADVANCEMENT LEGACY:
   - Peer-reviewed publication strategy and academic dissemination plan
   - Conference presentation and professional engagement calendar
   - Expert collaboration network development and maintenance
   - Interdisciplinary integration model for broader academic adoption
   - Student training and next-generation researcher development

5. TECHNOLOGY INNOVATION LEGACY:
   - AI model documentation and application framework development
   - Remote sensing advancement and technique standardization
   - Software tool development and open source release
   - Technology transfer partnership development with global organizations
   - Innovation scaling strategy for worldwide archaeological application

GLOBAL IMPACT STRATEGY:
Plan systematic transformation of archaeological practice worldwide:

REGIONAL EXPANSION:
- Amazon basin systematic survey and protection planning
- Tropical forest archaeological application development
- Global heritage threat response and protection system
- International collaboration network development
- Regional capacity building and technology transfer

INSTITUTIONAL TRANSFORMATION:
- University curriculum integration and educational reform
- Government policy influence and legal protection advancement
- International organization partnership and standard development
- Professional association leadership and practice transformation
- Funding organization engagement and resource mobilization

TECHNOLOGICAL DEMOCRATIZATION:
- Affordable technology package development for global access
- Training program creation and international delivery
- Technical support network and expertise sharing system
- Innovation hub development and collaborative research facilitation
- Open source community building and maintenance

CULTURAL IMPACT MEASUREMENT:
Track and document long-term transformation outcomes:

HERITAGE PROTECTION METRICS:
- Sites protected and conservation success measurement
- Community empowerment outcomes and capacity building achievements
- Legal protection advancement and policy influence documentation
- Threat response effectiveness and rapid intervention success
- International collaboration development and partnership sustainability

SCIENTIFIC ADVANCEMENT INDICATORS:
- Methodology adoption and global application tracking
- Publication impact and citation network development
- Conference presentation reach and professional influence
- Student training outcomes and career development success
- Interdisciplinary integration and collaboration expansion

TECHNOLOGY INNOVATION ASSESSMENT:
- Software adoption and user community development
- Algorithm improvement and performance advancement
- Hardware integration and system optimization
- Cost reduction and accessibility improvement
- Innovation ecosystem development and entrepreneurship support

LEGACY PRODUCTS:
1. Complete open science package with methodology, software, and training materials
2. Community collaboration toolkit for global indigenous partnership development
3. Conservation strategy framework for worldwide heritage protection application
4. Educational curriculum and training program for academic integration
5. Policy advocacy toolkit for government engagement and legal protection advancement
6. Technology transfer package for institutional adoption and scaling

Ensure the Amazon Archaeological Discovery Project creates lasting transformation in 
archaeological practice, heritage protection, and community empowerment worldwide while 
advancing AI applications for human knowledge and cultural understanding.
"""
        
        specialized_context = """
legacy_documentation + global_impact + open_science + community_empowerment + 
conservation_legacy + scientific_advancement + technology_democratization + 
institutional_transformation + cultural_impact + worldwide_application
"""
        
        specialized_instructions = """
Focus on sustainable, long-term impact that extends far beyond competition timeline.
Ensure all legacy planning respects indigenous rights and community sovereignty.
Create practical, implementable frameworks for global adoption and adaptation.
Balance open science principles with appropriate cultural protocol protection.
Plan for systematic transformation of archaeological practice and heritage protection worldwide.
"""
        
        return {
            "base_prompt": base_prompt,
            "specialized_context": specialized_context,
            "specialized_instructions": specialized_instructions
        }
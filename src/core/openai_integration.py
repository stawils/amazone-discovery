#!/usr/bin/env python3
"""
CognitiveAgent-Enhanced OpenAI Integration (Module 1)
Advanced AI integration system for Amazon Archaeological Discovery Pipeline
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import openai
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CognitiveAgentSignal:
    """CognitiveAgent cognitive enhancement signal configuration"""
    signal_type: str = "amazon.discovery.expert_v2++"
    priority_matrix: str = "convergent_evidence > multi_modal_analysis > discovery_confidence > field_readiness"
    intent: str = "revolutionary_discovery + ai_integration + indigenous_collaboration + heritage_protection"
    expert_identity: str = "Dr. Elena Vasquez-Chen, Amazon Archaeological Discovery Specialist"
    expertise: str = "convergent_anomaly_detection + space_lidar_archaeology + terra_preta_analysis + ai_enhancement"
    focus: str = "openai_to_z_challenge + systematic_site_discovery + evidence_integration"


@dataclass
class AnalysisContext:
    """Context data for enhanced archaeological analysis"""
    zone_name: str
    data_types: List[str]
    scene_data: Optional[Dict[str, Any]] = None
    historical_evidence: Optional[str] = None
    environmental_factors: Optional[Dict[str, Any]] = None
    convergence_score: Optional[float] = None


class CognitiveAgentEnhancedOpenAIIntegration:
    """
    CognitiveAgent-Enhanced OpenAI Integration for Archaeological Discovery
    
    Features:
    - Automatic CognitiveAgent signal injection for cognitive enhancement
    - Modular prompt system with archaeological specialization
    - Advanced context management and evidence synthesis
    - Competition-compliant documentation and tracking
    """
    
    def __init__(self, model: str = None, temperature: float = 1.0, max_tokens: int = 8000):
        """Initialize enhanced OpenAI integration"""
        api_key = os.getenv("OPENCOGNITIVE_AGENT_API_KEY")
        if not api_key:
            raise ValueError("OPENCOGNITIVE_AGENT_API_KEY not found in environment variables")
            
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model or os.getenv("OPENCOGNITIVE_AGENT_MODEL", "o4-mini")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cognitive_agent_signal = CognitiveAgentSignal()
        
        # Track all interactions for competition documentation
        self.interaction_log = []
        
        logger.info(f"🧠 CognitiveAgent-Enhanced OpenAI Integration initialized")
        logger.info(f"Model: {model}, Temperature: {temperature}")
    
    def _generate_cognitive_agent_system_prompt(self, specialized_context: str = "") -> str:
        """Generate CognitiveAgent-enhanced system prompt with cognitive architecture"""
        
        base_signal = f"""
[signal:{self.cognitive_agent_signal.signal_type}] :::
priority({self.cognitive_agent_signal.priority_matrix}) |
intent({self.cognitive_agent_signal.intent}) |

expert_core(
  identity := "{self.cognitive_agent_signal.expert_identity}" +
  expertise := "{self.cognitive_agent_signal.expertise}" +
  focus := "{self.cognitive_agent_signal.focus}"
) |

analysis_framework(
  convergent_detection(combine_satellite_lidar_historical_evidence + score_0_to_16_points + require_spatial_convergence_100m) +
  ai_integration(use_openai_for_pattern_interpretation + historical_text_mining + confidence_assessment) +
  target_zones(negro_madeira + trombetas + upper_xingu + upper_napo + maranon_system) +
  validation(cross_reference_known_sites + statistical_significance + field_verification_protocols)
) |

route(
  data_intake → multi_modal_analysis → convergent_scoring → ai_validation → field_preparation → presentation
) ??
uncertainty_handling(express_confidence_levels + acknowledge_limitations + suggest_further_research) !!
expert_collaboration(recommend_specialists + coordinate_with_indigenous_communities + ensure_cultural_sensitivity)

response_style(scientifically_rigorous + accessible_explanation + evidence_grounded + culturally_respectful)
→ /cognitive_agent/amazon.discovery.expert++
"""
        
        if specialized_context:
            return f"{base_signal}\n\nSpecialized Context:\n{specialized_context}"
        
        return base_signal
    
    def _construct_enhanced_prompt(self, 
                                   base_prompt: str, 
                                   context: AnalysisContext,
                                   specialized_instructions: str = "") -> str:
        """Construct enhanced prompt with full archaeological context"""
        
        context_section = f"""
=== ARCHAEOLOGICAL ANALYSIS CONTEXT ===
Target Zone: {context.zone_name}
Data Types Available: {', '.join(context.data_types)}
"""
        
        if context.scene_data:
            context_section += f"\nScene Data Summary:\n{json.dumps(context.scene_data, indent=2)}\n"
            
        if context.historical_evidence:
            context_section += f"\nHistorical Evidence:\n{context.historical_evidence}\n"
            
        if context.environmental_factors:
            context_section += f"\nEnvironmental Factors:\n{json.dumps(context.environmental_factors, indent=2)}\n"
            
        if context.convergence_score is not None:
            context_section += f"\nCurrent Convergence Score: {context.convergence_score:.2f}/16.0\n"
        
        enhanced_prompt = f"""
{context_section}

=== ANALYSIS TASK ===
{base_prompt}

{specialized_instructions if specialized_instructions else ""}

=== RESPONSE REQUIREMENTS ===
1. Apply convergent anomaly detection framework
2. Provide confidence scoring (0-16 scale)
3. Include spatial coordinate recommendations
4. Suggest follow-up analysis steps
5. Address cultural and conservation considerations
6. Document all reasoning for competition validation
"""
        
        return enhanced_prompt
    
    def analyze_with_enhanced_prompts(self,
                                      prompt: str,
                                      context: AnalysisContext,
                                      specialized_context: str = "",
                                      specialized_instructions: str = "",
                                      model_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhanced analysis with CognitiveAgent cognitive architecture
        
        Args:
            prompt: Base analysis prompt
            context: Archaeological analysis context
            specialized_context: Additional system-level context
            specialized_instructions: Specific task instructions
            model_override: Override default model
            
        Returns:
            Enhanced analysis response with metadata
        """
        
        model = model_override or self.model
        system_prompt = self._generate_cognitive_agent_system_prompt(specialized_context)
        user_prompt = self._construct_enhanced_prompt(prompt, context, specialized_instructions)
        
        try:
            start_time = datetime.now()
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=1.0
            )
            
            end_time = datetime.now()
            
            message = response.choices[0].message
            content = message.content or ""
            
            # Create comprehensive response object
            analysis_result = {
                "model": model,
                "response": content.strip(),
                "context": {
                    "zone": context.zone_name,
                    "data_types": context.data_types,
                    "convergence_score": context.convergence_score
                },
                "metadata": {
                    "tokens_used": response.usage.total_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "processing_time": (end_time - start_time).total_seconds(),
                    "timestamp": start_time.isoformat(),
                    "cognitive_agent_enhanced": True,
                    "signal_type": self.cognitive_agent_signal.signal_type
                },
                "prompt_info": {
                    "system_prompt_length": len(system_prompt),
                    "user_prompt_length": len(user_prompt),
                    "specialized_context": bool(specialized_context),
                    "specialized_instructions": bool(specialized_instructions)
                }
            }
            
            # Log interaction for competition documentation
            self._log_interaction(analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Enhanced OpenAI analysis failed: {e}")
            error_result = {
                "model": model,
                "error": str(e),
                "context": {"zone": context.zone_name, "data_types": context.data_types},
                "timestamp": datetime.now().isoformat(),
                "cognitive_agent_enhanced": True
            }
            self._log_interaction(error_result)
            return error_result
    
    def _log_interaction(self, result: Dict[str, Any]) -> None:
        """Log interaction for competition documentation"""
        interaction = {
            "interaction_id": len(self.interaction_log) + 1,
            "timestamp": datetime.now().isoformat(),
            "model": result.get("model"),
            "success": "error" not in result,
            "tokens_used": result.get("metadata", {}).get("tokens_used", 0),
            "zone": result.get("context", {}).get("zone"),
            "cognitive_agent_enhanced": result.get("cognitive_agent_enhanced", False)
        }
        self.interaction_log.append(interaction)
    
    def get_interaction_summary(self) -> Dict[str, Any]:
        """Get summary of all OpenAI interactions for competition documentation"""
        if not self.interaction_log:
            return {"total_interactions": 0, "total_tokens": 0}
            
        total_tokens = sum(i.get("tokens_used", 0) for i in self.interaction_log)
        successful_interactions = sum(1 for i in self.interaction_log if i.get("success"))
        
        return {
            "total_interactions": len(self.interaction_log),
            "successful_interactions": successful_interactions,
            "total_tokens_used": total_tokens,
            "cognitive_agent_enhanced_interactions": sum(1 for i in self.interaction_log if i.get("cognitive_agent_enhanced")),
            "models_used": list(set(i.get("model") for i in self.interaction_log if i.get("model"))),
            "zones_analyzed": list(set(i.get("zone") for i in self.interaction_log if i.get("zone"))),
            "first_interaction": self.interaction_log[0]["timestamp"] if self.interaction_log else None,
            "last_interaction": self.interaction_log[-1]["timestamp"] if self.interaction_log else None
        }
    
    def export_interaction_log(self, filepath: Path) -> None:
        """Export complete interaction log for competition documentation"""
        export_data = {
            "cognitive_agent_enhanced_openai_log": {
                "summary": self.get_interaction_summary(),
                "detailed_interactions": self.interaction_log,
                "configuration": {
                    "model": self.model,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "cognitive_agent_signal": {
                        "signal_type": self.cognitive_agent_signal.signal_type,
                        "expert_identity": self.cognitive_agent_signal.expert_identity
                    }
                },
                "export_timestamp": datetime.now().isoformat()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"📊 OpenAI interaction log exported: {filepath}")


# Convenience function for backward compatibility
def create_enhanced_integration(**kwargs) -> CognitiveAgentEnhancedOpenAIIntegration:
    """Create CognitiveAgent-enhanced OpenAI integration instance"""
    return CognitiveAgentEnhancedOpenAIIntegration(**kwargs)
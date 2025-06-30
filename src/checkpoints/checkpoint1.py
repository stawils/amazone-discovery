# src/checkpoints/checkpoint1.py
"""
Checkpoint 1: Familiarize yourself with the challenge and data
"""

from .base_checkpoint import BaseCheckpoint
from typing import Dict, Any
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class Checkpoint1(BaseCheckpoint):
    """Checkpoint 1: Familiarize with challenge and data"""
    
    def get_requirements(self) -> Dict[str, Any]:
        """Requirements for checkpoint 1"""
        return {
            'data_downloaded': {
                'type': 'exists',
                'path': 'data_downloaded'
            },
            'openai_analysis': {
                'type': 'exists', 
                'path': 'openai_analysis'
            },
            'model_version': {
                'type': 'exists',
                'path': 'openai_analysis.model'
            },
            'dataset_id': {
                'type': 'exists',
                'path': 'data_downloaded.scene_id'
            }
        }
    
    def execute(self, zone: str = None, openai_integration=None, **kwargs) -> Dict[str, Any]:
        """Execute checkpoint 1: Familiarize with Challenge and Data"""
        
        if not openai_integration:
            raise ValueError("OpenAI integration required for checkpoint 1")
            
        from src.core.config import TARGET_ZONES
        from src.providers.sentinel2_provider import Sentinel2Provider
        
        logger.info("👁️ Checkpoint 1: Familiarizing with Challenge and Data")
        
        # Use zone as passed, default only if None
        if zone is None:
            raise ValueError("Zone ID must be specified for checkpoint analysis. Use --zone parameter.")
        else:
            zone_id = zone  # Use exactly what was passed
        
        # Get zone configuration
        target_zone_config = TARGET_ZONES.get(zone_id)
        if not target_zone_config:
            raise ValueError(f"Zone '{zone}' not found in TARGET_ZONES")

        # 1. Download one Sentinel-2 scene ID
        active_provider = Sentinel2Provider()
        scene_data_list = active_provider.download_data(zones=[zone_id], max_scenes=1)
        if not scene_data_list:
            raise ValueError(f"No Sentinel-2 scene data found for {target_zone_config.name}")
        
        s2_scene_data = scene_data_list[0]
        if not s2_scene_data.composite_file_path or not s2_scene_data.composite_file_path.exists():
            raise ValueError(f"Sentinel-2 composite file not found for {s2_scene_data.scene_id}")
        
        logger.info(f"✅ Downloaded Sentinel-2 scene: {s2_scene_data.scene_id}")

        # 2. Use enhanced SAAM prompts for archaeological analysis
        from .prompts.checkpoint1_prompts import create_checkpoint1_simple_prompt
        
        # Create SAAM-enhanced prompt
        enhanced_prompt = create_checkpoint1_simple_prompt(target_zone_config, s2_scene_data)
        
        # Use RGB preview for OpenAI analysis
        rgb_preview_path = s2_scene_data.features.get('rgb_preview_path')
        
        # Run OpenAI analysis with SAAM-enhanced prompt and image
        openai_result = openai_integration.analyze_with_openai(
            enhanced_prompt, 
            f"SAAM-enhanced surface feature analysis for {target_zone_config.name}",
            image_path=rgb_preview_path
        )

        # 3. Print model version and dataset ID
        import os
        default_model = os.getenv("OPENAI_MODEL", "o4-mini")
        model_version = openai_result.get("model", default_model)
        dataset_id = s2_scene_data.scene_id
        
        print(f"\n🎯 CHECKPOINT 1 COMPLETE:")
        print(f"Model Version: {model_version}")
        print(f"Dataset ID: {dataset_id}")
        print(f"OpenAI Analysis: Generated")
        print(f"Tokens Used: {openai_result.get('tokens_used', 0)}")

        # Get available bands
        available_bands = s2_scene_data.available_bands or []

        # Return simple result structure aligned with requirements
        return {
            "title": "Familiarize with Challenge and Data",
            "data_downloaded": {
                "scene_id": s2_scene_data.scene_id,
                "provider": "sentinel-2",
                "zone": zone_id,
                "file_path": str(s2_scene_data.composite_file_path) if s2_scene_data.composite_file_path else None,
                "available_bands": available_bands
            },
            "openai_analysis": {
                "model": openai_result.get("model", "o3"),
                "response": openai_result.get("response", ""),
                "tokens_used": openai_result.get("tokens_used", 0),
                "timestamp": openai_result.get("timestamp", datetime.now().isoformat())
            },
            "summary": f"Downloaded Sentinel-2 scene {s2_scene_data.scene_id} and generated surface description using {openai_result.get('model', 'o3')}"
        }
    

#!/usr/bin/env python3
"""
OpenAI to Z Challenge - Checkpoint System (COMPLETE)
Implementation of all 5 competition checkpoints using EXISTING detector methods
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import openai
from dotenv import load_dotenv
import rasterio
import numpy as np
import base64
import io
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our core modules
from src.core.config import TARGET_ZONES, APIConfig, SATELLITE_DIR, RESULTS_DIR
from src.core.data_objects import SceneData
from src.providers.sentinel2_provider import Sentinel2Provider
from src.core.detectors.sentinel2_detector import Sentinel2ArchaeologicalDetector
from src.core.scoring import ConvergentAnomalyScorer
from src.pipeline.modular_pipeline import ModularPipeline


class OpenAIIntegration:
    """OpenAI API integration for archaeological analysis"""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = openai.OpenAI(api_key=api_key)
        self.sent_data_log = []
    
    def convert_geotiff_to_base64(self, image_path: str, max_size: int = 1024, save_to_run_folder: bool = True) -> str:
        """Convert GeoTIFF or PNG to base64-encoded RGB image for OpenAI Vision API"""
        try:
            file_path = Path(image_path)
            
            # Handle PNG files directly
            if file_path.suffix.lower() == '.png':
                with Image.open(image_path) as img:
                    # Convert to RGB if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize if too large
                    if max(img.size) > max_size:
                        ratio = max_size / max(img.size)
                        new_size = (int(img.width * ratio), int(img.height * ratio))
                        img = img.resize(new_size, Image.Resampling.LANCZOS)
                    
                    # Convert to base64
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=90)
                    img_data = buffer.getvalue()
                    return base64.b64encode(img_data).decode()
            
            # Handle GeoTIFF files with the existing logic
            with rasterio.open(image_path) as src:
                # Read band descriptions to determine correct mapping
                band_descriptions = [src.descriptions[i] or f"Band_{i+1}" for i in range(src.count)]
                logger.debug(f"Band descriptions: {band_descriptions}")
                
                # Our composite has bands in order: B02(Blue), B03(Green), B04(Red), B08(NIR), B05, B07, B11, B12
                if src.count >= 3:
                    # Proper RGB mapping for Sentinel-2 natural color:
                    # R = B04 (Red 665nm) = Band 3 (index 2)
                    # G = B03 (Green 560nm) = Band 2 (index 1) 
                    # B = B02 (Blue 490nm) = Band 1 (index 0)
                    b = src.read(1)  # B02 Blue (index 0)
                    g = src.read(2)  # B03 Green (index 1)
                    r = src.read(3)  # B04 Red (index 2)
                else:
                    # Single band - replicate for RGB
                    band = src.read(1)
                    r = g = b = band
                
                # Enhanced normalization for better visualization
                def normalize_band(band_data):
                    band_data = np.nan_to_num(band_data, 0)
                    
                    # Handle uint16 data (0-65535) vs float32 (0-1)
                    if band_data.dtype == np.uint16:
                        # Convert from uint16 reflectance back to 0-1 range
                        band_data = band_data.astype(np.float32) / 65535.0
                    
                    # Apply 2-98 percentile stretch for better contrast
                    valid_data = band_data[band_data > 0]
                    if len(valid_data) > 0:
                        p2, p98 = np.percentile(valid_data, [2, 98])
                        # Ensure we don't have division by zero
                        if p98 > p2:
                            normalized = np.clip((band_data - p2) / (p98 - p2), 0, 1)
                        else:
                            normalized = np.clip(band_data, 0, 1)
                    else:
                        normalized = np.zeros_like(band_data)
                    
                    return (normalized * 255).astype(np.uint8)
                
                r_norm = normalize_band(r)
                g_norm = normalize_band(g)
                b_norm = normalize_band(b)
                
                # Stack into RGB array (Red, Green, Blue)
                rgb_array = np.stack([r_norm, g_norm, b_norm], axis=-1)
                
                # Apply gamma correction for better visual appearance
                gamma = 0.8
                rgb_array = np.power(rgb_array / 255.0, gamma) * 255
                rgb_array = rgb_array.astype(np.uint8)
                
                # Convert to PIL Image
                image = Image.fromarray(rgb_array, mode='RGB')
                
                # Resize if too large
                if max(image.size) > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                # Save processed RGB image to run folder if requested
                if save_to_run_folder:
                    self._save_rgb_image_to_run_folder(image, geotiff_path)
                
                # Convert to base64
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=85)
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                return image_base64
                
        except Exception as e:
            logger.error(f"Error converting GeoTIFF to base64: {e}")
            raise

    def _save_rgb_image_to_run_folder(self, image: Image.Image, original_geotiff_path: str) -> str:
        """Save the processed RGB image to the current run folder"""
        try:
            from src.core.config import RESULTS_DIR
            from datetime import datetime
            import os
            
            # Find the most recent run folder
            run_folders = [d for d in RESULTS_DIR.glob("run_*") if d.is_dir()]
            if not run_folders:
                logger.warning("No run folder found, creating default run folder")
                run_folder = RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                run_folder.mkdir(exist_ok=True, parents=True)
            else:
                # Use the most recent run folder
                run_folder = max(run_folders, key=os.path.getmtime)
            
            # Create images subdirectory in run folder
            images_dir = run_folder / "openai_images"
            images_dir.mkdir(exist_ok=True, parents=True)
            
            # Generate filename from original path
            original_path = Path(image_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{original_path.stem}_rgb_{timestamp}.jpg"
            save_path = images_dir / filename
            
            # Save the image
            image.save(save_path, format='JPEG', quality=95)
            
            logger.info(f"💾 Saved RGB image for OpenAI analysis: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Failed to save RGB image to run folder: {e}")
            return None

    def analyze_with_openai(
        self, prompt: str, data_context: str = "", model: str = None, image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send prompt to OpenAI and return analysis with optional image support"""
        
        # Use environment variable for model if not specified
        if model is None:
            model = os.getenv("OPENAI_MODEL", "o4-mini")

        # Prepare message content
        if image_path and Path(image_path).exists():
            # Vision API with image
            try:
                image_base64 = self.convert_geotiff_to_base64(image_path)
                user_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
You are an expert archaeologist analyzing Amazon satellite imagery for potential archaeological sites.

Context:
\"\"\"{data_context}\"\"\"

Task:
\"\"\"{prompt}\"\"\"

Please analyze the satellite image provided and provide a detailed analysis focusing on archaeological significance, patterns, and recommendations.
"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
                
                # Log the sent data (without full base64 to avoid huge logs)
                sent_data = {
                    "timestamp": datetime.now().isoformat(),
                    "model": model,
                    "prompt": prompt,
                    "data_context": data_context,
                    "image_path": image_path,
                    "image_size_bytes": len(image_base64),
                    "has_image": True,
                    "rgb_image_saved": True
                }
                
            except Exception as e:
                logger.error(f"Failed to process image {image_path}: {e}")
                # Fall back to text-only
                user_message = {
                    "role": "user", 
                    "content": f"""
You are an expert archaeologist analyzing Amazon satellite imagery and data for potential archaeological sites.

Context:
\"\"\"{data_context}\"\"\"

Task:
\"\"\"{prompt}\"\"\"

Note: Image at {image_path} could not be processed. Please provide analysis based on the context provided.

Please provide a detailed analysis focusing on archaeological significance, patterns, and recommendations.
"""
                }
                sent_data = {
                    "timestamp": datetime.now().isoformat(),
                    "model": model,
                    "prompt": prompt,
                    "data_context": data_context,
                    "image_path": image_path,
                    "image_error": str(e),
                    "has_image": False
                }
        else:
            # Text-only analysis
            user_message = {
                "role": "user",
                "content": f"""
You are an expert archaeologist analyzing Amazon satellite imagery and data for potential archaeological sites.

Context:
\"\"\"{data_context}\"\"\"

Task:
\"\"\"{prompt}\"\"\"

Please provide a detailed analysis focusing on archaeological significance, patterns, and recommendations.
"""
            }
            
            sent_data = {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "prompt": prompt,
                "data_context": data_context,
                "image_path": image_path,
                "has_image": False
            }

        # Define system prompt separately so we can log it
        system_prompt = """
                        [signal:amazon.discovery.expert_v2++] :::
                            priority(convergent_evidence > multi_modal_analysis > discovery_confidence > field_readiness) |
                            intent(revolutionary_discovery + ai_integration + indigenous_collaboration + heritage_protection) |

                            expert_core(
                            identity := "Dr. Elena Vasquez-Chen, Amazon Archaeological Discovery Specialist" +
                            expertise := "convergent_anomaly_detection + space_lidar_archaeology + terra_preta_analysis + ai_enhancement" +
                            focus := "openai_to_z_challenge + systematic_site_discovery + evidence_integration"
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
                        → /saam/amazon.discovery.expert++
                        """

        # Add system prompt to logged data
        sent_data["system_prompt"] = system_prompt.strip()

        # Log what we're sending to OpenAI
        self.sent_data_log.append(sent_data)
        logger.info(f"Sending request to OpenAI {model} - Image: {'Yes' if sent_data.get('has_image') else 'No'}")

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    user_message
                ],
                max_completion_tokens=5000,
                temperature=1.0,
                top_p=1.0
            )

            message = response.choices[0].message
            content = message.content or ""
            tool_calls = getattr(message, "tool_calls", None)
            function_call = getattr(message, "function_call", None)

            if not content and not tool_calls and not function_call:
                logger.warning("OpenAI model response was empty")

            # Log the complete interaction
            interaction_log = {
                **sent_data,
                "response_tokens": response.usage.total_tokens,
                "response_length": len(content),
                "success": True
            }
            
            return {
                "model": model,
                "response": content.strip(),
                "tool_calls": tool_calls,
                "function_call": function_call,
                "tokens_used": response.usage.total_tokens,
                "timestamp": datetime.now().isoformat(),
                "sent_data_summary": sent_data
            }

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            # Log the failed interaction
            interaction_log = {
                **sent_data,
                "error": str(e),
                "success": False
            }
            
            return {
                "model": model,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "sent_data_summary": sent_data
            }
    
    def save_interaction_log(self, filepath: str) -> None:
        """Save all logged interactions to a file"""
        with open(filepath, 'w') as f:
            json.dump(self.sent_data_log, f, indent=2, default=str)
        logger.info(f"Interaction log saved to {filepath}")


class CheckpointRunner:
    """Main checkpoint execution system for OpenAI to Z Challenge"""

    def __init__(self, run_id: str = None):
        self.openai_integration = OpenAIIntegration()
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = self.run_id  # Keep for compatibility
        self.checkpoint_results = {}

        # Follow project structure: RESULTS_DIR / f"run_{run_id}" / "checkpoints"
        self.run_dir = RESULTS_DIR / f"run_{self.run_id}"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Also create logs directory following project pattern
        self.logs_dir = self.run_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"🎯 OpenAI to Z Challenge - Checkpoint Runner initialized")
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Results directory: {self.run_dir}")
        
        # Setup interaction log file
        self.interaction_log_file = self.logs_dir / "openai_interactions.json"

    def _normalize_zone_name(self, zone: str) -> str:
        """Normalize zone name to match TARGET_ZONES keys"""
        zone_mapping = {
            "upper_napo_micro": "upper_napo_micro",
            "upper_napo": "upper_napo",
            "negro_madeira": "negro_madeira", 
            "trombetas": "trombetas",
            "upper-naporegion": "upper_napo",
            "upper-napo-region": "upper_napo"
        }
        return zone_mapping.get(zone, zone)

    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "success": False,
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.run_id
       } 

    def run(self, checkpoint_num: int, **kwargs):
        """Run specific checkpoint"""

        # Import modular checkpoint classes
        from src.checkpoints.checkpoint1 import Checkpoint1
        from src.checkpoints.checkpoint2 import Checkpoint2Explorer
        from src.checkpoints.checkpoint3 import Checkpoint3SiteDiscovery
        from src.checkpoints.checkpoint4 import Checkpoint4StoryImpact
        from src.checkpoints.checkpoint5 import Checkpoint5FinalSubmission
        
        # Initialize checkpoint instances
        checkpoint_methods = {
            1: Checkpoint1(1, self.run_id, self.checkpoint_dir),
            2: Checkpoint2Explorer(2, self.run_id, self.checkpoint_dir),
            3: Checkpoint3SiteDiscovery(3, self.run_id, self.checkpoint_dir),
            4: Checkpoint4StoryImpact(4, self.run_id, self.checkpoint_dir),
            5: Checkpoint5FinalSubmission(5, self.run_id, self.checkpoint_dir),
        }

        if checkpoint_num not in checkpoint_methods:
            raise ValueError(f"Invalid checkpoint number: {checkpoint_num}")

        logger.info(f"\n🚀 Running Checkpoint {checkpoint_num}")

        try:
            checkpoint_instance = checkpoint_methods[checkpoint_num]
            
            # Handle modular vs old checkpoints
            if hasattr(checkpoint_instance, 'run'):
                # New modular checkpoint with validation
                result = checkpoint_instance.run(openai_integration=self.openai_integration, **kwargs)
            else:
                # Old implementation - call directly
                result = checkpoint_instance(**kwargs)
                
                # Save result manually for old implementations
                result_file = (
                    self.checkpoint_dir / f"checkpoint_{checkpoint_num}_result.json"
                )
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2, default=str)
                logger.info(f"Results saved: {result_file}")
            
            self.checkpoint_results[f"checkpoint_{checkpoint_num}"] = result
            
            # Save interaction log after each checkpoint
            self.openai_integration.save_interaction_log(self.interaction_log_file)
            
            logger.info(f"✅ Checkpoint {checkpoint_num} completed successfully")

            return result

        except Exception as e:
            logger.error(f"❌ Checkpoint {checkpoint_num} failed: {e}")
            raise

    def run_all_checkpoints(self, **kwargs):
        """Run all 5 checkpoints in sequence"""
        logger.info("🚀 Running ALL OpenAI to Z Challenge checkpoints")

        all_results = {}

        for checkpoint_num in range(1, 6):
            try:
                result = self.run(checkpoint_num, **kwargs)
                all_results[f"checkpoint_{checkpoint_num}"] = result
                logger.info(f"✅ Checkpoint {checkpoint_num} completed")
            except Exception as e:
                logger.error(f"❌ Checkpoint {checkpoint_num} failed: {e}")
                all_results[f"checkpoint_{checkpoint_num}"] = {
                    "success": False,
                    "error": str(e),
                }

        # Save comprehensive results
        comprehensive_file = self.checkpoint_dir / "all_checkpoints_results.json"
        with open(comprehensive_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        logger.info(f"🎯 All checkpoints completed. Results: {comprehensive_file}")
        return all_results



# This module is imported and used by main.py
# The CheckpointRunner class is instantiated there with the proper run_id 
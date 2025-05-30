from typing import Dict, List
from pathlib import Path
from src.visualizers import ArchaeologicalVisualizer
from src.config import MAPS_DIR
import logging

logger = logging.getLogger(__name__)

class VisualizationStep:
    """
    Modular visualization step for archaeological pipeline.
    Creates an interactive map of all discoveries.
    """
    def run(self, analysis_results: Dict[str, List[dict]], scoring_results: Dict[str, dict]) -> Path:
        """
        Create an interactive map of all discoveries.
        Args:
            analysis_results: Dictionary mapping zone_id to list of analysis result dicts.
            scoring_results: Dictionary mapping zone_id to scoring result dicts.
        Returns:
            Path to the generated map file.
        """
        logger.info("🗺️ Creating interactive discovery map...")
        try:
            visualizer = ArchaeologicalVisualizer()
            map_path = visualizer.create_discovery_map(
                analysis_results,
                scoring_results,
                MAPS_DIR / f"archaeological_discoveries_modular.html"
            )
            logger.info(f"✓ Interactive map created: {map_path}")
            return map_path
        except Exception as e:
            logger.error(f"Error creating map: {e}")
            return None 
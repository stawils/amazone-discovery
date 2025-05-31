from typing import List, Dict
from pathlib import Path
from src.core.data_objects import SceneData
from src.core.detectors.gee_detectors import ArchaeologicalDetector
from src.core.config import TARGET_ZONES, EXPORTS_DIR
import logging

logger = logging.getLogger(__name__)

REQUIRED_BANDS = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']

class AnalysisStep:
    """
    Modular analysis step for archaeological pipeline.
    Analyzes a list of SceneData objects and returns results grouped by zone_id.
    Skips scenes missing required bands.
    """
    def run(self, scene_data_list: List[SceneData]) -> Dict[str, List[dict]]:
        """
        Analyze each SceneData object and return results grouped by zone_id.
        Args:
            scene_data_list: List of SceneData objects to analyze.
        Returns:
            Dictionary mapping zone_id to list of analysis result dicts.
        """
        analysis_results: Dict[str, List[dict]] = {}
        for scene in scene_data_list:
            # Feature/band awareness: skip scenes missing required bands
            missing_bands = [b for b in REQUIRED_BANDS if not scene.has_band(b)]
            if missing_bands:
                logger.warning(f"Skipping scene {scene.scene_id} (zone {scene.zone_id}): missing bands {missing_bands}")
                continue
            zone_id = scene.zone_id
            zone = TARGET_ZONES.get(zone_id, None)
            detector = ArchaeologicalDetector(zone)
            scene_dir = None
            if scene.file_paths:
                for path in scene.file_paths.values():
                    if hasattr(path, 'is_dir') and path.is_dir():
                        scene_dir = path
                        break
                    elif hasattr(path, 'parent'):
                        scene_dir = path.parent
                        break
            if scene_dir and scene_dir.exists():
                logger.info(f"  Analyzing scene: {scene_dir}")
                try:
                    result = detector.analyze_scene(scene_dir)
                    if result.get('success'):
                        if zone_id not in analysis_results:
                            analysis_results[zone_id] = []
                        analysis_results[zone_id].append(result)
                        export_path = EXPORTS_DIR / f"{zone_id}_{scene_dir.name}_detections.geojson"
                        detector.export_detections_to_geojson(export_path)
                        logger.info(f"  ✓ Found {result['total_features']} features")
                    else:
                        logger.warning(f"  ❌ Analysis failed: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"  ❌ Error analyzing {scene_dir}: {e}")
            else:
                logger.warning(f"Scene directory not found or invalid for SceneData: {scene}")
        return analysis_results 
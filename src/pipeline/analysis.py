from typing import List, Dict
from pathlib import Path
from src.core.data_objects import SceneData
from src.core.detectors.gee_detectors import ArchaeologicalDetector
from src.core.detectors.sentinel2_detector import Sentinel2ArchaeologicalDetector
from src.core.config import TARGET_ZONES, EXPORTS_DIR
import logging

logger = logging.getLogger(__name__)

REQUIRED_BANDS = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']  # Landsat SR bands

class AnalysisStep:
    """
    Modular analysis step for archaeological pipeline.
    Now supports both GEE and Sentinel-2 data analysis.
    """
    def run(self, scene_data_list: List[SceneData]) -> Dict[str, List[dict]]:
        """
        Analyze each SceneData object using appropriate detector based on provider.
        """
        analysis_results: Dict[str, List[dict]] = {}
        
        for scene in scene_data_list:
            zone_id = scene.zone_id
            zone = TARGET_ZONES.get(zone_id, None)
            
            # Choose appropriate detector based on provider
            if scene.provider == 'sentinel-2':
                detector = Sentinel2ArchaeologicalDetector(zone)
                required_bands = ['B02', 'B03', 'B04', 'B08']  # Sentinel-2 band names
            else:  # GEE or other
                detector = ArchaeologicalDetector(zone)
                required_bands = ['blue', 'green', 'red', 'nir']  # GEE band names
            
            # Check for required bands
            missing_bands = [b for b in required_bands if not scene.has_band(b)]
            if missing_bands:
                logger.warning(f"Skipping scene {scene.scene_id}: missing bands {missing_bands}")
                continue
            
            # Find scene directory
            scene_dir = self._get_scene_directory(scene)
            
            if scene_dir and scene_dir.exists():
                logger.info(f"  Analyzing {scene.provider} scene: {scene_dir}")
                try:
                    result = detector.analyze_scene(scene_dir)
                    if result.get('success'):
                        if zone_id not in analysis_results:
                            analysis_results[zone_id] = []
                        analysis_results[zone_id].append(result)
                        
                        # Export detections
                        export_path = EXPORTS_DIR / f"{zone_id}_{scene_dir.name}_detections.geojson"
                        detector.export_detections_to_geojson(export_path)
                        
                        logger.info(f"  ✓ Found {result['total_features']} features")
                    else:
                        logger.warning(f"  ❌ Analysis failed: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.error(f"  ❌ Error analyzing {scene_dir}: {e}")
            else:
                logger.warning(f"Scene directory not found for: {scene}")
        
        return analysis_results
    
    def _get_scene_directory(self, scene: SceneData) -> Path:
        """Extract scene directory from SceneData object"""
        if scene.file_paths:
            for path in scene.file_paths.values():
                if hasattr(path, 'is_dir') and path.is_dir():
                    return path
                elif hasattr(path, 'parent'):
                    return path.parent
        
        # Fallback: use metadata directory
        if 'scene_directory' in scene.metadata:
            return Path(scene.metadata['scene_directory'])
        
        return None 
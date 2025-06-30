from typing import List, Dict
from pathlib import Path
from src.core.data_objects import SceneData
from src.core.detectors.sentinel2_detector import Sentinel2ArchaeologicalDetector
from src.core.detectors.gedi_detector import GEDIArchaeologicalDetector
from src.core.config import TARGET_ZONES, EXPORTS_DIR, RESULTS_DIR
import logging

logger = logging.getLogger(__name__)

REQUIRED_BANDS = [
    "SR_B2",
    "SR_B3",
    "SR_B4",
    "SR_B5",
    "SR_B6",
    "SR_B7",
]  # Landsat SR bands


class AnalysisStep:
    """
    Modular analysis step for archaeological pipeline.
    Now supports Sentinel-2 and GEDI data analysis.
    """
    
    def __init__(self, run_id=None):
        self.run_id = run_id
    
    def _get_zone_with_fallback(self, zone_id: str):
        """Get zone with proper fallback handling - NEW METHOD for GEDI."""
        
        # Try direct lookup first (existing behavior - don't change)
        zone = TARGET_ZONES.get(zone_id, None)
        if zone is not None:
            return zone
        
        # NEW: Try normalized zone ID (only for GEDI cases)
        normalized_id = self._normalize_zone_id(zone_id)
        zone = TARGET_ZONES.get(normalized_id, None)
        if zone is not None:
            return zone
            
        # NEW: Create default zone for unknown cases
        logger.warning(f"Zone not found for ID: {zone_id}, creating default zone")
        from src.core.config import TargetZone
        return TargetZone(
            id=zone_id,
            name=zone_id.replace("_", " ").title(),
            center=(0.0, 0.0),
            bbox=(-1.0, -73.0, 0.0, -72.0),
            priority=3,
            expected_features="Unknown",
            historical_evidence="Unknown"
        )

    def _normalize_zone_id(self, zone_id: str) -> str:
        """Normalize zone IDs - NEW METHOD for GEDI zone name fixes."""
        zone_mapping = {
            "upper-naporegion": "upper_napo",
            "upper_naporegion": "upper_napo", 
            "uppernaporegion": "upper_napo",
            "upper-napo": "upper_napo"
        }
        
        normalized = zone_id.lower().replace(" ", "_").replace("-", "_")
        return zone_mapping.get(normalized, normalized)

    def run(self, scene_data_list: List[SceneData]) -> Dict[str, List[dict]]:
        """
        Analyze each SceneData object using appropriate detector based on provider.
        """
        analysis_results: Dict[str, List[dict]] = {}
        detector = None # Initialize detector to None

        for scene in scene_data_list:
            zone_id = scene.zone_id
            zone = self._get_zone_with_fallback(zone_id)

            # Choose appropriate detector based on provider
            if scene.provider == "sentinel2":
                detector = Sentinel2ArchaeologicalDetector(zone, run_id=self.run_id)
                required_bands = ["B02", "B03", "B04", "B08"]  # Sentinel-2 band names
            elif scene.provider == "gedi":
                if zone is None:
                    logger.warning(f"Zone not found for ID: {zone_id}, using default zone for GEDI detector")
                    from src.core.config import TargetZone
                    zone = TargetZone(
                        name=zone_id,
                        center=(0.0, 0.0),
                        bbox=(-1.0, -73.0, 0.0, -72.0),
                        priority=3,
                        expected_features="Unknown",
                        historical_evidence="Unknown"
                    )
                detector = GEDIArchaeologicalDetector(zone, run_id=self.run_id)
                required_bands = []
            else:
                logger.warning(
                    f"Skipping scene {scene.scene_id} from unknown or unsupported provider: {scene.provider}. "
                    "No detector available."
                )
                detector = None # Ensure detector is None for this iteration
                continue # Skip to the next scene

            if detector is None: # Should not happen if logic above is correct, but as a safeguard
                 logger.error(f"Detector not initialized for scene {scene.scene_id} with provider {scene.provider}. Skipping analysis for this scene.")
                 continue

            # Check for required bands (skip for GEDI point clouds)
            if required_bands:
                missing_bands = [b for b in required_bands if not scene.has_band(b)]
                if missing_bands:
                    logger.warning(
                        f"Skipping scene {scene.scene_id}: missing bands {missing_bands} for provider {scene.provider}"
                    )
                    continue

            result = None
            if isinstance(detector, Sentinel2ArchaeologicalDetector):
                logger.info(f"  Analyzing Sentinel-2 scene data object for: {scene.scene_id}")
                try:
                    # Sentinel2Detector's detect_features_from_scene handles its own path logic from SceneData
                    result = detector.detect_features_from_scene(scene)
                except Exception as e:
                    logger.error(f"  ❌ Error analyzing scene data {scene.scene_id} with Sentinel2Detector: {e}", exc_info=True)
            else:
                # For GEDI detector, use the correct metrics directory from SceneData
                if isinstance(detector, GEDIArchaeologicalDetector):
                    # GEDI stores metrics in file_paths['processed_metrics_file']
                    metrics_file = scene.file_paths.get('processed_metrics_file')
                    if metrics_file and metrics_file.exists():
                        # Pass the directory containing the metrics file to the detector
                        metrics_dir = metrics_file.parent
                        logger.info(f"  Analyzing GEDI metrics directory: {metrics_dir} for scene {scene.scene_id}")
                        try:
                            result = detector.analyze_scene(metrics_dir)
                        except Exception as e:
                            logger.error(f"  ❌ Error analyzing GEDI metrics directory {metrics_dir} for scene {scene.scene_id}: {e}", exc_info=True)
                    else:
                        logger.warning(f"GEDI metrics file not found for scene {scene.scene_id}. Cannot analyze.")
                else:
                    # For other detectors, continue using _get_scene_directory approach
                    scene_dir = self._get_scene_directory(scene)
                    if scene_dir and scene_dir.exists():
                        logger.info(f"  Analyzing {scene.provider} scene directory: {scene_dir} for scene {scene.scene_id}")
                        try:
                            result = detector.analyze_scene(scene_dir) # Assumes other detectors take a dir path
                        except Exception as e:
                            logger.error(f"  ❌ Error analyzing directory {scene_dir} for scene {scene.scene_id}: {e}", exc_info=True)
                    else:
                        logger.warning(f"Scene directory not found or does not exist for: {scene.scene_id}. Cannot analyze.")

            # Common result processing logic
            if result and result.get("success"):
                if scene.zone_id not in analysis_results:
                    analysis_results[scene.zone_id] = []
                analysis_results[scene.zone_id].append(result)

                # Export detections - this might need adjustment based on what 'result' contains
                # and if export_detections_to_geojson is still relevant or handled by detect_features_from_scene cache.
                # For now, keeping it conditional and assuming detector might have populated detection_results.
                if hasattr(detector, 'export_detections_to_geojson'):
                    # Try to form a reasonable export path. The scene_dir might not be defined if Sentinel2 path was taken.
                    # We need a consistent way to define export paths based on scene_id and zone_id.
                    current_scene_id_for_export = scene.scene_id if scene.scene_id else "unknown_scene"
                    export_filename = f"{scene.zone_id}_{current_scene_id_for_export}_detections.geojson"
                    # Use unified export structure
                    if self.run_id:
                        run_specific_exports_dir = RESULTS_DIR / f"run_{self.run_id}" / "exports" / scene.provider
                    else:
                        run_specific_exports_dir = EXPORTS_DIR / scene.provider  # Fallback to global
                    export_path = run_specific_exports_dir / export_filename
                    try:
                        # Ensure exports directory exists
                        run_specific_exports_dir.mkdir(parents=True, exist_ok=True)
                        detector.export_detections_to_geojson(export_path)
                        logger.info(f"   Geospatial detections exported to {export_path}")
                    except Exception as e:
                        logger.error(f"Failed to export detections for {current_scene_id_for_export} to {export_path}: {e}", exc_info=True)
                
                total_features = result.get('total_features', 0) # S2 returns this in detect_features_from_scene summary
                if not total_features and isinstance(result.get('detection_summary'), dict): # GEDI might have it nested or named differently
                    summary = result.get('detection_summary', {})
                    if 'total_features' in summary: 
                        total_features = summary['total_features']
                    elif 'feature_counts' in summary: # S2 specific alternative in its summary
                        feature_counts = summary.get('feature_counts', {})
                        total_features = sum(feature_counts.values()) if isinstance(feature_counts, dict) else 0
                    else:
                        # Sum individual count fields for Sentinel-2 archaeological features
                        # Exclude non-feature counts like bands_loaded_count
                        feature_types = ['terra_preta_analysis_count', 'geometric_feature_analysis_count', 'crop_mark_analysis_count']
                        total_features = sum(summary.get(field, 0) for field in feature_types)
                
                # If still no total found, try summing counts from the main result object
                if not total_features:
                    # Check for individual analysis results with counts
                    analysis_keys = ['terra_preta_analysis', 'geometric_feature_analysis', 'crop_mark_analysis']
                    for key in analysis_keys:
                        if key in result and isinstance(result[key], dict):
                            total_features += result[key].get('count', 0)
                
                logger.info(f"  ✓ Found {total_features} features for scene {scene.scene_id}")
            elif result: # If result exists but not success
                logger.warning(
                    f"  ❌ Analysis failed for scene {scene.scene_id}: {result.get('status', result.get('error', 'Unknown error'))}"
                )
            # If result is None (e.g. scene_dir not found for non-S2), already logged.

        return analysis_results

    def _get_scene_directory(self, scene: SceneData) -> Path:
        """Extract scene directory from SceneData object"""
        if scene.file_paths:
            for path in scene.file_paths.values():
                if hasattr(path, "is_dir") and path.is_dir(): # If a path itself is the directory
                    return path
                elif hasattr(path, "parent"): # If it's a file, its parent is the directory
                    # Check if the first file_path's parent seems like a scene specific dir.
                    # This logic assumes file_paths are within a common scene directory.
                    # e.g. /path/to/scene_id/band1.tif, /path/to/scene_id/band2.tif
                    # A more robust way might be to find the common parent of all file_paths,
                    # or rely on SceneData having a dedicated scene_directory attribute.
                    # For now, taking parent of the first file path is a common case.
                    return path.parent 

        # Fallback: use metadata directory if explicitly set
        if "scene_directory" in scene.metadata: # Legacy or explicit override
            scene_dir_meta = scene.metadata["scene_directory"]
            if scene_dir_meta: # Ensure it's not None or empty
                return Path(scene_dir_meta)

        logger.warning(f"Could not determine scene directory for scene {scene.scene_id} from file_paths or metadata.")
        # Attempt to use composite_file_path's parent if it exists, as a last resort
        if scene.composite_file_path and scene.composite_file_path.exists():
            logger.info(f"Trying parent of composite_file_path for scene {scene.scene_id}")
            return scene.composite_file_path.parent
            
        return None

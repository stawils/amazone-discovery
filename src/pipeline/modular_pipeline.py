from typing import List, Dict, Optional, Any, Union, Tuple
from pathlib import Path
from datetime import datetime
from src.core.data_objects import SceneData, BaseProvider
from src.core.config import TARGET_ZONES, RESULTS_DIR
from src.core.scoring import ConvergentAnomalyScorer
# Enhanced modular visualization system (ONLY)
from src.visualization import ArchaeologicalMapGenerator
from src.pipeline.export_manager import UnifiedExportManager
from src.pipeline.analysis import AnalysisStep
from src.pipeline.report import ReportStep, get_results_directory
from src.providers.sentinel2_provider import Sentinel2Provider
from src.providers.gedi_provider import GEDIProvider
import logging
import json
import numpy as np

logger = logging.getLogger(__name__)


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy arrays and other numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def _deep_serialize_paths(data: Any) -> Any:
    """
    Recursively traverses a data structure and converts Path objects to strings.
    """
    if isinstance(data, Path):
        return str(data)
    elif isinstance(data, dict):
        return {key: _deep_serialize_paths(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_deep_serialize_paths(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(_deep_serialize_paths(item) for item in data)
    # Add other iterable types if necessary, e.g., set
    # elif isinstance(data, set):
    #     return {_deep_serialize_paths(item) for item in data}
    return data


class ModularPipeline:
    """
    Orchestrates the modular archaeological discovery pipeline steps.
    """

    def __init__(self, provider_instance, run_id: str, total_providers: int = 1):
        """
        Initialize the pipeline with a specific data provider and a run_id.
        Args:
            provider_instance: An instance of a data provider (e.g., GEDIProvider).
            run_id: The unique identifier for this pipeline run.
            total_providers: Total number of providers that will run (for cross-provider coordination).
        """
        self.provider_instance = provider_instance
        self.run_id = run_id
        self.total_providers = total_providers
        self.data_fetcher = provider_instance
        self.processor = provider_instance
        self.analyzer = provider_instance
        self.analysis_step = AnalysisStep(run_id=run_id)
        self.core_scorer = ConvergentAnomalyScorer()
        self.report_step = ReportStep(run_id=self.run_id)
        # Enhanced modular visualization system (ONLY)
        self.visualizer = ArchaeologicalMapGenerator(run_id=run_id, results_dir=RESULTS_DIR)
        
        # Initialize unified export manager
        self.export_manager = UnifiedExportManager(run_id=run_id, results_dir=RESULTS_DIR)
        
        # Determine provider name for directory organization early
        class_name = self.provider_instance.__class__.__name__
        self.provider_name_for_paths = class_name.lower()
        if self.provider_name_for_paths.endswith('provider'):
            self.provider_name_for_paths = self.provider_name_for_paths[:-8]
        self.base_run_provider_dir = RESULTS_DIR / f"run_{self.run_id}" / self.provider_name_for_paths
        self.base_run_provider_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ModularPipeline initialized for provider: {self.provider_name_for_paths}, Run ID: {self.run_id}. Base dir: {self.base_run_provider_dir}")

    def _get_staged_data_path(self, stage_output_name: str) -> Path:
        """Helper to get the standardized path for a stage's JSON output file."""
        return self.base_run_provider_dir / f"{stage_output_name}.json"
    
    def _cross_provider_analysis_enabled(self) -> bool:
        """Check if cross-provider analysis is enabled (multiple providers running)"""
        # Use the total_providers count to determine if multiple providers will run
        return self.total_providers > 1

    def _serialize_scenedata_list(self, scene_data_list: List[SceneData]) -> List[Dict[str, Any]]:
        """Serializes a list of SceneData objects to a list of dictionaries, ensuring deep path serialization."""
        serialized_list = []
        for sd in scene_data_list:
            # Basic serialization using vars() as SceneData does not have a custom to_dict()
            # that we need to preserve.
            data_dict = vars(sd).copy()
            
            # Perform deep serialization to convert all Path objects, even nested ones.
            serialized_item = _deep_serialize_paths(data_dict)
            serialized_list.append(serialized_item)
        return serialized_list

    def _deserialize_scenedata_list(self, data_list: List[Dict[str, Any]]) -> List[SceneData]:
        """Deserializes a list of dictionaries back into SceneData objects."""
        # This is a placeholder. Actual deserialization would require knowing
        # SceneData's structure and potentially a from_dict class method.
        # For now, we'll assume data is simple enough or this will be expanded.
        deserialized_list = []
        for data_dict in data_list:
            sd = SceneData(**data_dict) # This is a simplistic assumption
            # Convert path strings back to Path objects
            for key, value in vars(sd).items():
                if isinstance(value, str) and ('path' in key or 'dir' in key):
                     try:
                        setattr(sd, key, Path(value))
                     except TypeError: # Value might be None
                        pass 
            deserialized_list.append(sd)
        return deserialized_list

    def acquire_data(self, zones: Optional[List[str]], max_scenes: int) -> List[SceneData]:
        """Stage 1: Downloads raw data for the specified provider and zones."""
        logger.info(f"[Stage 1: Acquire Data] Provider: {self.provider_name_for_paths}, Zones: {zones}, Max Scenes: {max_scenes}")
        
        scene_data_list = self.data_fetcher.download_data(zones, max_scenes)
        logger.info(f"[Stage 1: Acquire Data] Downloaded {len(scene_data_list)} scenes/items.")
        
        # Serialize SceneData for saving
        # This requires SceneData to be JSON serializable or have a to_dict method.
        # Path objects within SceneData need to be converted to strings.
        serializable_scene_data = self._serialize_scenedata_list(scene_data_list)

        output_path = self._get_staged_data_path("acquired_scene_data")
        try:
            # Ensure the directory exists before writing
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(serializable_scene_data, f, indent=2, cls=NumpyJSONEncoder)
            logger.info(f"[Stage 1: Acquire Data] Saved acquired scene data to {output_path}")
        except Exception as e:
            logger.error(f"[Stage 1: Acquire Data] Failed to save scene data: {e}", exc_info=True)
            # Decide if we should raise error or return potentially unsaved list

        return scene_data_list # Return the original objects for in-memory chaining

    def analyze_scenes(self, scene_data_input: Union[List[SceneData], Path]) -> Dict[str, List[Dict[str, Any]]]:
        """Stage 2: Processes the acquired data using appropriate detectors."""
        logger.info(f"[Stage 2: Analyze Scenes] Provider: {self.provider_name_for_paths}")

        scene_data_list_to_process: List[SceneData]
        if isinstance(scene_data_input, Path):
            input_path = scene_data_input
            if not input_path.is_file(): # If just a name is given, assume it's in the run dir
                input_path = self._get_staged_data_path(str(scene_data_input))
            
            logger.info(f"[Stage 2: Analyze Scenes] Loading scene data from {input_path}")
            try:
                with open(input_path, 'r') as f:
                    loaded_serializable_data = json.load(f)
                scene_data_list_to_process = self._deserialize_scenedata_list(loaded_serializable_data)
                logger.info(f"[Stage 2: Analyze Scenes] Loaded {len(scene_data_list_to_process)} scenes for analysis.")
            except Exception as e:
                logger.error(f"[Stage 2: Analyze Scenes] Failed to load scene data from {input_path}: {e}", exc_info=True)
                return {} # Return empty if loading failed
        elif isinstance(scene_data_input, list):
            scene_data_list_to_process = scene_data_input
            logger.info(f"[Stage 2: Analyze Scenes] Using {len(scene_data_list_to_process)} scenes passed in memory.")
        else:
            logger.error("[Stage 2: Analyze Scenes] Invalid input type for scene_data_input.")
            return {}

        if not scene_data_list_to_process:
            logger.warning("[Stage 2: Analyze Scenes] No scene data to process.")
            return {}

        # The AnalysisStep().run method is expected to take List[SceneData]
        # and return Dict[str, List[Dict[str, Any]]] (analysis_results per zone)
        analysis_results = self.analysis_step.run(scene_data_list_to_process)
        logger.info(f"[Stage 2: Analyze Scenes] Analysis complete for {len(analysis_results)} zones.")

        output_path = self._get_staged_data_path("analysis_results")
        try:
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(analysis_results, f, indent=2, cls=NumpyJSONEncoder)
            logger.info(f"[Stage 2: Analyze Scenes] Saved analysis results to {output_path}")
        except Exception as e:
            logger.error(f"[Stage 2: Analyze Scenes] Failed to save analysis results: {e}", exc_info=True)

        return analysis_results

    def score_zones(self, analysis_results_input: Union[Dict[str, List[Dict[str, Any]]], Path]) -> Dict[str, Dict[str, Any]]:
        """Stage 3: Aggregates features from analysis results and scores zones."""
        logger.info(f"[Stage 3: Score Zones] Provider: {self.provider_name_for_paths}")

        analysis_results: Dict[str, List[Dict[str, Any]]]
        if isinstance(analysis_results_input, Path):
            input_path = analysis_results_input
            if not input_path.is_file(): # If just a name is given, assume it's in the run dir
                input_path = self._get_staged_data_path(str(analysis_results_input))
            
            logger.info(f"[Stage 3: Score Zones] Loading analysis results from {input_path}")
            try:
                with open(input_path, 'r') as f:
                    analysis_results = json.load(f)
                logger.info(f"[Stage 3: Score Zones] Loaded analysis results for {len(analysis_results)} zones.")
            except Exception as e:
                logger.error(f"[Stage 3: Score Zones] Failed to load analysis results from {input_path}: {e}", exc_info=True)
                return {}
        elif isinstance(analysis_results_input, dict):
            analysis_results = analysis_results_input
            logger.info(f"[Stage 3: Score Zones] Using analysis results for {len(analysis_results)} zones passed in memory.")
        else:
            logger.error("[Stage 3: Score Zones] Invalid input type for analysis_results_input.")
            return {}

        if not analysis_results:
            logger.warning("[Stage 3: Score Zones] No analysis results to score.")
            return {}

        logger.info("[Stage 3: Score Zones] Calculating convergent anomaly scores using core scorer...")
        scoring_results: Dict[str, Dict[str, Any]] = {}

        for zone_id, individual_analyses_for_zone in analysis_results.items():
            logger.info(f"  [Stage 3: Score Zones] Aggregating features and scoring zone: {zone_id}")
            
            # This aggregation logic is similar to the original run method
            aggregated_features_for_zone: Dict[str, List[Dict[str, Any]]] = {
                "terra_preta_patches": [],
                "geometric_features": [],
                "crop_marks": [],
                # Add other feature types if your CoreScorer expects them
            }

            if not individual_analyses_for_zone:
                logger.warning(f"  [Stage 3: Score Zones] No individual analyses found for zone {zone_id}. Skipping scoring.")
                scoring_results[zone_id] = self._create_scoring_error_result(zone_id, "No analysis data for zone")
                continue

            for analysis_output in individual_analyses_for_zone:
                if not analysis_output or not analysis_output.get('success', False):
                    logger.debug(f"  [Stage 3: Score Zones] Skipping unsuccessful or empty analysis item for zone {zone_id}")
                    continue

                # Extract real coordinates from detector GeoJSON outputs
                tp_analysis = analysis_output.get("terra_preta_analysis", {})
                if tp_analysis and tp_analysis.get("geojson_path"):
                    # Load real terra preta features from detector GeoJSON
                    try:
                        with open(tp_analysis["geojson_path"], 'r') as f:
                            tp_geojson = json.load(f)
                        
                        for feature in tp_geojson.get("features", []):
                            properties = feature.get("properties", {})
                            geometry = feature.get("geometry", {})
                            
                            # Use real coordinates from detector
                            real_coords = None
                            if "coordinates" in properties:
                                real_coords = properties["coordinates"]
                            elif geometry.get("type") == "Point" and "coordinates" in geometry:
                                real_coords = geometry["coordinates"]
                            
                            if real_coords and len(real_coords) == 2:
                                # Validate coordinates are in Amazon bounds
                                lon, lat = real_coords[0], real_coords[1]
                                if -80 <= lon <= -44 and -20 <= lat <= 10:
                                    aggregated_features_for_zone["terra_preta_patches"].append({
                                        "type": "terra_preta",
                                        "confidence": properties.get("confidence", 0.85),
                                        "area_m2": properties.get("area_m2", 5000),
                                        "centroid": real_coords,
                                        "mean_tp_index": properties.get("mean_tp_index", 0.15),
                                        "source": "real_detector_geojson"
                                    })
                    except Exception as e:
                        logger.error(f"Error loading terra preta GeoJSON {tp_analysis['geojson_path']}: {e}")

                # Extract real geometric features from detector GeoJSON
                geom_analysis = analysis_output.get("geometric_feature_analysis", {})
                if geom_analysis and geom_analysis.get("geojson_path"):
                    try:
                        with open(geom_analysis["geojson_path"], 'r') as f:
                            geom_geojson = json.load(f)
                        
                        for feature in geom_geojson.get("features", []):
                            properties = feature.get("properties", {})
                            geometry = feature.get("geometry", {})
                            
                            # Use real coordinates from detector
                            real_coords = None
                            if "coordinates" in properties:
                                real_coords = properties["coordinates"]
                            elif geometry.get("type") == "Point" and "coordinates" in geometry:
                                real_coords = geometry["coordinates"]
                            
                            if real_coords and len(real_coords) == 2:
                                # Validate coordinates are in Amazon bounds
                                lon, lat = real_coords[0], real_coords[1]
                                if -80 <= lon <= -44 and -20 <= lat <= 10:
                                    aggregated_features_for_zone["geometric_features"].append({
                                        "type": properties.get("type", "geometric_feature"),
                                        "confidence": properties.get("confidence", 0.70),
                                        "center": real_coords,
                                        "area_m2": properties.get("area_m2", 15000),
                                        "source": "real_detector_geojson"
                                    })
                    except Exception as e:
                        logger.error(f"Error loading geometric GeoJSON {geom_analysis['geojson_path']}: {e}")
                
                # Extract real crop mark features from detector GeoJSON
                crop_analysis = analysis_output.get("crop_mark_analysis", {})
                if crop_analysis and crop_analysis.get("geojson_path"):
                    try:
                        with open(crop_analysis["geojson_path"], 'r') as f:
                            crop_geojson = json.load(f)
                        
                        for feature in crop_geojson.get("features", []):
                            properties = feature.get("properties", {})
                            geometry = feature.get("geometry", {})
                            
                            # Use real coordinates from detector
                            real_coords = None
                            if "coordinates" in properties:
                                real_coords = properties["coordinates"]
                            elif geometry.get("type") == "Point" and "coordinates" in geometry:
                                real_coords = geometry["coordinates"]
                            
                            if real_coords and len(real_coords) == 2:
                                # Validate coordinates are in Amazon bounds
                                lon, lat = real_coords[0], real_coords[1]
                                if -80 <= lon <= -44 and -20 <= lat <= 10:
                                    # Crop marks are subsurface archaeological indicators - treat as terra preta evidence
                                    aggregated_features_for_zone["terra_preta_patches"].append({
                                        "type": "crop_mark_terra_preta",
                                        "confidence": properties.get("confidence", 0.75),
                                        "area_m2": properties.get("area_m2", 3000),
                                        "centroid": real_coords,
                                        "mean_tp_index": properties.get("mean_tp_index", 0.12),
                                        "detection_method": "crop_mark_analysis",
                                        "source": "real_detector_geojson"
                                    })
                    except Exception as e:
                        logger.error(f"Error loading crop mark GeoJSON {crop_analysis['geojson_path']}: {e}")

                # Extract real GEDI clearing and earthwork features from analysis results
                clearing_results = analysis_output.get("clearing_results", {})
                if clearing_results and clearing_results.get("gap_clusters"):
                    for cluster in clearing_results["gap_clusters"]:
                        center = cluster.get("center")
                        if center and len(center) == 2:
                            # GEDI coordinates might be in [lat, lon] format, check and fix
                            lat, lon = center[0], center[1]
                            if abs(lat) > abs(lon):  # Likely [lat, lon], swap to [lon, lat]
                                real_coords = [lon, lat]
                            else:
                                real_coords = [lat, lon]
                            
                            # Validate coordinates are in Amazon bounds
                            if -80 <= real_coords[0] <= -44 and -20 <= real_coords[1] <= 10:
                                aggregated_features_for_zone["geometric_features"].append({
                                    "type": "gedi_clearing",
                                    "confidence": 0.8,  # High confidence for clustered clearings
                                    "center": real_coords,
                                    "area_m2": cluster.get("area_km2", 0.0) * 1000000,
                                    "count": cluster.get("count", 0),
                                    "source": "real_gedi_analysis"
                                })

                earthwork_results = analysis_output.get("earthwork_results", {})
                if earthwork_results and earthwork_results.get("mound_clusters"):
                    for cluster in earthwork_results["mound_clusters"]:
                        center = cluster.get("center")
                        if center and len(center) == 2:
                            # GEDI coordinates might be in [lat, lon] format, check and fix
                            lat, lon = center[0], center[1]
                            if abs(lat) > abs(lon):  # Likely [lat, lon], swap to [lon, lat]
                                real_coords = [lon, lat]
                            else:
                                real_coords = [lat, lon]
                            
                            # Validate coordinates are in Amazon bounds
                            if -80 <= real_coords[0] <= -44 and -20 <= real_coords[1] <= 10:
                                aggregated_features_for_zone["geometric_features"].append({
                                    "type": "gedi_earthwork",
                                    "confidence": 0.75,
                                    "center": real_coords,
                                    "area_m2": cluster.get("area_km2", 0.0) * 1000000,
                                    "count": cluster.get("count", 0),
                                    "source": "real_gedi_analysis"
                                })

                # Fallback: Handle old structure if it exists
                tp_data = analysis_output.get("terra_preta", {})
                if isinstance(tp_data, dict) and tp_data.get("patches"):
                    patches = tp_data.get("patches", [])
                    if isinstance(patches, list):
                        aggregated_features_for_zone["terra_preta_patches"].extend(patches)
                    else:
                        logger.warning(f"  [Stage 3: Score Zones] Expected list for terra_preta patches in zone {zone_id}, got {type(patches)}")

                geom_features = analysis_output.get("geometric_features", [])
                if isinstance(geom_features, list):
                        aggregated_features_for_zone["geometric_features"].extend(geom_features)
                else:
                    logger.warning(f"  [Stage 3: Score Zones] Expected list for geometric_features in zone {zone_id}, got {type(geom_features)}")
            
            try:
                zone_scoring_result = self.core_scorer.calculate_zone_score(
                    zone_id=zone_id,
                    features=aggregated_features_for_zone
                )
                scoring_results[zone_id] = zone_scoring_result
                score_val = zone_scoring_result.get('total_score', 0)
                classification = zone_scoring_result.get('classification', 'Unknown')
                logger.info(f"    [Stage 3: Score Zones] Zone {zone_id}: Core score {score_val} ({classification})")
            except Exception as e:
                logger.error(f"  [Stage 3: Score Zones] Error using core scorer for zone {zone_id}: {e}", exc_info=True)
                scoring_results[zone_id] = self._create_scoring_error_result(zone_id, str(e))
        
        logger.info(f"[Stage 3: Score Zones] Core scoring complete for {len(scoring_results)} zones.")

        output_path = self._get_staged_data_path("scoring_results")
        try:
            # Ensure the directory exists before writing
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(scoring_results, f, indent=2, cls=NumpyJSONEncoder)
            logger.info(f"[Stage 3: Score Zones] Saved scoring results to {output_path}")
        except Exception as e:
            logger.error(f"[Stage 3: Score Zones] Failed to save scoring results: {e}", exc_info=True)

        return scoring_results

    def generate_outputs(self, 
                         analysis_results_input: Union[Dict[str, List[Dict[str, Any]]], Path],
                         scoring_results_input: Union[Dict[str, Dict[str, Any]], Path]
                        ) -> Dict[str, Any]:
        """Stage 4: Generates reports and visualizations from analysis and scoring results."""
        logger.info(f"[Stage 4: Generate Outputs] Provider: {self.provider_name_for_paths}")

        analysis_results: Dict[str, List[Dict[str, Any]]]
        scoring_results: Dict[str, Dict[str, Any]]

        # Load Analysis Results
        if isinstance(analysis_results_input, Path):
            input_path = analysis_results_input
            if not input_path.is_file(): input_path = self._get_staged_data_path(str(analysis_results_input))
            logger.info(f"[Stage 4: Generate Outputs] Loading analysis results from {input_path}")
            try:
                with open(input_path, 'r') as f: analysis_results = json.load(f)
                logger.info(f"[Stage 4: Generate Outputs] Loaded analysis results for {len(analysis_results)} zones.")
            except Exception as e:
                logger.error(f"[Stage 4: Generate Outputs] Failed to load analysis results: {e}", exc_info=True)
                return {"report_path": None, "map_paths": {}}
        elif isinstance(analysis_results_input, dict):
            analysis_results = analysis_results_input
        else:
            logger.error("[Stage 4: Generate Outputs] Invalid type for analysis_results_input.")
            return {"report_path": None, "map_paths": {}}

        # Load Scoring Results
        if isinstance(scoring_results_input, Path):
            input_path = scoring_results_input
            if not input_path.is_file(): input_path = self._get_staged_data_path(str(scoring_results_input))
            logger.info(f"[Stage 4: Generate Outputs] Loading scoring results from {input_path}")
            try:
                with open(input_path, 'r') as f: scoring_results = json.load(f)
                logger.info(f"[Stage 4: Generate Outputs] Loaded scoring results for {len(scoring_results)} zones.")
            except Exception as e:
                logger.error(f"[Stage 4: Generate Outputs] Failed to load scoring results: {e}", exc_info=True)
                return {"report_path": None, "map_paths": {}}
        elif isinstance(scoring_results_input, dict):
            scoring_results = scoring_results_input
        else:
            logger.error("[Stage 4: Generate Outputs] Invalid type for scoring_results_input.")
            return {"report_path": None, "map_paths": {}}

        if not analysis_results or not scoring_results:
            logger.warning("[Stage 4: Generate Outputs] Missing analysis or scoring results. Cannot generate outputs.")
            return {"report_path": None, "map_paths": {}}

        # Report Generation
        # ReportStep is initialized with run_id and provider_name_for_paths is passed to its run method.
        # It saves reports to .../run_ID/provider/reports/
        logger.info("[Stage 4: Generate Outputs] Generating report...")
        report_output_data = self.report_step.run(scoring_results, analysis_results, provider=self.provider_name_for_paths)
        # The main report file path can be constructed based on convention if not directly returned by report_step.run in a structured way
        # Assuming ReportStep saves to self.current_reports_subdir / "discovery_report.json"
        main_report_path = self.report_step.current_reports_subdir / "discovery_report.json" 
        logger.info(f"[Stage 4: Generate Outputs] Report generation complete. Main report: {main_report_path}")

        # 🏛️ ENHANCED: Automatic Academic Report Generation
        academic_report_path = None
        try:
            from src.core.academic_validation import AcademicValidatedScoring
            
            # Check if we have academic validation data in scoring results
            has_academic_data = any(
                result.get('academic_validation') is not None 
                for result in scoring_results.values()
            )
            
            if has_academic_data:
                logger.info("📊 [Stage 4] Generating automatic academic report...")
                academic_validator = AcademicValidatedScoring()
                
                # Generate comprehensive academic report
                academic_report = academic_validator.generate_academic_report(scoring_results)
                
                # Save academic report to results directory
                academic_reports_dir = self.base_run_provider_dir / "academic_reports"
                academic_reports_dir.mkdir(parents=True, exist_ok=True)
                
                academic_report_path = academic_reports_dir / f"academic_analysis_report_{self.run_id}.json"
                
                with open(academic_report_path, 'w') as f:
                    json.dump(academic_report, f, indent=2, cls=NumpyJSONEncoder)
                
                # Log academic report summary
                pub_stats = academic_report['publication_statistics']
                logger.info(f"📊 Academic Report Generated: {academic_report_path}")
                logger.info(f"    📈 Sites Meeting Standards: {pub_stats['sites_meeting_standards']}/{pub_stats['total_sites_analyzed']}")
                logger.info(f"    📊 Mean Effect Size (Cohen's d): {pub_stats['mean_effect_size']:.3f}")
                logger.info(f"    🎯 Statistical Power: {pub_stats['statistical_power']:.3f}")
                logger.info(f"    📚 Publication Ready: {academic_report['peer_review_ready']}")
                
                # Log recommended journals
                if academic_report['recommended_journals']:
                    logger.info(f"    📝 Recommended Journals: {', '.join(academic_report['recommended_journals'][:3])}")
                
            else:
                logger.info("📊 [Stage 4] No academic validation data found - skipping academic report")
                
        except ImportError:
            logger.warning("📊 [Stage 4] Academic validation module not available")
        except Exception as e:
            logger.error(f"📊 [Stage 4] Error generating academic report: {e}", exc_info=True)

        # Export generation (always needed for cross-provider analysis)
        logger.info(f"[Stage 4: Generate Outputs] Generating exports for {self.provider_name_for_paths}...")
        if analysis_results:
            for zone_id_key in analysis_results.keys():
                try:
                    # Generate exports for this provider (required for cross-provider analysis)
                    self._generate_unified_exports(analysis_results, zone_id_key)
                except Exception as e:
                    logger.error(f"Failed to generate exports for zone {zone_id_key}: {e}")

        # Visualization
        logger.info(f"[Stage 4: Generate Outputs] Creating visualizations for {self.provider_name_for_paths}...")
        map_paths: Dict[str, Optional[str]] = {}
        if not analysis_results:
            logger.warning("[Stage 4: Generate Outputs] No analysis results to visualize, skipping map generation.")
        elif self.provider_name_for_paths in ['gedi', 'sentinel2'] and self._cross_provider_analysis_enabled():
            # Skip individual provider maps when cross-provider analysis will create unified maps
            logger.info(f"[Stage 4: Generate Outputs] Skipping individual {self.provider_name_for_paths} map generation - unified cross-provider map will be created")
            map_paths = {zone_id: None for zone_id in analysis_results.keys()}
        else:
            output_maps_base_dir = self.base_run_provider_dir / "maps"
            output_maps_base_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[Stage 4: Generate Outputs] Saving maps to: {output_maps_base_dir}")

            for zone_id_key, zone_data_list in analysis_results.items():
                logger.info(f"  [Stage 4: Generate Outputs] Visualizing zone: {zone_id_key}")
                map_filename = f"{zone_id_key}_{self.provider_name_for_paths}_discovery_map.html"
                output_map_path = output_maps_base_dir / map_filename
                current_zone_analysis_data = {zone_id_key: zone_data_list}
                current_zone_scoring_data = {zone_id_key: scoring_results.get(zone_id_key)} if scoring_results.get(zone_id_key) else None

                try:
                    # STEP 1.5: Load checkpoint results for map integration
                    checkpoint_results = self._load_checkpoint_results()
                    
                    # Create enhanced map using new modular system (ONLY)
                    enhanced_map_path = self.visualizer.generate_enhanced_map(
                        zone_name=zone_id_key,
                        theme="professional",
                        include_analysis=True,
                        interactive_features=True
                    )
                    
                    # Enhanced map contains all data from all providers
                    
                    # Store enhanced map result
                    if enhanced_map_path:
                        map_paths[zone_id_key] = str(enhanced_map_path)
                        logger.info(f"    🎯 Enhanced map created for zone {zone_id_key}: {enhanced_map_path}")
                    else:
                        logger.warning(f"    ❌ Failed to create enhanced map for zone {zone_id_key}")
                        map_paths[zone_id_key] = None
                        
                except Exception as e:
                    map_paths[zone_id_key] = None
                    logger.error(f"    ✗ Error creating map for zone {zone_id_key}: {e}", exc_info=True)
            
        logger.info(f"[Stage 4: Generate Outputs] Visualization generation complete.")

        return {
            "report_path": str(main_report_path) if main_report_path.exists() else None,
            "academic_report_path": str(academic_report_path) if academic_report_path and academic_report_path.exists() else None,
            "map_paths": map_paths
        }
    
    def _load_checkpoint_results(self) -> Optional[Dict]:
        """Load checkpoint results if they exist for this run_id"""
        try:
            checkpoint_dir = RESULTS_DIR / f"run_{self.run_id}" / "checkpoints"
            checkpoint_file = checkpoint_dir / "checkpoint_2_result.json"
            
            if checkpoint_file.exists():
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                logger.info(f"✅ Loaded checkpoint results for run {self.run_id}")
                return checkpoint_data
            else:
                logger.debug(f"No checkpoint results found for run {self.run_id}")
                return None
        except Exception as e:
            logger.error(f"Error loading checkpoint results: {e}")
            return None

    def _create_scoring_error_result(self, zone_id: str, error_msg: str) -> Dict[str, Any]:
        """Helper to create a consistent error score dictionary for scoring step.
        Moved from the former ScoringStep class.
        """
        # Ensure TARGET_ZONES is accessible, or adapt to get zone_name differently if needed
        zone_config = TARGET_ZONES.get(zone_id)
        zone_name = zone_config.name if zone_config else zone_id.replace('_', ' ').title()
        return {
            'zone_id': zone_id,
            'zone_name': zone_name,
            'total_score': 0,
            'classification': 'ERROR IN SCORING',
            'error': error_msg,
            'evidence_summary': [f"Error: {error_msg}"],
            'evidence_count': 0,
            'score_breakdown': {},
            'confidence_metrics': {},
            'feature_details': {},
            'recommendation': {'title': 'Error', 'message': 'Scoring could not be completed.'}
        }

    def _get_provider_run_results_dir(self) -> Path:
        """Gets the base results directory for the current provider and run_id."""
        # This path structure anticipates ReportStep creating a similar base
        path = RESULTS_DIR / f"run_{self.run_id}" / self.provider_name_for_paths
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run(
        self, zones: Optional[List[str]] = None, max_scenes: int = 3
    ) -> Dict[str, object]:
        """
        Run the full modular pipeline by executing all stages in sequence:
        1. Acquire Data
        2. Analyze Scenes
        3. Score Zones
        4. Generate Outputs (Report and Visualizations)
        
        Args:
            zones: List of zone IDs to process.
            max_scenes: Maximum number of scenes per zone for data acquisition.
        Returns:
            Dictionary with final outputs (e.g., report path, map paths).
        """
        logger.info(f"🚀 Starting Full Modular Pipeline Run for provider: {self.provider_name_for_paths}, Run ID: {self.run_id}")
        logger.info(f"Zones: {zones or 'All priority zones'}, Max Scenes: {max_scenes}")

        # Stage 1: Acquire Data
        acquired_scene_data = self.acquire_data(zones=zones, max_scenes=max_scenes)
        if not acquired_scene_data:
            logger.error("Pipeline execution stopped: Data acquisition failed or returned no data.")
            return {
                "status": "Error",
                "message": "Data acquisition failed.",
                "scene_data": [],
                "analysis": {},
                "scores": {},
                "report_path": None,
                "map_paths": {}
            }

        # Stage 2: Analyze Scenes
        analysis_results = self.analyze_scenes(scene_data_input=acquired_scene_data)
        if not analysis_results:
            logger.error("Pipeline execution stopped: Scene analysis failed or returned no results.")
            return {
                "status": "Error",
                "message": "Scene analysis failed.",
                "scene_data": self._serialize_scenedata_list(acquired_scene_data), # Log what was acquired
                "analysis": {},
                "scores": {},
                "report_path": None,
                "map_paths": {}
            }
        
        # Stage 3: Score Zones
        scoring_results = self.score_zones(analysis_results_input=analysis_results)
        if not scoring_results:
            logger.error("Pipeline execution stopped: Zone scoring failed or returned no results.")
            # Still try to generate a report with what we have up to analysis
            logger.info("Attempting to generate partial report and maps with available analysis results...")
            partial_outputs = self.generate_outputs(
                analysis_results_input=analysis_results, 
                scoring_results_input={} # Pass empty scores
            )
            return {
                "status": "Error",
                "message": "Zone scoring failed. Partial outputs generated.",
                "scene_data": self._serialize_scenedata_list(acquired_scene_data),
                "analysis": analysis_results,
                "scores": {},
                "report_path": partial_outputs.get("report_path"),
                "map_paths": partial_outputs.get("map_paths", {})
            }

        # Stage 4: Generate Outputs
        final_outputs = self.generate_outputs(
            analysis_results_input=analysis_results, 
            scoring_results_input=scoring_results
        )
        
        logger.info(f"✅ Full Modular Pipeline Run Completed for provider: {self.provider_name_for_paths}, Run ID: {self.run_id}")
        
        # The main run method should return a comprehensive dictionary of results.
        # The individual stage outputs (JSON files) serve as checkpoints or inputs for partial runs.
        return {
            "status": "Success",
            "run_id": self.run_id,
            "provider": self.provider_name_for_paths,
            # Key outputs that might be useful programmatically:
            "scene_data": self._serialize_scenedata_list(acquired_scene_data), # Include actual scene data
            "analysis": analysis_results, # Include actual analysis results for checkpoint processing
            "scores": scoring_results, # Include actual scoring results
            "acquired_scene_data_summary": { # Summary, actual data is in the JSON file
                "count": len(acquired_scene_data),
                "path": str(self._get_staged_data_path("acquired_scene_data"))
            },
            "analysis_results_summary": { # Summary
                "zone_count": len(analysis_results),
                "path": str(self._get_staged_data_path("analysis_results"))
            },
            "scoring_results_summary": { # Summary
                "zone_count": len(scoring_results),
                "path": str(self._get_staged_data_path("scoring_results"))
            },
            "report_path": final_outputs.get("report_path"),
            "academic_report_path": final_outputs.get("academic_report_path"),  # 🏛️ ENHANCED: Academic report path
            "map_paths": final_outputs.get("map_paths", {}),
            # Optionally include the full data if small enough, or confirm they are saved.
            # For now, confirming saved by referring to their paths.
        }
    
    def _generate_unified_exports(self, analysis_results: Dict[str, List[Dict]], zone_id: str) -> None:
        """Generate unified exports from analysis results for a specific zone"""
        try:
            zone_results = analysis_results.get(zone_id, [])
            if not zone_results:
                logger.warning(f"No analysis results found for zone {zone_id}")
                return
            
            # Extract features by provider type
            gedi_features = []
            sentinel2_features = []
            combined_features = []
            
            logger.info(f"📊 Processing {len(zone_results)} scene results for zone {zone_id}")
            
            # CROSS-PROVIDER ENHANCEMENT: Load features from other providers that have already run
            parent_run_dir = self.base_run_provider_dir.parent
            logger.info(f"🔍 Checking for cross-provider data in: {parent_run_dir}")
            
            # Load GEDI features if we're not the GEDI provider and GEDI has run
            if self.provider_name_for_paths != 'gedi':
                gedi_results_file = parent_run_dir / 'gedi' / 'analysis_results.json'
                if gedi_results_file.exists():
                    logger.info(f"🛰️ Loading GEDI features from: {gedi_results_file}")
                    try:
                        with open(gedi_results_file, 'r') as f:
                            gedi_analysis_results = json.load(f)
                        gedi_zone_results = gedi_analysis_results.get(zone_id, [])
                        logger.info(f"🛰️ Found {len(gedi_zone_results)} GEDI scene results")
                    except Exception as e:
                        logger.warning(f"Failed to load GEDI data: {e}")
                        gedi_zone_results = []
                else:
                    gedi_zone_results = []
            else:
                gedi_zone_results = zone_results
            
            # Load Sentinel-2 features if we're not the Sentinel-2 provider and Sentinel-2 has run
            if self.provider_name_for_paths != 'sentinel2':
                s2_results_file = parent_run_dir / 'sentinel2' / 'analysis_results.json'
                if s2_results_file.exists():
                    logger.info(f"🛰️ Loading Sentinel-2 features from: {s2_results_file}")
                    try:
                        with open(s2_results_file, 'r') as f:
                            s2_analysis_results = json.load(f)
                        s2_zone_results = s2_analysis_results.get(zone_id, [])
                        logger.info(f"🛰️ Found {len(s2_zone_results)} Sentinel-2 scene results")
                    except Exception as e:
                        logger.warning(f"Failed to load Sentinel-2 data: {e}")
                        s2_zone_results = []
                else:
                    s2_zone_results = []
            else:
                s2_zone_results = zone_results
            
            # Consolidate all results to be processed uniformly.
            all_results_to_process = []
            if gedi_zone_results: all_results_to_process.extend(gedi_zone_results)
            if s2_zone_results: all_results_to_process.extend(s2_zone_results)

            # Deduplicate results to avoid processing the same scene data twice.
            unique_results = []
            seen_keys = set()
            for r in all_results_to_process:
                gap_clusters = r.get("clearing_results", {}).get("gap_clusters", [])
                gap_center = gap_clusters[0].get("center") if gap_clusters else None
                key = (r.get("terra_preta_analysis", {}).get("geojson_path") or
                       r.get("geometric_feature_analysis", {}).get("geojson_path") or
                       r.get("crop_mark_analysis", {}).get("geojson_path") or
                       str(gap_center))
                if key and key not in seen_keys and key not in ["None", "[{'center': None}]"]:
                    unique_results.append(r)
                    seen_keys.add(key)

            logger.info(f"📊 Processing {len(unique_results)} unique scene results for zone {zone_id} from all providers.")

            # Extract features using the new unified method
            gedi_features, sentinel2_features = self._extract_features_from_results(unique_results, zone_id)

            logger.info(f"🌐 Cross-provider feature totals: {len(gedi_features)} GEDI + {len(sentinel2_features)} Sentinel-2")
            
            # Create combined/convergence features and calculate convergent scores
            all_features = gedi_features + sentinel2_features
            combined_features = []  # Will only contain truly cross-validated features
            
            logger.info(f"🔍 Starting convergence detection with {len(all_features)} features ({len(gedi_features)} GEDI + {len(sentinel2_features)} Sentinel-2)")
            
            # Import convergent score calculator
            from ..core.scoring import ConvergentAnomalyScorer
            scorer = ConvergentAnomalyScorer()
            
            # Initialize convergent scores and cross-provider support for all features
            for feature in all_features:
                feature['convergent_score'] = 0.0
                feature['gedi_support'] = False
                feature['sentinel2_support'] = False
                feature['convergence_distance_m'] = None
                feature['combined_confidence'] = None
                feature['convergence_type'] = None
            
            # Enhanced convergence detection with proper scoring
            convergence_pairs = []
            pairs_checked = 0
            different_provider_pairs = 0
            
            for i, feature1 in enumerate(all_features):
                for j, feature2 in enumerate(all_features[i+1:], i+1):
                    pairs_checked += 1
                    if feature1.get('provider') != feature2.get('provider'):
                        different_provider_pairs += 1
                        # Calculate precise distance
                        coord1 = feature1['coordinates']
                        coord2 = feature2['coordinates']
                        spatial_distance = ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**0.5
                        
                        # Check if features are within convergence distance (500m max)
                        if spatial_distance <= 0.005:  # ~500m in degrees
                            
                            # Calculate proper convergent score
                            convergent_score = scorer.calculate_feature_convergent_score(
                                feature1, feature2, spatial_distance, {'zone': zone_id}
                            )
                            
                            # Update both features with convergent scores and cross-provider support
                            feature1['convergent_score'] = max(feature1.get('convergent_score', 0.0), convergent_score)
                            feature2['convergent_score'] = max(feature2.get('convergent_score', 0.0), convergent_score)
                            
                            # Set cross-provider support flags
                            if feature1.get('provider') == 'gedi':
                                feature1['sentinel2_support'] = True
                                feature2['gedi_support'] = True
                            elif feature1.get('provider') == 'sentinel2':
                                feature1['gedi_support'] = True
                                feature2['sentinel2_support'] = True
                            
                            # Set convergence metadata
                            distance_m = spatial_distance * 111000
                            feature1['convergence_distance_m'] = distance_m
                            feature2['convergence_distance_m'] = distance_m
                            feature1['combined_confidence'] = (feature1['confidence'] + feature2['confidence']) / 2
                            feature2['combined_confidence'] = (feature1['confidence'] + feature2['confidence']) / 2
                            feature1['convergence_type'] = 'spatial_overlap'
                            feature2['convergence_type'] = 'spatial_overlap'
                            
                            convergence_pairs.append((i, j, convergent_score, distance_m))
                            
                            logger.info(f"🎯 CONVERGENCE: {feature1.get('type')} + {feature2.get('type')} at {distance_m:.0f}m (score: {convergent_score:.1f})")
                            logger.info(f"   🔗 Cross-provider support: {feature1.get('type')} ← → {feature2.get('type')}")
                            
                            # Add both cross-validated features to combined_features
                            if feature1 not in combined_features:
                                combined_features.append(feature1)
                            if feature2 not in combined_features:
                                combined_features.append(feature2)
                            
                            # Create convergence feature if score is high enough (>8.0)
                            if convergent_score >= 8.0:
                                avg_coords = [
                                    (coord1[0] + coord2[0]) / 2,
                                    (coord1[1] + coord2[1]) / 2
                                ]
                                convergence_feature = {
                                    'coordinates': avg_coords,
                                    'provider': 'multi_sensor',
                                    'confidence': (feature1['confidence'] + feature2['confidence']) / 2,
                                    'type': f"{feature1.get('type', 'unknown')}+{feature2.get('type', 'unknown')}",
                                    'area_m2': max(feature1['area_m2'], feature2['area_m2']),
                                    'zone': zone_id,
                                    'run_id': self.run_id,
                                    'convergent_score': convergent_score,
                                    'archaeological_grade': 'high',
                                    'source_features': [i, j],  # Track source features
                                    'spatial_distance_m': spatial_distance * 111000  # Convert to meters
                                }
                                combined_features.append(convergence_feature)
            
            logger.info(f"🔗 Convergence detection results for {zone_id}:")
            logger.info(f"   📊 Total pairs checked: {pairs_checked}")
            logger.info(f"   🔄 Different provider pairs: {different_provider_pairs}")
            logger.info(f"   🎯 Convergent pairs found: {len(convergence_pairs)}")
            
            if convergence_pairs:
                avg_score = sum(pair[2] for pair in convergence_pairs) / len(convergence_pairs)
                logger.info(f"📊 Average convergent score: {avg_score:.2f}/15")
            
            # Export each type if features exist
            if gedi_features:
                self.export_manager.export_gedi_features(gedi_features, zone_id)
                logger.info(f"📍 Exported {len(gedi_features)} GEDI features for {zone_id}")
            
            if sentinel2_features:
                self.export_manager.export_sentinel2_features(sentinel2_features, zone_id)
                logger.info(f"📍 Exported {len(sentinel2_features)} Sentinel-2 features for {zone_id}")
            
            # Export combined features only if there are cross-validated features
            if combined_features:
                self.export_manager.export_combined_features(combined_features, zone_id)
                logger.info(f"📍 Exported {len(combined_features)} cross-validated features for {zone_id}")
            else:
                logger.info(f"📍 No cross-validated features found for {zone_id}")
                
            # Export top candidates - defer to cross-provider analysis if multiple providers
            if self._cross_provider_analysis_enabled():
                logger.info(f"🏆 Skipping individual provider top candidates - will be handled by cross-provider analysis")
            else:
                # Single provider mode: Export top candidates with enhanced ranking from ALL features
                # Priority: gedi_support=True (cross-provider validation) gets major boost
                def calculate_priority_score(feature):
                    confidence = feature.get('confidence', 0.0)
                    convergent_score = feature.get('convergent_score', 0.0)
                    area_m2 = feature.get('area_m2', 0.0)
                    gedi_support = feature.get('gedi_support', False)
                    sentinel2_support = feature.get('sentinel2_support', False)
                    
                    # Handle None values that can cause comparison errors
                    if area_m2 is None:
                        area_m2 = 0.0
                    if confidence is None:
                        confidence = 0.0
                    if convergent_score is None:
                        convergent_score = 0.0
                    
                    # Base score from confidence
                    score = confidence
                    
                    # MAJOR boost for cross-provider validation (the most important factor)
                    if gedi_support or sentinel2_support:
                        score += 10.0  # Massive boost for cross-provider validation
                        logger.debug(f"🎯 Cross-provider boost: {feature.get('type')} +10.0 points")
                    
                    # Additional boost for convergent score
                    if convergent_score > 0:
                        score += convergent_score * 1.5  # Weight convergence score
                    
                    # Boost for larger areas (archaeological significance)
                    if area_m2 and area_m2 > 50000:  # > 5 hectares
                        score += 1.0
                    elif area_m2 and area_m2 > 10000:  # > 1 hectare
                        score += 0.5
                    
                    # Boost for high-significance archaeological types
                    feature_type = feature.get('type', '').lower()
                    if 'terra_preta' in feature_type:
                        score += 0.5
                    if 'gedi_clearing' in feature_type:
                        score += 0.3
                    if 'multi_sensor' in feature.get('provider', ''):
                        score += 2.0  # High boost for multi-sensor features
                    
                    return score
                
                # Select top candidates from ALL features (not just combined)
                if all_features:
                    top_candidates = sorted(all_features, 
                                          key=calculate_priority_score, 
                                          reverse=True)[:5]
                    
                    logger.info(f"🏆 Top 5 candidate ranking:")
                    for i, candidate in enumerate(top_candidates):
                        conv_score = candidate.get('convergent_score', 0.0)
                        priority_score = calculate_priority_score(candidate)
                        logger.info(f"  {i+1}. {candidate.get('type')} (conf: {candidate['confidence']:.2f}, conv: {conv_score:.1f}, priority: {priority_score:.2f})")
                    for i, candidate in enumerate(top_candidates):
                        candidate['priority_rank'] = i + 1
                        candidate['field_investigation_priority'] = 'high' if i < 3 else 'medium'
                    
                    self.export_manager.export_top_candidates(top_candidates, zone_id, len(top_candidates))
                    logger.info(f"🎯 Exported {len(top_candidates)} top candidates for {zone_id}")
            
        except Exception as e:
            logger.error(f"Error generating unified exports for {zone_id}: {e}", exc_info=True)
    
    def _extract_features_from_results(self, scene_results: List[Dict], zone_id: str) -> Tuple[List[Dict], List[Dict]]:
        """Extracts GEDI and Sentinel-2 features from a list of scene analysis results."""
        gedi_features, sentinel2_features = [], []
        
        for scene_result in scene_results:
            if not scene_result.get("success", False): continue
            
            # Extract GEDI features from clusters
            if "clearing_results" in scene_result:
                for c in scene_result.get("clearing_results", {}).get("gap_clusters", []):
                    if "center" in c: gedi_features.append({'coordinates': c["center"], 'provider': 'gedi', 'confidence': 0.8, 'type': 'gedi_clearing', 'area_m2': c.get('area_km2', 0.0) * 1e6, 'zone': zone_id, 'run_id': self.run_id, 'archaeological_grade': 'high'})
            
            if "earthwork_results" in scene_result:
                for c in scene_result.get("earthwork_results", {}).get("mound_clusters", []):
                    if "center" in c: gedi_features.append({'coordinates': c["center"], 'provider': 'gedi', 'confidence': 0.75, 'type': 'gedi_earthwork', 'area_m2': c.get('area_km2', 0.0) * 1e6, 'zone': zone_id, 'run_id': self.run_id, 'archaeological_grade': 'high'})

            # Extract Sentinel-2 features from GeoJSON paths
            for key, f_type, grade, conf in [("terra_preta_analysis", "terra_preta", "high", 0.6), ("geometric_feature_analysis", "geometric_feature", "medium", 0.55), ("crop_mark_analysis", "crop_mark", "high", 0.65)]:
                analysis = scene_result.get(key, {})
                if analysis.get("geojson_path"):
                    features = self._load_features_from_geojson(analysis["geojson_path"])
                    for f in features:
                        if "coordinates" in f: 
                            # Preserve all geometric data for proper polygon export
                            feature_data = {
                                'coordinates': f["coordinates"], 
                                'provider': 'sentinel2', 
                                'confidence': f.get('confidence', conf), 
                                'type': f.get('type', f_type), 
                                'area_m2': f.get('area_m2', 0.0), 
                                'zone': zone_id, 
                                'run_id': self.run_id, 
                                'archaeological_grade': grade
                            }
                            
                            # Preserve geometric data needed for polygon export
                            if 'geographic_polygon_coords' in f:
                                feature_data['geographic_polygon_coords'] = f['geographic_polygon_coords']
                            if 'geographic_line_coords' in f:
                                feature_data['geographic_line_coords'] = f['geographic_line_coords']
                            if 'radius_m' in f:
                                feature_data['radius_m'] = f['radius_m']
                                
                            sentinel2_features.append(feature_data)
        
        return gedi_features, sentinel2_features

    def _load_features_from_geojson(self, geojson_path: str) -> List[Dict]:
        """Load features from a GeoJSON file and extract coordinate data"""
        try:
            import json
            with open(geojson_path, 'r') as f:
                geojson_data = json.load(f)
            
            features = []
            for feature in geojson_data.get("features", []):
                properties = feature.get("properties", {})
                geometry = feature.get("geometry", {})
                
                # Extract coordinates from properties first, then fallback to geometry
                coordinates = None
                if "coordinates" in properties:
                    coordinates = properties["coordinates"]
                elif geometry.get("type") == "Point" and "coordinates" in geometry:
                    coordinates = geometry["coordinates"]
                elif geometry.get("type") == "Polygon" and "coordinates" in geometry:
                    # For polygons, use the centroid from properties if available
                    if "geo_center_calculated" in properties:
                        coord_str = properties["geo_center_calculated"].strip("()")
                        coord_parts = coord_str.split(", ")
                        if len(coord_parts) == 2:
                            coordinates = [float(coord_parts[0]), float(coord_parts[1])]
                    else:
                        # Calculate centroid of polygon
                        polygon_coords = geometry["coordinates"][0]  # First ring
                        avg_lon = sum(coord[0] for coord in polygon_coords) / len(polygon_coords)
                        avg_lat = sum(coord[1] for coord in polygon_coords) / len(polygon_coords)
                        coordinates = [avg_lon, avg_lat]
                elif geometry.get("type") == "LineString" and "coordinates" in geometry:
                    # For lines, use center point
                    line_coords = geometry["coordinates"]
                    if len(line_coords) >= 2:
                        avg_lon = sum(coord[0] for coord in line_coords) / len(line_coords)
                        avg_lat = sum(coord[1] for coord in line_coords) / len(line_coords)
                        coordinates = [avg_lon, avg_lat]
                
                if coordinates and len(coordinates) == 2:
                    # Validate coordinates are in Amazon bounds
                    lon, lat = coordinates[0], coordinates[1]
                    if -80 <= lon <= -44 and -20 <= lat <= 10:
                        # Handle None/null values from GeoJSON
                        area_m2 = properties.get("area_m2", 0.0)
                        if area_m2 is None:
                            area_m2 = 0.0
                        
                        confidence = properties.get("confidence", 0.6)
                        if confidence is None:
                            confidence = 0.6
                        
                        feature_data = {
                            "coordinates": coordinates,
                            "confidence": confidence,
                            "type": properties.get("type", "unknown"),
                            "area_m2": area_m2,
                            **properties  # Include all other properties
                        }
                        features.append(feature_data)
                    else:
                        logger.warning(f"Coordinates {coordinates} outside Amazon bounds in {geojson_path}")
                else:
                    logger.debug(f"Could not extract valid coordinates from feature in {geojson_path}")
            
            logger.debug(f"Loaded {len(features)} valid features from {geojson_path}")
            return features
            
        except Exception as e:
            logger.error(f"Error loading features from {geojson_path}: {e}")
            return []

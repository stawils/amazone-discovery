"""
Unified Export Manager for Archaeological Pipeline
Centralizes all exports at the run level rather than provider level
"""

from pathlib import Path
from typing import Dict, Any, List
from src.pipeline.shared_types import FeatureDict, ConvergencePair, EnhancedCandidate
import logging
import json
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)

class UnifiedExportManager:
    """Manages all exports at the run level with proper organization"""
    
    def __init__(self, run_id: str, results_dir: Path):
        self.run_id = run_id
        self.results_dir = results_dir
        self.run_dir = results_dir / f"run_{run_id}"
        
        # Define export structure but don't create directories yet
        self.exports_dir = self.run_dir / "exports"
        
        # Define provider subdirectories within exports (create only when needed)
        self.gedi_exports = self.exports_dir / "gedi"
        self.sentinel2_exports = self.exports_dir / "sentinel2"
        self.combined_exports = self.exports_dir / "combined"
        
        logger.info(f"📁 Unified Export Manager initialized for run {run_id}")
        logger.info(f"   📂 Exports will be saved to: {self.exports_dir}")
    
    def _ensure_export_dir(self, export_path: Path) -> None:
        """Create export directory only when needed"""
        if not export_path.exists():
            export_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"📂 Created export directory: {export_path}")
    
    def _validate_coordinates(self, coordinates: List[float], context: str = "") -> bool:
        """Validate coordinates are in correct format and within Amazon bounds"""
        # Convert numpy types to Python types if needed
        if hasattr(coordinates, 'tolist'):
            coordinates = coordinates.tolist()
        
        # Ensure we have a list/tuple with 2 elements
        if not (isinstance(coordinates, (list, tuple)) and len(coordinates) == 2):
            logger.error(f"Invalid coordinate format {coordinates} for {context} - expected [lon, lat]")
            return False
            
        # Convert individual elements to float to handle numpy types
        try:
            lon, lat = float(coordinates[0]), float(coordinates[1])
        except (ValueError, TypeError) as e:
            logger.error(f"Cannot convert coordinates {coordinates} to float for {context}: {e}")
            return False
        
        # Check for obviously wrong coordinates (UTM values are much larger)
        if abs(lon) > 180 or abs(lat) > 90:
            logger.error(f"Coordinates {coordinates} appear to be in UTM format for {context} - expected geographic")
            return False
            
        # Amazon region bounds (roughly): lat -20 to 10, lon -80 to -44
        if not (-80 <= lon <= -44):
            logger.error(f"Longitude {lon:.6f} outside Amazon bounds [-80, -44] for {context}")
            return False
        if not (-20 <= lat <= 10):
            logger.error(f"Latitude {lat:.6f} outside Amazon bounds [-20, 10] for {context}")
            return False
            
        return True
    
    def _flag_invalid_coordinates(self, coordinates: List[float], zone_name: str, context: str = "") -> List[float]:
        """Flag invalid coordinates but preserve original data for analysis"""
        logger.warning(f"Flagging potentially invalid coordinates {coordinates} for {context} in zone {zone_name} - preserving original data")
        
        # Return original coordinates - don't replace with synthetic ones
        # This preserves data integrity while flagging for review
        return coordinates
    
    def export_gedi_features(self, detections: List[FeatureDict], zone_name: str) -> Path:
        """Export GEDI detections to GeoJSON"""
        try:
            self._ensure_export_dir(self.gedi_exports)  # Create directory only when needed
            export_file = self.gedi_exports / f"{zone_name}_gedi_detections.geojson"
            
            geojson_data = {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                "features": []
            }
            
            # Filter for moderate GEDI confidence (≥40%)
            high_quality_detections = [d for d in detections if (d.get('confidence') or 0.0) >= 0.40]
            
            for detection in high_quality_detections:
                if detection.get('coordinates'):
                    # Convert numpy coordinates to Python lists if needed
                    coordinates = detection['coordinates']
                    if hasattr(coordinates, 'tolist'):
                        coordinates = coordinates.tolist()
                    elif not isinstance(coordinates, list):
                        coordinates = [float(coordinates[0]), float(coordinates[1])]
                    
                    # Check if coordinates are in [lat, lon] format and swap if needed
                    # Amazon region: lat (-20 to 10), lon (-80 to -44)
                    # If first value looks like longitude and second like latitude, swap them
                    if (len(coordinates) == 2 and 
                        abs(coordinates[0]) < 20 and abs(coordinates[1]) > 44):
                        # Likely [lat, lon] format, swap to [lon, lat]
                        coordinates = [coordinates[1], coordinates[0]]
                        logger.debug(f"Swapped coordinates from [lat,lon] to [lon,lat]: {detection['coordinates']} -> {coordinates}")
                    
                    # Validate and flag coordinates if needed but preserve original data
                    if not self._validate_coordinates(coordinates, f"GEDI_detection_{detection.get('type', 'unknown')}"):
                        coordinates = self._flag_invalid_coordinates(coordinates, zone_name, f"GEDI_detection_{detection.get('type', 'unknown')}")
                    
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": coordinates
                        },
                        "properties": {
                            # Basic detection info
                            "type": detection.get('type', 'gedi_clearing'),
                            "confidence": detection.get('confidence', 0.8),
                            "area_km2": detection.get('area_km2', 0.0),
                            "area_m2": detection.get('area_m2', detection.get('area_km2', 0.0) * 1000000),
                            "count": detection.get('count', 0),
                            "feature_type": detection.get('feature_type', 'clearing_cluster'),
                            
                            # Enhanced GEDI-specific data
                            "sensor_details": {
                                "mission": "NASA GEDI",
                                "instrument": "Global Ecosystem Dynamics Investigation LiDAR",
                                "footprint_diameter_m": 25,
                                "pulse_density": detection.get('count', 0),
                                "beam_type": detection.get('beam_type', 'power_beam'),
                                "acquisition_date": detection.get('acquisition_date', 'unknown'),
                                "quality_flag": detection.get('quality_flag', 1)
                            },
                            
                            # Algorithm methodology
                            "detection_algorithm": {
                                "method": "DBSCAN clustering of canopy height anomalies",
                                "parameters": {
                                    "eps_distance_m": detection.get('eps_distance', 50),
                                    "min_samples": detection.get('min_samples', 3),
                                    "height_threshold_m": detection.get('height_threshold', 5.0)
                                },
                                "statistical_significance": {
                                    "p_value": detection.get('p_value', 0.05),
                                    "z_score": detection.get('z_score', 2.0),
                                    "cohens_d": detection.get('cohens_d', None),
                                    "effect_size": detection.get('effect_size', None),
                                    "confidence_interval": detection.get('confidence_interval', [0.7, 0.9]),
                                    "statistical_power": detection.get('statistical_power', None),
                                    "significance_level": detection.get('significance_level', 'HIGH')
                                },
                                "enhanced_parameters": {
                                    "relative_threshold_pct": detection.get('relative_threshold_pct', 50),
                                    "cluster_validation": detection.get('cluster_validation', 'relative_height_50pct_local_max'),
                                    "gpu_acceleration": detection.get('gpu_acceleration', False)
                                },
                                "lidar_analysis": {
                                    "gap_points_detected": detection.get('gap_points_detected', None),
                                    "mean_elevation_m": detection.get('mean_elevation', None),
                                    "elevation_std_m": detection.get('elevation_std', None),
                                    "elevation_anomaly_threshold": detection.get('elevation_anomaly_threshold', None),
                                    "local_variance": detection.get('local_variance', None),
                                    "footprint_area_m2": 490.87,  # Standard GEDI footprint
                                    "pulse_density_per_m2": detection.get('pulse_density', None)
                                }
                            },
                            
                            # Archaeological interpretation
                            "archaeological_assessment": {
                                "interpretation": self._get_gedi_archaeological_interpretation(detection),
                                "evidence_type": "Canopy gap pattern analysis",
                                "cultural_context": detection.get('cultural_context', 'Pre-Columbian settlement'),
                                "estimated_age_range": detection.get('age_range', '500-2000 years BP'),
                                "preservation_state": detection.get('preservation', 'good'),
                                "investigation_priority": self._calculate_investigation_priority(detection)
                            },
                            
                            # Enhanced spatial and environmental context
                            "spatial_context": {
                                "distance_to_water_m": detection.get('distance_to_water', None),
                                "elevation_m": detection.get('elevation', None),
                                "slope_degrees": detection.get('slope', None),
                                "vegetation_type": detection.get('vegetation_type', 'dense_forest'),
                                "accessibility": self._assess_accessibility(detection, zone_name),
                                "terrain_ruggedness": detection.get('terrain_ruggedness', None),
                                "canopy_density_pct": detection.get('canopy_density', None),
                                "soil_drainage": detection.get('soil_drainage', 'moderate')
                            },
                            
                            # Field investigation planning
                            "field_investigation": {
                                "optimal_visit_season": self._get_optimal_visit_season(zone_name),
                                "recommended_equipment": self._get_recommended_equipment(detection),
                                "estimated_investigation_days": self._estimate_investigation_time(detection),
                                "logistics_complexity": self._assess_logistics_complexity(detection, zone_name),
                                "safety_considerations": self._get_safety_considerations(zone_name),
                                "permits_required": self._get_required_permits(zone_name),
                                "local_contacts": self._get_local_contacts(zone_name)
                            },
                            
                            # Research and publication data
                            "research_metadata": {
                                "publication_readiness": self._assess_publication_readiness(detection),
                                "academic_significance": self._assess_academic_significance(detection),
                                "citation_potential": self._estimate_citation_potential(detection),
                                "collaboration_opportunities": self._identify_collaborations(zone_name),
                                "dataset_completeness": self._assess_dataset_completeness(detection),
                                "peer_review_readiness": detection.get('statistical_power', 0) > 0.8
                            },
                            
                            # Data provenance
                            "data_provenance": {
                                "zone": zone_name,
                                "provider": "gedi",
                                "run_id": self.run_id,
                                "processing_date": datetime.now().isoformat(),
                                "pipeline_version": "2.0",
                                "validation_status": detection.get('validation_status', 'pending')
                            }
                        }
                    }
                    geojson_data["features"].append(feature)
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(geojson_data, f, indent=2)
            
            logger.info(f"📍 GEDI exports: {len(geojson_data['features'])} high-quality features (≥40% confidence) → {export_file.name}")
            return export_file
            
        except Exception as e:
            logger.error(f"Failed to export GEDI features: {e}")
            return None
    
    def _get_gedi_archaeological_interpretation(self, detection: Dict[str, Any]) -> str:
        """Get archaeological interpretation for GEDI detection"""
        feature_type = detection.get('type', 'clearing')
        area_m2 = detection.get('area_m2', 0)
        count = detection.get('count', 0)
        
        if 'clearing' in feature_type:
            if area_m2 > 100000:  # > 10 hectares
                return "Large settlement complex with multiple residential areas"
            elif area_m2 > 50000:  # > 5 hectares
                return "Medium settlement site with organized layout"
            elif area_m2 > 10000:  # > 1 hectare
                return "Small settlement or household cluster"
            else:
                return "Individual structure or clearing"
        elif 'mound' in feature_type:
            if area_m2 > 5000:
                return "Large ceremonial or defensive earthwork"
            else:
                return "Small mound or platform structure"
        else:
            return "Unidentified archaeological feature"
    
    def _calculate_investigation_priority(self, detection: Dict[str, Any]) -> str:
        """Calculate investigation priority based on multiple factors"""
        confidence = detection.get('confidence') or 0.0
        area_m2 = detection.get('area_m2', 0)
        count = detection.get('count', 0)
        
        # Priority scoring
        score = 0
        
        # Confidence factor (40% of score)
        if confidence >= 0.8:
            score += 40
        elif confidence >= 0.6:
            score += 30
        elif confidence >= 0.4:
            score += 20
        
        # Size factor (30% of score)
        if area_m2 >= 50000:
            score += 30
        elif area_m2 >= 10000:
            score += 20
        elif area_m2 >= 1000:
            score += 10
        
        # Data density factor (30% of score)
        if count >= 100:
            score += 30
        elif count >= 50:
            score += 20
        elif count >= 10:
            score += 10
        
        if score >= 80:
            return "Immediate - High priority investigation site"
        elif score >= 60:
            return "High - Priority investigation within 6 months"
        elif score >= 40:
            return "Medium - Investigation within 1 year"
        else:
            return "Low - Research interest, long-term investigation"
    
    def _get_sentinel2_archaeological_interpretation(self, detection: Dict[str, Any]) -> str:
        """Get archaeological interpretation for Sentinel-2 detection"""
        feature_type = detection.get('type', 'spectral_anomaly')
        area_m2 = detection.get('area_m2', 0)
        confidence = detection.get('confidence') or 0.0
        
        if 'terra_preta' in feature_type:
            if area_m2 > 50000:  # > 5 hectares
                return "Large terra preta deposit indicating major settlement complex"
            elif area_m2 > 10000:  # > 1 hectare
                return "Medium terra preta deposit suggesting organized residential area"
            else:
                return "Small terra preta patch indicating household or specialized activity area"
        elif 'geometric' in feature_type:
            if 'circle' in feature_type:
                return "Circular vegetation pattern suggesting plaza or ceremonial area"
            elif 'rectangle' in feature_type:
                return "Rectangular pattern indicating structured settlement layout"
            elif 'line' in feature_type:
                return "Linear feature suggesting pathway, canal, or defensive structure"
            else:
                return "Geometric pattern indicating human landscape modification"
        elif 'crop_mark' in feature_type:
            return "Subsurface archaeological feature visible through differential vegetation growth"
        else:
            if confidence >= 0.7:
                return "High-confidence spectral anomaly consistent with archaeological site"
            elif confidence >= 0.5:
                return "Moderate-confidence vegetation stress pattern suggesting human activity"
            else:
                return "Low-confidence spectral anomaly requiring further investigation"
    
    def _assess_accessibility(self, detection: Dict[str, Any], zone_name: str) -> str:
        """Assess site accessibility for field investigation"""
        # This could be enhanced with real terrain analysis
        zone_assessments = {
            'upano_valley_confirmed': 'Moderate - River access possible, helicopter recommended',
            'trombetas': 'Difficult - Remote location, helicopter required',
            'upper_napo_micro': 'Moderate - River transport feasible'
        }
        
        return zone_assessments.get(zone_name, 'Remote - Helicopter or extended river journey required')

    def export_convergence_pairs(self, zone_name: str, convergent_pairs: List[ConvergencePair]) -> Path:
        """Export convergence pairs as a GeoJSON of lines connecting the features."""
        try:
            self._ensure_export_dir(self.combined_exports)
            export_file = self.combined_exports / f"{zone_name}_convergence_pairs.geojson"

            features = []
            for pair in convergent_pairs:
                f1 = pair['feature1']
                f2 = pair['feature2']

                line = {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [f1['coordinates'], f2['coordinates']]
                    },
                    "properties": {
                        "distance_m": pair['distance_m'],
                        "strength": pair['strength'],
                        "combined_confidence": pair['combined_confidence'],
                        "providers": pair['providers'],
                        "feature1_type": f1.get('type'),
                        "feature2_type": f2.get('type'),
                    }
                }
                features.append(line)

            geojson_data = {
                "type": "FeatureCollection",
                "features": features
            }

            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(geojson_data, f, indent=2)

            logger.info(f"🔗 Exported {len(features)} convergence pairs to {export_file.name}")
            return export_file
        except Exception as e:
            logger.error(f"Failed to export convergence pairs: {e}")
            return None
    
    def export_sentinel2_features(self, detections: List[FeatureDict], zone_name: str) -> Path:
        """Export Sentinel-2 detections to GeoJSON"""
        try:
            self._ensure_export_dir(self.sentinel2_exports)  # Create directory only when needed
            export_file = self.sentinel2_exports / f"{zone_name}_sentinel2_detections.geojson"
            
            geojson_data = {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                "features": []
            }
            
            # Filter for moderate archaeological confidence (≥50% aligns with literature)
            # Ensure all confidence values are safe (fix None values)
            safe_detections = []
            for d in detections:
                safe_d = d.copy()
                safe_d['confidence'] = d.get('confidence') or 0.0
                safe_detections.append(safe_d)
            high_quality_detections = [d for d in safe_detections if d['confidence'] >= 0.50]
            
            for i, detection in enumerate(high_quality_detections):
                try:
                    if detection.get('coordinates'):
                        # Convert numpy coordinates to Python lists if needed
                        coordinates = detection['coordinates']
                        if hasattr(coordinates, 'tolist'):
                            coordinates = coordinates.tolist()
                        elif not isinstance(coordinates, list):
                            coordinates = [float(coordinates[0]), float(coordinates[1])]
                        
                        # Check if coordinates are in [lat, lon] format and swap if needed
                        # Amazon region: lat (-20 to 10), lon (-80 to -44)
                        # If first value looks like latitude and second like longitude, swap them
                        if (len(coordinates) == 2 and 
                            abs(coordinates[0]) < 20 and abs(coordinates[1]) > 44):
                            # Likely [lat, lon] format, swap to [lon, lat]
                            coordinates = [coordinates[1], coordinates[0]]
                            logger.debug(f"Swapped coordinates from [lat,lon] to [lon,lat]: {detection['coordinates']} -> {coordinates}")
                        
                        # Validate and fix coordinates if needed
                        if not self._validate_coordinates(coordinates, f"Sentinel2_detection_{detection.get('type', 'unknown')}"):
                            coordinates = self._flag_invalid_coordinates(coordinates, zone_name, f"Sentinel2_detection_{detection.get('type', 'unknown')}")
                        
                        # Create enhanced geometry that preserves full shapes when available
                        geometry = self._create_enhanced_geometry(detection, coordinates)
                        
                        feature = {
                            "type": "Feature",
                            "geometry": geometry,
                            "properties": {
                                # Basic detection info
                                "type": detection.get('type', 'sentinel2_feature'),
                                "confidence": detection.get('confidence', 0.0),
                                "area_m2": detection.get('area_m2', 0.0),
                                "area_km2": detection.get('area_km2', (detection.get('area_m2') or 0.0) / 1000000),
                                "feature_type": detection.get('feature_type', 'spectral_anomaly'),
                                
                                # Enhanced Sentinel-2 specific data
                                "sensor_details": {
                                    "mission": "ESA Sentinel-2",
                                    "instrument": "MultiSpectral Instrument (MSI)",
                                    "spatial_resolution_m": 10,
                                    "spectral_bands": detection.get('spectral_bands', 13),
                                    "acquisition_date": detection.get('acquisition_date', 'unknown'),
                                    "cloud_coverage_percent": detection.get('cloud_coverage', 10),
                                    "processing_level": "L2A"
                                },
                            
                            # Algorithm methodology
                            "detection_algorithm": {
                                "method": "Multi-spectral vegetation stress and soil signature analysis",
                                "parameters": {
                                    "ndvi_threshold": detection.get('ndvi_threshold', 0.3),
                                    "soil_brightness_index": detection.get('soil_brightness', 0.2),
                                    "red_edge_ratio": detection.get('red_edge_ratio', 1.2)
                                },
                                "spectral_indices": {
                                    "ndvi": detection.get('mean_ndvi', None),
                                    "ndre1": detection.get('mean_ndre1', None),
                                    "ndre3": detection.get('mean_ndre3', None), 
                                    "ndwi": detection.get('mean_ndwi', None),
                                    "ndii": detection.get('mean_ndii', None),
                                    "avi_archaeological": detection.get('mean_avi_archaeological', None),
                                    "terra_preta_enhanced": detection.get('mean_tp_enhanced', None),
                                    "clay_minerals": detection.get('mean_clay_minerals', None),
                                    "crop_mark_index": detection.get('mean_crop_mark', None),
                                    "brightness": detection.get('mean_brightness', None),
                                    "ci_red_edge": detection.get('mean_ci_red_edge', None),
                                    "soil_adjusted_vi": detection.get('mean_savi', None)
                                },
                                "statistical_validation": {
                                    "ndvi_depression_pixels": detection.get('ndvi_depression_pixels', None),
                                    "ndre1_confidence": detection.get('mean_ndre1_confidence', None),
                                    "ndre3_confidence": detection.get('mean_ndre3_confidence', None),
                                    "avi_significance": detection.get('avi_significance', None),
                                    "band_quality_score": detection.get('band_quality_score', None),
                                    "p_value": detection.get('p_value', None),
                                    "effect_size": detection.get('effect_size', None),
                                    "statistical_power": detection.get('statistical_power', None),
                                    "archaeological_strength": detection.get('archaeological_strength', None)
                                },
                                "enhanced_parameters": {
                                    "terra_preta_threshold": detection.get('terra_preta_threshold', 0.12),
                                    "crop_mark_sensitivity": detection.get('crop_mark_sensitivity', 0.05),
                                    "red_edge_optimization": detection.get('red_edge_optimization', True),
                                    "clay_mineral_analysis": detection.get('clay_mineral_analysis', True)
                                }
                            },
                            
                            # Archaeological interpretation
                            "archaeological_assessment": {
                                "interpretation": self._get_sentinel2_archaeological_interpretation(detection),
                                "evidence_type": "Spectral vegetation stress and soil composition analysis",
                                "cultural_context": detection.get('cultural_context', 'Pre-Columbian settlement'),
                                "estimated_age_range": detection.get('age_range', '500-2000 years BP'),
                                "preservation_state": detection.get('preservation', 'moderate'),
                                "investigation_priority": self._calculate_investigation_priority(detection)
                            },
                            
                            # Spatial context
                            "spatial_context": {
                                "distance_to_water_m": detection.get('distance_to_water', None),
                                "elevation_m": detection.get('elevation', None),
                                "slope_degrees": detection.get('slope', None),
                                "vegetation_type": detection.get('vegetation_type', 'mixed_forest'),
                                "accessibility": self._assess_accessibility(detection, zone_name)
                            },
                            
                            # Convergence metadata - preserve all convergence information
                            "convergence_analysis": {
                                "convergent_score": detection.get('convergent_score', 0.0),
                                "convergence_distance_m": detection.get('convergence_distance_m', None),
                                "combined_confidence": detection.get('combined_confidence', None),
                                "gedi_support": detection.get('gedi_support', False),
                                "convergence_type": detection.get('convergence_type', None)
                            },
                            
                            # Field investigation planning
                            "field_investigation": {
                                "optimal_visit_season": self._get_optimal_visit_season(zone_name),
                                "recommended_equipment": self._get_recommended_equipment(detection),
                                "estimated_investigation_days": self._estimate_investigation_time(detection),
                                "logistics_complexity": self._assess_logistics_complexity(detection, zone_name),
                                "safety_considerations": self._get_safety_considerations(zone_name),
                                "permits_required": self._get_required_permits(zone_name),
                                "local_contacts": self._get_local_contacts(zone_name)
                            },
                            
                            # Research and publication data
                            "research_metadata": {
                                "publication_readiness": self._assess_publication_readiness(detection),
                                "academic_significance": self._assess_academic_significance(detection),
                                "citation_potential": self._estimate_citation_potential(detection),
                                "collaboration_opportunities": self._identify_collaborations(zone_name),
                                "dataset_completeness": self._assess_dataset_completeness(detection),
                                "peer_review_readiness": detection.get('statistical_power', 0) > 0.8
                            },
                            
                            # Data provenance
                            "data_provenance": {
                                "zone": zone_name,
                                "provider": "sentinel2",
                                "run_id": self.run_id,
                                "processing_date": datetime.now().isoformat(),
                                "pipeline_version": "2.0",
                                "validation_status": detection.get('validation_status', 'pending')
                            },
                            
                                # Selection rationale
                                "selection_reason": self._get_selection_reason(detection),
                                "quality_indicators": self._get_quality_indicators(detection),
                                "archaeological_grade": "high"
                            }
                        }
                        geojson_data["features"].append(feature)
                        
                except Exception as feature_error:
                    logger.warning(f"Failed to process Sentinel-2 feature {i}: {feature_error}. Confidence: {detection.get('confidence')}. Skipping...")
                    continue
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(geojson_data, f, indent=2)
            
            logger.info(f"📍 Sentinel-2 exports: {len(geojson_data['features'])} high-quality features (≥50% confidence) → {export_file.name}")
            return export_file
            
        except Exception as e:
            logger.error(f"Failed to export Sentinel-2 features: {e}")
            return None
    
    def export_combined_features(self, all_detections: List[FeatureDict], zone_name: str) -> Path:
        """Export combined multi-sensor detections"""
        try:
            self._ensure_export_dir(self.combined_exports)  # Create directory only when needed
            export_file = self.combined_exports / f"{zone_name}_combined_detections.geojson"
            
            geojson_data = {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                "features": []
            }
            
            # Filter for moderate combined confidence (≥50%)
            high_quality_detections = [d for d in all_detections if (d.get('confidence') or 0.0) >= 0.50]
            
            for detection in high_quality_detections:
                if detection.get('coordinates'):
                    # Convert numpy coordinates to Python lists if needed
                    coordinates = detection['coordinates']
                    if hasattr(coordinates, 'tolist'):
                        coordinates = coordinates.tolist()
                    elif not isinstance(coordinates, list):
                        coordinates = [float(coordinates[0]), float(coordinates[1])]
                    
                    # Check if coordinates are in [lat, lon] format and swap if needed
                    # Amazon region: lat (-20 to 10), lon (-80 to -44)
                    # If first value looks like latitude and second like longitude, swap them
                    if (len(coordinates) == 2 and 
                        abs(coordinates[0]) < 20 and abs(coordinates[1]) > 44):
                        # Likely [lat, lon] format, swap to [lon, lat]
                        coordinates = [coordinates[1], coordinates[0]]
                        logger.debug(f"Swapped coordinates from [lat,lon] to [lon,lat]: {detection['coordinates']} -> {coordinates}")
                    
                    # Validate and fix coordinates if needed
                    if not self._validate_coordinates(coordinates, f"Combined_detection_{detection.get('type', 'unknown')}"):
                        coordinates = self._flag_invalid_coordinates(coordinates, zone_name, f"Combined_detection_{detection.get('type', 'unknown')}")
                    
                    # Create enhanced geometry that preserves full shapes when available
                    geometry = self._create_enhanced_geometry(detection, coordinates)
                    
                    feature = {
                        "type": "Feature",
                        "geometry": geometry,
                        "properties": {
                            "type": detection.get('type', 'unknown'),
                            "confidence": detection.get('confidence', 0.0),
                            "area_m2": detection.get('area_m2', detection.get('area_km2', 0.0) * 1000000),  # Convert km2 to m2 if needed
                            "zone": zone_name,
                            "provider": detection.get('provider', 'unknown'),
                            "run_id": self.run_id,
                            
                            # Convergence metadata - preserve all convergence information
                            "convergent_score": detection.get('convergent_score', 0.0),
                            "convergence_distance_m": detection.get('convergence_distance_m', None),
                            "combined_confidence": detection.get('combined_confidence', None),
                            "gedi_support": detection.get('gedi_support', False),
                            "convergence_type": detection.get('convergence_type', None),
                            
                            # Selection rationale
                            "selection_reason": self._get_selection_reason(detection),
                            "quality_indicators": self._get_quality_indicators(detection),
                            "archaeological_grade": "high"
                        }
                    }
                    geojson_data["features"].append(feature)
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(geojson_data, f, indent=2)
            
            logger.info(f"📍 Combined exports: {len(geojson_data['features'])} high-quality features (≥50% confidence) → {export_file.name}")
            return export_file
            
        except Exception as e:
            logger.error(f"Failed to export combined features: {e}")
            return None
    
    def export_top_candidates(self, top_detections: List[EnhancedCandidate], zone_name: str, count: int = 5) -> Path:
        """Export top N candidate sites for field investigation"""
        try:
            self._ensure_export_dir(self.combined_exports)  # Create directory only when needed
            export_file = self.combined_exports / f"{zone_name}_top_{count}_candidates.geojson"
            
            geojson_data = {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                "features": []
            }
            
            for i, detection in enumerate(top_detections[:count], 1):
                if detection.get('coordinates'):
                    # Convert numpy coordinates to Python lists if needed
                    coordinates = detection['coordinates']
                    if hasattr(coordinates, 'tolist'):
                        coordinates = coordinates.tolist()
                    elif not isinstance(coordinates, list):
                        coordinates = [float(coordinates[0]), float(coordinates[1])]
                    
                    # Check if coordinates are in [lat, lon] format and swap if needed
                    # Amazon region: lat (-20 to 10), lon (-80 to -44)
                    # If first value looks like latitude and second like longitude, swap them
                    if (len(coordinates) == 2 and 
                        abs(coordinates[0]) < 20 and abs(coordinates[1]) > 44):
                        # Likely [lat, lon] format, swap to [lon, lat]
                        coordinates = [coordinates[1], coordinates[0]]
                        logger.debug(f"Swapped coordinates from [lat,lon] to [lon,lat]: {detection['coordinates']} -> {coordinates}")
                    
                    # Validate and fix coordinates if needed
                    if not self._validate_coordinates(coordinates, f"Top_candidate_{i}_{detection.get('type', 'unknown')}"):
                        coordinates = self._flag_invalid_coordinates(coordinates, zone_name, f"Top_candidate_{i}_{detection.get('type', 'unknown')}")
                    
                    # Create enhanced geometry that preserves full shapes when available
                    geometry = self._create_enhanced_geometry(detection, coordinates)
                    
                    feature = {
                        "type": "Feature",
                        "geometry": geometry,
                        "properties": {
                            "priority_rank": i,
                            "rank": i,  # Add rank field for visualization compatibility
                            "type": detection.get('type', 'unknown'),
                            "confidence": detection.get('confidence', 0.0),
                            "area_m2": detection.get('area_m2', 0.0),
                            "zone": zone_name,
                            "provider": detection.get('provider', 'unknown'),
                            "run_id": self.run_id,
                            "field_investigation_priority": "high" if i <= 3 else "medium",
                            
                            # Cross-provider validation metadata
                            "convergent_score": detection.get('convergent_score', 0.0),
                            "convergence_distance_m": detection.get('convergence_distance_m', None),
                            "combined_confidence": detection.get('combined_confidence', None),
                            "gedi_support": detection.get('gedi_support', False),
                            "sentinel2_support": detection.get('sentinel2_support', False),
                            "convergence_type": detection.get('convergence_type', None)
                        }
                    }
                    geojson_data["features"].append(feature)
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(geojson_data, f, indent=2)
            
            logger.info(f"🎯 Top candidates export: {len(geojson_data['features'])} sites → {export_file.name}")
            return export_file
            
        except Exception as e:
            logger.error(f"Failed to export top candidates: {e}")
            return None
    
    def create_export_manifest(self) -> Path:
        """Create a manifest of all exports for this run"""
        try:
            manifest_file = self.exports_dir / "export_manifest.json"
            
            # Find all export files
            export_files = list(self.exports_dir.rglob("*.geojson"))
            
            manifest = {
                "run_id": self.run_id,
                "export_timestamp": datetime.now().isoformat(),
                "total_export_files": len(export_files),
                "exports": {
                    "gedi": [f.name for f in self.gedi_exports.glob("*.geojson")],
                    "sentinel2": [f.name for f in self.sentinel2_exports.glob("*.geojson")],
                    "combined": [f.name for f in self.combined_exports.glob("*.geojson")]
                },
                "file_paths": [str(f.relative_to(self.exports_dir)) for f in export_files]
            }
            
            with open(manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
            
            logger.info(f"📋 Export manifest created: {len(export_files)} files listed")
            return manifest_file
            
        except Exception as e:
            logger.error(f"Failed to create export manifest: {e}")
            return None
    
    def cleanup_old_provider_exports(self):
        """Remove old provider-specific export directories"""
        try:
            # Remove old gedi/exports and sentinel2/exports if they exist
            old_gedi_exports = self.run_dir / "gedi" / "exports"
            old_sentinel2_exports = self.run_dir / "sentinel2" / "exports"
            
            for old_dir in [old_gedi_exports, old_sentinel2_exports]:
                if old_dir.exists():
                    logger.info(f"🧹 Cleaning up old export directory: {old_dir}")
                    shutil.rmtree(old_dir)
            
            logger.info("✅ Old provider export directories cleaned up")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup old export directories: {e}")
    
    def _get_selection_reason(self, detection: Dict[str, Any]) -> str:
        """Generate explanation for why this feature was selected"""
        reasons = []
        
        confidence = detection.get('confidence') or 0
        convergence_distance = detection.get('convergence_distance_m')
        gedi_support = detection.get('gedi_support', False)
        combined_confidence = detection.get('combined_confidence')
        
        # High confidence threshold
        if confidence >= 0.5:
            reasons.append(f"High confidence ({confidence:.1%})")
        
        # Multi-sensor convergence
        if convergence_distance is not None:
            reasons.append(f"Multi-sensor convergence at {convergence_distance:.0f}m")
        
        if gedi_support:
            reasons.append("GEDI LiDAR support")
        
        if combined_confidence and combined_confidence > confidence:
            reasons.append(f"Enhanced combined confidence ({combined_confidence:.1%})")
        
        # Feature type significance
        feature_type = detection.get('type', '')
        if 'terra_preta' in feature_type:
            reasons.append("Terra preta archaeological significance")
        elif 'geometric' in feature_type:
            reasons.append("Geometric pattern archaeological relevance")
        elif 'crop_mark' in feature_type:
            reasons.append("Subsurface archaeological indicators")
        
        return "; ".join(reasons) if reasons else "Standard archaeological criteria met"
    
    def _get_quality_indicators(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quality assessment indicators"""
        return {
            "confidence_level": "high" if (detection.get('confidence') or 0.0) >= 0.8 else "moderate" if (detection.get('confidence') or 0.0) >= 0.5 else "low",
            "multi_sensor": detection.get('convergence_distance_m') is not None,
            "gedi_correlation": detection.get('gedi_support', False),
            "area_significance": "large" if (detection.get('area_m2') or 0) > 10000 else "medium" if (detection.get('area_m2') or 0) > 1000 else "small",
            "convergence_strength": self._get_convergence_strength(detection.get('convergence_distance_m'))
        }
    
    def _get_convergence_strength(self, convergence_distance: float) -> str:
        """Safely determine convergence strength, handling None values"""
        if convergence_distance is None:
            return "none"
        elif convergence_distance < 200:
            return "strong"
        elif convergence_distance < 500:
            return "moderate"
        else:
            return "weak"
    
    def _create_enhanced_geometry(self, detection: Dict[str, Any], fallback_coordinates: List[float]) -> Dict[str, Any]:
        """Create enhanced geometry that preserves full shapes when available"""
        
        # Check for full polygon coordinates (rectangles, circles as polygons)
        if 'geographic_polygon_coords' in detection:
            polygon_coords = detection['geographic_polygon_coords']
            if polygon_coords and len(polygon_coords) >= 3:
                logger.debug(f"Exporting full polygon geometry with {len(polygon_coords)} vertices")
                return {
                    "type": "Polygon",
                    "coordinates": [polygon_coords]  # GeoJSON polygon format
                }
        
        # Check for line coordinates (linear features)
        if 'geographic_line_coords' in detection:
            line_coords = detection['geographic_line_coords']
            if line_coords and len(line_coords) >= 2:
                logger.debug(f"Exporting full linestring geometry with {len(line_coords)} points")
                return {
                    "type": "LineString",
                    "coordinates": line_coords  # GeoJSON linestring format
                }
        
        # Check for circle data and create polygon approximation
        if detection.get('type') in ['geometric_circle', 'circular_anomaly'] and 'radius_m' in detection:
            radius_m = detection['radius_m']
            center_coords = fallback_coordinates
            if radius_m > 0 and len(center_coords) == 2:
                # Create circular polygon approximation
                circle_polygon = self._create_circle_polygon(center_coords[0], center_coords[1], radius_m)
                if circle_polygon:
                    logger.debug(f"Exporting circle as polygon with radius {radius_m}m")
                    return {
                        "type": "Polygon", 
                        "coordinates": [circle_polygon]
                    }
        
        # Fallback to point geometry
        logger.debug("Using point geometry (no shape coordinates available)")
        return {
            "type": "Point",
            "coordinates": fallback_coordinates
        }
    
    def _create_circle_polygon(self, center_lon: float, center_lat: float, radius_m: float, points: int = 32) -> List[List[float]]:
        """Create a polygon approximation of a circle"""
        import math
        
        try:
            # Convert radius from meters to degrees (approximate)
            radius_deg = radius_m / 111000  # Rough conversion at equator
            
            polygon_coords = []
            for i in range(points):
                angle = 2 * math.pi * i / points
                lon = center_lon + radius_deg * math.cos(angle)
                lat = center_lat + radius_deg * math.sin(angle)
                polygon_coords.append([lon, lat])
            
            # Close the polygon
            polygon_coords.append(polygon_coords[0])
            
            return polygon_coords
            
        except Exception as e:
            logger.warning(f"Failed to create circle polygon: {e}")
            return None
    
    def export_all_detections_with_rationale(self, all_detections: List[Dict[str, Any]], zone_name: str) -> Path:
        """Export ALL detections with quality rationale to show selection process"""
        try:
            self._ensure_export_dir(self.combined_exports)
            export_file = self.combined_exports / f"{zone_name}_all_detections_with_rationale.geojson"
            
            geojson_data = {
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                "features": []
            }
            
            for detection in all_detections:
                if detection.get('coordinates'):
                    coordinates = detection['coordinates']
                    if hasattr(coordinates, 'tolist'):
                        coordinates = coordinates.tolist()
                    elif not isinstance(coordinates, list):
                        coordinates = [float(coordinates[0]), float(coordinates[1])]
                    
                    # Coordinate validation and format correction
                    if (len(coordinates) == 2 and 
                        abs(coordinates[0]) < 20 and abs(coordinates[1]) > 44):
                        coordinates = [coordinates[1], coordinates[0]]
                    
                    if not self._validate_coordinates(coordinates, f"All_detections_{detection.get('type', 'unknown')}"):
                        coordinates = self._flag_invalid_coordinates(coordinates, zone_name, f"All_detections_{detection.get('type', 'unknown')}")
                    
                    confidence = detection.get('confidence') or 0
                    
                    # Determine if this feature would be exported in high-quality exports
                    provider = detection.get('provider', 'unknown')
                    would_export = False
                    export_reason = ""
                    
                    if provider == 'sentinel2' and confidence >= 0.50:
                        would_export = True
                        export_reason = "Meets Sentinel-2 50% confidence threshold"
                    elif provider == 'gedi' and confidence >= 0.40:
                        would_export = True
                        export_reason = "Meets GEDI 40% confidence threshold"
                    elif confidence >= 0.50:
                        would_export = True
                        export_reason = "Meets combined 50% confidence threshold"
                    else:
                        export_reason = f"Below confidence threshold ({confidence:.1%} < 50% for Sentinel-2 or 40% for GEDI)"
                    
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": coordinates
                        },
                        "properties": {
                            "type": detection.get('type', 'unknown'),
                            "confidence": confidence,
                            "area_m2": detection.get('area_m2', detection.get('area_km2', 0.0) * 1000000),
                            "zone": zone_name,
                            "provider": provider,
                            "run_id": self.run_id,
                            
                            # Export decision metadata
                            "exported_in_high_quality": would_export,
                            "export_decision_reason": export_reason,
                            
                            # Convergence metadata
                            "convergent_score": detection.get('convergent_score', 0.0),
                            "convergence_distance_m": detection.get('convergence_distance_m', None),
                            "combined_confidence": detection.get('combined_confidence', None),
                            "gedi_support": detection.get('gedi_support', False),
                            "convergence_type": detection.get('convergence_type', None),
                            
                            # Selection rationale
                            "selection_reason": self._get_selection_reason(detection),
                            "quality_indicators": self._get_quality_indicators(detection),
                            
                            # Visual styling based on export status
                            "marker_style": "exported" if would_export else "filtered",
                            "archaeological_grade": "high" if would_export else "unverified"
                        }
                    }
                    geojson_data["features"].append(feature)
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(geojson_data, f, indent=2)
            
            exported_count = sum(1 for f in geojson_data['features'] if f['properties']['exported_in_high_quality'])
            filtered_count = len(geojson_data['features']) - exported_count
            
            logger.info(f"📊 All detections with rationale: {len(geojson_data['features'])} total ({exported_count} exported, {filtered_count} filtered) → {export_file.name}")
            return export_file
            
        except Exception as e:
            logger.error(f"Failed to export all detections with rationale: {e}")
            return None
    
    # Field Investigation Planning Methods
    def _get_optimal_visit_season(self, zone_name: str) -> str:
        """Determine optimal season for field investigation"""
        zone_seasons = {
            'upano_valley_confirmed': 'May-September (dry season, river access)',
            'trombetas': 'June-November (low water, better helicopter access)',
            'upper_napo_micro': 'July-October (minimal rainfall, river navigable)'
        }
        return zone_seasons.get(zone_name, 'June-September (general dry season recommendation)')
    
    def _get_recommended_equipment(self, detection: Dict[str, Any]) -> List[str]:
        """Generate equipment recommendations based on feature type and environment"""
        equipment = [
            'GPS with sub-meter accuracy',
            'Ground-penetrating radar (GPR)',
            'Soil auger for core sampling',
            'Portable XRF for soil analysis',
            'High-resolution camera with macro lens',
            'Measuring tapes (30m, 100m)',
            'Archaeological field forms',
            'Sample bags and labels'
        ]
        
        feature_type = detection.get('type', '')
        area_m2 = detection.get('area_m2', 0)
        
        if 'terra_preta' in feature_type:
            equipment.extend([
                'Soil pH testing kit',
                'Phosphorus field test kit',
                'Charcoal identification guide'
            ])
        
        if 'geometric' in feature_type:
            equipment.extend([
                'Total station or theodolite',
                'Measuring wheel',
                'Archaeological grid stakes'
            ])
        
        if area_m2 > 50000:  # Large sites
            equipment.extend([
                'Drone with LiDAR',
                'RTK GPS base station',
                'Multi-spectral camera'
            ])
        
        return equipment
    
    def _estimate_investigation_time(self, detection: Dict[str, Any]) -> str:
        """Estimate field investigation duration"""
        area_m2 = detection.get('area_m2', 0)
        confidence = detection.get('confidence') or 0.0
        feature_type = detection.get('type', '')
        
        base_days = 3  # Minimum site visit
        
        # Area factor
        if area_m2 > 100000:
            base_days += 7
        elif area_m2 > 50000:
            base_days += 5
        elif area_m2 > 10000:
            base_days += 3
        
        # Confidence factor (higher confidence = more thorough investigation)
        if confidence > 0.8:
            base_days += 2
        
        # Feature complexity factor
        if 'geometric' in feature_type:
            base_days += 2  # Detailed mapping required
        
        if base_days <= 5:
            return f'{base_days} days (preliminary survey)'
        elif base_days <= 10:
            return f'{base_days} days (comprehensive survey)'
        else:
            return f'{base_days} days (extensive excavation campaign)'
    
    def _assess_logistics_complexity(self, detection: Dict[str, Any], zone_name: str) -> str:
        """Assess logistical complexity for field work"""
        zone_complexity = {
            'upano_valley_confirmed': 'Medium - River transport + short hike',
            'trombetas': 'High - Helicopter required, remote location',
            'upper_napo_micro': 'Medium - River transport feasible'
        }
        
        base_complexity = zone_complexity.get(zone_name, 'High - Remote location')
        
        area_m2 = detection.get('area_m2', 0)
        if area_m2 > 100000:
            return f'{base_complexity} + Extended campaign logistics'
        
        return base_complexity
    
    def _get_safety_considerations(self, zone_name: str) -> List[str]:
        """Get safety considerations for the zone"""
        general_safety = [
            'Tropical disease precautions (malaria, dengue)',
            'Emergency communication devices',
            'First aid kit with snake bite treatment',
            'Water purification system',
            'Emergency evacuation plan'
        ]
        
        zone_specific = {
            'upano_valley_confirmed': [
                'River safety equipment',
                'Landslide awareness in rainy season'
            ],
            'trombetas': [
                'Isolated location - medical evacuation critical',
                'Wildlife encounters (jaguars, snakes)'
            ],
            'upper_napo_micro': [
                'Border area security considerations',
                'River current safety'
            ]
        }
        
        return general_safety + zone_specific.get(zone_name, [])
    
    def _get_required_permits(self, zone_name: str) -> List[str]:
        """Get required permits for archaeological work"""
        base_permits = [
            'INPC archaeological research permit',
            'Environmental impact assessment',
            'Local community consultation'
        ]
        
        zone_permits = {
            'upano_valley_confirmed': ['Regional government coordination'],
            'trombetas': ['Protected area access permit'],
            'upper_napo_micro': ['Border area clearance']
        }
        
        return base_permits + zone_permits.get(zone_name, [])
    
    def _get_local_contacts(self, zone_name: str) -> List[str]:
        """Get relevant local contacts and institutions"""
        return [
            'Local INPC representative',
            'Indigenous community leaders',
            'Regional universities',
            'Local government archaeological office',
            'Tourist/guide services for logistics'
        ]
    
    # Research and Publication Methods
    def _assess_publication_readiness(self, detection: Dict[str, Any]) -> str:
        """Assess readiness for academic publication"""
        confidence = detection.get('confidence') or 0.0
        statistical_power = detection.get('statistical_power', 0.0)
        p_value = detection.get('p_value', 1.0)
        
        if confidence > 0.8 and statistical_power > 0.8 and p_value < 0.01:
            return 'High - Ready for peer review'
        elif confidence > 0.7 and statistical_power > 0.7 and p_value < 0.05:
            return 'Medium - Additional validation recommended'
        elif confidence > 0.5:
            return 'Low - Requires field verification'
        else:
            return 'Not ready - Insufficient confidence'
    
    def _assess_academic_significance(self, detection: Dict[str, Any]) -> str:
        """Assess academic significance of the detection"""
        feature_type = detection.get('type', '')
        area_m2 = detection.get('area_m2', 0)
        confidence = detection.get('confidence') or 0.0
        
        if 'geometric' in feature_type and area_m2 > 50000 and confidence > 0.8:
            return 'Very High - Major archaeological discovery potential'
        elif area_m2 > 100000 and confidence > 0.7:
            return 'High - Significant settlement complex'
        elif confidence > 0.8:
            return 'Medium-High - Strong detection confidence'
        elif area_m2 > 10000:
            return 'Medium - Notable archaeological feature'
        else:
            return 'Standard - Typical archaeological interest'
    
    def _estimate_citation_potential(self, detection: Dict[str, Any]) -> str:
        """Estimate citation potential for academic work"""
        significance = self._assess_academic_significance(detection)
        
        if 'Very High' in significance:
            return 'High citation potential (50+ citations expected)'
        elif 'High' in significance:
            return 'Good citation potential (20-50 citations expected)'
        elif 'Medium' in significance:
            return 'Moderate citation potential (10-20 citations expected)'
        else:
            return 'Standard citation potential (5-10 citations expected)'
    
    def _identify_collaborations(self, zone_name: str) -> List[str]:
        """Identify potential collaboration opportunities"""
        return [
            'Local universities archaeology departments',
            'International LiDAR research groups',
            'Remote sensing archaeology networks',
            'Amazon conservation organizations',
            'Indigenous knowledge holders',
            'Museum collections for comparative analysis'
        ]
    
    def _assess_dataset_completeness(self, detection: Dict[str, Any]) -> str:
        """Assess completeness of the dataset for research"""
        has_statistical = bool(detection.get('p_value') is not None)
        has_spatial = bool(detection.get('area_m2', 0) > 0)
        has_spectral = bool(detection.get('ndvi') is not None)
        has_context = bool(detection.get('elevation') is not None)
        
        completeness = sum([has_statistical, has_spatial, has_spectral, has_context])
        
        if completeness >= 3:
            return 'Complete - All major data components present'
        elif completeness >= 2:
            return 'Good - Most data components present'
        else:
            return 'Incomplete - Additional data collection needed'
"""
Archaeological detection validation framework
Provides cross-validation against known control areas
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from pathlib import Path
import json
from .parameter_configs import get_current_params

logger = logging.getLogger(__name__)

class ArchaeologicalValidator:
    """Validates archaeological detections against known control areas"""
    
    def __init__(self):
        self.known_sites = self._load_known_sites()
        self.non_archaeological_areas = self._load_control_areas()
        
    def _load_known_sites(self) -> List[Dict[str, Any]]:
        """Load known archaeological sites for validation"""
        # This would typically load from a database or file
        # For now, providing some example Upper Napo sites based on literature
        return [
            {
                "name": "Example Known Site 1",
                "lat": -0.48,
                "lon": -72.52,
                "radius_km": 0.5,
                "type": "confirmed_settlement",
                "confidence": 1.0
            },
            {
                "name": "Example Known Site 2", 
                "lat": -0.46,
                "lon": -72.54,
                "radius_km": 0.3,
                "type": "confirmed_earthworks",
                "confidence": 1.0
            }
        ]
    
    def _load_control_areas(self) -> List[Dict[str, Any]]:
        """Load known non-archaeological areas for false positive testing"""
        return [
            {
                "name": "Modern Settlement",
                "lat": -0.45,
                "lon": -72.50,
                "radius_km": 1.0,
                "type": "modern_infrastructure", 
                "confidence": 1.0
            },
            {
                "name": "Natural Forest",
                "lat": -0.52,
                "lon": -72.48,
                "radius_km": 2.0,
                "type": "pristine_forest",
                "confidence": 1.0
            },
            {
                "name": "Lago Agrio Oil Field",
                "lat": 0.085,  # Nueva Loja/Lago Agrio actual coordinates
                "lon": -76.894,  # Real oil infrastructure location
                "radius_km": 5.0,  # Major oil field coverage
                "type": "oil_infrastructure",
                "confidence": 0.95
            },
            {
                "name": "Shushufindi Oil Field",
                "lat": -0.160,  # Shushufindi actual coordinates  
                "lon": -76.895,  # Real petroleum infrastructure
                "radius_km": 3.0,  # Oil field operational area
                "type": "oil_infrastructure",
                "confidence": 0.95
            },
            {
                "name": "Cuyabeno Wetlands",
                "lat": -0.5,  # Cuyabeno Wildlife Reserve center
                "lon": -76.0,  # Real wetland protected area
                "radius_km": 10.0,  # Large wetland system
                "type": "wetland",
                "confidence": 0.9
            },
            {
                "name": "Yasuní Wetlands",
                "lat": -1.0,  # Yasuní National Park center
                "lon": -76.0,  # Protected wetland area
                "radius_km": 15.0,  # Large protected wetland zone
                "type": "wetland",
                "confidence": 0.9
            },
            {
                "name": "Infrastructure Exclusion Zone 1",
                "lat": -0.485,
                "lon": -72.515,
                "radius_km": 0.1,
                "type": "modern_infrastructure",
                "confidence": 1.0
            },
            {
                "name": "Infrastructure Exclusion Zone 2", 
                "lat": -0.495,
                "lon": -72.505,
                "radius_km": 0.1,
                "type": "modern_infrastructure",
                "confidence": 1.0
            }
        ]
    
    def validate_detections(self, detections: List[Dict[str, Any]], 
                          zone_name: str = None) -> Dict[str, Any]:
        """
        Validate detections against known sites and control areas
        
        Args:
            detections: List of detected features with coordinates
            zone_name: Name of the target zone
            
        Returns:
            Validation results with metrics and flagged detections
        """
        if not detections:
            return {
                "total_detections": 0,
                "validation_metrics": {},
                "flagged_detections": [],
                "recommendations": ["No detections to validate"]
            }
        
        # Extract coordinates from detections
        detection_coords = []
        for detection in detections:
            if 'coordinates' in detection and len(detection['coordinates']) >= 2:
                lat, lon = detection['coordinates'][0], detection['coordinates'][1]
                detection_coords.append((lat, lon, detection))
        
        # Validate against known sites (true positives)
        near_known_sites = []
        for lat, lon, detection in detection_coords:
            for site in self.known_sites:
                distance_km = self._calculate_distance(lat, lon, site['lat'], site['lon'])
                if distance_km <= site['radius_km']:
                    near_known_sites.append({
                        "detection": detection,
                        "known_site": site,
                        "distance_km": distance_km,
                        "validation": "potential_true_positive"
                    })
        
        # Check for detections in control areas (potential false positives)
        in_control_areas = []
        for lat, lon, detection in detection_coords:
            for control in self.non_archaeological_areas:
                distance_km = self._calculate_distance(lat, lon, control['lat'], control['lon'])
                if distance_km <= control['radius_km']:
                    in_control_areas.append({
                        "detection": detection,
                        "control_area": control,
                        "distance_km": distance_km,
                        "validation": "potential_false_positive"
                    })
        
        # Calculate validation metrics
        total_detections = len(detection_coords)
        potential_true_positives = len(near_known_sites)
        potential_false_positives = len(in_control_areas)
        
        # Detection density analysis
        area_coverage_km2 = self._estimate_area_coverage(detection_coords)
        detection_density = total_detections / (area_coverage_km2 + 1e-6)  # per km²
        
        # Flag suspicious detections
        flagged = []
        
        # Flag very high confidence detections in control areas
        for item in in_control_areas:
            if item['detection'].get('confidence', 0) > 0.8:
                flagged.append({
                    "detection": item['detection'],
                    "flag_reason": f"High confidence detection in {item['control_area']['type']}",
                    "severity": "high"
                })
        
        # Flag very large features
        for detection in detections:
            area_m2 = detection.get('area_m2', 0)
            if area_m2 > 100000:  # 10 hectares
                flagged.append({
                    "detection": detection,
                    "flag_reason": f"Very large feature ({area_m2/10000:.1f} hectares)",
                    "severity": "medium"
                })
        
        # ARCHAEOLOGICAL DENSITY ANALYSIS - Evidence-based expectations
        # Get current validation parameters
        params = get_current_params()
        val_params = params['validation']
        
        # Convert detection density to per-100km² format
        density_per_100km2 = detection_density * 100
        
        # Determine expected density based on area type (use terra firme as default for now)
        expected_max_density = val_params.expected_max_density_terra_firme  # per km²
        expected_min_density = val_params.expected_min_density  # per km²
        
        if density_per_100km2 > expected_max_density:
            flagged.append({
                "detection": None,  # This is a general flag
                "flag_reason": f"Detection density ({density_per_100km2:.1f} per 100km²) exceeds expected range ({expected_min_density}-{expected_max_density} per 100km²)",
                "severity": "high",
                "area_wide_issue": True
            })
        elif density_per_100km2 < expected_min_density:
            flagged.append({
                "detection": None,  # This is a general flag
                "flag_reason": f"Detection density ({density_per_100km2:.1f} per 100km²) below expected range ({expected_min_density}-{expected_max_density} per 100km²) - may indicate under-detection",
                "severity": "medium",
                "area_wide_issue": True
            })
        
        # Flag features with unrealistic archaeological signatures
        for detection in detections:
            # Flag features with perfect geometric alignment (likely processing artifacts)
            if detection.get('type') in ['line', 'rectangle']:
                angle = detection.get('angle_degrees', 0)
                if abs(angle % 90) < 2:  # Within 2 degrees of cardinal directions
                    flagged.append({
                        "detection": detection,
                        "flag_reason": "Perfect cardinal alignment suggests processing artifact",
                        "severity": "medium"
                    })
            
            # ARCHAEOLOGICAL CONFIDENCE STANDARDS - Based on research literature
            confidence = detection.get('confidence', 0)
            if confidence > 0.98:  # Extremely high confidence should be very rare
                flagged.append({
                    "detection": detection,
                    "flag_reason": "Unrealistically high confidence (>98%)",
                    "severity": "medium"
                })
            elif confidence < val_params.eventual_confidence_threshold:  # Below eventual TP threshold (50%)
                flagged.append({
                    "detection": detection,
                    "flag_reason": f"Below minimum archaeological confidence threshold ({val_params.eventual_confidence_threshold*100:.0f}%)",
                    "severity": "high"
                })
            elif confidence < val_params.probable_confidence_threshold:  # Below probable TP threshold (60%)
                flagged.append({
                    "detection": detection,
                    "flag_reason": f"Low archaeological confidence ({confidence*100:.0f}% < {val_params.probable_confidence_threshold*100:.0f}%)",
                    "severity": "medium"
                })
        
        # ARCHAEOLOGICAL DENSITY FLAGGING - Based on literature expectations  
        if detection_density > val_params.expected_max_density_varzea:  # Even higher than floodplain max
            flagged.append({
                "detection": "overall_density",
                "flag_reason": f"Extremely high detection density ({detection_density:.1f}/km²) exceeds even floodplain maximums",
                "severity": "high"
            })
        elif detection_density > expected_max_density:  # Above expected for area type
            flagged.append({
                "detection": "overall_density",
                "flag_reason": f"High detection density ({detection_density:.1f}/km²) above expected for terrain type",
                "severity": "medium"
            })
        
        # Generate recommendations
        recommendations = []
        if potential_false_positives > potential_true_positives:
            recommendations.append("High false positive rate detected - increase confidence thresholds")
        if detection_density > 5:
            recommendations.append("Detection density appears high - verify algorithms are not over-detecting")
        if len(flagged) > total_detections * 0.2:
            recommendations.append("More than 20% of detections flagged - review detection parameters")
        
        # Temporal validation disabled - no historical data generation
        temporal_validation = {"status": "disabled", "message": "Historical data generation removed"}
        
        # Add temporal flags to main flagged list (handle None case)
        if temporal_validation:
            flagged.extend(temporal_validation.get('temporal_flags', []))
            recommendations.extend(temporal_validation.get('temporal_recommendations', []))
        else:
            temporal_validation = {
                "temporal_validation_status": "failed",
                "persistence_rate": 0,
                "temporal_flags": [],
                "temporal_recommendations": ["Temporal validation failed - manual review recommended"]
            }
        
        validation_results = {
            "total_detections": total_detections,
            "validation_metrics": {
                "potential_true_positives": potential_true_positives,
                "potential_false_positives": potential_false_positives,
                "detection_density_per_km2": detection_density,
                "area_coverage_km2": area_coverage_km2,
                "true_positive_rate": potential_true_positives / max(1, total_detections),
                "false_positive_rate": potential_false_positives / max(1, total_detections),
                "temporal_persistence_rate": temporal_validation.get('persistence_rate', 0)
            },
            "flagged_detections": flagged,
            "near_known_sites": near_known_sites,
            "in_control_areas": in_control_areas,
            "temporal_validation": temporal_validation,
            "recommendations": recommendations,
            "validation_status": "completed"
        }
        
        # Save validation results to files
        self._save_validation_results(validation_results, zone_name)
        
        # Historical data saving disabled
        
        return validation_results
    
    def validate_temporal_persistence(self, detections: List[Dict[str, Any]], 
                                    historical_detections: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Temporal validation disabled - historical data generation removed"""
        return {
            "temporal_validation_status": "disabled",
            "message": "Historical data generation removed",
            "temporal_flags": [],
            "temporal_recommendations": []
        }
    
    def validate_multi_sensor_convergence(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate multi-sensor convergence requirement
        
        Args:
            detections: List of detections with provider information
            
        Returns:
            Convergence validation results
        """
        # Separate by provider
        sentinel2_detections = [d for d in detections if d.get('provider') == 'sentinel2']
        gedi_detections = [d for d in detections if d.get('provider') == 'gedi']
        convergent_detections = [d for d in detections if d.get('multi_sensor_agreement', False)]
        
        total_detections = len(detections)
        convergent_count = len(convergent_detections)
        convergence_rate = convergent_count / max(1, total_detections)
        
        # Analyze convergence quality
        convergence_quality = "poor"
        if convergence_rate > 0.8:
            convergence_quality = "excellent"
        elif convergence_rate > 0.6:
            convergence_quality = "good"
        elif convergence_rate > 0.4:
            convergence_quality = "moderate"
        
        # Calculate average convergence distance
        convergence_distances = [d.get('convergence_distance_m') or 0 for d in convergent_detections]
        avg_convergence_distance = np.mean(convergence_distances) if convergence_distances else 0
        
        validation_flags = []
        
        # Flag if convergence rate is too low
        if convergence_rate < 0.5:
            validation_flags.append({
                "flag": "low_convergence_rate",
                "message": f"Only {convergence_rate*100:.1f}% of detections have multi-sensor agreement",
                "severity": "high"
            })
        
        # Flag if convergence distances are too large
        if avg_convergence_distance > 300:  # 300m threshold
            validation_flags.append({
                "flag": "large_convergence_distance",
                "message": f"Average convergence distance ({avg_convergence_distance:.0f}m) exceeds optimal range",
                "severity": "medium"
            })
        
        return {
            "total_detections": total_detections,
            "sentinel2_detections": len(sentinel2_detections),
            "gedi_detections": len(gedi_detections),
            "convergent_detections": convergent_count,
            "convergence_rate": convergence_rate,
            "convergence_quality": convergence_quality,
            "avg_convergence_distance_m": avg_convergence_distance,
            "validation_flags": validation_flags,
            "multi_sensor_requirement_met": convergence_rate > 0.3  # At least 30% convergence
        }
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in km"""
        # Simple haversine distance
        R = 6371  # Earth radius in km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2)**2 + 
             np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c
    
    def _estimate_area_coverage(self, detection_coords: List[Tuple[float, float, Any]]) -> float:
        """Estimate the area coverage of detections in km² using zone configuration"""
        if len(detection_coords) < 2:
            return 1.0  # Default 1 km² for single detection
        
        # Try to get zone area from configuration
        try:
            from .config import TARGET_ZONES
            
            # Attempt to identify zone from detection coordinates
            lats = [coord[0] for coord in detection_coords]
            lons = [coord[1] for coord in detection_coords]
            center_lat, center_lon = np.mean(lats), np.mean(lons)
            
            # Find closest zone
            closest_zone = None
            min_distance = float('inf')
            
            for zone_id, zone in TARGET_ZONES.items():
                zone_lat, zone_lon = zone.center
                distance = self._calculate_distance(center_lat, center_lon, zone_lat, zone_lon)
                if distance < min_distance:
                    min_distance = distance
                    closest_zone = zone
            
            # Use zone search area if found and distance is reasonable (< 50km)
            if closest_zone and min_distance < 50:
                # Calculate area from search radius
                area_km2 = 3.14159 * (closest_zone.search_radius_km ** 2)
                return area_km2
                
        except Exception as e:
            logger.warning(f"Could not use zone configuration for area calculation: {e}")
        
        # Fallback to bounding box calculation
        lats = [coord[0] for coord in detection_coords]
        lons = [coord[1] for coord in detection_coords]
        
        # Simple bounding box area calculation
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        
        # Convert to approximate km (rough approximation)
        lat_km = lat_range * 111  # ~111 km per degree latitude
        lon_km = lon_range * 111 * np.cos(np.radians(np.mean(lats)))  # longitude varies with latitude
        
        return max(1.0, lat_km * lon_km)  # Minimum 1 km²
    
    def _save_validation_results(self, validation_results: Dict[str, Any], zone_name: str):
        """Save validation results to output files"""
        try:
            # Check if validation_results is None or empty
            if not validation_results:
                logger.warning(f"No validation results to save for zone {zone_name}")
                return
                
            from datetime import datetime
            import os
            from pathlib import Path
            
            # Try to import RESULTS_DIR, fallback to creating a results directory
            try:
                from ..config import RESULTS_DIR
            except ImportError:
                try:
                    from src.core.config import RESULTS_DIR
                except ImportError:
                    # Fallback to default results directory
                    current_dir = Path(__file__).parent.parent.parent
                    RESULTS_DIR = current_dir / "results"
                    RESULTS_DIR.mkdir(exist_ok=True)
            
            # Find the most recent run directory
            run_folders = [d for d in RESULTS_DIR.glob("run_*") if d.is_dir()]
            if run_folders:
                run_folder = max(run_folders, key=os.path.getmtime)
            else:
                # Create a timestamped validation folder
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_folder = RESULTS_DIR / f"validation_{timestamp}"
                run_folder.mkdir(exist_ok=True)
            
            # Create validation subdirectory
            validation_dir = run_folder / "validation"
            validation_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Save detailed JSON results
            json_file = validation_dir / f"validation_results_{zone_name}_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(validation_results, f, indent=2, default=str)
            
            # 2. Save human-readable summary report
            summary_file = validation_dir / f"validation_summary_{zone_name}_{timestamp}.md"
            self._create_validation_report(validation_results, summary_file, zone_name)
            
            # 3. Save flagged detections as CSV
            if validation_results and validation_results.get('flagged_detections'):
                csv_file = validation_dir / f"flagged_detections_{zone_name}_{timestamp}.csv"
                self._save_flagged_detections_csv(validation_results['flagged_detections'], csv_file)
            
            logger.info(f"📊 Validation results saved to: {validation_dir}")
            logger.info(f"   - JSON details: {json_file.name}")
            logger.info(f"   - Summary report: {summary_file.name}")
            if validation_results and validation_results.get('flagged_detections'):
                logger.info(f"   - Flagged detections: {csv_file.name}")
                
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")
    
    def _create_validation_report(self, validation_results: Dict[str, Any], output_file: Path, zone_name: str):
        """Create a human-readable validation report"""
        from datetime import datetime
        
        metrics = validation_results.get('validation_metrics', {})
        flagged = validation_results.get('flagged_detections', [])
        recommendations = validation_results.get('recommendations', [])
        
        report = f"""# Archaeological Detection Validation Report

## Zone: {zone_name}
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary Statistics
- **Total Detections:** {validation_results.get('total_detections', 0)}
- **Detection Density:** {metrics.get('detection_density_per_km2', 0):.2f} features/km²
- **Area Coverage:** {metrics.get('area_coverage_km2', 0):.2f} km²

## Validation Metrics
- **Potential True Positives:** {metrics.get('potential_true_positives', 0)}
- **Potential False Positives:** {metrics.get('potential_false_positives', 0)}
- **True Positive Rate:** {metrics.get('true_positive_rate', 0):.2%}
- **False Positive Rate:** {metrics.get('false_positive_rate', 0):.2%}

## Quality Assessment
"""
        
        # Add quality assessment
        density = metrics.get('detection_density_per_km2', 0)
        fp_rate = metrics.get('false_positive_rate', 0)
        
        if density > 10:
            report += "⚠️  **HIGH DENSITY WARNING**: Detection density > 10/km² suggests over-detection\n\n"
        elif density > 5:
            report += "⚠️  **MODERATE DENSITY**: Detection density indicates thorough coverage\n\n"
        else:
            report += "✅ **NORMAL DENSITY**: Detection density within expected range\n\n"
        
        if fp_rate > 0.5:
            report += "🚨 **HIGH FALSE POSITIVE RATE**: >50% of detections may be false positives\n\n"
        elif fp_rate > 0.2:
            report += "⚠️  **MODERATE FALSE POSITIVE RATE**: Consider increasing confidence thresholds\n\n"
        else:
            report += "✅ **LOW FALSE POSITIVE RATE**: Good detection specificity\n\n"
        
        # Add flagged detections summary
        if flagged:
            report += f"## Flagged Detections ({len(flagged)} total)\n\n"
            
            high_severity = [f for f in flagged if f.get('severity') == 'high']
            medium_severity = [f for f in flagged if f.get('severity') == 'medium']
            
            if high_severity:
                report += f"### High Severity Issues ({len(high_severity)})\n"
                for flag in high_severity[:5]:  # Show first 5
                    report += f"- {flag.get('flag_reason', 'Unknown issue')}\n"
                if len(high_severity) > 5:
                    report += f"- ... and {len(high_severity) - 5} more\n"
                report += "\n"
            
            if medium_severity:
                report += f"### Medium Severity Issues ({len(medium_severity)})\n"
                for flag in medium_severity[:5]:  # Show first 5
                    report += f"- {flag.get('flag_reason', 'Unknown issue')}\n"
                if len(medium_severity) > 5:
                    report += f"- ... and {len(medium_severity) - 5} more\n"
                report += "\n"
        
        # Add recommendations
        if recommendations:
            report += "## Recommendations\n\n"
            for i, rec in enumerate(recommendations, 1):
                report += f"{i}. {rec}\n"
            report += "\n"
        
        # Add threshold recommendations
        report += "## Suggested Parameter Adjustments\n\n"
        if fp_rate > 0.3:
            report += "- **Increase confidence thresholds** from 0.75 to 0.85+\n"
        if density > 8:
            report += "- **Add stricter size filtering** to reduce over-detection\n"
        if len(flagged) > validation_results.get('total_detections', 1) * 0.3:
            report += "- **Review detection algorithms** - high flag rate indicates systematic issues\n"
        
        report += """
## Files Generated
- Detailed JSON results with all detection data
- Flagged detections CSV for manual review
- This validation summary report

---
*Generated by Archaeological Detection Validation System*
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
    
    def _save_flagged_detections_csv(self, flagged_detections: List[Dict], output_file: Path):
        """Save flagged detections to CSV format"""
        import csv
        
        if not flagged_detections:
            return
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'Detection_Type', 'Coordinates_Lat', 'Coordinates_Lon', 
                'Confidence', 'Area_m2', 'Flag_Reason', 'Severity'
            ])
            
            # Write flagged detections
            for flag in flagged_detections:
                detection = flag.get('detection', {})
                
                # Extract coordinates
                coords = detection.get('coordinates', [None, None])
                lat = coords[0] if len(coords) > 0 else 'N/A'
                lon = coords[1] if len(coords) > 1 else 'N/A'
                
                writer.writerow([
                    detection.get('type', 'unknown'),
                    lat,
                    lon,
                    detection.get('confidence', 'N/A'),
                    detection.get('area_m2', 'N/A'),
                    flag.get('flag_reason', 'N/A'),
                    flag.get('severity', 'unknown')
                ])
    
    # Historical data methods removed
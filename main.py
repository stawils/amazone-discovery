#!/usr/bin/env python3
"""
Amazon Archaeological Discovery Pipeline - Main Execution
Complete automated archaeological site detection system
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import os
from dotenv import load_dotenv
from types import SimpleNamespace

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import TARGET_ZONES, RESULTS_DIR, EXPORTS_DIR, REPORTS_DIR, MAPS_DIR
from src.usgs_api import USGSArchaeologyAPI, USGSAPIError, USGSProvider
from src.gee_provider import GEEProvider
from src.detectors import ArchaeologicalDetector
from src.scoring import ConvergentAnomalyScorer
from src.visualizers import ArchaeologicalVisualizer
from src.data_objects import SceneData
from src.pipeline_steps.modular_pipeline import ModularPipeline

# Setup logging
def setup_logging(verbose: bool = False):
    """Configure logging for the pipeline"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # File handler
    log_file = RESULTS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    return log_file

logger = logging.getLogger(__name__)

# Add this helper for custom zone
def ensure_custom_zone():
    if 'custom' not in TARGET_ZONES:
        TARGET_ZONES['custom'] = SimpleNamespace(
            name='Custom Data',
            center=(0.0, 0.0),
            priority=3,
            expected_features='(custom input)',
            historical_evidence='(user-supplied)',
            search_radius_km=0,
            min_feature_size_m=0,
            max_feature_size_m=0,
            bbox=(-10, -70, 10, -50)
        )

class ArchaeologicalPipeline:
    """Main archaeological discovery pipeline"""
    
    def __init__(self, provider='usgs'):
        self.usgs_api = None
        self.results = {}
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._provider = provider
        self.provider_instance = None  # New: holds the BaseProvider instance
        # Instantiate provider abstraction if possible
        if provider == 'usgs':
            try:
                self.provider_instance = USGSProvider()
            except Exception as e:
                logger.error(f"Failed to initialize USGSProvider: {e}")
                self.provider_instance = None
        elif provider == 'gee':
            try:
                self.provider_instance = GEEProvider()
            except Exception as e:
                logger.error(f"Failed to initialize GEEProvider: {e}")
                self.provider_instance = None
        elif provider == 'both':
            # For 'both', keep both providers
            try:
                self.provider_instance = [USGSProvider(), GEEProvider()]
            except Exception as e:
                logger.error(f"Failed to initialize both providers: {e}")
                self.provider_instance = None
        # Fallback to old logic if needed
        if provider in ['usgs', 'both'] and self.provider_instance is None:
            try:
                self.usgs_api = USGSArchaeologyAPI()
            except USGSAPIError as e:
                if provider == 'usgs':
                    logger.error(f"Failed to initialize USGS API: {e}")
                    logger.error("Please check your USGS_USERNAME and USGS_TOKEN environment variables")
                    sys.exit(1)
                else:
                    logger.warning(f"USGS API not available: {e}")
                    logger.warning("Continuing with other providers only")
                    self.usgs_api = None
        else:
            logger.info("USGS API not initialized (provider is GEE only)")
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.usgs_api:
            self.usgs_api.logout()
    
    def list_target_zones(self):
        """Display all configured target zones"""
        print("\n🎯 AMAZON ARCHAEOLOGICAL DISCOVERY - TARGET ZONES")
        print("=" * 60)
        
        # Sort by priority
        sorted_zones = sorted(TARGET_ZONES.items(), key=lambda x: x[1].priority)
        
        for zone_id, zone in sorted_zones:
            print(f"\n📍 {zone.name.upper()}")
            print(f"   ID: {zone_id}")
            print(f"   Coordinates: {zone.center[0]:.4f}°, {zone.center[1]:.4f}°")
            print(f"   Priority: {zone.priority} {'⭐' * (4 - zone.priority)}")
            print(f"   Expected: {zone.expected_features}")
            print(f"   Evidence: {zone.historical_evidence}")
            print(f"   Search Area: {zone.search_radius_km} km radius")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total zones: {len(TARGET_ZONES)}")
        print(f"   Priority 1 (Highest): {sum(1 for z in TARGET_ZONES.values() if z.priority == 1)}")
        print(f"   Priority 2 (High): {sum(1 for z in TARGET_ZONES.values() if z.priority == 2)}")
        print(f"   Priority 3 (Medium): {sum(1 for z in TARGET_ZONES.values() if z.priority == 3)}")
    
    def download_satellite_data(self, zones: list = None, max_scenes: int = 3, provider: str = 'usgs'):
        """Download satellite data for target zones using specified provider. Also returns SceneData objects."""
        if zones is None:
            # Default to Priority 1 zones
            zones = [k for k, v in TARGET_ZONES.items() if v.priority == 1]
        elif zones == ['all']:
            zones = list(TARGET_ZONES.keys())
        logger.info(f"Downloading satellite data for zones: {zones} using provider: {provider}")
        downloads = {}
        all_scene_data = []
        # New: Use provider abstraction if available
        if self.provider_instance is not None:
            if isinstance(self.provider_instance, list):
                # Both providers
                for prov in self.provider_instance:
                    try:
                        prov_scene_data = prov.download_data(zones, max_scenes)
                        all_scene_data.extend(prov_scene_data)
                    except Exception as e:
                        logger.error(f"Provider {prov.__class__.__name__} failed: {e}")
            else:
                try:
                    all_scene_data = self.provider_instance.download_data(zones, max_scenes)
                except Exception as e:
                    logger.error(f"Provider {self.provider_instance.__class__.__name__} failed: {e}")
            # For backward compatibility, keep downloads dict empty or fill if needed
            return downloads, all_scene_data
        # Fallback: old logic
        if provider == 'usgs' or provider == 'both':
            if self.usgs_api:
                logger.info("📡 Using USGS M2M API...")
                try:
                    usgs_result = self.usgs_api.batch_download_zones(zones, max_scenes)
                    for zone, result in usgs_result.items():
                        downloads[zone] = result.get('file_paths', [])
                        all_scene_data.extend(result.get('scene_data', []))
                except Exception as e:
                    logger.error(f"USGS download failed: {e}")
                    if provider == 'usgs':  # If USGS-only, return error
                        return {}, []
            else:
                logger.warning("USGS API not available - skipping USGS download")
                if provider == 'usgs':
                    logger.error("USGS provider requested but API not initialized")
                    return {}, []
        if provider == 'gee' or provider == 'both':
            logger.info("🌍 Using Google Earth Engine...")
            try:
                from src.gee_provider import GoogleEarthEngineProvider
                gee_provider = GoogleEarthEngineProvider()
                gee_result = gee_provider.batch_analyze_zones(zones, max_scenes)
                for zone, result in gee_result.items():
                    downloads[f"{zone}_gee"] = [result.get('result', {})]
                    all_scene_data.extend(result.get('scene_data', []))
            except Exception as e:
                logger.error(f"Google Earth Engine failed: {e}")
                if provider == 'gee':  # If GEE-only, return error
                    return {}, []
        # Log results
        total_items = sum(len(items) for items in downloads.values())
        logger.info(f"✓ Downloaded/processed {total_items} items across {len(downloads)} zones")
        # Save download manifest
        manifest = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'provider': provider,
            'zones_processed': zones,
            'downloads': {k: [str(p) if hasattr(p, '__str__') else str(p) for p in v] 
                         for k, v in downloads.items()},
            'total_items': total_items
        }
        manifest_path = RESULTS_DIR / f"download_manifest_{self.session_id}.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Download manifest saved: {manifest_path}")
        # New: return both downloads and all_scene_data for modular pipeline
        return downloads, all_scene_data
    
    def analyze_downloaded_data(self, zones: list = None, data_path: str = None, scene_data_list: list = None):
        """Analyze downloaded satellite data for archaeological features (raster and vector, flexible path or SceneData list)"""
        analysis_results = {}
        # New: If scene_data_list is provided, analyze these scenes directly
        if scene_data_list is not None:
            logger.info("Analyzing provided SceneData objects...")
            for scene in scene_data_list:
                zone_id = scene.zone_id
                zone = TARGET_ZONES.get(zone_id, None)
                detector = ArchaeologicalDetector(zone)
                # Try to use file_paths if available, else skip
                scene_dir = None
                if scene.file_paths:
                    # If file_paths contains a directory, use it; else, use the first file's parent
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
            self.results['analysis'] = analysis_results
            return analysis_results
        # Old logic: file system search
        if zones is None:
            zones = [k for k, v in TARGET_ZONES.items() if v.priority <= 2]
        elif zones == ['all']:
            zones = list(TARGET_ZONES.keys())
        analysis_results = {}
        for zone_id in zones:
            if zone_id not in TARGET_ZONES:
                logger.warning(f"Unknown zone: {zone_id}")
                continue
            zone = TARGET_ZONES[zone_id]
            logger.info(f"\n🔍 Analyzing {zone.name}")
            zone_dir = Path("data/satellite") / zone.name.lower().replace(' ', '_')
            if not zone_dir.exists():
                logger.warning(f"No data directory found for {zone.name}: {zone_dir}")
                continue
            # Find scene directories (raster) and vector files
            scene_dirs = [d for d in zone_dir.iterdir() if d.is_dir()]
            vector_files = list(zone_dir.glob("*.kml")) + list(zone_dir.glob("*.geojson"))
            zone_results = []
            detector = ArchaeologicalDetector(zone)
            # Process raster scenes if present
            if scene_dirs:
                for scene_dir in scene_dirs[:3]:  # Analyze top 3 scenes
                    logger.info(f"  Analyzing scene: {scene_dir.name}")
                    try:
                        result = detector.analyze_scene(scene_dir)
                        if result.get('success'):
                            zone_results.append(result)
                            export_path = EXPORTS_DIR / f"{zone_id}_{scene_dir.name}_detections.geojson"
                            detector.export_detections_to_geojson(export_path)
                            logger.info(f"  ✓ Found {result['total_features']} features")
                        else:
                            logger.warning(f"  ❌ Analysis failed: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        logger.error(f"  ❌ Error analyzing {scene_dir}: {e}")
            # Process vector files if present
            if vector_files:
                for vf in vector_files:
                    logger.info(f"  Analyzing vector file: {vf.name}")
                    try:
                        result = detector.analyze_vector_scene(vf)
                        if result.get('success'):
                            zone_results.append(result)
                            export_path = EXPORTS_DIR / f"{zone_id}_{vf.stem}_detections.geojson"
                            detector.export_detections_to_geojson(export_path)
                            logger.info(f"  ✓ Found {result['total_features']} vector features")
                        else:
                            logger.warning(f"  ❌ Vector analysis failed: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        logger.error(f"  ❌ Error analyzing vector file {vf}: {e}")
            if not scene_dirs and not vector_files:
                logger.warning(f"No scenes or vector files found in {zone_dir}")
            analysis_results[zone_id] = zone_results
            logger.info(f"✓ Completed analysis for {zone.name}: {len(zone_results)} successful scenes/files")
        self.results['analysis'] = analysis_results
        return analysis_results
    
    def calculate_convergent_scores(self, analysis_results: dict = None):
        """Calculate convergent anomaly scores for all detections"""
        
        if analysis_results is None:
            analysis_results = self.results.get('analysis', {})
        
        if not analysis_results:
            logger.warning("No analysis results available for scoring")
            return {}
        
        logger.info("🧮 Calculating convergent anomaly scores...")
        
        scorer = ConvergentAnomalyScorer()
        scoring_results = {}
        
        for zone_id, zone_results in analysis_results.items():
            if not zone_results:
                continue
                
            zone = TARGET_ZONES[zone_id]
            logger.info(f"  Scoring {zone.name}...")
            
            # Combine all detections from multiple scenes
            all_features = {
                'terra_preta_patches': [],
                'geometric_features': []
            }
            
            for scene_result in zone_results:
                if scene_result.get('success'):
                    # Terra preta patches
                    tp_patches = scene_result.get('terra_preta', {}).get('patches', [])
                    all_features['terra_preta_patches'].extend(tp_patches)
                    
                    # Geometric features
                    geom_features = scene_result.get('geometric_features', [])
                    all_features['geometric_features'].extend(geom_features)
            
            # Calculate score for this zone
            zone_score = scorer.calculate_zone_score(zone_id, all_features)
            scoring_results[zone_id] = zone_score
            
            logger.info(f"  ✓ {zone.name}: {zone_score['total_score']}/15 points ({zone_score['classification']})")
        
        self.results['scores'] = scoring_results
        return scoring_results
    
    def generate_discovery_report(self):
        """Generate comprehensive archaeological discovery report"""
        
        logger.info("📊 Generating discovery report...")
        
        if not self.results:
            logger.warning("No results available for report generation")
            return None
        
        analysis_results = self.results.get('analysis', {})
        scoring_results = self.results.get('scores', {})
        
        # Create comprehensive report
        report = {
            'session_info': {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'pipeline_version': '1.0',
                'zones_analyzed': list(analysis_results.keys())
            },
            'executive_summary': self._generate_executive_summary(scoring_results),
            'zone_details': self._generate_zone_details(analysis_results, scoring_results),
            'high_priority_sites': self._extract_high_priority_sites(scoring_results),
            'recommendations': self._generate_recommendations(scoring_results),
            'technical_summary': self._generate_technical_summary(analysis_results)
        }
        
        # Save detailed JSON report
        report_path = REPORTS_DIR / f"archaeological_discovery_report_{self.session_id}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate human-readable report
        readable_report_path = REPORTS_DIR / f"discovery_report_{self.session_id}.md"
        self._write_readable_report(report, readable_report_path)
        
        # Print executive summary to console
        self._print_executive_summary(report['executive_summary'])
        
        logger.info(f"✓ Reports saved:")
        logger.info(f"  Detailed JSON: {report_path}")
        logger.info(f"  Readable markdown: {readable_report_path}")
        
        return report
    
    def _generate_executive_summary(self, scoring_results: dict) -> dict:
        """Generate executive summary of discoveries"""
        # Always return all expected keys, even if no results
        if not scoring_results:
            return {
                'total_zones_analyzed': 0,
                'high_confidence_sites': 0,
                'probable_archaeological_features': 0,
                'total_features_detected': 0,
                'highest_scoring_zone': None,
                'success_rate': "0%"
            }
        high_confidence = sum(1 for r in scoring_results.values() if r['total_score'] >= 10)
        probable_features = sum(1 for r in scoring_results.values() if r['total_score'] >= 7)
        total_features = sum(len(r.get('feature_details', [])) for r in scoring_results.values())
        # Find highest scoring zone
        best_zone = max(scoring_results.items(), key=lambda x: x[1]['total_score'])
        return {
            'total_zones_analyzed': len(scoring_results),
            'high_confidence_sites': high_confidence,
            'probable_archaeological_features': probable_features,
            'total_features_detected': total_features,
            'highest_scoring_zone': {
                'zone_id': best_zone[0],
                'zone_name': TARGET_ZONES[best_zone[0]].name,
                'score': best_zone[1]['total_score'],
                'classification': best_zone[1]['classification']
            },
            'success_rate': f"{(probable_features / len(scoring_results) * 100):.1f}%" if scoring_results else "0%"
        }
    
    def _generate_zone_details(self, analysis_results: dict, scoring_results: dict) -> dict:
        """Generate detailed information for each zone"""
        
        zone_details = {}
        
        for zone_id in analysis_results.keys():
            zone = TARGET_ZONES[zone_id]
            analysis = analysis_results.get(zone_id, [])
            scores = scoring_results.get(zone_id, {})
            
            # Count features across all scenes
            total_tp_patches = sum(len(scene.get('terra_preta', {}).get('patches', [])) 
                                 for scene in analysis if scene.get('success'))
            total_geometric = sum(len(scene.get('geometric_features', [])) 
                                for scene in analysis if scene.get('success'))
            
            zone_details[zone_id] = {
                'zone_info': {
                    'name': zone.name,
                    'coordinates': zone.center,
                    'priority': zone.priority,
                    'expected_features': zone.expected_features,
                    'historical_evidence': zone.historical_evidence
                },
                'analysis_summary': {
                    'scenes_analyzed': len([s for s in analysis if s.get('success')]),
                    'terra_preta_patches': total_tp_patches,
                    'geometric_features': total_geometric,
                    'total_detections': total_tp_patches + total_geometric
                },
                'convergent_score': scores.get('total_score', 0),
                'classification': scores.get('classification', 'No analysis'),
                'evidence_summary': scores.get('evidence_summary', []),
                'recommended_action': self._get_recommended_action(scores.get('total_score', 0))
            }
        
        return zone_details
    
    def _extract_high_priority_sites(self, scoring_results: dict) -> list:
        """Extract sites requiring immediate ground verification"""
        
        high_priority = []
        
        for zone_id, scores in scoring_results.items():
            if scores['total_score'] >= 7:  # Probable or high confidence
                zone = TARGET_ZONES[zone_id]
                
                site_info = {
                    'zone_id': zone_id,
                    'zone_name': zone.name,
                    'coordinates': zone.center,
                    'anomaly_score': scores['total_score'],
                    'classification': scores['classification'],
                    'evidence_types': scores.get('evidence_summary', []),
                    'priority_ranking': 1 if scores['total_score'] >= 10 else 2,
                    'expected_features': zone.expected_features,
                    'access_difficulty': self._assess_access_difficulty(zone),
                    'verification_cost_estimate': self._estimate_verification_cost(zone)
                }
                
                high_priority.append(site_info)
        
        # Sort by score (highest first)
        high_priority.sort(key=lambda x: x['anomaly_score'], reverse=True)
        
        return high_priority
    
    def _generate_recommendations(self, scoring_results: dict) -> dict:
        """Generate actionable recommendations"""
        
        recommendations = {
            'immediate_actions': [],
            'follow_up_analysis': [],
            'methodology_improvements': [],
            'resource_allocation': {}
        }
        
        high_conf_zones = [k for k, v in scoring_results.items() if v['total_score'] >= 10]
        probable_zones = [k for k, v in scoring_results.items() if 7 <= v['total_score'] < 10]
        
        # Immediate actions
        if high_conf_zones:
            for zone_id in high_conf_zones:
                zone_name = TARGET_ZONES[zone_id].name
                recommendations['immediate_actions'].append(
                    f"Ground verification expedition to {zone_name} - High confidence archaeological site"
                )
        
        if probable_zones:
            recommendations['immediate_actions'].append(
                f"Acquire high-resolution imagery for {len(probable_zones)} probable sites"
            )
        
        # Follow-up analysis
        low_score_zones = [k for k, v in scoring_results.items() if v['total_score'] < 4]
        if low_score_zones:
            recommendations['follow_up_analysis'].append(
                f"Re-analyze {len(low_score_zones)} zones with different seasonal imagery"
            )
        
        # Methodology improvements
        recommendations['methodology_improvements'] = [
            "Integrate LiDAR data when available for improved detection",
            "Apply machine learning models trained on known archaeological sites",
            "Cross-reference with indigenous oral histories and place names",
            "Use multi-temporal analysis to detect subtle vegetation patterns"
        ]
        
        # Resource allocation
        total_high_priority = len(high_conf_zones) + len(probable_zones)
        recommendations['resource_allocation'] = {
            'ground_verification_teams': max(1, total_high_priority // 3),
            'estimated_expedition_duration': f"{total_high_priority * 3-5} days per site",
            'priority_order': [TARGET_ZONES[z].name for z in high_conf_zones + probable_zones]
        }
        
        return recommendations
    
    def _generate_technical_summary(self, analysis_results: dict) -> dict:
        """Generate technical analysis summary"""
        
        total_scenes = sum(len(scenes) for scenes in analysis_results.values())
        successful_analyses = sum(len([s for s in scenes if s.get('success')]) 
                                for scenes in analysis_results.values())
        
        return {
            'processing_statistics': {
                'total_scenes_processed': total_scenes,
                'successful_analyses': successful_analyses,
                'success_rate': f"{(successful_analyses/total_scenes*100):.1f}%" if total_scenes else "0%",
                'processing_time': 'Variable - depends on scene size and complexity'
            },
            'detection_algorithms': {
                'terra_preta_detection': 'NIR-SWIR spectral analysis with NDVI filtering',
                'geometric_detection': 'OpenCV Hough transforms + edge detection',
                'scoring_method': '15-point convergent anomaly system'
            },
            'data_quality': {
                'max_cloud_cover': '20%',
                'preferred_season': 'Dry season (June-September)',
                'pixel_resolution': '30m (Landsat)',
                'spectral_bands': 'Blue, Green, Red, NIR, SWIR1, SWIR2'
            }
        }
    
    def _get_recommended_action(self, score: float) -> str:
        """Get recommended action based on anomaly score"""
        if score >= 10:
            return "IMMEDIATE GROUND VERIFICATION - High confidence archaeological site"
        elif score >= 7:
            return "HIGH-RESOLUTION IMAGERY + GROUND RECONNAISSANCE"
        elif score >= 4:
            return "Additional remote sensing analysis recommended"
        else:
            return "Continue monitoring with different seasonal imagery"
    
    def _assess_access_difficulty(self, zone) -> str:
        """Assess difficulty of ground access to zone"""
        # Simplified assessment based on zone characteristics
        if 'Upper' in zone.name:
            return "High - Remote headwater region"
        elif 'Confluence' in zone.name:
            return "Medium - River access possible"
        else:
            return "Medium - Standard Amazon access challenges"
    
    def _estimate_verification_cost(self, zone) -> str:
        """Estimate cost for ground verification"""
        # Rough estimates based on location and access
        if 'High' in self._assess_access_difficulty(zone):
            return "$50,000-80,000 USD"
        else:
            return "$25,000-40,000 USD"
    
    def _write_readable_report(self, report: dict, output_path: Path):
        """Write human-readable markdown report"""
        
        with open(output_path, 'w') as f:
            f.write("# Amazon Archaeological Discovery Report\n\n")
            f.write(f"**Session ID:** {report['session_info']['session_id']}\n")
            f.write(f"**Generated:** {report['session_info']['timestamp']}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            summary = report['executive_summary']
            f.write(f"- **Zones Analyzed:** {summary['total_zones_analyzed']}\n")
            f.write(f"- **High Confidence Sites:** {summary['high_confidence_sites']}\n")
            f.write(f"- **Total Features Detected:** {summary['total_features_detected']}\n")
            f.write(f"- **Success Rate:** {summary['success_rate']}\n\n")
            
            if summary.get('highest_scoring_zone'):
                best = summary['highest_scoring_zone']
                f.write(f"**Highest Scoring Zone:** {best['zone_name']} ({best['score']}/15 points)\n\n")
            
            # High Priority Sites
            f.write("## High Priority Sites for Ground Verification\n\n")
            for site in report['high_priority_sites']:
                f.write(f"### {site['zone_name']}\n")
                f.write(f"- **Score:** {site['anomaly_score']}/15 ({site['classification']})\n")
                f.write(f"- **Coordinates:** {site['coordinates'][0]:.4f}°, {site['coordinates'][1]:.4f}°\n")
                f.write(f"- **Expected Features:** {site['expected_features']}\n")
                f.write(f"- **Verification Cost:** {site['verification_cost_estimate']}\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("### Immediate Actions\n")
            for action in report['recommendations']['immediate_actions']:
                f.write(f"- {action}\n")
            f.write("\n")
            
            f.write("### Resource Allocation\n")
            resources = report['recommendations']['resource_allocation']
            f.write(f"- **Ground Teams Needed:** {resources['ground_verification_teams']}\n")
            f.write(f"- **Estimated Duration:** {resources['estimated_expedition_duration']}\n")
            f.write("- **Priority Order:**\n")
            for i, zone_name in enumerate(resources['priority_order'], 1):
                f.write(f"  {i}. {zone_name}\n")
    
    def _print_executive_summary(self, summary: dict):
        """Print executive summary to console"""
        
        print("\n" + "="*70)
        print("🏛️  AMAZON ARCHAEOLOGICAL DISCOVERY - EXECUTIVE SUMMARY")
        print("="*70)
        
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"   Zones Analyzed: {summary['total_zones_analyzed']}")
        print(f"   Features Detected: {summary['total_features_detected']}")
        print(f"   Success Rate: {summary['success_rate']}")
        
        print(f"\n🎯 DISCOVERY CLASSIFICATION:")
        print(f"   High Confidence Sites: {summary['high_confidence_sites']} ⭐⭐⭐")
        print(f"   Probable Features: {summary['probable_archaeological_features']} ⭐⭐")
        
        if summary.get('highest_scoring_zone'):
            best = summary['highest_scoring_zone']
            print(f"\n🏆 TOP DISCOVERY:")
            print(f"   {best['zone_name']}")
            print(f"   Score: {best['score']}/15 points")
            print(f"   Classification: {best['classification']}")
        
        print("\n" + "="*70)
    
    def create_interactive_map(self):
        """Create interactive map of all discoveries"""
        
        if not self.results.get('analysis'):
            logger.warning("No analysis results available for mapping")
            return None
        
        logger.info("🗺️ Creating interactive discovery map...")
        
        try:
            visualizer = ArchaeologicalVisualizer()
            map_path = visualizer.create_discovery_map(
                self.results['analysis'],
                self.results.get('scores', {}),
                MAPS_DIR / f"archaeological_discoveries_{self.session_id}.html"
            )
            
            logger.info(f"✓ Interactive map created: {map_path}")
            return map_path
            
        except Exception as e:
            logger.error(f"Error creating map: {e}")
            return None
    
    def run_full_pipeline(self, zones: list = None, download: bool = True, 
                         analyze: bool = True, score: bool = True, 
                         report: bool = True, visualize: bool = True):
        """Run the complete archaeological discovery pipeline"""
        
        logger.info("🚀 Starting Amazon Archaeological Discovery Pipeline...")
        
        try:
            # Step 1: Download satellite data
            if download:
                logger.info("\n📡 STEP 1: Downloading satellite data...")
                downloads, all_scene_data = self.download_satellite_data(zones, provider=self._provider)
                if not downloads:
                    logger.error("No data downloaded - cannot continue")
                    return False
            
            # Step 2: Analyze for archaeological features
            if analyze:
                logger.info("\n🔍 STEP 2: Analyzing archaeological features...")
                analysis_results = self.analyze_downloaded_data(zones)
                if not analysis_results:
                    logger.error("No analysis results - cannot continue")
                    return False
            
            # Step 3: Calculate convergent anomaly scores
            if score:
                logger.info("\n🧮 STEP 3: Calculating convergent anomaly scores...")
                scoring_results = self.calculate_convergent_scores()
            
            # Step 4: Generate discovery report
            if report:
                logger.info("\n📊 STEP 4: Generating discovery report...")
                report_data = self.generate_discovery_report()
            
            # Step 5: Create interactive visualizations
            if visualize:
                logger.info("\n🗺️ STEP 5: Creating interactive map...")
                map_path = self.create_interactive_map()
            
            logger.info("\n✅ PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Session ID: {self.session_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="Amazon Archaeological Discovery Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --list-zones                    # List all target zones
  python main.py --zone negro_madeira --download # Download data for specific zone
  python main.py --zone all --full-pipeline      # Run complete pipeline
  python main.py --analyze-existing --report     # Analyze existing data and report
        """
    )
    
    # Optional: Add Google Earth Engine as alternative provider
    parser.add_argument('--provider', choices=['usgs', 'gee', 'both'], default='gee',
                       help='Data provider to use (usgs=USGS M2M, gee=Google Earth Engine, both=try both)')
    parser.add_argument('--gee-auth', action='store_true',
                       help='Force Google Earth Engine authentication')
    
    # Zone selection
    parser.add_argument('--zone', nargs='+', 
                       choices=list(TARGET_ZONES.keys()) + ['all'],
                       help='Target zones to process')
    
    # Pipeline steps
    parser.add_argument('--list-zones', action='store_true',
                       help='List all configured target zones')
    parser.add_argument('--download', action='store_true',
                       help='Download satellite data')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze downloaded data')
    parser.add_argument('--analyze-existing', action='store_true',
                       help='Analyze existing downloaded data')
    parser.add_argument('--score', action='store_true',
                       help='Calculate convergent anomaly scores')
    parser.add_argument('--report', action='store_true',
                       help='Generate discovery report')
    parser.add_argument('--visualize', action='store_true',
                       help='Create interactive visualizations')
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run complete pipeline (download + analyze + score + report)')
    parser.add_argument('--modular-pipeline', action='store_true',
                       help='Run the new modular pipeline (experimental)')
    
    # Options
    parser.add_argument('--max-scenes', type=int, default=3,
                       help='Maximum scenes to download per zone (default: 3)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to a file or directory to analyze (overrides default zone-based search)')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = setup_logging(args.verbose)
    
    # Handle list zones
    if args.list_zones:
        pipeline = ArchaeologicalPipeline()
        pipeline.list_target_zones()
        return
    
    # Main pipeline execution
    with ArchaeologicalPipeline(provider=args.provider) as pipeline:
        
        # Handle GEE authentication if requested
        if hasattr(args, 'gee_auth') and args.gee_auth:
            try:
                from src.gee_provider import GoogleEarthEngineProvider
                gee_provider = GoogleEarthEngineProvider()
                print("✅ Google Earth Engine authentication successful")
                return
            except Exception as e:
                print(f"❌ GEE authentication failed: {e}")
                return
        # New: Modular pipeline option
        if hasattr(args, 'modular_pipeline') and args.modular_pipeline:
            print("\n🚀 Running Modular Archaeological Discovery Pipeline...")
            modular = ModularPipeline(provider=args.provider)
            zones = args.zone if args.zone else None
            results = modular.run(zones=zones, max_scenes=args.max_scenes)
            print("\nModular pipeline complete.")
            if results.get('report'):
                print(f"📄 Report saved: {results['report']['session_info']['session_id']}")
            if results.get('map_path'):
                print(f"🗺️  Map saved: {results['map_path']}")
            return
        # Legacy pipeline logic
        if args.full_pipeline:
            # Run complete pipeline
            zones = args.zone if args.zone else None
            success = pipeline.run_full_pipeline(
                zones=zones,
                download=True,
                analyze=True, 
                score=True,
                report=True,
                visualize=True
            )
            if not success:
                logger.error("Pipeline execution failed")
                sys.exit(1)
        else:
            # Run individual steps
            zones = args.zone if args.zone else None
            if args.download:
                pipeline.download_satellite_data(zones, args.max_scenes, provider=pipeline._provider)
            if args.analyze or args.analyze_existing:
                pipeline.analyze_downloaded_data(zones, args.data_path)
            if args.score:
                pipeline.calculate_convergent_scores()
            if args.report:
                pipeline.generate_discovery_report()
            if args.visualize:
                pipeline.create_interactive_map()
    
    print(f"\n📋 Session log saved: {log_file}")

if __name__ == "__main__":
    main()
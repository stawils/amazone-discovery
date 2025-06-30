from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime
from src.core.config import TARGET_ZONES, RESULTS_DIR
import logging
import json

logger = logging.getLogger(__name__)


def get_results_directory(run_id: str, provider: str, create_subdirs: bool = False) -> Path:
    """Get the provider-specific directory for a given run.
    
    Args:
        run_id: The unique identifier for the pipeline run.
        provider: The data provider name (e.g., 'gedi', 'sentinel2').
        create_subdirs: Whether to create subdirectories (reports, maps) immediately
        
    Returns:
        Path to the provider-specific directory for this run (RESULTS_DIR / f"run_{run_id}" / provider).
    """
    provider_run_dir = RESULTS_DIR / f"run_{run_id}" / provider
    
    # Only create subdirectories if explicitly requested
    if create_subdirs:
        reports_dir = provider_run_dir / "reports"
        maps_dir = provider_run_dir / "maps"
        
        for directory in [provider_run_dir, reports_dir, maps_dir]:
            directory.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Created results directory structure for Run ID '{run_id}', Provider '{provider}': {provider_run_dir}")
    
    return provider_run_dir


class ReportStep:
    """Enhanced report step for archaeological pipeline - MULTI-PROVIDER VERSION."""
    
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.provider = None
        self.results_dir = None
        logger.info(f"ReportStep initialized for Run ID: {self.run_id}")
    
    def run(self, scoring_results: Dict[str, dict], analysis_results: Dict[str, List[dict]], provider: str = "multi") -> dict:
        """Generate a comprehensive archaeological discovery report - ENHANCED VERSION."""
        
        logger.info(f"📊 Generating discovery report for provider: {provider}, Run ID: {self.run_id}...")
        
        # Get the base directory for this provider and run_id
        # This directory (e.g., .../run_xxxx/gedi/) will contain reports, maps, exports subfolders
        current_provider_run_dir = get_results_directory(run_id=self.run_id, provider=provider, create_subdirs=True)
        self.current_reports_subdir = current_provider_run_dir / "reports" # Specific subdir for reports
        
        logger.info(f"📁 Saving report results to {self.current_reports_subdir}")
        
        if not scoring_results and not analysis_results:
            logger.warning("No results available for report generation")
            # Pass the specific reports subdirectory for the empty report
            return self._create_empty_report(reports_dir=self.current_reports_subdir)
        
        # Create comprehensive report
        report = {
            'session_info': self._generate_session_info(analysis_results),
            'executive_summary': self._generate_executive_summary(scoring_results, analysis_results),
            'zone_details': self._generate_zone_details(analysis_results, scoring_results),
            'high_priority_sites': self._extract_high_priority_sites(scoring_results),
            'provider_analysis': self._analyze_providers(analysis_results),
            'feature_inventory': self._create_feature_inventory(analysis_results, scoring_results),
            'recommendations': self._generate_recommendations(scoring_results, analysis_results),
            'technical_summary': self._generate_technical_summary(analysis_results),
            'data_quality_assessment': self._assess_data_quality(analysis_results)
        }
        
        # Save detailed JSON report to the reports subdirectory
        report_path = self._save_report(report, reports_dir=self.current_reports_subdir)
        
        # Generate summary markdown in the same reports subdirectory
        self._save_summary_markdown(report, report_path, reports_dir=self.current_reports_subdir)
        
        return report

    def _generate_session_info(self, analysis_results: Dict[str, List[dict]]) -> dict:
        """Generate session information."""
        
        # Use the run_id as the session_id
        session_id = self.run_id
        
        # Determine providers used
        providers_used = set()
        for zone_analyses in analysis_results.values():
            for analysis in zone_analyses:
                if 'clearing_results' in analysis or 'earthwork_results' in analysis:
                    providers_used.add('GEDI')
                elif 'terra_preta' in analysis:
                    providers_used.add('Sentinel-2/GEE')
        
        return {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': 'multi-provider-2.0',
            'zones_analyzed': list(analysis_results.keys()),
            'providers_used': list(providers_used),
            'total_analyses': sum(len(analyses) for analyses in analysis_results.values())
        }

    def _generate_executive_summary(self, scoring_results: dict, analysis_results: dict) -> dict:
        """Generate executive summary with provider-specific insights."""
        
        if not scoring_results:
            return {
                'total_zones_analyzed': len(analysis_results),
                'high_confidence_sites': 0,
                'probable_archaeological_features': 0,
                'total_features_detected': 0,
                'highest_scoring_zone': None,
                'success_rate': "0%",
                'key_findings': ["No scoring results available"]
            }
        
        # Calculate key metrics
        high_confidence = sum(1 for r in scoring_results.values() if r.get('total_score', 0) >= 10)
        probable_features = sum(1 for r in scoring_results.values() if r.get('total_score', 0) >= 7)
        
        # Calculate total features by provider
        feature_counts = self._count_features_by_provider(analysis_results)
        # Use the provider totals to avoid double counting
        total_features = feature_counts.get('total_sentinel2', 0) + feature_counts.get('total_gedi', 0)
        
        # Find highest scoring zone
        best_zone = None
        if scoring_results:
            best_zone_item = max(scoring_results.items(), key=lambda x: x[1].get('total_score', 0))
            best_zone = {
                'zone_id': best_zone_item[0],
                'zone_name': best_zone_item[1].get('zone_name', best_zone_item[0]),
                'score': best_zone_item[1].get('total_score', 0),
                'classification': best_zone_item[1].get('classification', 'Unknown')
            }
        
        # Generate key findings
        key_findings = self._generate_key_findings(scoring_results, analysis_results, feature_counts)
        
        return {
            'total_zones_analyzed': len(analysis_results),
            'high_confidence_sites': high_confidence,
            'probable_archaeological_features': probable_features,
            'total_features_detected': total_features,
            'feature_breakdown': feature_counts,
            'highest_scoring_zone': best_zone,
            'success_rate': f"{(probable_features / len(scoring_results) * 100):.1f}%" if scoring_results else "0%",
            'key_findings': key_findings
        }

    def _count_features_by_provider(self, analysis_results: dict) -> dict:
        """Count features detected by each provider - FIXED to match actual data structures."""
        
        feature_counts = {
            'gedi_clearings': 0,
            'gedi_earthworks': 0, 
            'gedi_linear_features': 0,
            'terra_preta_sites': 0,
            'geometric_patterns': 0,
            'crop_marks': 0,
            'total_sentinel2': 0,
            'total_gedi': 0
        }
        
        for zone_analyses in analysis_results.values():
            for analysis in zone_analyses:
                if not analysis.get('success'):
                    continue
                
                provider = analysis.get('provider', 'unknown')
                
                if provider == 'gedi':
                    # Count GEDI features - Updated for actual structure
                    total_features = analysis.get('total_features', 0)
                    feature_counts['total_gedi'] += total_features
                    
                    clearing_results = analysis.get('clearing_results', {})
                    # GEDI gap_clusters represent clearings
                    gap_clusters = clearing_results.get('gap_clusters', [])
                    feature_counts['gedi_clearings'] += len(gap_clusters)
                    
                    earthwork_results = analysis.get('earthwork_results', {})
                    feature_counts['gedi_earthworks'] += len(earthwork_results.get('mound_clusters', []))
                    feature_counts['gedi_linear_features'] += len(earthwork_results.get('linear_features', []))
                    
                elif provider == 'sentinel2':
                    # Count Sentinel-2 features - Updated for actual structure
                    detection_summary = analysis.get('detection_summary', {})
                    if detection_summary:
                        # Use detection_summary for accurate counts
                        tp_count = detection_summary.get('terra_preta_analysis_count', 0)
                        geom_count = detection_summary.get('geometric_feature_analysis_count', 0) 
                        crop_count = detection_summary.get('crop_mark_analysis_count', 0)
                        
                        feature_counts['terra_preta_sites'] += tp_count
                        feature_counts['geometric_patterns'] += geom_count
                        feature_counts['crop_marks'] += crop_count
                        feature_counts['total_sentinel2'] += (tp_count + geom_count + crop_count)
                    else:
                        # Fallback: Use direct analysis fields
                        terra_preta = analysis.get('terra_preta_analysis', {})
                        feature_counts['terra_preta_sites'] += terra_preta.get('count', 0)
                        
                        geometric_features = analysis.get('geometric_feature_analysis', {})
                        feature_counts['geometric_patterns'] += geometric_features.get('count', 0)
                        
                        crop_marks = analysis.get('crop_mark_analysis', {})
                        feature_counts['crop_marks'] += crop_marks.get('count', 0)
                
                # Legacy support for vegetation anomalies
                if 'ndvi_anomalies' in analysis or 'vegetation_analysis' in analysis:
                    feature_counts['crop_marks'] += 1
        
        return feature_counts

    def _generate_key_findings(self, scoring_results: dict, analysis_results: dict, feature_counts: dict) -> List[str]:
        """Generate key findings from the analysis."""
        
        findings = []
        
        # Provider-specific findings
        if feature_counts['gedi_clearings'] > 0:
            findings.append(f"GEDI LiDAR detected {feature_counts['gedi_clearings']} potential archaeological clearings")
        
        if feature_counts['gedi_earthworks'] > 0:
            findings.append(f"Elevation analysis revealed {feature_counts['gedi_earthworks']} possible earthwork complexes")
        
        if feature_counts['terra_preta_sites'] > 0:
            findings.append(f"Spectral analysis identified {feature_counts['terra_preta_sites']} terra preta (anthropogenic soil) signatures")
        
        if feature_counts['geometric_patterns'] > 0:
            findings.append(f"Geometric pattern recognition found {feature_counts['geometric_patterns']} structured archaeological features")
        
        if feature_counts['crop_marks'] > 0:
            findings.append(f"Crop mark analysis detected {feature_counts['crop_marks']} subsurface archaeological features")
        
        # High-confidence sites
        high_conf_zones = [zone for zone, score in scoring_results.items() 
                          if score.get('total_score', 0) >= 10]
        if high_conf_zones:
            zone_names = [TARGET_ZONES.get(z, type('obj', (object,), {'name': z})).name for z in high_conf_zones]
            findings.append(f"High-confidence archaeological sites identified in: {', '.join(zone_names)}")
        
        # Convergent evidence
        multi_evidence_zones = [zone for zone, score in scoring_results.items() 
                               if len(score.get('evidence_summary', [])) >= 3]
        if multi_evidence_zones:
            findings.append(f"Multiple evidence types converge in {len(multi_evidence_zones)} zones, increasing confidence")
        
        # Data coverage
        total_data_points = 0
        for zone_analyses in analysis_results.values():
            for analysis in zone_analyses:
                data_quality = analysis.get('data_quality', {})
                total_data_points += data_quality.get('total_points', 0)
        
        if total_data_points > 0:
            findings.append(f"Analysis processed {total_data_points:,} satellite/LiDAR data points across all zones")
        
        if not findings:
            findings.append("Analysis completed but no significant archaeological indicators detected")
        
        return findings

    def _generate_zone_details(self, analysis_results: dict, scoring_results: dict) -> dict:
        """Generate detailed analysis for each zone."""
        
        zone_details = {}
        
        for zone_id, zone_analyses in analysis_results.items():
            zone = TARGET_ZONES.get(zone_id)
            scores = scoring_results.get(zone_id, {})
            
            # Aggregate analysis results
            analysis_summary = self._summarize_zone_analysis(zone_analyses)
            
            zone_details[zone_id] = {
                'zone_info': {
                    'name': zone.name if zone else zone_id.replace('_', ' ').title(),
                    'coordinates': zone.center if zone else [0, 0],
                    'priority': zone.priority if zone else 3,
                    'expected_features': zone.expected_features if zone else 'Unknown',
                    'historical_evidence': zone.historical_evidence if zone else 'Unknown'
                },
                'analysis_summary': analysis_summary,
                'convergent_score': scores.get('total_score', 0),
                'classification': scores.get('classification', 'No analysis'),
                'confidence': scores.get('confidence', 0.0),
                'evidence_summary': scores.get('evidence_summary', []),
                'feature_details': scores.get('feature_details', []),
                'provider_breakdown': self._get_provider_breakdown(zone_analyses),
                'recommended_action': self._get_recommended_action(scores.get('total_score', 0), scores.get('confidence', 0))
            }
        
        return zone_details

    def _summarize_zone_analysis(self, zone_analyses: List[dict]) -> dict:
        """Summarize analysis results for a zone."""
        
        successful_analyses = [a for a in zone_analyses if a.get('success')]
        
        summary = {
            'total_analyses': len(zone_analyses),
            'successful_analyses': len(successful_analyses),
            'providers_used': [],
            'data_coverage': {},
            'feature_summary': {}
        }
        
        # Determine providers and coverage
        for analysis in successful_analyses:
            if 'clearing_results' in analysis or 'earthwork_results' in analysis:
                if 'GEDI' not in summary['providers_used']:
                    summary['providers_used'].append('GEDI')
                data_quality = analysis.get('data_quality', {})
                summary['data_coverage']['gedi_points'] = data_quality.get('total_points', 0)
            
            if 'terra_preta' in analysis:
                if 'Sentinel-2/GEE' not in summary['providers_used']:
                    summary['providers_used'].append('Sentinel-2/GEE')
        
        # Feature summary
        feature_summary = {}
        for analysis in successful_analyses:
            # GEDI features
            clearing_results = analysis.get('clearing_results', {})
            if clearing_results:
                feature_summary['clearings'] = clearing_results.get('total_clearings', 0)
            
            earthwork_results = analysis.get('earthwork_results', {})
            if earthwork_results:
                feature_summary['earthworks'] = len(earthwork_results.get('mound_clusters', []))
                feature_summary['linear_features'] = len(earthwork_results.get('linear_features', []))
            
            # Spectral features
            terra_preta = analysis.get('terra_preta', {})
            if terra_preta:
                feature_summary['terra_preta_patches'] = len(terra_preta.get('patches', []))
            
            geometric_features = analysis.get('geometric_features', [])
            if geometric_features:
                feature_summary['geometric_patterns'] = len(geometric_features)
        
        summary['feature_summary'] = feature_summary
        
        return summary

    def _get_provider_breakdown(self, zone_analyses: List[dict]) -> dict:
        """Get breakdown of results by provider."""
        
        breakdown = {
            'gedi': {'analyses': 0, 'features': 0, 'success_rate': 0},
            'spectral': {'analyses': 0, 'features': 0, 'success_rate': 0}
        }
        
        for analysis in zone_analyses:
            if not analysis.get('success'):
                continue
            
            if 'clearing_results' in analysis or 'earthwork_results' in analysis:
                breakdown['gedi']['analyses'] += 1
                breakdown['gedi']['features'] += analysis.get('total_features', 0)
            
            if 'terra_preta' in analysis:
                breakdown['spectral']['analyses'] += 1
                breakdown['spectral']['features'] += analysis.get('total_features', 0)
        
        # Calculate success rates
        total_gedi = breakdown['gedi']['analyses']
        total_spectral = breakdown['spectral']['analyses']
        
        if total_gedi > 0:
            breakdown['gedi']['success_rate'] = (total_gedi / total_gedi) * 100  # All successful if we got here
        
        if total_spectral > 0:
            breakdown['spectral']['success_rate'] = (total_spectral / total_spectral) * 100
        
        return breakdown

    def _extract_high_priority_sites(self, scoring_results: dict) -> list:
        """Extract high-priority sites for immediate investigation."""
        
        high_priority = []
        
        for zone_id, scores in scoring_results.items():
            total_score = scores.get('total_score', 0)
            if total_score >= 7:  # Probable or higher
                zone = TARGET_ZONES.get(zone_id)
                
                site_info = {
                    'zone_id': zone_id,
                    'zone_name': scores.get('zone_name', zone_id),
                    'coordinates': zone.center if zone else [0, 0],
                    'anomaly_score': total_score,
                    'classification': scores.get('classification', 'Unknown'),
                    'confidence': scores.get('confidence', 0.0),
                    'evidence_types': scores.get('evidence_summary', []),
                    'feature_count': len(scores.get('feature_details', [])),
                    'priority_ranking': 1 if total_score >= 10 else 2,
                    'expected_features': zone.expected_features if zone else 'Unknown',
                    'access_difficulty': self._assess_access_difficulty(zone),
                    'verification_cost_estimate': self._estimate_verification_cost(zone, total_score),
                    'recommended_survey_methods': self._recommend_survey_methods(scores)
                }
                high_priority.append(site_info)
        
        # Sort by score descending
        high_priority.sort(key=lambda x: x['anomaly_score'], reverse=True)
        
        return high_priority

    def _analyze_providers(self, analysis_results: dict) -> dict:
        """Analyze the effectiveness of different data providers."""
        
        provider_analysis = {
            'gedi': {
                'zones_covered': 0,
                'total_features': 0,
                'avg_data_points': 0,
                'strengths': [],
                'limitations': []
            },
            'spectral': {
                'zones_covered': 0,
                'total_features': 0,
                'coverage_quality': 0,
                'strengths': [],
                'limitations': []
            }
        }
        
        gedi_data_points = []
        spectral_zones = 0
        
        for zone_analyses in analysis_results.values():
            zone_has_gedi = False
            zone_has_spectral = False
            
            for analysis in zone_analyses:
                if not analysis.get('success'):
                    continue
                
                if 'clearing_results' in analysis or 'earthwork_results' in analysis:
                    if not zone_has_gedi:
                        provider_analysis['gedi']['zones_covered'] += 1
                        zone_has_gedi = True
                    
                    provider_analysis['gedi']['total_features'] += analysis.get('total_features', 0)
                    
                    data_quality = analysis.get('data_quality', {})
                    points = data_quality.get('total_points', 0)
                    if points > 0:
                        gedi_data_points.append(points)
                
                if 'terra_preta' in analysis:
                    if not zone_has_spectral:
                        provider_analysis['spectral']['zones_covered'] += 1
                        zone_has_spectral = True
                    
                    provider_analysis['spectral']['total_features'] += analysis.get('total_features', 0)
        
        # Calculate averages
        if gedi_data_points:
            provider_analysis['gedi']['avg_data_points'] = sum(gedi_data_points) / len(gedi_data_points)
        
        # Add strengths and limitations
        if provider_analysis['gedi']['zones_covered'] > 0:
            provider_analysis['gedi']['strengths'] = [
                "Excellent for detecting canopy clearings and settlement patterns",
                "Precise elevation data reveals earthworks and mounds",
                "Can penetrate cloud cover with LiDAR technology",
                "25m resolution footprints provide detailed coverage"
            ]
            provider_analysis['gedi']['limitations'] = [
                "Limited spatial coverage compared to optical imagery",
                "Point cloud data requires specialized processing",
                "May miss spectral signatures of anthropogenic soils"
            ]
        
        if provider_analysis['spectral']['zones_covered'] > 0:
            provider_analysis['spectral']['strengths'] = [
                "Excellent for detecting terra preta and soil anomalies",
                "Large-scale coverage enables regional surveys",
                "Multiple spectral bands reveal material composition",
                "Regular revisit times enable temporal analysis"
            ]
            provider_analysis['spectral']['limitations'] = [
                "Limited by cloud cover in tropical regions",
                "May miss features hidden under dense canopy",
                "Resolution limitations for small archaeological features"
            ]
        
        return provider_analysis

    def _create_feature_inventory(self, analysis_results: dict, scoring_results: dict) -> dict:
        """Create a comprehensive inventory of all detected features."""
        
        inventory = {
            'by_type': {},
            'by_zone': {},
            'by_confidence': {'high': 0, 'medium': 0, 'low': 0},
            'spatial_distribution': {},
            'total_count': 0
        }
        
        for zone_id, zone_analyses in analysis_results.items():
            zone_features = []
            
            for analysis in zone_analyses:
                if not analysis.get('success'):
                    continue
                
                # Extract GEDI features
                clearing_results = analysis.get('clearing_results', {})
                for cluster in clearing_results.get('gap_clusters', []):
                    feature = {
                        'type': 'archaeological_clearing',
                        'location': cluster.get('center'),
                        'size': cluster.get('count', 0),
                        'area_km2': cluster.get('area_km2', 0),
                        'confidence': 'high' if cluster.get('count', 0) >= 5 else 'medium',
                        'provider': 'GEDI'
                    }
                    zone_features.append(feature)
                    inventory['by_confidence'][feature['confidence']] += 1
                
                earthwork_results = analysis.get('earthwork_results', {})
                for cluster in earthwork_results.get('mound_clusters', []):
                    feature = {
                        'type': 'earthwork_mound',
                        'location': cluster.get('center'),
                        'size': cluster.get('count', 0),
                        'confidence': 'high' if cluster.get('count', 0) >= 4 else 'medium',
                        'provider': 'GEDI'
                    }
                    zone_features.append(feature)
                    inventory['by_confidence'][feature['confidence']] += 1
                
                for feature in earthwork_results.get('linear_features', []):
                    linear_feature = {
                        'type': 'linear_causeway',
                        'r2_score': feature.get('r2', 0),
                        'length_km': feature.get('length_km', 0),
                        'confidence': 'high' if feature.get('r2', 0) > 0.9 else 'medium',
                        'provider': 'GEDI'
                    }
                    zone_features.append(linear_feature)
                    inventory['by_confidence'][linear_feature['confidence']] += 1
                
                # Extract spectral features
                terra_preta = analysis.get('terra_preta', {})
                for patch in terra_preta.get('patches', []):
                    feature = {
                        'type': 'terra_preta',
                        'location': patch.get('coordinates'),
                        'confidence_score': patch.get('confidence', 0),
                        'confidence': 'high' if patch.get('confidence', 0) > 0.8 else 'medium' if patch.get('confidence', 0) > 0.6 else 'low',
                        'provider': 'Spectral'
                    }
                    zone_features.append(feature)
                    inventory['by_confidence'][feature['confidence']] += 1
            
            inventory['by_zone'][zone_id] = zone_features
            
            # Count by type
            for feature in zone_features:
                feature_type = feature['type']
                if feature_type not in inventory['by_type']:
                    inventory['by_type'][feature_type] = 0
                inventory['by_type'][feature_type] += 1
        
        inventory['total_count'] = sum(inventory['by_type'].values())
        
        return inventory

    def _generate_recommendations(self, scoring_results: dict, analysis_results: dict) -> dict:
        """Generate actionable recommendations based on results."""
        
        recommendations = {
            'immediate_actions': [],
            'follow_up_analysis': [],
            'methodology_improvements': [],
            'resource_allocation': {},
            'field_work_priorities': []
        }
        
        # Immediate actions for high-confidence sites
        high_conf_zones = [k for k, v in scoring_results.items() if v.get('total_score', 0) >= 10]
        probable_zones = [k for k, v in scoring_results.items() if 7 <= v.get('total_score', 0) < 10]
        
        if high_conf_zones:
            for zone_id in high_conf_zones:
                zone_name = scoring_results[zone_id].get('zone_name', zone_id)
                recommendations['immediate_actions'].append(
                    f"🎯 PRIORITY: Ground verification expedition to {zone_name} - High confidence archaeological site"
                )
                recommendations['field_work_priorities'].append({
                    'zone': zone_name,
                    'priority': 'URGENT',
                    'estimated_duration': '5-7 days',
                    'team_size': '4-6 archaeologists + local guides',
                    'equipment_needed': ['GPS units', 'Metal detectors', 'Survey equipment', 'Documentation cameras']
                })
        
        if probable_zones:
            recommendations['immediate_actions'].append(
                f"📡 Acquire high-resolution imagery for {len(probable_zones)} probable archaeological sites"
            )
            for zone_id in probable_zones:
                zone_name = scoring_results[zone_id].get('zone_name', zone_id)
                recommendations['field_work_priorities'].append({
                    'zone': zone_name,
                    'priority': 'HIGH',
                    'estimated_duration': '3-4 days',
                    'team_size': '3-4 archaeologists',
                    'equipment_needed': ['GPS units', 'Survey equipment', 'Sampling tools']
                })
        
        # Follow-up analysis recommendations
        low_score_zones = [k for k, v in scoring_results.items() if v.get('total_score', 0) < 4]
        if low_score_zones:
            recommendations['follow_up_analysis'].append(
                f"🔄 Re-analyze {len(low_score_zones)} zones with different seasonal imagery"
            )
            recommendations['follow_up_analysis'].append(
                "⏰ Implement temporal analysis to detect seasonal archaeological signatures"
            )
        
        # Check for data gaps
        gedi_zones = []
        spectral_zones = []
        for zone_id, zone_analyses in analysis_results.items():
            has_gedi = any('clearing_results' in a for a in zone_analyses if a.get('success'))
            has_spectral = any('terra_preta' in a for a in zone_analyses if a.get('success'))
            
            if has_gedi:
                gedi_zones.append(zone_id)
            if has_spectral:
                spectral_zones.append(zone_id)
        
        missing_gedi = set(analysis_results.keys()) - set(gedi_zones)
        missing_spectral = set(analysis_results.keys()) - set(spectral_zones)
        
        if missing_gedi:
            recommendations['follow_up_analysis'].append(
                f"🛰️ Acquire GEDI LiDAR data for zones: {', '.join(missing_gedi)}"
            )
        
        if missing_spectral:
            recommendations['follow_up_analysis'].append(
                f"🌈 Acquire spectral imagery for zones: {', '.join(missing_spectral)}"
            )
        
        # Methodology improvements
        recommendations['methodology_improvements'] = [
            "🤖 Integrate machine learning models trained on confirmed archaeological sites",
            "📊 Implement multi-temporal analysis for better change detection",
            "🗺️ Cross-reference findings with indigenous oral histories and place names",
            "🔬 Develop provider-specific confidence calibration algorithms",
            "📡 Combine GEDI point clouds with high-resolution optical imagery",
            "🌿 Use phenological analysis to detect subtle vegetation patterns over archaeological sites"
        ]
        
        # Resource allocation
        total_high_priority = len(high_conf_zones) + len(probable_zones)
        recommendations['resource_allocation'] = {
            'ground_verification_teams': max(1, total_high_priority // 2),
            'estimated_expedition_duration': f"{max(3, total_high_priority * 4)} to {total_high_priority * 6} days total across all sites",
            'priority_order': [scoring_results[z].get('zone_name', z) for z in high_conf_zones + probable_zones],
            'estimated_budget': f"${(len(high_conf_zones) * 60000) + (len(probable_zones) * 35000):,} USD",
            'recommended_partnerships': [
                "Local indigenous communities for traditional knowledge",
                "Brazilian/Peruvian archaeological institutions",
                "Remote sensing research centers",
                "Conservation organizations for site protection"
            ]
        }
        
        return recommendations

    def _generate_technical_summary(self, analysis_results: dict) -> dict:
        """Generate technical summary of processing and methods."""
        
        total_scenes = sum(len(scenes) for scenes in analysis_results.values())
        successful_analyses = sum(len([s for s in scenes if s.get('success')]) 
                                for scenes in analysis_results.values())
        
        # Count data points processed
        total_gedi_points = 0
        spectral_scenes = 0
        
        for zone_analyses in analysis_results.values():
            for analysis in zone_analyses:
                if analysis.get('success'):
                    data_quality = analysis.get('data_quality', {})
                    points = data_quality.get('total_points', 0)
                    if points > 0:
                        total_gedi_points += points
                    
                    if 'terra_preta' in analysis:
                        spectral_scenes += 1
        
        return {
            'processing_statistics': {
                'total_scenes_processed': total_scenes,
                'successful_analyses': successful_analyses,
                'success_rate': f"{(successful_analyses/total_scenes*100):.1f}%" if total_scenes else "0%",
                'gedi_points_processed': f"{total_gedi_points:,}",
                'spectral_scenes_analyzed': spectral_scenes
            },
            'detection_algorithms': {
                'gedi_clearing_detection': 'Canopy gap analysis with clustering (DBSCAN)',
                'gedi_earthwork_detection': 'Elevation anomaly detection with statistical thresholds',
                'terra_preta_detection': 'NIR-SWIR spectral analysis with NDVI filtering',
                'geometric_detection': 'OpenCV Hough transforms + edge detection',
                'scoring_method': '15-point convergent anomaly system with provider weighting'
            },
            'data_quality_parameters': {
                'gedi_min_points_per_cluster': 3,
                'gedi_elevation_anomaly_threshold': '2.0 standard deviations',
                'spectral_cloud_cover_limit': '20%',
                'preferred_season': 'Dry season (June-September)',
                'gedi_footprint_size': '25m diameter',
                'spectral_pixel_resolution': '10-30m depending on provider'
            },
            'confidence_metrics': {
                'high_confidence_threshold': 'Score ≥ 10/15 points',
                'probable_threshold': 'Score ≥ 7/15 points',
                'minimum_evidence_types': 2,
                'convergence_bonus_applied': 'Multiple evidence types in same location'
            }
        }

    def _assess_data_quality(self, analysis_results: dict) -> dict:
        """Assess the quality of data used in analysis."""
        
        quality_assessment = {
            'overall_quality': 'Unknown',
            'provider_quality': {},
            'recommendations': [],
            'data_gaps': []
        }
        
        gedi_quality_scores = []
        spectral_quality_scores = []
        
        for zone_id, zone_analyses in analysis_results.items():
            for analysis in zone_analyses:
                if not analysis.get('success'):
                    continue
                
                data_quality = analysis.get('data_quality', {})
                
                # GEDI quality assessment
                if 'total_points' in data_quality:
                    points = data_quality['total_points']
                    has_canopy = data_quality.get('has_canopy_data', False)
                    has_elevation = data_quality.get('has_elevation_data', False)
                    
                    quality_score = 0
                    if points > 1000:
                        quality_score += 30
                    elif points > 500:
                        quality_score += 20
                    elif points > 100:
                        quality_score += 10
                    
                    if has_canopy:
                        quality_score += 35
                    if has_elevation:
                        quality_score += 35
                    
                    gedi_quality_scores.append(quality_score)
                
                # Spectral quality assessment
                if 'terra_preta' in analysis:
                    # Assume good quality if analysis was successful
                    spectral_quality_scores.append(75)
        
        # Calculate averages
        if gedi_quality_scores:
            avg_gedi = sum(gedi_quality_scores) / len(gedi_quality_scores)
            quality_assessment['provider_quality']['GEDI'] = {
                'average_score': avg_gedi,
                'quality_level': 'Excellent' if avg_gedi >= 80 else 'Good' if avg_gedi >= 60 else 'Fair' if avg_gedi >= 40 else 'Poor'
            }
        
        if spectral_quality_scores:
            avg_spectral = sum(spectral_quality_scores) / len(spectral_quality_scores)
            quality_assessment['provider_quality']['Spectral'] = {
                'average_score': avg_spectral,
                'quality_level': 'Excellent' if avg_spectral >= 80 else 'Good' if avg_spectral >= 60 else 'Fair' if avg_spectral >= 40 else 'Poor'
            }
        
        # Overall quality
        all_scores = gedi_quality_scores + spectral_quality_scores
        if all_scores:
            overall_avg = sum(all_scores) / len(all_scores)
            quality_assessment['overall_quality'] = (
                'Excellent' if overall_avg >= 80 else 
                'Good' if overall_avg >= 60 else 
                'Fair' if overall_avg >= 40 else 'Poor'
            )
        
        return quality_assessment

    def _get_recommended_action(self, score: float, confidence: float) -> str:
        """Get recommended action based on score and confidence."""
        
        if score >= 10:
            return "🎯 IMMEDIATE GROUND VERIFICATION - High confidence archaeological site"
        elif score >= 7:
            return "📡 HIGH-RESOLUTION IMAGERY + GROUND RECONNAISSANCE"
        elif score >= 4:
            return "🔄 Additional remote sensing analysis recommended"
        else:
            return "📊 Continue monitoring with different seasonal imagery"

    def _assess_access_difficulty(self, zone) -> str:
        """Assess difficulty of accessing the zone."""
        
        if not zone:
            return "Unknown - No zone information"
        
        if 'Upper' in zone.name:
            return "High - Remote headwater region, requires river transport"
        elif 'Confluence' in zone.name:
            return "Medium - River access possible, some overland travel"
        else:
            return "Medium - Standard Amazon access challenges"

    def _estimate_verification_cost(self, zone, score: float) -> str:
        """Estimate cost of field verification."""
        
        base_cost = 30000  # Base expedition cost
        
        if score >= 10:
            base_cost += 20000  # More thorough investigation
        
        if zone and 'Upper' in zone.name:
            base_cost += 15000  # Remote location premium
        
        return f"${base_cost:,} - ${base_cost + 10000:,} USD"

    def _recommend_survey_methods(self, scores: dict) -> List[str]:
        """Recommend survey methods based on detected features."""
        
        methods = ["GPS mapping", "Photographic documentation", "Artifact surface collection"]
        
        evidence = scores.get('evidence_summary', [])
        features = scores.get('feature_details', [])
        
        if any('clearing' in str(e).lower() for e in evidence):
            methods.append("Vegetation pattern analysis")
            methods.append("Soil sampling for anthrosol detection")
        
        if any('earthwork' in str(e).lower() or 'mound' in str(e).lower() for e in evidence):
            methods.append("Topographic survey")
            methods.append("Ground-penetrating radar")
        
        if any('terra preta' in str(e).lower() for e in evidence):
            methods.append("Soil chemistry analysis")
            methods.append("Ceramic fragment collection")
        
        if len(features) >= 3:
            methods.append("Drone aerial photography")
            methods.append("Systematic test excavation")
        
        return methods

    def _save_report(self, report: dict, reports_dir: Path) -> Path:
        """Save the detailed JSON report to the specified reports directory."""
        # Filename can now be simpler as it's within a run_id and provider specific path
        report_filename = f"discovery_report.json"
        report_path = reports_dir / report_filename
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Saved detailed JSON report: {report_path}")
        except Exception as e:
            logger.error(f"Failed to save JSON report {report_path}: {e}", exc_info=True)
            # Consider re-raising or returning a status
        return report_path

    def _generate_markdown_summary(self, report: dict) -> str:
        """Generate a markdown summary of the report."""
        
        session_info = report.get('session_info', {})
        exec_summary = report.get('executive_summary', {})
        high_priority = report.get('high_priority_sites', [])
        recommendations = report.get('recommendations', {})
        
        md = f"""# Archaeological Discovery Report (Run ID: {session_info.get('session_id', 'N/A')})
        
**Generated:** {session_info.get('timestamp', 'N/A')}  
**Pipeline Version:** {session_info.get('pipeline_version', 'N/A')}

## Executive Summary

- **Zones Analyzed:** {exec_summary.get('total_zones_analyzed', 0)}
- **High Confidence Sites:** {exec_summary.get('high_confidence_sites', 0)}
- **Probable Archaeological Features:** {exec_summary.get('probable_archaeological_features', 0)}
- **Total Features Detected:** {exec_summary.get('total_features_detected', 0)}
- **Success Rate:** {exec_summary.get('success_rate', "0%")}

### Key Findings
"""
        
        for finding in exec_summary.get('key_findings', []):
            md += f"- {finding}\n"
        
        highest_scoring_zone = exec_summary.get('highest_scoring_zone')
        if highest_scoring_zone:
            md += f"\n**Highest Scoring Zone:** {highest_scoring_zone.get('zone_name', 'N/A')} ({highest_scoring_zone.get('score', 0)} points) - {highest_scoring_zone.get('classification', 'Unknown')}\n"
        
        md += "\n## High Priority Sites\n\n"
        
        if high_priority:
            for i, site in enumerate(high_priority[:5], 1):  # Top 5 sites
                md += f"### {i}. {site.get('zone_name', 'N/A')}\n"
                md += f"- **Score:** {site.get('anomaly_score', 0.0):.1f}/15\n" # Assuming max score 15
                md += f"- **Classification:** {site.get('classification', 'Unknown')}\n"
                md += f"- **Confidence:** {site.get('confidence', 0.0):.1f}\n"
                md += f"- **Features:** {site.get('feature_count', 0)}\n"
                md += f"- **Recommended Action:** {site.get('recommended_action', 'Investigation needed')}\n\n"
        else:
            md += "No high-priority sites identified in this analysis.\n\n"
        
        md += "## Immediate Recommendations\n\n"
        
        for action in recommendations.get('immediate_actions', [])[:5]:
            md += f"- {action}\n"
        
        md += "\n## Follow-up Analysis\n\n"
        
        for action in recommendations.get('follow_up_analysis', [])[:5]:
            md += f"- {action}\n"
        
        resource_allocation = recommendations.get('resource_allocation', {})
        if resource_allocation:
            md += f"\n## Resource Requirements\n\n"
            md += f"- **Ground Teams Needed:** {resource_allocation.get('ground_verification_teams', 'TBD')}\n"
            md += f"- **Estimated Duration:** {resource_allocation.get('estimated_expedition_duration', 'TBD')}\n"
            md += f"- **Estimated Budget:** {resource_allocation.get('estimated_budget', 'TBD')}\n"
        
        md += f"\n---\n*Generated by Amazon Archaeological Discovery Pipeline v{session_info.get('pipeline_version', 'N/A')}*\n"
        
        return md

    def _save_summary_markdown(self, report: dict, report_path: Path, reports_dir: Path):
        """Generate a summary markdown file in the specified reports directory."""
        summary_filename = f"discovery_summary.md"
        summary_path = reports_dir / summary_filename
        
        try:
            markdown_content = self._generate_markdown_summary(report)
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            logger.info(f"Saved summary markdown: {summary_path}")
        except Exception as e:
            logger.error(f"Error saving markdown summary {summary_path}: {e}", exc_info=True)

    def _create_empty_report(self, reports_dir: Path) -> dict:
        """Create an empty report when no data is available. Saves an empty report file."""
        
        empty_report_data = {
            'session_info': {
                'session_id': f"empty_{self.run_id}", # Use run_id
                'timestamp': datetime.now().isoformat(),
                'pipeline_version': 'multi-provider-2.0',
                'zones_analyzed': [],
                'providers_used': [],
                'total_analyses': 0
            },
            'executive_summary': {
                'total_zones_analyzed': 0,
                'high_confidence_sites': 0,
                'probable_archaeological_features': 0,
                'total_features_detected': 0,
                'highest_scoring_zone': None,
                'success_rate': "0%",
                'key_findings': ["No analysis results available"]
            },
            'zone_details': {},
            'high_priority_sites': [],
            'provider_analysis': {},
            'feature_inventory': {'total_count': 0},
            'recommendations': {
                'immediate_actions': ["Run archaeological analysis on target zones"],
                'follow_up_analysis': [],
                'methodology_improvements': [],
                'resource_allocation': {}
            },
            'technical_summary': {},
            'data_quality_assessment': {'overall_quality': 'No data processed'}
        }
        # Save the empty report as well for traceability
        empty_report_filename = "empty_report.json"
        empty_report_path = reports_dir / empty_report_filename
        try:
            with open(empty_report_path, 'w') as f:
                json.dump(empty_report_data, f, indent=2, default=str)
            logger.info(f"Saved empty report: {empty_report_path}")
        except Exception as e:
            logger.error(f"Failed to save empty report {empty_report_path}: {e}", exc_info=True)
        return empty_report_data
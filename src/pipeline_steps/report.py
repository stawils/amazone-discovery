from typing import Dict, List
from pathlib import Path
from datetime import datetime
from src.config import TARGET_ZONES, REPORTS_DIR
import logging

logger = logging.getLogger(__name__)

class ReportStep:
    """
    Modular report step for archaeological pipeline.
    Generates a comprehensive discovery report from scoring and analysis results.
    """
    def run(self, scoring_results: Dict[str, dict], analysis_results: Dict[str, List[dict]]) -> dict:
        """
        Generate a comprehensive archaeological discovery report.
        Args:
            scoring_results: Dictionary mapping zone_id to scoring result dicts.
            analysis_results: Dictionary mapping zone_id to list of analysis result dicts.
        Returns:
            Dictionary containing the full report.
        """
        logger.info("📊 Generating discovery report...")
        if not scoring_results or not analysis_results:
            logger.warning("No results available for report generation")
            return None
        # Create comprehensive report
        report = {
            'session_info': {
                'session_id': f"modular_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'timestamp': datetime.now().isoformat(),
                'pipeline_version': 'modular-1.0',
                'zones_analyzed': list(analysis_results.keys())
            },
            'executive_summary': self._generate_executive_summary(scoring_results),
            'zone_details': self._generate_zone_details(analysis_results, scoring_results),
            'high_priority_sites': self._extract_high_priority_sites(scoring_results),
            'recommendations': self._generate_recommendations(scoring_results),
            'technical_summary': self._generate_technical_summary(analysis_results)
        }
        # Save detailed JSON report
        report_path = REPORTS_DIR / f"archaeological_discovery_report_{report['session_info']['session_id']}.json"
        with open(report_path, 'w') as f:
            import json
            json.dump(report, f, indent=2)
        logger.info(f"✓ Report saved: {report_path}")
        return report

    def _generate_executive_summary(self, scoring_results: dict) -> dict:
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
        zone_details = {}
        for zone_id in analysis_results.keys():
            zone = TARGET_ZONES[zone_id]
            analysis = analysis_results.get(zone_id, [])
            scores = scoring_results.get(zone_id, {})
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
        high_priority = []
        for zone_id, scores in scoring_results.items():
            if scores['total_score'] >= 7:
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
        high_priority.sort(key=lambda x: x['anomaly_score'], reverse=True)
        return high_priority

    def _generate_recommendations(self, scoring_results: dict) -> dict:
        recommendations = {
            'immediate_actions': [],
            'follow_up_analysis': [],
            'methodology_improvements': [],
            'resource_allocation': {}
        }
        high_conf_zones = [k for k, v in scoring_results.items() if v['total_score'] >= 10]
        probable_zones = [k for k, v in scoring_results.items() if 7 <= v['total_score'] < 10]
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
        low_score_zones = [k for k, v in scoring_results.items() if v['total_score'] < 4]
        if low_score_zones:
            recommendations['follow_up_analysis'].append(
                f"Re-analyze {len(low_score_zones)} zones with different seasonal imagery"
            )
        recommendations['methodology_improvements'] = [
            "Integrate LiDAR data when available for improved detection",
            "Apply machine learning models trained on known archaeological sites",
            "Cross-reference with indigenous oral histories and place names",
            "Use multi-temporal analysis to detect subtle vegetation patterns"
        ]
        total_high_priority = len(high_conf_zones) + len(probable_zones)
        recommendations['resource_allocation'] = {
            'ground_verification_teams': max(1, total_high_priority // 3),
            'estimated_expedition_duration': f"{total_high_priority * 3-5} days per site",
            'priority_order': [TARGET_ZONES[z].name for z in high_conf_zones + probable_zones]
        }
        return recommendations

    def _generate_technical_summary(self, analysis_results: dict) -> dict:
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
        if score >= 10:
            return "IMMEDIATE GROUND VERIFICATION - High confidence archaeological site"
        elif score >= 7:
            return "HIGH-RESOLUTION IMAGERY + GROUND RECONNAISSANCE"
        elif score >= 4:
            return "Additional remote sensing analysis recommended"
        else:
            return "Continue monitoring with different seasonal imagery"

    def _assess_access_difficulty(self, zone) -> str:
        if 'Upper' in zone.name:
            return "High - Remote headwater region"
        elif 'Confluence' in zone.name:
            return "Medium - River access possible"
        else:
            return "Medium - Standard Amazon access challenges"

    def _estimate_verification_cost(self, zone) -> str:
        if 'High' in self._assess_access_difficulty(zone):
            return "$50,000-80,000 USD"
        else:
            return "$25,000-40,000 USD" 
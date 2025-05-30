from typing import Dict, List
from src.scoring import ConvergentAnomalyScorer
from src.config import TARGET_ZONES
import logging

logger = logging.getLogger(__name__)

class ScoringStep:
    """
    Modular scoring step for archaeological pipeline.
    Calculates convergent anomaly scores for each zone based on analysis results.
    Skips zones with no valid features.
    """
    def run(self, analysis_results: Dict[str, List[dict]]) -> Dict[str, dict]:
        """
        Calculate convergent anomaly scores for all detections.
        Args:
            analysis_results: Dictionary mapping zone_id to list of analysis result dicts.
        Returns:
            Dictionary mapping zone_id to scoring result dicts.
        """
        scorer = ConvergentAnomalyScorer()
        scoring_results: Dict[str, dict] = {}
        for zone_id, zone_results in analysis_results.items():
            # Feature-awareness: skip zones with no valid features
            valid_results = [r for r in zone_results if r.get('success')]
            if not valid_results:
                logger.warning(f"Skipping scoring for zone {zone_id}: no valid analysis results (all scenes may be missing required bands)")
                continue
            zone = TARGET_ZONES[zone_id]
            logger.info(f"  Scoring {zone.name}...")
            # Combine all detections from multiple scenes
            all_features = {
                'terra_preta_patches': [],
                'geometric_features': []
            }
            for scene_result in valid_results:
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
        return scoring_results 
from typing import Dict, List
from src.core.scoring import ConvergentAnomalyScorer
from src.core.config import TARGET_ZONES
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
        logger.info("🔢 Calculating convergent anomaly scores...")
        scoring_results = {}
        for zone_id, results in analysis_results.items():
            scorer = ConvergentAnomalyScorer()
            score = scorer.calculate_score(results)
            scoring_results[zone_id] = score
            logger.info(f"  ✓ Zone {zone_id}: Score {score['total_score']}")
        return scoring_results 
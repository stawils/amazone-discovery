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
        """
        logger.info("🔢 Calculating convergent anomaly scores...")
        
        # Use existing batch scoring function
        from src.core.scoring import batch_score_zones
        scoring_results = batch_score_zones(analysis_results)
        
        for zone_id, result in scoring_results.items():
            score = result.get('total_score', 0)
            logger.info(f"  ✓ Zone {zone_id}: Score {score}/15")
        
        return scoring_results 
from typing import List, Dict, Optional
from src.core.data_objects import SceneData, BaseProvider
from src.core.config import TARGET_ZONES
from src.pipeline.analysis import AnalysisStep
from src.pipeline.scoring import ScoringStep
from src.pipeline.report import ReportStep
from src.pipeline.visualization import VisualizationStep
from src.providers.gee_provider import GEEProvider
from src.providers.sentinel2_provider import Sentinel2Provider
import logging

logger = logging.getLogger(__name__)

class ModularPipeline:
    """
    Orchestrates the modular archaeological discovery pipeline steps.
    """
    def __init__(self, provider: str = 'gee'):
        self.provider = provider
        if provider == 'gee':
            self.provider_instance = GEEProvider()
        elif provider == 'sentinel2':
            self.provider_instance = Sentinel2Provider()
        else:
            raise ValueError(f"Unknown provider: {provider}. Supported: 'gee', 'sentinel2'")
        self.analysis_step = AnalysisStep()
        self.scoring_step = ScoringStep()
        self.report_step = ReportStep()
        self.visualization_step = VisualizationStep()

    def run(self, zones: Optional[List[str]] = None, max_scenes: int = 3) -> Dict[str, object]:
        """
        Run the full modular pipeline: download, analyze, score, report, visualize.
        Args:
            zones: List of zone IDs to process.
            max_scenes: Maximum number of scenes per zone.
        Returns:
            Dictionary with all major outputs (scene_data, analysis, scores, report, map_path).
        """
        logger.info("\n🚀 Starting Modular Archaeological Discovery Pipeline...")
        # Download
        logger.info(f"Using provider: {self.provider_instance.__class__.__name__}")
        
        # Let the provider handle zones parameter - it now has proper handling for None and string cases
        all_scene_data = self.provider_instance.download_data(zones, max_scenes)
        logger.info(f"✓ Downloaded {len(all_scene_data)} scenes.")
        # Analyze
        analysis_results = self.analysis_step.run(all_scene_data)
        logger.info(f"✓ Analysis complete for {len(analysis_results)} zones.")
        # Score
        scoring_results = self.scoring_step.run(analysis_results)
        logger.info(f"✓ Scoring complete for {len(scoring_results)} zones.")
        # Report
        report = self.report_step.run(scoring_results, analysis_results)
        logger.info(f"✓ Report generated.")
        # Visualize
        map_path = self.visualization_step.run(analysis_results, scoring_results)
        logger.info(f"✓ Visualization complete: {map_path}")
        return {
            'scene_data': all_scene_data,
            'analysis': analysis_results,
            'scores': scoring_results,
            'report': report,
            'map_path': map_path
        } 
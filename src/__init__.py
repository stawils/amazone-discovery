"""
Teratosoft Archaeological Discovery Pipeline
Advanced remote sensing and AI-powered archaeological site detection
"""

__version__ = "1.0.0"
__author__ = "Teratosoft"
__description__ = "AI-powered archaeological discovery system for the Amazon Basin"

# Core imports for easy access
from .config import TARGET_ZONES, APIConfig, DetectionConfig, ScoringConfig
from .usgs_api import USGSArchaeologyAPI
from .detectors import ArchaeologicalDetector
from .scoring import ConvergentAnomalyScorer
from .processors import ImageProcessor, BandProcessor
from .visualizers import ArchaeologicalVisualizer

__all__ = [
    'TARGET_ZONES',
    'APIConfig', 
    'DetectionConfig',
    'ScoringConfig',
    'USGSArchaeologyAPI',
    'ArchaeologicalDetector', 
    'ConvergentAnomalyScorer',
    'ImageProcessor',
    'BandProcessor',
    'ArchaeologicalVisualizer'
]
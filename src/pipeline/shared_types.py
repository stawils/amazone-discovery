"""
Centralized type definitions for the archaeological pipeline.
"""

from typing import List, TypedDict
from shapely.geometry.base import BaseGeometry

# Define precise TypedDicts for structured data
class FeatureDict(TypedDict, total=False):
    """
    Represents a feature with required and optional keys.
    'total=False' makes all keys optional by default.
    We specify required keys below.
    """
    # Required keys
    id: str
    type: str
    coordinates: List[float]
    provider: str
    
    # Optional keys added during analysis
    confidence: float
    area_m2: float
    original_geometry: BaseGeometry
    gedi_support: bool
    sentinel2_support: bool
    convergence_score: float
    enhanced_score: float

class ConvergencePair(TypedDict):
    """Represents a pair of convergent features."""
    feature1: FeatureDict
    feature2: FeatureDict
    distance_m: float
    strength: float
    combined_confidence: float
    providers: List[str]

class EnhancedCandidate(FeatureDict):
    """
    Represents an enhanced candidate, inheriting from FeatureDict.
    No new fields needed currently, but provides clear type separation.
    """
    pass

class ConvergenceResult(TypedDict):
    """Represents the result of convergence analysis."""
    convergent_pairs: List[ConvergencePair]
    features_with_convergence: List[FeatureDict]

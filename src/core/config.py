"""
Enhanced Configuration for Amazon Archaeological Discovery Project
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
)

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories
for dir_path in [DATA_DIR, RESULTS_DIR, NOTEBOOKS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Subdirectories for organized data storage
SATELLITE_DIR = DATA_DIR / "satellite"
LIDAR_DIR = DATA_DIR / "lidar"
PROCESSED_DIR = DATA_DIR / "processed"
EXPORTS_DIR = RESULTS_DIR / "exports"
REPORTS_DIR = RESULTS_DIR / "reports"
MAPS_DIR = RESULTS_DIR / "maps"

for subdir in [
    SATELLITE_DIR,
    LIDAR_DIR,
    PROCESSED_DIR,
    EXPORTS_DIR,
    REPORTS_DIR,
    MAPS_DIR,
]:
    subdir.mkdir(exist_ok=True, parents=True)


@dataclass
class TargetZone:
    """Target zone configuration"""

    name: str
    center: Tuple[float, float]  # (lat, lon)
    bbox: Tuple[float, float, float, float]  # (south, west, north, east)
    priority: int
    expected_features: str
    historical_evidence: str
    search_radius_km: float = 25.0
    min_feature_size_m: int = 50
    max_feature_size_m: int = 500


# Complete target zones from your research
TARGET_ZONES = {
    "negro_madeira": TargetZone(
        name="Negro-Madeira Confluence",
        center=(-3.1667, -60.0000),
        bbox=(-3.4, -60.3, -2.9, -59.7),
        priority=1,
        expected_features="Large ceremonial complexes, fortified settlements",
        historical_evidence="Orellana 1542 battle site + High terra preta probability",
        search_radius_km=30.0,
        min_feature_size_m=100,
        max_feature_size_m=800,
    ),
    "trombetas": TargetZone(
        name="Trombetas River Junction",
        center=(-1.5000, -56.0000),
        bbox=(-1.45, -56.5, -1.15, -55.5),
        priority=1,
        expected_features="Fortified settlements, 100-300m diameter earthworks",
        historical_evidence="Amazon warrior encounter + Eastern Amazon optimal zone",
        search_radius_km=25.0,
        min_feature_size_m=80,
        max_feature_size_m=400,
    ),
    "upper_xingu": TargetZone(
        name="upper-xingu-dead-horse-camp",
        center=(-11.7167, -54.5833),
        bbox=(-12.0, -55.0, -11.3, -54.0),
        priority=2,
        expected_features="Mound villages, road networks",
        historical_evidence="Fawcett target + 81 sites already found nearby",
        search_radius_km=35.0,
        min_feature_size_m=50,
        max_feature_size_m=300,
    ),
    "upper_napo": TargetZone(
        name="upper-naporegion",
        center=(-0.5000, -72.5000),
        bbox=(-1.0, -73.0, 0.0, -72.0),
        priority=3,
        expected_features="Circular settlements, defensive works",
        historical_evidence="Multiple expedition reports + Major confluence",
        search_radius_km=20.0,
    ),
    "maranon": TargetZone(
        name="maranon-river-system",
        center=(-4.0000, -75.0000),
        bbox=(-4.5, -75.5, -3.5, -74.5),
        priority=3,
        expected_features="Large settlement complexes",
        historical_evidence="60+ Jesuit missions + 200,000+ documented population",
        search_radius_km=40.0,
        min_feature_size_m=150,
        max_feature_size_m=1000,
    ),
}


# API Configuration
class APIConfig:
    # OpenAI for AI enhancement
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # Copernicus (if available)
    COPERNICUS_USER = os.getenv("COPERNICUS_USER")
    COPERNICUS_PASSWORD = os.getenv("COPERNICUS_PASSWORD")


class GEDIConfig:
    """Configuration for NASA GEDI access."""

    EARTHDATA_USERNAME = os.getenv("EARTHDATA_USERNAME")
    EARTHDATA_PASSWORD = os.getenv("EARTHDATA_PASSWORD")

    L2A_COLLECTION = "GEDI02_A.002"
    L3_COLLECTION = "GEDI03.002"

    CANOPY_GAP_THRESHOLD = 15.0
    ELEVATION_ANOMALY_THRESHOLD = 1.5


# Detection Parameters
class DetectionConfig:
    # Satellite imagery settings
    MAX_CLOUD_COVER = 20  # percent
    PREFERRED_MONTHS = [6, 7, 8, 9]  # Dry season (June-September)
    MIN_PIXEL_RESOLUTION = 30  # meters

    # Spectral analysis thresholds
    TERRA_PRETA_NDVI_MIN = 0.3
    TERRA_PRETA_INDEX_MIN = 0.1
    MIN_ANOMALY_PIXELS = 100

    # Geometric detection parameters
    CIRCLE_DETECTION_PARAMS = {
        "dp": 1,
        "min_dist_ratio": 0.8,  # minimum distance between circles as ratio of radius
        "param1": 50,
        "param2": 30,
        "blur_kernel": (5, 5),
    }

    # LiDAR processing (when available)
    LIDAR_GROUND_THRESHOLD = 2.0  # meters above ground
    LIDAR_VEGETATION_REMOVAL = True
    LIDAR_RESOLUTION = 1.0  # target resolution in meters


# Scoring System Configuration
class ScoringConfig:
    # Evidence weights for convergent anomaly scoring
    WEIGHTS = {
        "historical_reference": 2,
        "geometric_pattern": 3,  # per pattern, max 6
        "spectral_anomaly": 2,
        "terra_preta_signature": 2,
        "environmental_suitability": 1,
        "multiple_periods": 2,
        "indigenous_placename": 3,
        "priority_bonus": 1,
    }

    # Classification thresholds
    THRESHOLDS = {
        "high_confidence": 10,
        "probable_feature": 7,
        "possible_anomaly": 4,
        "natural_variation": 0,
    }

    # Maximum scores to prevent over-weighting
    MAX_GEOMETRIC_SCORE = 6
    MAX_TOTAL_SCORE = 15


# Visualization Configuration
class VisualizationConfig:
    # Map settings
    DEFAULT_ZOOM = 12
    TILE_LAYERS = ["OpenStreetMap", "Esri WorldImagery", "CartoDB Positron"]

    # Color schemes for different confidence levels
    CONFIDENCE_COLORS = {
        "high_confidence": "#ff0000",  # Red
        "probable_feature": "#ff8800",  # Orange
        "possible_anomaly": "#ffff00",  # Yellow
        "natural_variation": "#cccccc",  # Gray
    }

    # Export formats
    EXPORT_FORMATS = ["geojson", "kml", "shapefile", "csv"]


# Processing Configuration
class ProcessingConfig:
    # Multiprocessing
    MAX_WORKERS = min(8, os.cpu_count())

    # Memory management
    CHUNK_SIZE = 1024  # pixels for processing large images
    MAX_MEMORY_GB = 8

    # File formats
    RASTER_FORMAT = "GTiff"
    VECTOR_FORMAT = "GeoJSON"

    # Compression
    COMPRESSION = "lzw"
    PREDICTOR = 2


# Quality Control
class QualityConfig:
    # Minimum requirements for analysis
    MIN_SCENE_COVERAGE = 0.8  # 80% of AOI must be covered
    MAX_DATA_GAPS = 0.1  # 10% data gaps allowed
    MIN_CONFIDENCE_THRESHOLD = 0.6

    # Validation settings
    CROSS_VALIDATION_FOLDS = 5
    TEST_SIZE = 0.2
    RANDOM_STATE = 42


# Export all configurations
__all__ = [
    "TARGET_ZONES",
    "APIConfig",
    "DetectionConfig",
    "ScoringConfig",
    "VisualizationConfig",
    "ProcessingConfig",
    "QualityConfig",
    "PROJECT_ROOT",
    "DATA_DIR",
    "RESULTS_DIR",
]

MAPS_DIR = Path("./results/maps")

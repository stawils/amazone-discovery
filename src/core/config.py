"""
Enhanced Configuration for Amazon Archaeological Discovery Project
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from dotenv import load_dotenv

# Import Provider Classes
from src.providers.gedi_provider import GEDIProvider
from src.providers.sentinel2_provider import Sentinel2Provider
# Add other provider imports here if you have more, e.g.:
# from src.providers.landsat_provider import LandsatProvider

# Load environment variables
load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
)

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"  # Changed to be under project root
RESULTS_DIR = PROJECT_ROOT / "results"  # Changed to be under project root
EXPORTS_DIR = PROJECT_ROOT / "exports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories
for dir_path in [DATA_DIR, RESULTS_DIR, EXPORTS_DIR, NOTEBOOKS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Subdirectories for organized data storage
SATELLITE_DIR = DATA_DIR / "satellite"



# Only create essential data directories, not output directories
for subdir in [
    SATELLITE_DIR
]:
    subdir.mkdir(exist_ok=True, parents=True)


@dataclass
class TargetZone:
    """Target zone configuration"""

    id: str
    name: str
    center: Tuple[float, float]  # (lat, lon)
    bbox: Tuple[float, float, float, float]  # (south, west, north, east)
    priority: int
    expected_features: str
    historical_evidence: str
    search_radius_km: float = 7.0
    min_feature_size_m: int = 50
    max_feature_size_m: int = 500
    zone_type: str = "forested_buried_sites"        # NEW FIELD
    detection_strategy: str = "balanced"            # NEW FIELD


TARGET_ZONES = {
    "xingu_deep_forest": TargetZone(
    id="xingu_deep_forest",
    name="Xingu Deep Forest - Protected Interior",
    center=(-12.2, -53.1),
    bbox=(-12.29, -53.17, -12.11, -53.03),
    priority=1,  # HIGHEST PRIORITY
    expected_features="Hidden mound complexes, forest settlements",
    historical_evidence="Fawcett route interior - zero modern access",
    search_radius_km=7.0,
    min_feature_size_m=40,
    max_feature_size_m=300,
    zone_type="deep_forest_isolation",
    detection_strategy="mound_detection"
    ),
    
    "acre_capixaba_gap": TargetZone(
        id="acre_capixaba_gap",
        name="Acre-Capixaba Archaeological Gap",
        center=(-9.2456, -67.8934),
        bbox=(-9.3156, -67.9634, -9.1756, -67.8234),
        priority=1,
        expected_features="Geometric earthworks, ditched enclosures, ring villages, raised fields",
        historical_evidence="Strategic gap between documented Acre geoglyph clusters and Rondônia archaeological sites",
        search_radius_km=7.0,
        min_feature_size_m=100,
        max_feature_size_m=800,
        zone_type="cultural_corridor",
        detection_strategy="geometric_earthwork_detection"
    ),


    "negro_madeira_interfluve": TargetZone(
        id="negro_madeira_interfluve",
        name="Negro–Madeira Interfluve – Strategic Central Gap",
        center=(-4.1, -62.8),
        bbox=(-4.19, -62.89, -4.01, -62.71),
        priority=2,
        expected_features="Interfluvial ADE soils, possible ring villages, anthropogenic forest patches",
        historical_evidence="DOCUMENTED: Predicted high site density in gap by 2023 LiDAR modeling (McMichael et al."
        "Science Advances 2023)",
        search_radius_km=7.0,
        min_feature_size_m=30,
        max_feature_size_m=800,
        zone_type="technology_optimal",
        detection_strategy="terra_preta_focused"
    ),


    "acre_corridor_gap": TargetZone(
        id="acre_corridor_gap",
        name="Southern Acre - Geoglyph Corridor Gap",
        center=(-11.0, -68.9),
        bbox=(-11.07, -68.97, -10.93, -68.83),
        priority=1,
        expected_features="Geometric ditched enclosures (circles, squares), surrounding mounds, linear ditches",
        historical_evidence="DOCUMENTED: >450 geometric earthworks in Acre (Nature 2018); Gap in data where earth-building cultures likely connected",
        search_radius_km=7.0,
        min_feature_size_m=30,
        max_feature_size_m=800,
        zone_type="deforested_visible_earthworks",
        detection_strategy="geometric_primary"
    ),


    "bolivian_pantanal_edge": TargetZone(
        id="bolivian_pantanal_edge",
        name="Bolivian Pantanal Edge - Seasonal Settlement Zone",
        center=(-16.0, -60.0),
        bbox=(-16.07, -60.07, -15.93, -59.93),
        priority=3,
        expected_features="Seasonal camps, fish weirs, raised field systems, shell mounds",
        historical_evidence="STRATEGIC: Pantanal edge optimal for seasonal settlements. Underexplored transition zone",
        search_radius_km=7.0,
        min_feature_size_m=30,
        max_feature_size_m=600,
        zone_type="forested_buried_sites",
        detection_strategy="balanced"
    ),

    "upper_napo_quijos": TargetZone(
        id="upper_napo_quijos",
        name="Upper Napo Quijos - Terra Preta Archaeological Complex",
        center=(-0.4876, -76.8234),
        bbox=(-0.5576, -76.8934, -0.4176, -76.7534),
        priority=1,
        expected_features="Terra preta settlements, obsidian trade sites, pre-Columbian villages, anthropogenic soils",
        historical_evidence="DOCUMENTED: Quijos/Upper Napo regional system with documented terra preta formations and obsidian trade networks (1040-1210 AD); Pedro Porras archaeological surveys since 1950s; Alto Coca Reserve excavations with ceramic and obsidian artifacts",
        search_radius_km=7.0,
        min_feature_size_m=30,
        max_feature_size_m=400,
        zone_type="confirmed_extension",
        detection_strategy="terra_preta_focused"
    ),

    "envira_humaita_bridge": TargetZone(
        id="envira_humaita_bridge",
        name="Envira-Humaitá Cultural Bridge",
        center=(-7.8234, -66.4567),
        bbox=(-7.8934, -66.5267, -7.7534, -66.3867),
        priority=1,
        expected_features="Terra preta settlements, river confluence sites, trade network nodes",
        historical_evidence="Critical connection zone between Upper Purus and Middle Madeira cultural spheres",
        search_radius_km=7.0,
        min_feature_size_m=80,
        max_feature_size_m=600,
        zone_type="environmental_optimal",
        detection_strategy="riverine_settlement_pattern"
    ),

    "guapore_lamego_complex": TargetZone(
        id="guapore_lamego_complex",
        name="Guaporé-Lamego Archaeological Complex",
        center=(-12.5678, -63.2345),
        bbox=(-12.6378, -63.3045, -12.4978, -63.1645),
        priority=2,
        expected_features="Mounded villages, raised fields, ceramic traditions, fortified sites",
        historical_evidence="Extension of documented Llanos de Mojos cultural landscape into Brazilian territory",
        search_radius_km=7.0,
        min_feature_size_m=120,
        max_feature_size_m=1000,
        zone_type="confirmed_extension",
        detection_strategy="multi_sensor_convergence"
    ),

    "labrea_terra_preta_belt": TargetZone(
        id="labrea_terra_preta_belt",
        name="Lábrea Terra Preta Belt",
        center=(-7.2345, -64.7890),
        bbox=(-7.3045, -64.8590, -7.1645, -64.7190),
        priority=1,
        expected_features="Extensive terra preta deposits, village sites, riverine adaptations",
        historical_evidence="High-density anthropogenic soil formation zone along Purus River system",
        search_radius_km=7.0,
        min_feature_size_m=60,
        max_feature_size_m=500,
        zone_type="forested_buried_sites",
        detection_strategy="terra_preta_focused"
    ),

    "xingu_iriri_plateau": TargetZone(
        id="xingu_iriri_plateau",
        name="Xingu-Iriri Plateau Interface",
        center=(-8.9876, -55.1234),
        bbox=(-9.0576, -55.1934, -8.9176, -55.0534),
        priority=2,
        expected_features="Rock art sites, seasonal camps, lithic workshops, plateau edge settlements",
        historical_evidence="Interface zone between Xingu cultural sphere and Central Brazilian plateau peoples",
        search_radius_km=7.0,
        min_feature_size_m=40,
        max_feature_size_m=300,
        zone_type="technology_optimal",
        detection_strategy="multi_sensor_convergence"
    ),

    "upper_purus_peru": TargetZone(
        id="upper_purus_peru",
        name="Upper Purus Peru Extension",
        center=(-9.3456, -70.8901),
        bbox=(-9.4156, -70.9601, -9.2756, -70.8201),
        priority=1,
        expected_features="Fortified hilltop sites, terraced agriculture, stone constructions",
        historical_evidence="Andean-Amazon transition zone with documented pre-Inca and early colonial period sites",
        search_radius_km=7.0,
        min_feature_size_m=150,
        max_feature_size_m=1200,
        zone_type="environmental_optimal",
        detection_strategy="multi_sensor_convergence"
    ),

    "mojos_east_extension": TargetZone(
        id="mojos_east_extension",
        name="Mojos Eastern Extension",
        center=(-14.7890, -64.5678),
        bbox=(-14.8590, -64.6378, -14.7190, -64.4978),
        priority=2,
        expected_features="Raised field systems, water management, settlement mounds, causeways",
        historical_evidence="Eastern extension of documented Llanos de Mojos hydraulic landscape",
        search_radius_km=7.0,
        min_feature_size_m=200,
        max_feature_size_m=1500,
        zone_type="confirmed_extension",
        detection_strategy="geometric_earthwork_detection"
    )
}


# API Configuration
class APIConfig:
    # OpenAI for AI enhancement
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # Copernicus (if available)
    COPERNICUS_USER = os.getenv("COPERNICUS_USER")
    COPERNICUS_PASSWORD = os.getenv("COPERNICUS_PASSWORD")
    
    @classmethod
    def validate_credentials(cls) -> Dict[str, bool]:
        """Validate API credentials and return status"""
        validation_results = {
            "openai": bool(cls.OPENAI_API_KEY and len(cls.OPENAI_API_KEY.strip()) > 10),
            "copernicus": bool(cls.COPERNICUS_USER and cls.COPERNICUS_PASSWORD),
            "earthdata": bool(os.getenv("EARTHDATA_USERNAME") and os.getenv("EARTHDATA_PASSWORD"))
        }
        return validation_results
    
    @classmethod
    def get_missing_credentials(cls) -> List[str]:
        """Get list of missing credential sets"""
        validation = cls.validate_credentials()
        return [name for name, valid in validation.items() if not valid]


class GEDIConfig:
    """Configuration for NASA GEDI access."""

    EARTHDATA_USERNAME = os.getenv("EARTHDATA_USERNAME")
    EARTHDATA_PASSWORD = os.getenv("EARTHDATA_PASSWORD")
    
    @classmethod
    def is_configured(cls) -> bool:
        """Check if GEDI credentials are properly configured"""
        return bool(cls.EARTHDATA_USERNAME and cls.EARTHDATA_PASSWORD)

    L2A_COLLECTION = "GEDI02_A.002"
    L3_COLLECTION = "GEDI03.002"

    CANOPY_GAP_THRESHOLD = 15.0
    ELEVATION_ANOMALY_THRESHOLD = 1.5


# Detection Parameters
class DetectionConfig:
    # Satellite imagery settings
    MAX_CLOUD_COVER = 60  # percent (relaxed for tropical Amazon conditions)
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
    MAX_TOTAL_SCORE = 13  # Reduced from 15 (removed 2 points for historical reference)


# Zone-specific detection parameters
ZONE_DETECTION_CONFIG = {
    "deep_forest_isolation": {
        "detection_weights": {
            "terra_preta_weight": 3.0,    # PRIMARY: Focus on anthropogenic soils
            "crop_mark_weight": 2.0,      # Secondary: Vegetation stress indicators  
            "geometric_weight": 1.0       # Tertiary: Some geometric patterns
        },
        "thresholds": {
            "crop_mark_threshold": 0.06,  # Sensitive: Need to find hidden features
            "terra_preta_threshold": 0.15, # Standard: Proven terra preta detection
            "geometric_min_size": 10,     # Smaller: Allow smaller buried features
            "density_factor": 1.0         # Standard: Realistic archaeological density
        }
    },
    "cloud_forest_isolation": {
        "detection_weights": {
            "terra_preta_weight": 3.0,    # PRIMARY: Focus on anthropogenic soils
            "crop_mark_weight": 2.0,      # Secondary: Vegetation stress indicators  
            "geometric_weight": 1.0       # Tertiary: Some geometric patterns
        },
        "thresholds": {
            "crop_mark_threshold": 0.06,  # Sensitive: Need to find hidden features
            "terra_preta_threshold": 0.15, # Standard: Proven terra preta detection
            "geometric_min_size": 10,     # Smaller: Allow smaller buried features
            "density_factor": 1.0         # Standard: Realistic archaeological density
        }
    },
    "deforested_visible_earthworks": {
        "detection_weights": {
            "geometric_weight": 3.0,      # PRIMARY: Focus on geometric patterns
            "terra_preta_weight": 2.0,    # Secondary: Still check for villages
            "crop_mark_weight": 0.5       # Minimal: Don't need vegetation stress for visible features
        },
        "thresholds": {
            "crop_mark_threshold": 0.12,  # Conservative: Features already visible
            "geometric_threshold": 0.08,  # Lower: Easier to detect clear earthworks
            "geometric_min_size": 50,     # Larger: Focus on documented earthwork sizes
            "density_factor": 2.0         # Higher: Hundreds of earthworks expected
        }
    },
    "forested_buried_sites": {
        "detection_weights": {
            "terra_preta_weight": 3.0,    # PRIMARY: Focus on anthropogenic soils
            "crop_mark_weight": 2.0,      # Secondary: Vegetation stress indicators
            "geometric_weight": 1.0       # Tertiary: Some geometric patterns
        },
        "thresholds": {
            "crop_mark_threshold": 0.06,  # Sensitive: Need to find hidden features
            "terra_preta_threshold": 0.15, # Standard: Proven terra preta detection
            "geometric_min_size": 10,     # Smaller: Allow smaller buried features
            "density_factor": 1.0         # Standard: Realistic archaeological density
        }
    }
}


# Visualization Configuration
class VisualizationConfig:
    # Map settings - Very high zoom for detailed satellite imagery
    DEFAULT_ZOOM = 17  # Very high zoom for detailed archaeological features
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
    MAX_WORKERS = min(8, os.cpu_count() or 4)  # Default to 4 workers if cpu_count is None

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
    "SATELLITE_PROVIDERS",
    "DEFAULT_PROVIDERS",
    "APIConfig",
    "DetectionConfig",
    "ScoringConfig",
    "VisualizationConfig",
    "ProcessingConfig",
    "QualityConfig",
    "ZONE_DETECTION_CONFIG",  # NEW: Export the zone-specific config
    "PROJECT_ROOT",
    "DATA_DIR",
    "RESULTS_DIR",
]

# MAPS_DIR already defined above - removed duplicate

# Define SATELLITE_PROVIDERS dictionary
SATELLITE_PROVIDERS: Dict[str, type] = {
    "gedi": GEDIProvider,
    "sentinel2": Sentinel2Provider,
    # "gee": GEEProvider, # Removed GEEProvider
    # "landsat": LandsatProvider, # Example for another provider
}

# Define DEFAULT_PROVIDERS list (using keys from SATELLITE_PROVIDERS)
# This ensures DEFAULT_PROVIDERS only contains known, configured providers.
DEFAULT_PROVIDERS: List[str] = ["gedi", "sentinel2"] # Adjust as needed, e.g., add "gee"

# Ensure all DEFAULT_PROVIDERS are in SATELLITE_PROVIDERS
for provider_key in DEFAULT_PROVIDERS:
    if provider_key not in SATELLITE_PROVIDERS:
        raise ValueError(
            f"Error in config: Provider '{provider_key}' in DEFAULT_PROVIDERS "
            f"is not defined in SATELLITE_PROVIDERS. Available: {list(SATELLITE_PROVIDERS.keys())}"
        )

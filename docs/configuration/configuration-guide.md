 # Configuration Management Guide

## Overview

The Amazon Archaeological Discovery Pipeline uses a centralized configuration system that manages all aspects of the archaeological detection workflow, from target zone definitions to provider settings and detection parameters. This guide provides comprehensive documentation for understanding, configuring, and customizing the system.

## Configuration Architecture

The configuration system is built around dataclasses defined in `src/core/config.py`, providing type-safe, validated parameter management across all pipeline components.

### Core Configuration Principles

1. **Centralized Management**: All configuration in a single module
2. **Type Safety**: Dataclass-based configuration with type hints
3. **Environment Integration**: Support for environment variables and external configuration
4. **Default Values**: Sensible defaults based on archaeological research
5. **Validation**: Built-in parameter validation and range checking

---

## 1. Target Zone Configuration

### `TargetZone` Definition

Target zones define geographic areas for archaeological investigation with associated metadata and search parameters.

```python
@dataclass
class TargetZone:
    id: str                              # Unique zone identifier
    name: str                            # Human-readable zone name
    center: Tuple[float, float]          # (latitude, longitude) center point
    bbox: Tuple[float, float, float, float]  # (south, west, north, east)
    priority: int                        # Priority level (1=highest, 3=lowest)
    expected_features: str               # Expected archaeological features
    historical_evidence: str             # Historical documentation
    search_radius_km: float = 7.0        # Search radius in kilometers (default 7.0)
    min_feature_size_m: int = 50          # Minimum feature size in meters
    max_feature_size_m: int = 500         # Maximum feature size in meters
    zone_type: str = "forested_buried_sites"  # Zone classification for detection strategy
    detection_strategy: str = "balanced"      # Detection approach for this zone
```

### Pre-Configured Archaeological Zones

The system includes 9 pre-configured archaeological zones based on recent discoveries, historical documentation, and environmental analysis:

#### Xingu Deep Forest - Protected Interior
```python
"xingu_deep_forest": TargetZone(
    id="xingu_deep_forest",
    name="Xingu Deep Forest - Protected Interior",
    center=(-12.2, -53.1),
    bbox=(-12.29, -53.17, -12.11, -53.03),
    priority=1,
    expected_features="Hidden mound complexes, forest settlements",
    historical_evidence="Fawcett route interior - zero modern access",
    search_radius_km=7.0,
    min_feature_size_m=40,
    max_feature_size_m=300,
    zone_type="deep_forest_isolation",
    detection_strategy="mound_detection"
)
```

#### Casarabe Culture Core - Northern Extension
```python
"casarabe_north": TargetZone(
    id="casarabe_north",
    name="Casarabe Core North – Urban Extension",
    center=(-13.0, -65.7),
    bbox=(-13.09, -65.79, -12.91, -65.61),
    priority=1,
    expected_features="Monumental mound complexes, ring-ditch sites, raised causeways, artificial reservoirs",
    historical_evidence="Archaeological research documented by Prümers et al., Nature 2022",
    search_radius_km=7.0,
    min_feature_size_m=30,
    max_feature_size_m=800,
    zone_type="confirmed_extension",
    detection_strategy="multi_sensor_convergence"
)
```

#### Upper Napo Micro Region
```python
"upper_napo_micro": TargetZone(
    id="upper_napo_micro",
    name="Upper Napo Micro Region",
    center=(-0.50, -72.50),
    bbox=(-0.75, -72.75, -0.25, -72.25),
    priority=1,
    expected_features="Pre-Columbian settlements, terra preta, geometric earthworks",
    historical_evidence="Multiple 16th-17th century expedition reports",
    search_radius_km=25.0,
    min_feature_size_m=50.0,
    max_feature_size_m=20000.0
)
```

#### Upper Napo Micro Small (Testing Zone)
```python
"upper_napo_micro_small": TargetZone(
    id="upper_napo_micro_small",
    name="Upper Napo Micro Small",
    center=(-0.50, -72.50),
    bbox=(-0.55, -72.55, -0.45, -72.45),
    priority=1,
    expected_features="Pre-Columbian settlements, terra preta",
    historical_evidence="Testing zone derived from upper_napo_micro",
    search_radius_km=5.0,
    min_feature_size_m=50.0,
    max_feature_size_m=5000.0
)
```

### Creating Custom Target Zones

#### Basic Zone Definition

```python
from src.core.config import TargetZone

# Define custom archaeological zone
custom_zone = TargetZone(
    id="custom_site_001",
    name="Custom Archaeological Site",
    center=(-2.15, -68.30),  # Latitude, Longitude
    bbox=(-2.25, -68.40, -2.05, -68.20),  # min_lat, min_lon, max_lat, max_lon
    priority=2,
    expected_features="Possible earthworks and clearings",
    historical_evidence="Local indigenous oral histories",
    search_radius_km=15.0,
    min_feature_size_m=100.0,
    max_feature_size_m=10000.0
)

# Add to configuration
from src.core.config import TARGET_ZONES
TARGET_ZONES["custom_site_001"] = custom_zone
```

#### Geographic Coordinate Guidelines

**Coordinate System**: WGS84 (EPSG:4326)
**Format**: Decimal degrees
**Precision**: 6 decimal places recommended (±0.1m accuracy)

**Amazon Region Bounds**:
- **Latitude Range**: -20° to 10° (Southern/Northern bounds)
- **Longitude Range**: -85° to -45° (Western/Eastern bounds)

**Bounding Box Calculation**:
```python
def calculate_bbox(center_lat: float, center_lon: float, radius_km: float) -> Tuple[float, float, float, float]:
    """Calculate bounding box from center point and radius."""
    # Approximate degrees per kilometer at equator
    lat_deg_per_km = 1 / 111.32
    lon_deg_per_km = 1 / (111.32 * cos(radians(center_lat)))
    
    lat_offset = radius_km * lat_deg_per_km
    lon_offset = radius_km * lon_deg_per_km
    
    return (
        center_lat - lat_offset,  # min_lat
        center_lon - lon_offset,  # min_lon
        center_lat + lat_offset,  # max_lat
        center_lon + lon_offset   # max_lon
    )
```

### Zone Priority Levels

**Priority 1 (Highest)**:
- Zones with strong historical documentation
- Multiple evidence convergence
- High archaeological potential
- Immediate investigation priority

**Priority 2 (Medium)**:
- Zones with moderate evidence
- Single-source documentation
- Exploratory investigation targets

**Priority 3 (Low)**:
- Zones with limited evidence
- Speculative targets
- Long-term monitoring areas

### Zone-Specific Detection Strategies

The system uses zone-specific detection parameters based on environmental and archaeological characteristics:

```python
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
```

**Detection Strategy Usage**:
- **deep_forest_isolation**: For protected interior zones with dense forest cover
- **deforested_visible_earthworks**: For areas with visible earthworks and clear satellite access
- **forested_buried_sites**: For general Amazon forest archaeological surveys

---

## 2. API Configuration

### `APIConfig` Class

Manages authentication credentials for external services.

```python
@dataclass
class APIConfig:
    openai_api_key: Optional[str] = None
    copernicus_user: Optional[str] = None
    copernicus_password: Optional[str] = None
    earthdata_username: Optional[str] = None
    earthdata_password: Optional[str] = None
```

### Environment Variable Integration

The system automatically loads credentials from environment variables:

```bash
# OpenAI API (for AI-enhanced analysis)
export OPENAI_API_KEY="sk-your-openai-api-key"

# NASA Earthdata (required for GEDI data access)
export EARTHDATA_USERNAME="your_nasa_username"
export EARTHDATA_PASSWORD="your_nasa_password"

# Copernicus (optional, for Sentinel-2 backup access)
export COPERNICUS_USER="your_copernicus_username"
export COPERNICUS_PASSWORD="your_copernicus_password"
```

### Credential Validation

```python
from src.core.config import get_api_config

def validate_credentials():
    """Validate required API credentials."""
    api_config = get_api_config()
    
    # Required for GEDI data access
    if not api_config.earthdata_username or not api_config.earthdata_password:
        raise ValueError("NASA Earthdata credentials required for GEDI access")
    
    # Optional but recommended for AI features
    if not api_config.openai_api_key:
        print("Warning: OpenAI API key not set - AI features disabled")
    
    return True
```

---

## 3. GEDI Configuration

### `GEDIConfig` Class

NASA GEDI satellite configuration and archaeological detection parameters.

```python
@dataclass
class GEDIConfig:
    gap_threshold: float = 15.0           # Canopy gap threshold (meters)
    anomaly_threshold: float = 2.0        # Elevation anomaly (standard deviations)
    min_cluster_size: int = 3             # Minimum cluster size for clearings
    max_feature_size_km2: float = 50.0    # Maximum archaeological site size
    linear_r2_threshold: float = 0.8      # Linear feature R² threshold
    clustering_eps: float = 0.0015        # DBSCAN clustering radius (degrees)
    footprint_area_m2: float = 625.0      # GEDI footprint area (25m diameter)
    elevation_data_required: bool = True  # Require elevation data for analysis
    quality_threshold: float = 0.9        # Minimum GEDI quality flag
```

### Archaeological Detection Thresholds

#### Canopy Gap Analysis

**Gap Threshold (15.0 meters)**:
- Based on Amazon forest ecology research
- Distinguishes human clearings from natural gaps
- Validated against known archaeological sites

```python
# Customizing gap detection
gedi_config = GEDIConfig(
    gap_threshold=12.0,  # More sensitive detection
    min_cluster_size=4   # Require larger clearings
)
```

#### Elevation Anomaly Detection

**Anomaly Threshold (2.0 standard deviations)**:
- Statistical threshold for earthwork detection
- Balances sensitivity vs. false positives
- Based on Amazon topographic variation studies

```python
# Conservative earthwork detection
gedi_config = GEDIConfig(
    anomaly_threshold=2.5,  # Higher threshold
    max_feature_size_km2=25.0  # Smaller maximum size
)
```

#### Linear Feature Analysis

**R² Threshold (0.8)**:
- Linear regression goodness of fit
- Distinguishes constructed features from natural patterns
- Calibrated on known causeway networks

```python
# Strict linear feature detection
gedi_config = GEDIConfig(
    linear_r2_threshold=0.9,  # Very high linearity required
    clustering_eps=0.001      # Tighter clustering
)
```

### GEDI Data Quality Parameters

```python
@dataclass
class GEDIQualityConfig:
    beam_types: List[str] = field(default_factory=lambda: ["BEAM0000", "BEAM0001", "BEAM0010", "BEAM0011"])
    quality_flag_threshold: float = 0.9
    algorithm_run_flag: int = 1
    degrade_flag: int = 0
    surface_flag: List[int] = field(default_factory=lambda: [0, 1])  # Land surfaces only
    sensitivity: float = 0.9
    solar_elevation_threshold: float = 0.0  # Day/night data
```

---

## 4. Detection Configuration

### `DetectionConfig` Class

Satellite imagery analysis and detection parameters.

```python
@dataclass
class DetectionConfig:
    min_confidence: float = 0.5                    # Minimum detection confidence
    cloud_cover_threshold: float = 0.2             # Maximum cloud cover (20%)
    preferred_months: List[int] = field(default_factory=lambda: [6, 7, 8, 9])  # Dry season
    require_elevation_data: bool = True            # Require elevation data
    enable_gpu_acceleration: bool = True           # Enable GPU processing
    max_feature_size_m2: float = 500000.0         # Maximum feature size (50 hectares)
    min_feature_size_m2: float = 100.0            # Minimum feature size
    spectral_threshold_ndvi: Tuple[float, float] = (0.2, 0.8)  # NDVI range
    terra_preta_threshold: float = 0.12            # Terra preta spectral threshold
    vegetation_stress_threshold: float = 0.1       # Vegetation stress detection
```

### Seasonal Considerations

#### Optimal Detection Periods

**Dry Season (June-September)**:
- Reduced cloud cover
- Enhanced spectral contrast
- Minimal vegetation interference
- Optimal archaeological visibility

```python
# Dry season configuration
detection_config = DetectionConfig(
    preferred_months=[6, 7, 8, 9],  # June through September
    cloud_cover_threshold=0.15,     # Stricter cloud filtering
    vegetation_stress_threshold=0.08 # More sensitive during dry season
)
```

**Transitional Periods (May, October)**:
- Moderate cloud cover
- Good spectral conditions
- Acceptable for analysis

```python
# Extended season configuration
detection_config = DetectionConfig(
    preferred_months=[5, 6, 7, 8, 9, 10],  # Extended dry season
    cloud_cover_threshold=0.25,            # Relaxed cloud filtering
    require_elevation_data=False            # More flexible requirements
)
```

### Spectral Analysis Parameters

#### NDVI Thresholds

**Healthy Vegetation**: 0.6-0.8
**Moderate Vegetation**: 0.4-0.6
**Sparse Vegetation**: 0.2-0.4
**Archaeological Targets**: 0.35-0.7 (intermediate range)

```python
# Archaeological NDVI optimization
detection_config = DetectionConfig(
    spectral_threshold_ndvi=(0.35, 0.7),  # Archaeological sweet spot
    vegetation_stress_threshold=0.1,       # Detect subtle stress
    terra_preta_threshold=0.12             # Sensitive soil detection
)
```

---

## 5. Scoring Configuration

### `ScoringConfig` Class

Evidence weighting and classification thresholds for convergent anomaly detection.

```python
@dataclass
class ScoringConfig:
    # Evidence weights
    historical_weight: int = 2              # Historical documentation
    geometric_pattern_weight: int = 3       # Geometric patterns (each)
    geometric_pattern_max: int = 6          # Maximum geometric points
    terra_preta_weight: int = 2             # Anthropogenic soil
    environmental_weight: int = 1           # Environmental suitability
    priority_zone_bonus: int = 1            # High-priority zone bonus
    convergence_bonus_max: int = 2          # Spatial convergence bonus
    
    # Classification thresholds
    high_confidence_threshold: int = 10     # HIGH CONFIDENCE (≥10 points)
    probable_threshold: int = 7             # PROBABLE (7-9 points)
    possible_threshold: int = 4             # POSSIBLE (4-6 points)
    
    # Spatial convergence parameters
    convergence_radius_m: float = 100.0     # Convergence proximity (meters)
    min_convergence_evidence: int = 2       # Minimum evidence for convergence
```

### Evidence Weight Rationale

#### Historical Evidence (2 points)
- Documentary sources from 16th-17th centuries
- Indigenous oral histories
- Previous archaeological investigations
- High reliability but limited spatial precision

#### Geometric Patterns (3 points each, max 6)
- Circular features (plazas, defensive structures)
- Linear features (causeways, field boundaries)
- Rectangular features (platform mounds, structures)
- Strong indicators of human planning and construction

#### Terra Preta Signatures (2 points)
- Anthropogenic soil formation
- Indicates sustained human occupation
- Well-established archaeological indicator
- Detectable via multispectral analysis

#### Environmental Suitability (1 point)
- Suitable terrain for settlement
- Water access and soil quality
- Defensive advantages
- Supportive but not definitive evidence

#### Spatial Convergence Bonus (0.5-2 points)
- Multiple evidence types in proximity
- Increases archaeological confidence
- Validates independent detection methods
- Strongest indicator when multiple sensors agree

### Classification System Calibration

#### HIGH CONFIDENCE (≥10 points)
**Immediate ground verification recommended**
- Multiple evidence convergence
- Strong archaeological indicators
- High success probability (>85%)
- Priority for field expeditions

#### PROBABLE (7-9 points)
**High-resolution follow-up analysis**
- Solid evidence base
- Moderate archaeological indicators
- Good success probability (60-85%)
- Secondary priority for investigation

#### POSSIBLE (4-6 points)
**Additional remote sensing analysis**
- Limited evidence
- Weak archaeological indicators
- Lower success probability (30-60%)
- Monitor with seasonal imagery

#### NATURAL VARIATION (0-3 points)
**Continue monitoring**
- Minimal evidence
- Natural explanation likely
- Low success probability (<30%)
- Long-term observation

### Custom Scoring Configuration

```python
# Conservative scoring (reduce false positives)
conservative_scoring = ScoringConfig(
    historical_weight=3,                # Higher weight for documentation
    geometric_pattern_weight=4,         # Stronger geometric evidence
    high_confidence_threshold=12,       # Higher threshold
    probable_threshold=9,               # Stricter classification
    convergence_radius_m=50.0          # Tighter convergence requirement
)

# Exploratory scoring (increase sensitivity)
exploratory_scoring = ScoringConfig(
    terra_preta_weight=3,               # Higher weight for spectral
    environmental_weight=2,             # More environmental consideration
    high_confidence_threshold=8,        # Lower threshold
    probable_threshold=5,               # More inclusive classification
    convergence_radius_m=200.0         # Looser convergence requirement
)
```

---

## 6. Visualization Configuration

### `VisualizationConfig` Class

Map settings, export formats, and visualization parameters.

```python
@dataclass
class VisualizationConfig:
    # Map settings
    default_zoom: int = 10                       # Default map zoom level
    basemap_provider: str = "CartoDB.Positron"   # Leaflet basemap
    feature_color_scheme: Dict[str, str] = field(default_factory=lambda: {
        "gedi": "#FF6B6B",           # Red for GEDI features
        "sentinel2": "#4ECDC4",      # Teal for Sentinel-2 features
        "convergent": "#45B7D1",     # Blue for convergent features
        "high_confidence": "#96CEB4", # Green for high confidence
        "probable": "#FFEAA7",       # Yellow for probable
        "possible": "#DDA0DD"        # Purple for possible
    })
    
    # Export formats
    export_formats: List[str] = field(default_factory=lambda: ["geojson", "kml", "shapefile"])
    coordinate_precision: int = 6                # Decimal places for coordinates
    include_metadata: bool = True               # Include detection metadata
    
    # Interactive features
    enable_clustering: bool = True              # Cluster nearby features
    cluster_radius: int = 50                   # Clustering radius (pixels)
    popup_content: List[str] = field(default_factory=lambda: [
        "type", "confidence", "provider", "coordinates"
    ])
    
    # Performance settings
    max_features_per_layer: int = 1000         # Maximum features per map layer
    enable_heatmaps: bool = True               # Generate density heatmaps
    simplify_geometries: bool = True           # Simplify complex geometries
```

### Map Customization

#### Basemap Options

**Scientific Visualization**:
```python
viz_config = VisualizationConfig(
    basemap_provider="Esri.WorldImagery",  # Satellite imagery basemap
    default_zoom=12,                       # Detailed zoom level
    enable_clustering=False                # Show individual features
)
```

**Presentation Maps**:
```python
viz_config = VisualizationConfig(
    basemap_provider="CartoDB.Positron",   # Clean, minimal basemap
    enable_clustering=True,                # Group nearby features
    include_metadata=False                 # Simplified popups
)
```

#### Color Scheme Customization

```python
# Custom archaeological color scheme
archaeological_colors = {
    "settlements": "#8B4513",      # Brown for settlements
    "earthworks": "#228B22",       # Forest green for earthworks
    "terra_preta": "#DAA520",      # Goldenrod for soil signatures
    "causeways": "#4169E1",        # Royal blue for linear features
    "high_priority": "#DC143C"     # Crimson for priority sites
}

viz_config = VisualizationConfig(
    feature_color_scheme=archaeological_colors
)
```

---

## 7. Processing Configuration

### `ProcessingConfig` Class

Memory management, file formats, and performance settings.

```python
@dataclass
class ProcessingConfig:
    # Memory management
    max_memory_gb: float = 8.0               # Maximum memory usage
    chunk_size: int = 10000                  # Processing chunk size
    enable_multiprocessing: bool = True      # Enable parallel processing
    max_workers: int = 4                     # Maximum worker processes
    
    # File format preferences
    preferred_image_format: str = "GeoTIFF"  # Raster data format
    compression: str = "LZW"                 # Compression algorithm
    nodata_value: float = -9999.0           # No-data value
    
    # Caching settings
    enable_caching: bool = True              # Enable result caching
    cache_expiry_days: int = 30              # Cache expiration
    cache_compression: bool = True           # Compress cached data
    
    # GPU acceleration
    gpu_memory_fraction: float = 0.8         # GPU memory allocation
    gpu_batch_size: int = 1000               # GPU processing batch size
    fallback_to_cpu: bool = True             # CPU fallback if GPU fails
```

### Performance Optimization

#### Memory-Constrained Systems

```python
# Low-memory configuration
low_memory_config = ProcessingConfig(
    max_memory_gb=2.0,          # Limited memory
    chunk_size=5000,            # Smaller chunks
    enable_multiprocessing=False, # Single-threaded
    gpu_memory_fraction=0.4,    # Conservative GPU usage
    cache_compression=True      # Aggressive compression
)
```

#### High-Performance Systems

```python
# High-performance configuration
hpc_config = ProcessingConfig(
    max_memory_gb=32.0,         # Abundant memory
    chunk_size=50000,           # Large chunks
    max_workers=16,             # Many workers
    gpu_memory_fraction=0.9,    # Aggressive GPU usage
    enable_caching=True         # Full caching
)
```

### GPU Optimization Settings

```python
@dataclass
class GPUConfig:
    device_id: int = 0                      # GPU device ID
    memory_growth: bool = True              # Dynamic memory allocation
    mixed_precision: bool = True            # Use FP16 for speed
    synchronous_execution: bool = False     # Asynchronous processing
    profiling_enabled: bool = False         # Performance profiling
```

---

## 8. Quality Configuration

### `QualityConfig` Class

Validation and quality control parameters.

```python
@dataclass
class QualityConfig:
    # Detection validation
    min_detection_confidence: float = 0.5   # Minimum detection confidence
    max_false_positive_rate: float = 0.15   # Maximum acceptable false positives
    require_multi_sensor: bool = False      # Require multi-sensor validation
    
    # Coordinate validation
    coordinate_precision_m: float = 10.0    # Coordinate precision requirement
    bounds_validation: bool = True          # Validate geographic bounds
    datum_validation: bool = True           # Validate coordinate datum
    
    # Data quality metrics
    completeness_threshold: float = 0.8     # Minimum data completeness
    temporal_consistency: bool = True       # Check temporal consistency
    spatial_consistency: bool = True        # Check spatial consistency
    
    # Archaeological validity
    size_range_validation: bool = True      # Validate feature sizes
    density_validation: bool = True         # Validate site density
    environmental_validation: bool = True   # Validate environmental context
```

### Validation Rules

#### Archaeological Site Density

**Amazon Research-Based Limits**:
- **High Density**: 5-10 sites per 100 km²
- **Moderate Density**: 2-5 sites per 100 km²
- **Low Density**: 1-2 sites per 100 km²

```python
quality_config = QualityConfig(
    density_validation=True,
    max_sites_per_100km2=5,    # Conservative estimate
    min_site_spacing_km=2.0    # Minimum inter-site distance
)
```

#### Feature Size Validation

**Archaeological Size Ranges**:
- **Individual Houses**: 50-200 m²
- **Villages**: 0.5-5 hectares
- **Large Settlements**: 5-50 hectares
- **Urban Complexes**: 50-500+ hectares

```python
quality_config = QualityConfig(
    min_feature_size_m2=50.0,      # Minimum house size
    max_feature_size_m2=500000.0,  # Maximum complex size (50 ha)
    size_range_validation=True
)
```

---

## 9. Environment Integration

### Configuration Loading

```python
from src.core.config import (
    get_target_zones, get_api_config, get_gedi_config,
    get_detection_config, get_scoring_config, get_visualization_config
)

# Load all configurations
target_zones = get_target_zones()
api_config = get_api_config()
gedi_config = get_gedi_config()
detection_config = get_detection_config()
scoring_config = get_scoring_config()
viz_config = get_visualization_config()
```

### Environment Variable Override

Create `.env` file in project root:

```bash
# API Configuration
OPENAI_API_KEY=sk-your-api-key
EARTHDATA_USERNAME=your_username
EARTHDATA_PASSWORD=your_password

# Detection Parameters
GEDI_GAP_THRESHOLD=12.0
GEDI_ANOMALY_THRESHOLD=2.5
DETECTION_MIN_CONFIDENCE=0.6
CLOUD_COVER_THRESHOLD=0.15

# Processing Configuration
MAX_MEMORY_GB=16.0
ENABLE_GPU=true
MAX_WORKERS=8

# Quality Control
MIN_DETECTION_CONFIDENCE=0.6
MAX_FALSE_POSITIVE_RATE=0.10
REQUIRE_MULTI_SENSOR=false
```

### Configuration Validation

```python
def validate_configuration():
    """Comprehensive configuration validation."""
    
    # Validate target zones
    for zone_id, zone in TARGET_ZONES.items():
        assert -90 <= zone.center[0] <= 90, f"Invalid latitude for {zone_id}"
        assert -180 <= zone.center[1] <= 180, f"Invalid longitude for {zone_id}"
        assert zone.search_radius_km > 0, f"Invalid search radius for {zone_id}"
    
    # Validate API configuration
    api_config = get_api_config()
    if not api_config.earthdata_username:
        warnings.warn("GEDI data access requires NASA Earthdata credentials")
    
    # Validate detection parameters
    detection_config = get_detection_config()
    assert 0 <= detection_config.min_confidence <= 1, "Invalid confidence range"
    assert 0 <= detection_config.cloud_cover_threshold <= 1, "Invalid cloud cover threshold"
    
    # Validate scoring thresholds
    scoring_config = get_scoring_config()
    assert scoring_config.high_confidence_threshold >= scoring_config.probable_threshold
    assert scoring_config.probable_threshold >= scoring_config.possible_threshold
    
    return True
```

---

## 10. Best Practices

### Configuration Management

1. **Version Control**: Include configuration in version control
2. **Environment Separation**: Use different configs for development/production
3. **Validation**: Always validate configuration at startup
4. **Documentation**: Document all configuration changes
5. **Backup**: Maintain backups of working configurations

### Parameter Tuning

1. **Start Conservative**: Begin with restrictive thresholds
2. **Iterative Refinement**: Gradually adjust based on results
3. **Ground Truth**: Validate against known archaeological sites
4. **Cross-Validation**: Test across multiple zones
5. **Documentation**: Record successful parameter combinations

### Performance Optimization

1. **Memory Monitoring**: Monitor memory usage during processing
2. **GPU Utilization**: Optimize GPU settings for your hardware
3. **Cache Management**: Regularly clean cache to free disk space
4. **Parallel Processing**: Tune worker count for your CPU
5. **Profiling**: Use profiling to identify bottlenecks

This configuration guide provides the foundation for customizing the Amazon Archaeological Discovery Pipeline to specific research needs, environmental conditions, and computational resources.
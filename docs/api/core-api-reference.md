# Core API Reference

## Overview

The Amazon Archaeological Analysis Pipeline core API provides a comprehensive framework for multi-sensor archaeological detection in the Amazon rainforest. This reference documents the essential classes, methods, and interfaces that form the foundation of the system.

## Core Architecture

The system follows a modular architecture with five primary components:

1. **Configuration System** (`config.py`) - Centralized parameter management
2. **Data Objects** (`data_objects.py`) - Standardized data containers and provider interfaces
3. **Pipeline Orchestration** (`modular_pipeline.py`) - Four-stage processing workflow
4. **Convergent Scoring** (`scoring.py`) - Multi-evidence anomaly detection
5. **Export Management** (`export_manager.py`) - Unified GeoJSON output system

---

## 1. Configuration System (`src/core/config.py`)

### Target Zone Configuration

#### `TargetZone` (dataclass)

Defines archaeological search zones with geographic and contextual parameters.

```python
@dataclass
class TargetZone:
    id: str                     # Unique zone identifier
    name: str                   # Human-readable zone name
    center: Tuple[float, float] # [latitude, longitude] center point
    bbox: Tuple[float, float, float, float]  # [min_lat, min_lon, max_lat, max_lon]
    priority: int               # Priority level (1=highest, 3=lowest)
    expected_features: str      # Expected archaeological features
    historical_evidence: str    # Historical documentation
    search_radius_km: float = 50.0        # Search radius in kilometers
    min_feature_size_m: float = 100.0     # Minimum feature size in meters
    max_feature_size_m: float = 50000.0   # Maximum feature size in meters
```

**Usage Example:**
```python
from src.core.config import TARGET_ZONES

# Access pre-configured zones
upper_napo = TARGET_ZONES["upper_napo_micro"]
print(f"Zone: {upper_napo.name}")
print(f"Center: {upper_napo.center}")
print(f"Expected: {upper_napo.expected_features}")
```

### Configuration Classes

#### `APIConfig`
API credentials and authentication configuration.

```python
@dataclass
class APIConfig:
    openai_api_key: Optional[str] = None
    copernicus_user: Optional[str] = None
    copernicus_password: Optional[str] = None
    earthdata_username: Optional[str] = None
    earthdata_password: Optional[str] = None
```

#### `GEDIConfig`
NASA GEDI satellite configuration and detection thresholds.

```python
@dataclass
class GEDIConfig:
    gap_threshold: float = 15.0           # Canopy gap threshold (meters)
    anomaly_threshold: float = 2.0        # Elevation anomaly (std deviations)
    min_cluster_size: int = 3             # Minimum cluster size for clearings
    max_feature_size_km2: float = 50.0    # Maximum archaeological site size
    linear_r2_threshold: float = 0.8      # Linear feature R² threshold
```

#### `DetectionConfig`
Satellite imagery analysis parameters.

```python
@dataclass
class DetectionConfig:
    min_confidence: float = 0.5
    cloud_cover_threshold: float = 0.2
    preferred_months: List[int] = field(default_factory=lambda: [6, 7, 8, 9])  # Dry season
    require_elevation_data: bool = True
    enable_gpu_acceleration: bool = True
```

#### `ScoringConfig`
Evidence weighting and classification thresholds.

```python
@dataclass
class ScoringConfig:
    historical_weight: int = 2
    geometric_pattern_weight: int = 3
    geometric_pattern_max: int = 6
    terra_preta_weight: int = 2
    environmental_weight: int = 1
    priority_zone_bonus: int = 1
    convergence_bonus_max: int = 2
    
    # Classification thresholds
    high_confidence_threshold: int = 10
    probable_threshold: int = 7
    possible_threshold: int = 4
```

### Global Constants

```python
# Pre-configured archaeological zones
TARGET_ZONES: Dict[str, TargetZone] = {
    "upper_napo_micro": TargetZone(...),
    "upper_napo_micro_small": TargetZone(...),
    "negro_madeira_confluence": TargetZone(...),
    # ... additional zones
}

# Provider registration system
SATELLITE_PROVIDERS: Dict[str, type] = {
    "gedi": GEDIProvider,
    "sentinel2": Sentinel2Provider
}

# Active providers for pipeline execution
DEFAULT_PROVIDERS: List[str] = ["gedi", "sentinel2"]

# Directory structure
RESULTS_DIR = Path("results")
EXPORTS_DIR = Path("exports")
CACHE_DIR = Path("cache")
```

---

## 2. Data Objects (`src/core/data_objects.py`)

### Core Data Container

#### `SceneData`

Standardized container for satellite scene data across all providers.

```python
class SceneData:
    def __init__(self, 
                 zone_id: str, 
                 provider: str, 
                 scene_id: str, 
                 file_paths: Dict[str, Path], 
                 available_bands: List[str], 
                 metadata: Optional[Dict] = None, 
                 features: Optional[List] = None, 
                 composite_file_path: Optional[Path] = None, 
                 provider_name: Optional[str] = None):
```

**Attributes:**
- `zone_id`: Target zone identifier
- `provider`: Data provider name ("gedi", "sentinel2")
- `scene_id`: Unique scene identifier
- `file_paths`: Dictionary mapping band names to file paths
- `available_bands`: List of available spectral/data bands
- `metadata`: Optional scene metadata
- `features`: Optional pre-extracted features
- `composite_file_path`: Path to composite/processed data file

**Key Methods:**

```python
def has_band(self, band: str) -> bool:
    """Check if specific band is available."""
    return band in self.available_bands

def get_band_path(self, band: str) -> Optional[Path]:
    """Get file path for specific band."""
    return self.file_paths.get(band)

def get_default_path(self, provider_name: str) -> Path:
    """Get default storage path for provider."""
    return CACHE_DIR / provider_name / self.zone_id / self.scene_id
```

**Usage Example:**
```python
# Check data availability
if scene.has_band("B04"):  # Red band for Sentinel-2
    red_band_path = scene.get_band_path("B04")

# Access metadata
cloud_cover = scene.metadata.get("cloud_cover", 0)
acquisition_date = scene.metadata.get("acquisition_date")
```

### Provider Interface

#### `BaseProvider` (Abstract Base Class)

Abstract interface that all data providers must implement.

```python
from abc import ABC, abstractmethod

class BaseProvider(ABC):
    @abstractmethod
    def download_data(self, zones: List[str], max_scenes: int = 3) -> List[SceneData]:
        """Download data for specified zones.
        
        Args:
            zones: List of zone IDs to process
            max_scenes: Maximum scenes per zone
            
        Returns:
            List of SceneData objects
        """
        pass
```

**Implementation Pattern:**
```python
class CustomProvider(BaseProvider):
    def download_data(self, zones: List[str], max_scenes: int = 3) -> List[SceneData]:
        scene_data_list = []
        
        for zone_id in zones:
            # Download logic here
            scene = SceneData(
                zone_id=zone_id,
                provider="custom",
                scene_id=f"custom_{zone_id}_{timestamp}",
                file_paths={"data": data_file_path},
                available_bands=["custom_band"],
                metadata={"acquisition_date": datetime.now()}
            )
            scene_data_list.append(scene)
        
        return scene_data_list
```

---

## 3. Pipeline Orchestration (`src/pipeline/modular_pipeline.py`)

### Main Pipeline Class

#### `ModularPipeline`

Four-stage archaeological analysis pipeline with provider isolation and intermediate persistence.

```python
class ModularPipeline:
    def __init__(self, provider_instance: BaseProvider, run_id: str):
        """Initialize pipeline with provider and unique run identifier.
        
        Args:
            provider_instance: Data provider (GEDI, Sentinel-2, etc.)
            run_id: Unique identifier for this pipeline run
        """
```

**Attributes:**
- `provider_instance`: Data provider implementation
- `run_id`: Unique run identifier for result isolation
- `provider_name`: Provider name for path management
- `analysis_step`: Scene analysis coordinator
- `scoring_step`: Convergent anomaly scorer
- `report_step`: Report and visualization generator

### Pipeline Stages

#### Stage 1: Data Acquisition

```python
def acquire_data(self, zones: List[str], max_scenes: int = 3) -> List[SceneData]:
    """Download and prepare data for analysis.
    
    Args:
        zones: List of zone IDs to process
        max_scenes: Maximum scenes per zone
        
    Returns:
        List of SceneData objects
        
    Side Effects:
        - Downloads raw data via provider
        - Saves SceneData objects to JSON files
        - Creates provider-specific directory structure
    """
```

**Data Flow:**
```
Provider.download_data() → List[SceneData] → JSON persistence → Return SceneData list
```

#### Stage 2: Scene Analysis

```python
def analyze_scenes(self, scene_data_input: Union[List[SceneData], Path]) -> Dict[str, List[dict]]:
    """Apply archaeological detectors to scenes.
    
    Args:
        scene_data_input: Either SceneData list or path to saved scene data
        
    Returns:
        Dict mapping zone_id to list of analysis results
        
    Side Effects:
        - Applies provider-specific detectors
        - Exports individual detection GeoJSON files
        - Saves analysis results to JSON
    """
```

**Analysis Workflow:**
```
SceneData → Provider-Specific Detector → Detection Results → GeoJSON Export → JSON persistence
```

#### Stage 3: Convergent Scoring

```python
def score_zones(self, analysis_results_input: Union[Dict[str, List[dict]], Path]) -> Dict[str, Dict[str, Any]]:
    """Calculate convergent anomaly scores for zones.
    
    Args:
        analysis_results_input: Analysis results or path to saved results
        
    Returns:
        Dict mapping zone_id to scoring results
        
    Side Effects:
        - Aggregates features across providers
        - Calculates multi-evidence convergence scores
        - Saves scoring results to JSON
    """
```

**Scoring Process:**
```
Analysis Results → Feature Aggregation → Evidence Weighting → Convergence Calculation → Scoring Results
```

#### Stage 4: Output Generation

```python
def generate_outputs(self, analysis_results: Dict[str, List[dict]], 
                    scoring_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Generate reports and visualizations.
    
    Args:
        analysis_results: Scene analysis results
        scoring_results: Zone scoring results
        
    Returns:
        Dict containing paths to generated files
        
    Side Effects:
        - Generates comprehensive JSON report
        - Creates markdown summary
        - Produces interactive visualizations
    """
```

### Full Pipeline Execution

```python
def run(self, zones: Optional[List[str]] = None, max_scenes: int = 3) -> Dict[str, object]:
    """Execute complete four-stage pipeline.
    
    Args:
        zones: Zone IDs to process (None = default zones)
        max_scenes: Maximum scenes per zone
        
    Returns:
        Dict containing all pipeline results and output paths
    """
```

**Complete Workflow:**
```python
# Example usage
from src.providers.gedi_provider import GEDIProvider

provider = GEDIProvider()
pipeline = ModularPipeline(provider, run_id="archaeological_survey_001")

# Execute full pipeline
results = pipeline.run(zones=["upper_napo_micro"], max_scenes=5)

# Access results
scene_data = results["scene_data"]
analysis_results = results["analysis_results"]
scoring_results = results["scoring_results"]
report = results["report"]
```

### Stage Independence

Each stage can accept file paths or in-memory data, enabling flexible execution:

```python
# Sequential execution with file persistence
scene_data = pipeline.acquire_data(["upper_napo_micro"])
analysis_results = pipeline.analyze_scenes(scene_data)
scoring_results = pipeline.score_zones(analysis_results)
outputs = pipeline.generate_outputs(analysis_results, scoring_results)

# Resume from saved results
analysis_results = pipeline.analyze_scenes(Path("results/run_001/scene_data.json"))
scoring_results = pipeline.score_zones(Path("results/run_001/analysis_results.json"))
```

---

## 4. Convergent Scoring (`src/core/scoring.py`)

### Evidence Framework

#### `EvidenceItem` (dataclass)

Individual piece of archaeological evidence with weight and confidence.

```python
@dataclass
class EvidenceItem:
    type: str           # Evidence type ("clearing", "terra_preta", etc.)
    weight: int         # Point value (1-6 depending on type)
    confidence: float   # Provider confidence (0.0-1.0)
    description: str    # Human-readable description
    coordinates: Tuple[float, float]  # [latitude, longitude]
```

### Convergent Anomaly Scorer

#### `ConvergentAnomalyScorer`

Multi-evidence archaeological anomaly detection system.

```python
class ConvergentAnomalyScorer:
    def __init__(self):
        """Initialize with evidence weights from configuration."""
        self.scoring_config = get_scoring_config()
        self.weights = {
            'historical': self.scoring_config.historical_weight,
            'geometric': self.scoring_config.geometric_pattern_weight,
            'terra_preta': self.scoring_config.terra_preta_weight,
            'environmental': self.scoring_config.environmental_weight,
            'priority': self.scoring_config.priority_zone_bonus
        }
```

#### Core Scoring Method

```python
def calculate_zone_score(self, zone_id: str, features: Dict[str, List]) -> Dict[str, Any]:
    """Calculate convergent anomaly score for archaeological zone.
    
    Args:
        zone_id: Target zone identifier
        features: Dict of detected features by provider
        
    Returns:
        Dict containing:
        - total_score: int (0-16 point scale)
        - classification: str (HIGH_CONFIDENCE, PROBABLE, POSSIBLE, NATURAL)
        - confidence: float (0.0-1.0)
        - evidence_summary: List[str] (evidence descriptions)
        - feature_details: List[Dict] (detailed feature information)
        - convergence_analysis: Dict (spatial convergence metrics)
    """
```

### Evidence Types and Scoring

**Historical Reference Evidence** (+2 points):
```python
# Based on TARGET_ZONES historical documentation
if zone.historical_evidence and zone.historical_evidence != "Unknown":
    evidence_list.append(EvidenceItem(
        type="historical_reference",
        weight=2,
        confidence=1.0,
        description=f"Historical documentation: {zone.historical_evidence}",
        coordinates=zone.center
    ))
```

**Geometric Pattern Evidence** (+3 points each, max 6):
```python
# Circles, lines, rectangles from detectors
for pattern_type in ["circles", "lines", "rectangles"]:
    if pattern_type in geometric_features:
        for feature in geometric_features[pattern_type]:
            evidence_list.append(EvidenceItem(
                type=f"geometric_{pattern_type}",
                weight=3,
                confidence=feature.get("confidence", 0.7),
                description=f"Geometric {pattern_type} pattern detected",
                coordinates=feature["coordinates"]
            ))
```

**Terra Preta Signatures** (+2 points):
```python
# Anthropogenic soil indicators from Sentinel-2
for tp_feature in terra_preta_features:
    evidence_list.append(EvidenceItem(
        type="terra_preta",
        weight=2,
        confidence=tp_feature.get("confidence", 0.8),
        description=f"Anthropogenic soil signature",
        coordinates=tp_feature["coordinates"]
    ))
```

**Environmental Suitability** (+1 point):
```python
# Settlement-friendly environmental conditions
if self._assess_environmental_suitability(zone):
    evidence_list.append(EvidenceItem(
        type="environmental_suitability",
        weight=1,
        confidence=0.6,
        description="Suitable environment for archaeological sites",
        coordinates=zone.center
    ))
```

**Convergence Bonus** (+0.5-2 points):
```python
# Multiple evidence types clustering spatially
convergence_bonus = self._calculate_convergence_bonus(evidence_list)
if convergence_bonus > 0:
    evidence_list.append(EvidenceItem(
        type="spatial_convergence",
        weight=convergence_bonus,
        confidence=0.9,
        description=f"Multiple evidence types converge spatially",
        coordinates=convergence_center
    ))
```

### Classification System

```python
def _classify_anomaly_score(self, total_score: int) -> Tuple[str, float]:
    """Classify archaeological significance based on total score.
    
    Score Ranges:
    - 10+ points: HIGH CONFIDENCE - Immediate ground verification
    - 7-9 points: PROBABLE FEATURE - High-resolution follow-up  
    - 4-6 points: POSSIBLE ANOMALY - Additional remote sensing
    - 0-3 points: NATURAL VARIATION - Continue monitoring
    """
    
    if total_score >= self.scoring_config.high_confidence_threshold:
        return "HIGH_CONFIDENCE", 0.9
    elif total_score >= self.scoring_config.probable_threshold:
        return "PROBABLE", 0.7
    elif total_score >= self.scoring_config.possible_threshold:
        return "POSSIBLE", 0.5
    else:
        return "NATURAL", 0.2
```

### Batch Processing

```python
def batch_score_zones(self, analysis_results: Dict[str, List[dict]]) -> Dict[str, Dict[str, Any]]:
    """Score multiple zones efficiently.
    
    Args:
        analysis_results: Dict mapping zone_id to analysis results
        
    Returns:
        Dict mapping zone_id to scoring results
    """
```

### Scoring Summary

```python
def generate_scoring_summary(self, scoring_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Generate statistical summary of scoring results.
    
    Returns:
        Dict containing:
        - total_zones_scored: int
        - high_confidence_count: int
        - probable_count: int
        - average_score: float
        - top_zones: List[Dict] (highest scoring zones)
        - evidence_distribution: Dict (evidence type frequencies)
    """
```

---

## 5. Export Management (`src/pipeline/export_manager.py`)

### Unified Export System

#### `UnifiedExportManager`

Centralized GeoJSON export system with quality filtering and standards compliance.

```python
class UnifiedExportManager:
    def __init__(self, run_id: str, results_dir: Path):
        """Initialize export manager with run-specific isolation.
        
        Args:
            run_id: Unique identifier for pipeline run
            results_dir: Base directory for results storage
        """
        self.run_id = run_id
        self.results_dir = results_dir
        self.export_dir = results_dir / f"run_{run_id}" / "exports"
        self.unified_dir = self.export_dir / "unified"
```

### Provider-Specific Exports

#### GEDI Feature Export

```python
def export_gedi_features(self, detections: Dict[str, Any], zone_name: str) -> Path:
    """Export GEDI LiDAR detections to GeoJSON.
    
    Args:
        detections: GEDI detection results
        zone_name: Target zone name
        
    Returns:
        Path to exported GeoJSON file
        
    Quality Filtering:
        - Minimum 40% confidence threshold
        - Validates coordinate format
        - Excludes oversized features (>50 km²)
    """
```

**GEDI GeoJSON Schema:**
```json
{
    "type": "FeatureCollection",
    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
    "features": [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
        "properties": {
            "type": "archaeological_clearing",
            "confidence": 0.85,
            "area_m2": 1250.0,
            "count": 5,
            "provider": "gedi",
            "detection_method": "canopy_gap_analysis"
        }
    }]
}
```

#### Sentinel-2 Feature Export

```python
def export_sentinel2_features(self, detections: Dict[str, Any], zone_name: str) -> Path:
    """Export Sentinel-2 detections to GeoJSON.
    
    Args:
        detections: Sentinel-2 detection results
        zone_name: Target zone name
        
    Returns:
        Path to exported GeoJSON file
        
    Quality Filtering:
        - Minimum 50% confidence threshold
        - NDVI range validation (0.2-0.8)
        - Size constraints (100m² - 50km²)
    """
```

**Sentinel-2 GeoJSON Schema:**
```json
{
    "type": "Feature",
    "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
    "properties": {
        "type": "terra_preta_enhanced",
        "confidence": 0.935,
        "area_m2": 74000.0,
        "ndvi": 0.65,
        "terra_preta_index": 0.12,
        "provider": "sentinel2",
        "detection_method": "enhanced_red_edge"
    }
}
```

### Multi-Sensor Integration

#### Combined Feature Export

```python
def export_combined_features(self, all_detections: Dict[str, Any], zone_name: str) -> Path:
    """Export multi-sensor combined features.
    
    Args:
        all_detections: Combined detection results from all providers
        zone_name: Target zone name
        
    Returns:
        Path to combined GeoJSON file
        
    Features:
        - Merges GEDI and Sentinel-2 detections
        - Applies spatial proximity clustering
        - Enhanced confidence for convergent features
        - Unified archaeological grading system
    """
```

#### Top Candidates Export

```python
def export_top_candidates(self, top_detections: List[Dict], zone_name: str, count: int = 5) -> Path:
    """Export top archaeological candidates for analysis validation.
    
    Args:
        top_detections: Highest confidence detections
        zone_name: Target zone name
        count: Number of top candidates to export
        
    Returns:
        Path to priority candidates GeoJSON
        
    Selection Criteria:
        - Minimum 70% confidence
        - Multi-sensor convergence preferred
        - Archaeological size range (1-50 hectares)
        - Accessibility considerations
    """
```

### Export Manifest and Metadata

#### Manifest Generation

```python
def create_export_manifest(self) -> Path:
    """Create comprehensive manifest of all export files.
    
    Returns:
        Path to manifest JSON file
        
    Manifest Contents:
        - Export file inventory
        - Feature count summaries
        - Quality statistics
        - Coordinate system information
        - Export timestamps
    """
```

**Manifest Schema:**
```json
{
    "run_id": "archaeological_survey_001",
    "export_timestamp": "2025-12-06T15:30:00Z",
    "coordinate_system": "WGS84",
    "total_features": 127,
    "by_provider": {
        "gedi": {"files": 3, "features": 89},
        "sentinel2": {"files": 3, "features": 38}
    },
    "export_files": [
        {
            "filename": "upper_napo_micro_gedi_features.geojson",
            "type": "gedi_features",
            "zone": "upper_napo_micro",
            "feature_count": 39,
            "file_size_mb": 0.8
        }
    ]
}
```

### Quality Control and Standards

#### Coordinate Validation

```python
def _validate_coordinates(self, coordinates: List[float]) -> bool:
    """Validate coordinate format and bounds.
    
    Checks:
        - Format: [longitude, latitude]
        - Longitude range: -180 to 180
        - Latitude range: -90 to 90
        - Amazon region bounds validation
    """
```

#### Confidence Filtering

```python
def _apply_confidence_filter(self, features: List[Dict], min_confidence: float) -> List[Dict]:
    """Filter features by confidence threshold.
    
    Provider-Specific Thresholds:
        - GEDI: 40% minimum (space-based LiDAR uncertainty)
        - Sentinel-2: 50% minimum (spectral analysis reliability)
        - Combined: 50% minimum (multi-sensor validation)
    """
```

### Legacy Compatibility

```python
def cleanup_old_provider_exports(self) -> None:
    """Remove deprecated provider-specific export directories.
    
    Migrates from:
        exports/gedi/, exports/sentinel2/
    To:
        exports/unified/run_{run_id}/
    """
```

---

## Integration Patterns

### Provider Registration

```python
# Register new provider
from src.core.config import SATELLITE_PROVIDERS
from src.providers.custom_provider import CustomProvider

SATELLITE_PROVIDERS["custom"] = CustomProvider
```

### Pipeline Execution

```python
# Complete workflow
from src.core.config import TARGET_ZONES
from src.providers.gedi_provider import GEDIProvider
from src.pipeline.modular_pipeline import ModularPipeline

# Initialize components
provider = GEDIProvider()
pipeline = ModularPipeline(provider, run_id="survey_2025_001")

# Execute pipeline
results = pipeline.run(
    zones=list(TARGET_ZONES.keys())[:3],  # First 3 zones
    max_scenes=5
)

# Access outputs
for zone_id, score_data in results["scoring_results"].items():
    print(f"{zone_id}: {score_data['total_score']} points ({score_data['classification']})")
```

### Custom Detector Integration

```python
# Integrate custom detector
from src.pipeline.analysis import AnalysisStep

class CustomDetector:
    def __init__(self, zone, run_id=None):
        self.zone = zone
        self.run_id = run_id
    
    def analyze_scene(self, scene_path):
        # Custom detection logic
        return {
            "success": True,
            "provider": "custom",
            "total_features": feature_count,
            "custom_analysis": detection_results
        }

# Register with analysis step
analysis_step = AnalysisStep(run_id="custom_001")
# Detector selection happens automatically based on SceneData.provider
```

This core API provides the foundation for archaeological analysis workflows, enabling flexible data processing, multi-sensor analysis, and quality-controlled archaeological candidate identification.
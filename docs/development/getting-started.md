# Getting Started Guide

**Amazon Archaeological Discovery Pipeline - Complete Developer Onboarding**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
3. [Installation Guide](#installation-guide)
4. [First Pipeline Run](#first-pipeline-run)
5. [Development Workflow](#development-workflow)
6. [Code Organization](#code-organization)
7. [Testing Framework](#testing-framework)
8. [Debugging Guide](#debugging-guide)
9. [Common Development Tasks](#common-development-tasks)
10. [Best Practices](#best-practices)

---

## Quick Start

Get the Amazon Archaeological Discovery Pipeline running in under 10 minutes:

### Prerequisites
- Python 3.8+
- 8GB+ RAM recommended
- NASA Earthdata account (free registration)
- OpenAI API key (optional, for AI features)

### 5-Minute Setup

```bash
# 1. Clone and setup environment
git clone <repository-url>
cd amazon-discovery
conda create -n amazon python=3.9
conda activate amazon

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials (create .env file)
cp .env.example .env
# Edit .env with your NASA Earthdata credentials

# 4. Run first test
python main.py --pipeline --zone upper_napo_micro_small

# 5. View results
ls results/run_*/
```

### Verify Installation

```python
# Quick verification script
from src.core.config import TARGET_ZONES, APIConfig
from src.providers.gedi_provider import GEDIProvider

print(f"✅ Loaded {len(TARGET_ZONES)} target zones")
print(f"✅ API config: {bool(APIConfig().EARTHDATA_USERNAME)}")
print("✅ Installation successful!")
```

---

## Environment Setup

### System Requirements

**Minimum Configuration:**
- **Operating System**: Linux (Ubuntu 18.04+), macOS (10.14+), or Windows 10/11
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM (16GB+ recommended)
- **Storage**: 10GB free space for dependencies and data
- **CPU**: 4+ cores (Intel/AMD x64)

**Recommended Configuration:**
- **Memory**: 32GB+ RAM for large-scale processing
- **GPU**: CUDA-compatible GPU for acceleration
- **Storage**: SSD with 50GB+ free space
- **CPU**: 16+ cores for parallel processing

### Required Accounts and Credentials

1. **NASA Earthdata Account** (Required for GEDI data):
   - Register at: https://urs.earthdata.nasa.gov/
   - Required for GEDI L2A/L2B satellite data access

2. **OpenAI API Key** (Optional, for AI features):
   - Get API key from: https://platform.openai.com/api-keys
   - Enables AI-enhanced archaeological interpretation

3. **Copernicus Account** (Optional, for Sentinel-2 backup):
   - Register at: https://scihub.copernicus.eu/dhus/#/self-registration
   - Backup access method for Sentinel-2 data

---

## Development Environment Setup

### 1. Repository Setup

```bash
# Clone the repository
git clone https://github.com/your-username/amazon-discovery.git
cd amazon-discovery

# Check current status
git status
git log --oneline -10
```

### 2. Python Environment Setup

#### Option A: Conda Environment (Recommended)

```bash
# Activate conda
source /home/tsuser/miniconda3/etc/profile.d/conda.sh

# Create dedicated environment
conda create -n amazon python=3.8
conda activate amazon

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import numpy, geopandas, rasterio; print('Core dependencies installed')"
```

#### Option B: Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```bash
# Copy template
cp .env.template .env

# Edit with your credentials
nano .env
```

**Required Environment Variables:**

```bash
# NASA Earthdata (Required for GEDI)
EARTHDATA_USERNAME=your_nasa_username
EARTHDATA_PASSWORD=your_nasa_password

# OpenAI API (Optional for AI features)
OPENAI_API_KEY=sk-your-openai-api-key

# Optional Copernicus (Backup Sentinel-2 access)
COPERNICUS_USER=your_copernicus_username
COPERNICUS_PASSWORD=your_copernicus_password

# Development Settings
ENABLE_GPU=true
MAX_WORKERS=4
DEBUG_MODE=true
```

### 4. Verify Installation

```bash
# Test core functionality
python -c "from src.core.config import TARGET_ZONES; print(f'Loaded {len(TARGET_ZONES)} target zones')"

# Check GPU availability (optional)
python -c "from src.core.enable_optimizations import check_optimization_requirements; print(check_optimization_requirements())"

# Verify provider functionality
python -c "from src.providers.gedi_provider import GEDIProvider; print('GEDI provider available')"
```

---

## Codebase Overview

### Project Structure

```
amazon-discovery/
├── src/                              # Main source code
│   ├── core/                         # Core system components
│   │   ├── config.py                    # Central configuration
│   │   ├── data_objects.py              # Data structures
│   │   ├── scoring.py                   # Convergent scoring
│   │   ├── validation.py                # Quality assurance
│   │   ├── detectors/                   # Detection algorithms
│   │   │   ├── gedi_detector.py            # GEDI LiDAR analysis
│   │   │   └── sentinel2_detector.py       # Multispectral analysis
│   │   └── enable_optimizations.py     # GPU acceleration
│   ├── providers/                    # Data acquisition
│   │   ├── gedi_provider.py             # NASA GEDI data access
│   │   └── sentinel2_provider.py        # Sentinel-2 data access
│   ├── pipeline/                     # Analysis workflow
│   │   ├── modular_pipeline.py          # Main orchestrator
│   │   ├── analysis.py                  # Multi-detector coordination
│   │   ├── export_manager.py            # Unified exports
│   │   └── report.py                    # Report generation
│   └── checkpoints/                  # OpenAI integration
│       ├── base_checkpoint.py           # Base checkpoint class
│       └── checkpoint[1-5].py           # Specific checkpoints
├── main.py                           # Primary entry point
├── openai_checkpoints.py             # OpenAI checkpoint runner
├── requirements.txt                  # Python dependencies
├── docs/                             # Documentation
├── results/                          # Processing results
├── exports/                          # GeoJSON exports
└── cache/                            # Data cache
```

### Key Development Patterns

#### 1. Provider Pattern
All data sources implement `BaseProvider` interface:

```python
from src.core.data_objects import BaseProvider

class CustomProvider(BaseProvider):
    def download_data(self, zones: List[str], max_scenes: int = 3) -> List[SceneData]:
        # Implement data acquisition logic
        return scene_data_list
```

#### 2. Detector Pattern
Archaeological detectors follow consistent interfaces:

```python
class CustomDetector:
    def __init__(self, zone, run_id=None):
        self.zone = zone
        self.run_id = run_id
    
    def analyze_scene(self, scene_path):
        # Implement detection algorithms
        return detection_results
```

#### 3. Configuration Pattern
All components use centralized configuration:

```python
from src.core.config import get_detection_config, TARGET_ZONES

config = get_detection_config()
zone = TARGET_ZONES["upper_napo_micro"]
```

---

## Development Workflow

### 1. Understanding the Pipeline

#### Basic Pipeline Execution

```bash
# Test with micro zone (quick execution)
python main.py --pipeline --zone upper_napo_micro_small

# Check available zones
python main.py --list-zones

# Run specific checkpoint
python main.py --checkpoint 1
```

#### Pipeline Stages

```bash
# Stage 1: Data acquisition
python main.py --pipeline --stage acquire_data --zone upper_napo_micro

# Stage 2: Scene analysis  
python main.py --pipeline --stage analyze_scenes --zone upper_napo_micro

# Stage 3: Convergent scoring
python main.py --pipeline --stage score_zones --zone upper_napo_micro

# Stage 4: Output generation
python main.py --pipeline --stage generate_outputs --zone upper_napo_micro
```

### 2. Development Testing

#### Quick Testing Workflow

```python
# test_development.py
from src.core.config import TARGET_ZONES
from src.providers.gedi_provider import GEDIProvider
from src.pipeline.modular_pipeline import ModularPipeline

# Quick test setup
zone_id = "upper_napo_micro_small"
provider = GEDIProvider()
pipeline = ModularPipeline(provider, run_id="dev_test_001")

# Test data acquisition
scene_data = pipeline.acquire_data([zone_id], max_scenes=1)
print(f"Acquired {len(scene_data)} scenes")

# Test analysis
if scene_data:
    analysis_results = pipeline.analyze_scenes(scene_data)
    print(f"Analysis results: {analysis_results}")
```

#### Component Testing

```python
# Test configuration loading
from src.core.config import get_gedi_config, get_detection_config

gedi_config = get_gedi_config()
detection_config = get_detection_config()

print(f"GEDI gap threshold: {gedi_config.gap_threshold}")
print(f"Detection confidence: {detection_config.min_confidence}")
```

### 3. Debugging Techniques

#### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in environment
export DEBUG_MODE=true
```

#### Common Debug Points

```python
# Check data availability
scene = scene_data[0]
print(f"Scene: {scene.scene_id}")
print(f"Provider: {scene.provider}")
print(f"Available bands: {scene.available_bands}")
print(f"File paths: {scene.file_paths}")

# Check detection results
for zone_id, analyses in analysis_results.items():
    for analysis in analyses:
        print(f"Zone {zone_id}: {analysis.get('total_features', 0)} features")
        if not analysis.get('success'):
            print(f"Error: {analysis.get('error', 'Unknown error')}")
```

#### GPU Debugging

```python
# Check GPU availability
from src.core.enable_optimizations import check_optimization_requirements
gpu_info = check_optimization_requirements()
print(gpu_info)

# Test CuPy installation
try:
    import cupy as cp
    print(f"CuPy available: {cp.cuda.is_available()}")
    print(f"GPU memory: {cp.cuda.MemoryPool().total_bytes() / 1e9:.1f}GB")
except ImportError:
    print("CuPy not installed - using CPU processing")
```

---

## Contributing Guidelines

### 1. Code Standards

#### Python Style
- Follow PEP 8 style guidelines
- Use type hints for function parameters and returns
- Maximum line length: 120 characters
- Use descriptive variable and function names

```python
# Good example
def calculate_archaeological_confidence(feature_type: str, 
                                      properties: Dict[str, Any]) -> float:
    """Calculate archaeological confidence based on feature properties."""
    base_confidence = FEATURE_TYPES[feature_type]['confidence_base']
    return min(1.0, base_confidence + calculate_bonuses(properties))

# Avoid
def calc_conf(ft, props):
    return min(1.0, TYPES[ft]['conf'] + bonus(props))
```

#### Documentation Standards
- All functions require docstrings with parameters and return types
- Complex algorithms need explanatory comments
- Use examples in docstrings for public APIs

```python
def detect_archaeological_clearings(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Detect potential archaeological clearings using canopy gap analysis.
    
    Args:
        data: GEDI data dictionary containing 'longitude', 'latitude', 'canopy_height'
        
    Returns:
        Dict containing:
        - gap_clusters: List of detected clearing clusters
        - total_gaps: Total number of gap points identified
        - metadata: Processing metadata and parameters
        
    Example:
        >>> detector = GEDIArchaeologicalDetector(zone)
        >>> data = {'longitude': [...], 'latitude': [...], 'canopy_height': [...]}
        >>> results = detector.detect_archaeological_clearings(data)
        >>> print(f"Found {len(results['gap_clusters'])} clearings")
    """
```

### 2. Testing Requirements

#### Unit Testing
Create tests for new functions in `tests/` directory:

```python
# tests/test_detectors.py
import pytest
import numpy as np
from src.core.detectors.gedi_detector import GEDIArchaeologicalDetector
from src.core.config import TARGET_ZONES

def test_gedi_clearing_detection():
    """Test GEDI clearing detection with known data."""
    zone = TARGET_ZONES["upper_napo_micro_small"]
    detector = GEDIArchaeologicalDetector(zone)
    
    # Create test data with clear gap signature
    test_data = {
        'longitude': np.array([-72.5, -72.5, -72.5]),
        'latitude': np.array([-0.5, -0.5, -0.5]),
        'canopy_height': np.array([5.0, 8.0, 12.0])  # Below 15m threshold
    }
    
    results = detector.detect_archaeological_clearings(test_data)
    
    assert 'gap_clusters' in results
    assert len(results['gap_clusters']) >= 1
    assert results['total_gaps'] == 3
```

#### Integration Testing
Test complete workflows:

```python
def test_pipeline_integration():
    """Test complete pipeline execution."""
    from src.providers.gedi_provider import GEDIProvider
    from src.pipeline.modular_pipeline import ModularPipeline
    
    provider = GEDIProvider()
    pipeline = ModularPipeline(provider, run_id="integration_test")
    
    # Test with small zone
    results = pipeline.run(zones=["upper_napo_micro_small"], max_scenes=1)
    
    assert 'scene_data' in results
    assert 'analysis_results' in results
    assert 'scoring_results' in results
```

### 3. Git Workflow

#### Branch Strategy
```bash
# Create feature branch
git checkout -b feature/new-detector-algorithm
git checkout -b fix/coordinate-transformation-bug
git checkout -b docs/api-reference-update

# Work on feature
git add src/core/detectors/new_detector.py
git commit -m "Add CNN-based archaeological feature detector

- Implement YOLOv3-based detection for geometric patterns
- Add confidence scoring and post-processing filters
- Include tests and documentation
- Performance: 15% improvement over traditional methods"

# Push and create PR
git push origin feature/new-detector-algorithm
```

#### Commit Message Standards
```
<type>: <description>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding tests
- `refactor`: Code refactoring
- `perf`: Performance improvements

**Example:**
```
feat: Add multi-temporal Sentinel-2 analysis capability

- Implement seasonal crop mark detection using time series
- Add temporal NDVI analysis for archaeological signatures  
- Include 6-month temporal window optimization
- Add tests for temporal detection algorithms

Closes #45
Improves detection accuracy by 23% for subsurface features
```

---

## Common Development Tasks

### 1. Adding a New Target Zone

```python
# 1. Define zone in src/core/config.py
new_zone = TargetZone(
    id="new_site_001",
    name="New Archaeological Site",
    center=(-3.5, -65.2),
    bbox=(-3.6, -65.3, -3.4, -65.1),
    priority=2,
    expected_features="Possible earthworks",
    historical_evidence="Local indigenous reports"
)

TARGET_ZONES["new_site_001"] = new_zone

# 2. Test the zone
python main.py --pipeline --zone new_site_001 --max-scenes 1
```

### 2. Customizing Detection Parameters

```python
# 1. Modify configuration in src/core/config.py
@dataclass
class GEDIConfig:
    gap_threshold: float = 12.0  # More sensitive (was 15.0)
    min_cluster_size: int = 4    # Larger clusters required (was 3)

# 2. Test parameter impact
from src.core.config import get_gedi_config
config = get_gedi_config()
print(f"New gap threshold: {config.gap_threshold}")
```

### 3. Adding a New Detection Algorithm

```python
# 1. Create detector in src/core/detectors/
class CustomDetector:
    def __init__(self, zone, run_id=None):
        self.zone = zone
        self.run_id = run_id
    
    def analyze_scene(self, scene_path):
        # Implement custom detection logic
        return {
            'success': True,
            'provider': 'custom',
            'total_features': feature_count,
            'custom_analysis': detection_results
        }

# 2. Integration happens automatically via AnalysisStep
# based on SceneData.provider attribute
```

### 4. Debugging Data Provider Issues

```python
# 1. Test provider directly
from src.providers.gedi_provider import GEDIProvider

provider = GEDIProvider()
try:
    scene_data = provider.download_data(["upper_napo_micro_small"], max_scenes=1)
    print(f"Success: {len(scene_data)} scenes downloaded")
    for scene in scene_data:
        print(f"Scene: {scene.scene_id}, Files: {len(scene.file_paths)}")
except Exception as e:
    print(f"Provider error: {e}")

# 2. Check credentials
import os
print(f"Earthdata user: {os.getenv('EARTHDATA_USERNAME', 'Not set')}")
print(f"OpenAI key: {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
```

### 5. Performance Profiling

```python
# 1. Profile pipeline execution
import cProfile
import pstats

def profile_pipeline():
    from src.pipeline.modular_pipeline import ModularPipeline
    from src.providers.gedi_provider import GEDIProvider
    
    provider = GEDIProvider()
    pipeline = ModularPipeline(provider, run_id="profile_test")
    
    # Profile execution
    results = pipeline.run(zones=["upper_napo_micro_small"], max_scenes=1)
    return results

# Run profiler
cProfile.run('profile_pipeline()', 'profile_results.prof')

# Analyze results
stats = pstats.Stats('profile_results.prof')
stats.sort_stats('cumulative').print_stats(20)
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'src'
# Solution: Ensure you're in the project root directory
cd /path/to/amazon-discovery
python main.py --help

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/amazon-discovery"
```

#### 2. GPU Acceleration Issues
```python
# Check GPU availability
python -c "
try:
    import cupy as cp
    print(f'CuPy available: {cp.cuda.is_available()}')
    print(f'Device count: {cp.cuda.runtime.getDeviceCount()}')
except ImportError:
    print('CuPy not installed')
except Exception as e:
    print(f'GPU error: {e}')
"

# Install CuPy if needed
pip install cupy-cuda11x  # For CUDA 11.x
pip install cupy-cuda12x  # For CUDA 12.x
```

#### 3. Memory Issues
```python
# Monitor memory usage
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

# Use smaller chunks for large datasets
config = ProcessingConfig(
    max_memory_gb=4.0,      # Reduce memory limit
    chunk_size=5000,        # Smaller chunks
    enable_multiprocessing=False  # Single-threaded
)
```

#### 4. Data Access Issues
```bash
# Test NASA Earthdata credentials
curl -u "$EARTHDATA_USERNAME:$EARTHDATA_PASSWORD" \
  "https://urs.earthdata.nasa.gov/api/users/user"

# Test network connectivity
ping gedi.umd.edu
ping scihub.copernicus.eu
```

### Getting Help

#### 1. Documentation Resources
- **API Reference**: `docs/repo-documentation/api/core-api-reference.md`
- **Configuration Guide**: `docs/repo-documentation/configuration/configuration-guide.md`  
- **System Architecture**: `docs/repo-documentation/architecture/system-overview.md`
- **Main README**: Comprehensive project overview

#### 2. Debugging Resources
```python
# Enable verbose logging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Check configuration
from src.core.config import *
print("Configuration loaded successfully")

# Validate setup
from src.core.validation import validate_configuration
validate_configuration()
```

#### 3. Community and Support
- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Documentation**: Update documentation for improvements you discover

### Development Best Practices

1. **Start Small**: Begin with `upper_napo_micro_small` zone for quick testing
2. **Use Caching**: Let the system cache intermediate results to speed up development
3. **Monitor Resources**: Keep an eye on memory and disk usage during development
4. **Test Incrementally**: Test each component before integrating with full pipeline
5. **Document Changes**: Update documentation for any new features or modifications

This getting started guide provides the foundation for productive development on the Amazon Archaeological Discovery Pipeline. The modular architecture and comprehensive configuration system make it straightforward to extend the system with new capabilities while maintaining scientific rigor and quality standards.
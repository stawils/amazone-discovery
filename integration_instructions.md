# Complete Sentinel-2 AWS Integration Instructions

## Phase 1: Environment Setup and Dependencies

### Step 1: Install Required Packages
```bash
# Add to your requirements.txt or install directly
pip install pystac-client>=0.7.0
pip install rioxarray>=0.15.0
pip install xarray>=2023.1.0
pip install geopandas>=0.14.0
pip install rasterio>=1.3.0
pip install shapely>=2.0.0
pip install pandas>=2.0.0
```

### Step 2: Create Sentinel-2 Provider File
**File:** `src/providers/sentinel2_provider.py`
**Action:** Create the file with the complete Sentinel-2 provider code from the artifact

### Step 3: Create Enhanced Detector
**File:** `src/core/sentinel2_detector.py`
**Action:** Create the file with the enhanced archaeological detector code

## Phase 2: Configuration Updates

### Step 4: Update Core Configuration
**File:** `src/core/config.py`
**Location:** Add after existing imports
**Add:**
```python
# Sentinel-2 specific configuration
class Sentinel2Config:
    """Sentinel-2 provider configuration"""
    STAC_API_URL = "https://earth-search.aws.element84.com/v1"
    L2A_COLLECTION = "sentinel-2-l2a"
    MAX_CLOUD_COVER = 20.0
    PREFERRED_MONTHS = [6, 7, 8, 9]  # Amazon dry season
    PRIORITY_BANDS = ['B02', 'B03', 'B04', 'B05', 'B07', 'B08', 'B11', 'B12']
```

### Step 5: Update Detection Configuration
**File:** `src/core/config.py`
**Location:** In DetectionConfig class
**Add:**
```python
# Sentinel-2 specific detection parameters
SENTINEL2_TERRA_PRETA_THRESHOLD = 0.12  # Higher threshold for red-edge enhanced
SENTINEL2_CROP_MARK_THRESHOLD = 0.05
SENTINEL2_MIN_FEATURE_SIZE = 20  # pixels at 10m resolution
RED_EDGE_SENSITIVITY = 0.1  # Minimum red-edge response
```

## Phase 3: Pipeline Integration

### Step 6: Update Analysis Step
**File:** `src/pipeline/analysis.py`
**Location:** Add import at top
**Add:**
```python
from src.core.sentinel2_detector import Sentinel2ArchaeologicalDetector
```

**Location:** In AnalysisStep.run method, after line that creates detector
**Replace:**
```python
detector = ArchaeologicalDetector(zone)
```
**With:**
```python
# Choose detector based on data provider
if scene.provider == 'sentinel-2':
    detector = Sentinel2ArchaeologicalDetector(zone)
else:
    detector = ArchaeologicalDetector(zone)
```

### Step 7: Update Main Entry Point
**File:** `main.py`
**Location:** Add new provider option in argument parser
**Find:**
```python
parser.add_argument('--provider', choices=['gee'], default='gee',
                   help='Data provider (default: gee)')
```
**Replace with:**
```python
parser.add_argument('--provider', choices=['gee', 'sentinel2'], default='gee',
                   help='Data provider (default: gee)')
```

**Location:** In `run_modular_pipeline` function
**Find:**
```python
if provider == 'gee':
    from src.providers.gee_provider import GEEProvider
    provider_instance = GEEProvider()
```
**Add after:**
```python
elif provider == 'sentinel2':
    from src.providers.sentinel2_provider import Sentinel2Provider
    provider_instance = Sentinel2Provider()
```

### Step 8: Update Modular Pipeline
**File:** `src/pipeline/modular_pipeline.py`
**Location:** In __init__ method
**Find:**
```python
if provider == 'gee':
    self.provider_instance = GEEProvider()
```
**Add after:**
```python
elif provider == 'sentinel2':
    from src.providers.sentinel2_provider import Sentinel2Provider
    self.provider_instance = Sentinel2Provider()
```

## Phase 4: OpenAI Checkpoint Integration

### Step 9: Update Checkpoint System for Sentinel-2
**File:** `openai_checkpoints.py`
**Location:** In checkpoint1_familiarize method
**Find:**
```python
if provider == 'gee':
    from src.providers.gee_provider import GEEProvider
    provider_instance = GEEProvider()
```
**Add after:**
```python
elif provider == 'sentinel2':
    from src.providers.sentinel2_provider import Sentinel2Provider
    provider_instance = Sentinel2Provider()
```

### Step 10: Enhance OpenAI Analysis for Sentinel-2
**File:** `openai_checkpoints.py`
**Location:** In OpenAIIntegration.analyze_with_openai method
**Find the prompt creation section and enhance it:**
**Add:**
```python
# Enhanced prompt for Sentinel-2 data
if 'sentinel-2' in data_context.lower():
    enhanced_context = f"""
    {data_context}
    
    SENTINEL-2 ADVANTAGES:
    - 13-band multispectral data with 5-day revisit cycle
    - Red-edge bands (705nm, 740nm, 783nm) critical for vegetation stress detection
    - 10m resolution core bands ideal for archaeological features
    - SWIR bands (1610nm, 2190nm) at 20m for soil composition analysis
    - Superior capability for detecting crop marks and burial-induced vegetation changes
    
    ARCHAEOLOGICAL ANALYSIS FOCUS:
    - Red-edge response indicates vegetation stress over buried features
    - SWIR signatures reveal soil composition changes (terra preta detection)
    - High spatial resolution enables detection of small archaeological features
    - Temporal analysis possible with 5-day repeat cycle
    """
    data_context = enhanced_context
```

## Phase 5: Enhanced Archaeological Analysis

### Step 11: Create Sentinel-2 Specific Indices
**File:** `src/core/processors.py`
**Location:** Add new method to BandProcessor class
**Add:**
```python
def calculate_sentinel2_archaeological_indices(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Calculate Sentinel-2 specific archaeological indices"""
    
    indices = {}
    eps = 1e-8
    
    # Red-edge archaeological indices (KEY for Sentinel-2)
    if 'B04' in bands and 'B05' in bands:  # Red and Red-edge 1
        indices['ndre1'] = (bands['B05'] - bands['B04']) / (bands['B05'] + bands['B04'] + eps)
    
    if 'B04' in bands and 'B07' in bands:  # Red and Red-edge 3
        indices['ndre3'] = (bands['B07'] - bands['B04']) / (bands['B07'] + bands['B04'] + eps)
    
    # Archaeological Vegetation Index (combines both red-edge bands)
    if 'B05' in bands and 'B07' in bands:
        indices['avi'] = (bands['B07'] - bands['B05']) / (bands['B07'] + bands['B05'] + eps)
    
    # Enhanced Terra Preta Index using red-edge
    if 'B07' in bands and 'B11' in bands:
        indices['terra_preta_enhanced'] = (bands['B07'] - bands['B11']) / (bands['B07'] + bands['B11'] + eps)
    
    # Crop Mark Index for archaeological crop marks
    if all(b in bands for b in ['B04', 'B05', 'B08']):
        red, re1, nir = bands['B04'], bands['B05'], bands['B08']
        indices['crop_mark'] = ((re1 - red) * (nir - re1)) / ((re1 + red) * (nir + re1) + eps)
    
    # S2 Archaeological Index
    if all(b in bands for b in ['B05', 'B07', 'B11']):
        re1, re3, swir1 = bands['B05'], bands['B07'], bands['B11']
        indices['s2_archaeological'] = ((re1 + re3) / 2 - swir1) / ((re1 + re3) / 2 + swir1 + eps)
    
    return indices
```

### Step 12: Update Visualization for Sentinel-2
**File:** `src/core/visualizers.py`
**Location:** In _create_tp_popup method
**Add Sentinel-2 specific information:**
```python
def _create_s2_popup(self, feature: Dict, zone_name: str) -> str:
    """Create enhanced popup for Sentinel-2 features"""
    
    detection_method = feature.get('detection_method', 'unknown')
    
    html = f"""
    <div style="width: 280px;">
        <h4 style="margin-bottom: 10px; color: #2E86AB;">🛰️ Sentinel-2 Detection</h4>
        <hr style="margin: 10px 0;">
        
        <p><strong>Zone:</strong> {zone_name}</p>
        <p><strong>Detection Method:</strong> {detection_method}</p>
        <p><strong>Confidence:</strong> {feature.get('confidence', 0):.2f}</p>
    """
    
    if 'red_edge_enhanced' in feature:
        html += f"<p><strong>Red-Edge Enhanced:</strong> {'Yes' if feature['red_edge_enhanced'] else 'No'}</p>"
    
    if 'crop_mark_index' in feature:
        html += f"<p><strong>Crop Mark Index:</strong> {feature['crop_mark_index']:.3f}</p>"
    
    if 'avi' in feature:
        html += f"<p><strong>Archaeological VI:</strong> {feature['avi']:.3f}</p>"
    
    html += """
        <hr style="margin: 10px 0;">
        <p style="font-size: 12px; color: #666;">
            Detected using Sentinel-2's 13-band multispectral data with red-edge sensitivity
            for enhanced archaeological feature detection.
        </p>
    </div>
    """
    return html
```

## Phase 6: Testing and Validation

### Step 13: Create Test Script
**File:** `test_sentinel2.py` (in project root)
**Create:**
```python
#!/usr/bin/env python3
"""Test script for Sentinel-2 integration"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from src.providers.sentinel2_provider import test_sentinel2_access
from src.core.config import TARGET_ZONES

def main():
    print("🛰️ Testing Sentinel-2 AWS Integration")
    print("=" * 50)
    
    # Test API access
    print("\n1. Testing STAC API access...")
    success = test_sentinel2_access('negro_madeira')
    
    if success:
        print("✅ Sentinel-2 STAC API access successful")
    else:
        print("❌ Sentinel-2 STAC API access failed")
        return False
    
    # Test provider integration
    print("\n2. Testing provider integration...")
    try:
        from src.providers.sentinel2_provider import Sentinel2Provider
        provider = Sentinel2Provider()
        print("✅ Sentinel-2 provider initialized")
    except Exception as e:
        print(f"❌ Provider initialization failed: {e}")
        return False
    
    # Test detector integration
    print("\n3. Testing enhanced detector...")
    try:
        from src.core.sentinel2_detector import Sentinel2ArchaeologicalDetector
        zone = TARGET_ZONES['negro_madeira']
        detector = Sentinel2ArchaeologicalDetector(zone)
        print("✅ Enhanced detector initialized")
    except Exception as e:
        print(f"❌ Detector initialization failed: {e}")
        return False
    
    print("\n✅ All Sentinel-2 integration tests passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

### Step 14: Test Individual Components
```bash
# Test Sentinel-2 provider
python test_sentinel2.py

# Test with main pipeline
python main.py --provider sentinel2 --zones negro_madeira --max-scenes 1

# Test with checkpoints
python main.py --checkpoint 1 --provider sentinel2 --zone negro_madeira
```

### Step 15: Test Full Integration
```bash
# Test complete pipeline with Sentinel-2
python main.py --pipeline --provider sentinel2 --zones negro_madeira trombetas --max-scenes 2

# Test all checkpoints with Sentinel-2
python main.py --all-checkpoints --provider sentinel2 --zone negro_madeira

# Test competition-ready workflow
python main.py --competition-ready --provider sentinel2
```

## Phase 7: Performance Optimization

### Step 16: Add Caching for STAC Searches
**File:** `src/providers/sentinel2_provider.py`
**Location:** Add caching functionality
**Add:**
```python
import pickle
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_stac_search(self, zone_bbox: Tuple, date_range: str, max_results: int):
    """Cache STAC search results to reduce API calls"""
    # Implementation here
```

### Step 17: Optimize Band Processing
**File:** `src/core/sentinel2_detector.py`
**Location:** Add memory-efficient processing
**Add:**
```python
def process_bands_chunked(self, bands: Dict[str, np.ndarray], chunk_size: int = 1024):
    """Process large Sentinel-2 scenes in chunks to manage memory"""
    # Implementation for chunked processing
```

## Phase 8: Documentation and Validation

### Step 18: Update Documentation
**File:** `README.md`
**Add section:**
```markdown
## Sentinel-2 Provider

The Sentinel-2 provider offers enhanced archaeological detection capabilities:

- **13-band multispectral analysis** with 5-day revisit cycle
- **Red-edge bands** (705nm, 740nm, 783nm) for vegetation stress detection
- **High spatial resolution** (10m) for small archaeological features
- **SWIR bands** for enhanced terra preta detection
- **Crop mark detection** using red-edge sensitivity

### Usage

```bash
# Use Sentinel-2 as primary provider
python main.py --provider sentinel2 --zones negro_madeira

# Compare with other providers
python main.py --provider sentinel2 --zones negro_madeira  # Sentinel-2
```

### Key Advantages for Archaeology

1. **Red-edge sensitivity**: Critical 705nm and 783nm bands detect vegetation stress over buried features
2. **Higher resolution**: 10m pixels vs 30m Landsat for better feature detection
3. **Faster revisit**: 5-day cycle vs 16-day Landsat for temporal analysis
4. **Enhanced indices**: Archaeological-specific indices using red-edge bands
5. **Cloud-free access**: AWS hosting eliminates download bottlenecks
```

### Step 19: Validate Archaeological Performance
**Create validation script:**
```python
# Compare detection performance between providers
python scripts/compare_providers.py --zone negro_madeira --providers sentinel2
```

### Step 20: Final Integration Test
```bash
# Complete end-to-end test
python main.py --competition-ready --provider sentinel2 --zones negro_madeira trombetas

# Validate all checkpoints pass
python -m src.checkpoints.validator
```

## Expected Results

After completing these steps, you should have:

1. ✅ **Sentinel-2 as primary provider** with 13-band multispectral access
2. ✅ **Red-edge enhanced detection** for superior crop mark identification  
3. ✅ **Enhanced terra preta detection** using 705nm and 783nm bands
4. ✅ **Higher resolution analysis** with 10m spatial resolution
5. ✅ **Faster data access** via AWS Cloud-Optimized GeoTIFFs
6. ✅ **OpenAI checkpoint integration** with Sentinel-2 specific analysis
7. ✅ **Competition-ready pipeline** with superior archaeological capabilities

## Key Benefits Achieved

- **3x higher spatial resolution** (10m vs 30m Landsat)
- **3x faster revisit cycle** (5 days vs 16 days)
- **Enhanced spectral resolution** with critical red-edge bands
- **Superior archaeological detection** using vegetation stress indicators
- **Cloud-optimized access** eliminating data transfer bottlenecks
- **Advanced crop mark detection** using red-edge sensitivity
- **Real-time processing** with AWS infrastructure

Your archaeological pipeline now has access to the most advanced multispectral satellite data optimized specifically for archaeological discovery in the Amazon basin!
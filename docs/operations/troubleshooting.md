# Troubleshooting Manual

## Overview

This manual provides comprehensive troubleshooting guidance for common issues encountered with the Amazon Archaeological Discovery Pipeline. Issues are organized by category with detailed diagnostic steps, solutions, and prevention strategies.

## Quick Diagnostic Checklist

### System Health Check

```bash
#!/bin/bash
# Quick health check script

echo "=== Amazon Archaeological Pipeline Health Check ==="

# 1. Environment Check
echo "1. Environment Status:"
if conda list | grep -q "gdal"; then
    echo "   ✓ GDAL installed"
else
    echo "   ✗ GDAL missing"
fi

if python -c "import earthaccess" 2>/dev/null; then
    echo "   ✓ NASA Earthdata access available"
else
    echo "   ✗ NASA Earthdata access unavailable"
fi

# 2. API Connectivity
echo "2. API Connectivity:"
if curl -s --max-time 10 https://urs.earthdata.nasa.gov/ > /dev/null; then
    echo "   ✓ NASA Earthdata reachable"
else
    echo "   ✗ NASA Earthdata unreachable"
fi

if [[ -n "$OPENAI_API_KEY" ]]; then
    echo "   ✓ OpenAI API key configured"
else
    echo "   ✗ OpenAI API key missing"
fi

# 3. Storage Check
echo "3. Storage Status:"
for dir in "$DATA_DIR" "$RESULTS_DIR" "$CACHE_DIR"; do
    if [[ -d "$dir" && -w "$dir" ]]; then
        available=$(df -h "$dir" | tail -1 | awk '{print $4}')
        echo "   ✓ $dir: $available available"
    else
        echo "   ✗ $dir: Not accessible or missing"
    fi
done

# 4. Memory Check
echo "4. System Resources:"
memory_percent=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
echo "   Memory Usage: ${memory_percent}%"

cpu_count=$(nproc)
echo "   CPU Cores: $cpu_count"

echo "=== Health Check Complete ==="
```

## Installation Issues

### Dependency Installation Problems

#### GDAL Installation Failures

**Symptoms:**
```
ERROR: Failed building wheel for GDAL
ImportError: No module named 'osgeo'
```

**Solutions:**

1. **Ubuntu/Debian Systems:**
```bash
# Install system dependencies first
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev build-essential

# Get GDAL version
gdal-config --version

# Install Python GDAL with matching version
pip install GDAL==$(gdal-config --version)
```

2. **macOS Systems:**
```bash
# Using Homebrew
brew install gdal

# Set environment variables
export CPPFLAGS=-I/usr/local/include
export LDFLAGS=-L/usr/local/lib

# Install Python GDAL
pip install GDAL==$(gdal-config --version)
```

3. **Conda Installation (Recommended):**
```bash
# Remove existing problematic installation
conda remove gdal rasterio geopandas -y

# Install through conda-forge
conda install -c conda-forge gdal rasterio geopandas

# Verify installation
python -c "from osgeo import gdal; print(gdal.__version__)"
```

#### CuPy/GPU Installation Issues

**Symptoms:**
```
ImportError: CuPy is not correctly installed
CUDA out of memory
```

**Solutions:**

1. **Verify CUDA Installation:**
```bash
# Check CUDA version
nvidia-smi
nvcc --version

# Install matching CuPy version
pip install cupy-cuda11x  # For CUDA 11.x
# or
pip install cupy-cuda12x  # For CUDA 12.x
```

2. **Memory Issues:**
```bash
# Set memory pool limit
export CUPY_MEMORY_POOL_LIMIT=8000000000  # 8GB

# Or disable GPU if issues persist
export ENABLE_GPU=false
```

3. **Fallback to CPU:**
```python
# In your Python code
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("GPU acceleration disabled, using CPU")
```

### Environment Configuration Issues

#### Authentication Problems

**NASA Earthdata Authentication:**

**Symptoms:**
```
HTTP 401: Unauthorized
Authentication failed for NASA Earthdata
```

**Solutions:**

1. **Verify Credentials:**
```bash
# Test authentication
python -c "
import earthaccess
try:
    auth = earthaccess.login()
    print('Authentication successful')
except Exception as e:
    print(f'Authentication failed: {e}')
"
```

2. **Configure Credentials:**
```bash
# Method 1: Environment variables
export EARTHDATA_USERNAME=your_username
export EARTHDATA_PASSWORD=your_password

# Method 2: Interactive login
python -c "import earthaccess; earthaccess.login(persist=True)"

# Method 3: Netrc file
echo "machine urs.earthdata.nasa.gov login your_username password your_password" >> ~/.netrc
chmod 600 ~/.netrc
```

**OpenAI API Authentication:**

**Symptoms:**
```
OpenAI API key not provided
Invalid API key provided
```

**Solutions:**

1. **Configure API Key:**
```bash
# Set environment variable
export OPENAI_API_KEY=your_api_key_here

# Verify in Python
python -c "
import os
import openai
openai.api_key = os.getenv('OPENAI_API_KEY')
print('OpenAI configured successfully' if openai.api_key else 'API key missing')
"
```

2. **Test API Access:**
```python
import openai
import os

openai.api_key = os.getenv('OPENAI_API_KEY')

try:
    response = openai.Model.list()
    print("OpenAI API accessible")
except Exception as e:
    print(f"OpenAI API error: {e}")
```

## Runtime Issues

### Memory Problems

#### Out of Memory Errors

**Symptoms:**
```
MemoryError: Unable to allocate array
Process killed (signal 9) - Out of memory
```

**Diagnostic Steps:**

1. **Check Memory Usage:**
```python
import psutil
import os

def check_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    print(f"RSS: {memory_info.rss / 1024 / 1024:.1f} MB")
    print(f"VMS: {memory_info.vms / 1024 / 1024:.1f} MB")
    
    system_memory = psutil.virtual_memory()
    print(f"System Memory: {system_memory.percent:.1f}% used")
    print(f"Available: {system_memory.available / 1024 / 1024:.1f} MB")

check_memory_usage()
```

**Solutions:**

1. **Reduce Memory Usage:**
```python
# config/memory_optimization.py
MEMORY_OPTIMIZED_CONFIG = {
    'MAX_WORKERS': 2,           # Reduce parallel processing
    'CHUNK_SIZE_MB': 64,        # Smaller processing chunks
    'ENABLE_GPU': False,        # Disable GPU to save memory
    'CACHE_SIZE_GB': 5,         # Smaller cache
    'GC_AGGRESSIVE': True       # Aggressive garbage collection
}
```

2. **Process in Chunks:**
```python
def process_large_dataset_in_chunks(data, chunk_size=64*1024*1024):
    """Process large datasets in memory-efficient chunks."""
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        result = process_chunk(chunk)
        yield result
        
        # Force garbage collection
        import gc
        gc.collect()
```

3. **Use Memory Mapping:**
```python
import numpy as np
import mmap

def memory_mapped_processing(file_path):
    """Use memory mapping for large files."""
    with open(file_path, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
            # Process without loading entire file into memory
            data = np.frombuffer(mmapped_file, dtype=np.float32)
            return process_data(data)
```

#### Memory Leaks

**Symptoms:**
```
Memory usage continuously increasing
System becomes unresponsive over time
```

**Diagnostic Steps:**

1. **Memory Profiling:**
```python
import tracemalloc
import gc

def profile_memory_usage():
    tracemalloc.start()
    
    # Your code here
    run_pipeline()
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
    
    tracemalloc.stop()

def check_object_counts():
    """Check for object leaks."""
    import gc
    
    for obj_type in [list, dict, tuple, set]:
        count = len([obj for obj in gc.get_objects() if isinstance(obj, obj_type)])
        print(f"{obj_type.__name__}: {count} objects")
```

**Solutions:**

1. **Explicit Memory Management:**
```python
def analyze_scene_with_cleanup(scene_data):
    """Analyze scene with explicit cleanup."""
    try:
        # Perform analysis
        result = perform_analysis(scene_data)
        return result
    finally:
        # Explicit cleanup
        del scene_data
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear matplotlib figure cache
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except ImportError:
            pass
```

2. **Context Managers:**
```python
from contextlib import contextmanager

@contextmanager
def managed_analysis(scene_data):
    """Context manager for memory-safe analysis."""
    try:
        yield scene_data
    finally:
        # Cleanup operations
        cleanup_scene_data(scene_data)
        gc.collect()

# Usage
with managed_analysis(scene_data) as data:
    result = analyze_scene(data)
```

### Performance Issues

#### Slow Processing

**Symptoms:**
```
Analysis taking much longer than expected
High CPU usage with low throughput
Frequent timeouts
```

**Diagnostic Steps:**

1. **Performance Profiling:**
```python
import cProfile
import pstats

def profile_analysis():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run your analysis
    result = run_analysis()
    
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 time-consuming functions
    
    return result
```

2. **I/O Monitoring:**
```bash
# Monitor I/O usage
iostat -x 1

# Check for I/O bottlenecks
iotop -a

# Monitor network usage for downloads
iftop
```

**Solutions:**

1. **Parallel Processing:**
```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

def parallel_scene_analysis(scenes, max_workers=None):
    """Process scenes in parallel."""
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(scenes))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(analyze_scene, scene): scene for scene in scenes}
        
        # Collect results
        results = []
        for future in as_completed(futures):
            scene = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Analysis failed for scene {scene.scene_id}: {e}")
        
        return results
```

2. **Caching Strategy:**
```python
import functools
import pickle
from pathlib import Path

def persistent_cache(cache_dir):
    """Decorator for persistent caching."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = str(hash((args, tuple(sorted(kwargs.items())))))
            cache_file = cache_path / f"{func.__name__}_{key}.pkl"
            
            # Check cache
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except:
                    pass  # Cache corrupted, recompute
            
            # Compute and cache result
            result = func(*args, **kwargs)
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            return result
        return wrapper
    return decorator

@persistent_cache('./cache')
def expensive_computation(data):
    # Time-consuming operation
    return complex_analysis(data)
```

### Network Issues

#### Download Failures

**Symptoms:**
```
Connection timeout
HTTP 503 Service Unavailable
Incomplete downloads
```

**Diagnostic Steps:**

1. **Test Connectivity:**
```bash
# Test NASA Earthdata
curl -I https://urs.earthdata.nasa.gov/

# Test with authentication
curl -u username:password https://e4ftl01.cr.usgs.gov/

# Check DNS resolution
nslookup urs.earthdata.nasa.gov
```

2. **Network Speed Test:**
```python
import time
import requests

def test_download_speed(url, test_size_mb=10):
    """Test download speed."""
    start_time = time.time()
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            downloaded += len(chunk)
            if downloaded >= test_size_mb * 1024 * 1024:
                break
        
        duration = time.time() - start_time
        speed_mbps = (downloaded / duration) / (1024 * 1024)
        
        print(f"Download speed: {speed_mbps:.2f} MB/s")
        return speed_mbps
        
    except Exception as e:
        print(f"Download test failed: {e}")
        return 0
```

**Solutions:**

1. **Retry Strategy:**
```python
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_resilient_session():
    """Create HTTP session with retry strategy."""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        method_whitelist=["HEAD", "GET", "OPTIONS"],
        backoff_factor=1
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def robust_download(url, file_path, max_retries=3):
    """Download with retry and resume capability."""
    session = create_resilient_session()
    
    for attempt in range(max_retries):
        try:
            # Check if partial file exists
            resume_header = {}
            if file_path.exists():
                resume_header['Range'] = f'bytes={file_path.stat().st_size}-'
            
            response = session.get(url, headers=resume_header, stream=True, timeout=300)
            response.raise_for_status()
            
            mode = 'ab' if resume_header else 'wb'
            with open(file_path, mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
            
        except Exception as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    
    return False
```

2. **Bandwidth Management:**
```python
import time
from threading import Semaphore

class BandwidthLimiter:
    def __init__(self, max_concurrent=4, rate_limit_mbps=None):
        self.semaphore = Semaphore(max_concurrent)
        self.rate_limit = rate_limit_mbps * 1024 * 1024 if rate_limit_mbps else None
        self.last_chunk_time = time.time()
    
    def download_with_limit(self, session, url, file_path):
        with self.semaphore:
            response = session.get(url, stream=True)
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if self.rate_limit:
                        # Rate limiting
                        current_time = time.time()
                        time_since_last = current_time - self.last_chunk_time
                        
                        expected_time = len(chunk) / self.rate_limit
                        if time_since_last < expected_time:
                            time.sleep(expected_time - time_since_last)
                        
                        self.last_chunk_time = time.time()
                    
                    f.write(chunk)
```

### Data Processing Issues

#### Coordinate Reference System (CRS) Problems

**Symptoms:**
```
Coordinate transformation errors
Features appearing in wrong locations
CRS mismatch warnings
```

**Diagnostic Steps:**

1. **Check CRS Information:**
```python
import rasterio
import geopandas as gpd

def diagnose_crs(file_path):
    """Diagnose CRS issues in geospatial data."""
    try:
        # For raster data
        with rasterio.open(file_path) as src:
            print(f"CRS: {src.crs}")
            print(f"Transform: {src.transform}")
            print(f"Bounds: {src.bounds}")
            
    except:
        try:
            # For vector data
            gdf = gpd.read_file(file_path)
            print(f"CRS: {gdf.crs}")
            print(f"Bounds: {gdf.bounds}")
        except Exception as e:
            print(f"Could not read CRS: {e}")
```

**Solutions:**

1. **CRS Standardization:**
```python
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def standardize_crs(input_file, output_file, target_crs='EPSG:4326'):
    """Standardize CRS to WGS84."""
    
    if input_file.suffix in ['.tif', '.tiff']:
        # Raster reprojection
        with rasterio.open(input_file) as src:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': target_crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            
            with rasterio.open(output_file, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest
                    )
    else:
        # Vector reprojection
        gdf = gpd.read_file(input_file)
        gdf_reprojected = gdf.to_crs(target_crs)
        gdf_reprojected.to_file(output_file)
```

2. **Coordinate Validation:**
```python
def validate_coordinates(coordinates, expected_crs='EPSG:4326'):
    """Validate coordinate values."""
    issues = []
    
    for i, (lon, lat) in enumerate(coordinates):
        # Check for valid latitude/longitude ranges
        if expected_crs == 'EPSG:4326':
            if not -180 <= lon <= 180:
                issues.append(f"Invalid longitude at index {i}: {lon}")
            if not -90 <= lat <= 90:
                issues.append(f"Invalid latitude at index {i}: {lat}")
        
        # Check for null island (0,0) coordinates
        if lon == 0 and lat == 0:
            issues.append(f"Suspicious null island coordinate at index {i}")
    
    return issues
```

#### File Format Issues

**Symptoms:**
```
Unsupported file format
Corrupted data files
Missing bands or metadata
```

**Solutions:**

1. **File Format Validation:**
```python
import rasterio
import geopandas as gpd
from pathlib import Path

def validate_file_format(file_path):
    """Validate and diagnose file format issues."""
    path = Path(file_path)
    
    if not path.exists():
        return {'valid': False, 'error': 'File does not exist'}
    
    try:
        # Try as raster
        with rasterio.open(path) as src:
            return {
                'valid': True,
                'type': 'raster',
                'format': src.driver,
                'bands': src.count,
                'crs': str(src.crs),
                'shape': (src.height, src.width)
            }
    except:
        pass
    
    try:
        # Try as vector
        gdf = gpd.read_file(path)
        return {
            'valid': True,
            'type': 'vector',
            'format': path.suffix,
            'features': len(gdf),
            'crs': str(gdf.crs),
            'columns': list(gdf.columns)
        }
    except Exception as e:
        return {'valid': False, 'error': str(e)}

def repair_corrupted_file(file_path, backup_path=None):
    """Attempt to repair corrupted geospatial files."""
    try:
        # Create backup
        if backup_path:
            import shutil
            shutil.copy2(file_path, backup_path)
        
        # Try to read and re-write the file
        with rasterio.open(file_path) as src:
            data = src.read()
            profile = src.profile
        
        # Write to temporary file first
        temp_path = file_path.with_suffix('.tmp')
        with rasterio.open(temp_path, 'w', **profile) as dst:
            dst.write(data)
        
        # Replace original if successful
        temp_path.replace(file_path)
        return True
        
    except Exception as e:
        logger.error(f"Could not repair file {file_path}: {e}")
        return False
```

## Debugging Strategies

### Logging Configuration

```python
# config/debug_logging.py
import logging
import sys
from pathlib import Path

def setup_debug_logging(log_level='DEBUG', log_file=None):
    """Setup comprehensive debug logging."""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Module-specific loggers
    debug_modules = [
        'src.pipeline',
        'src.providers',
        'src.detectors',
        'src.core'
    ]
    
    for module in debug_modules:
        logger = logging.getLogger(module)
        logger.setLevel('DEBUG')

# Usage
setup_debug_logging('DEBUG', './logs/debug.log')
```

### Step-by-Step Debugging

```python
# debug/step_by_step.py
import json
import traceback
from pathlib import Path

class PipelineDebugger:
    def __init__(self, debug_dir='./debug'):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(exist_ok=True)
        self.step_counter = 0
    
    def save_debug_data(self, data, name, description=""):
        """Save debug data with step information."""
        self.step_counter += 1
        
        filename = f"step_{self.step_counter:03d}_{name}.json"
        filepath = self.debug_dir / filename
        
        debug_info = {
            'step': self.step_counter,
            'name': name,
            'description': description,
            'data': data,
            'timestamp': time.time()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(debug_info, f, indent=2, default=str)
            
            logger.debug(f"Debug data saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save debug data: {e}")
    
    def debug_analysis_pipeline(self, scene_data):
        """Debug the analysis pipeline step by step."""
        try:
            # Step 1: Input data
            self.save_debug_data(
                {
                    'scene_id': scene_data.scene_id,
                    'zone_id': scene_data.zone_id,
                    'provider': scene_data.provider,
                    'file_paths': {k: str(v) for k, v in scene_data.file_paths.items()},
                    'available_bands': scene_data.available_bands
                },
                'input_scene_data',
                'Original scene data from provider'
            )
            
            # Step 2: Data loading
            loaded_data = load_scene_data(scene_data)
            self.save_debug_data(
                {
                    'shape': loaded_data.shape if hasattr(loaded_data, 'shape') else 'unknown',
                    'dtype': str(loaded_data.dtype) if hasattr(loaded_data, 'dtype') else 'unknown',
                    'min_value': float(loaded_data.min()) if hasattr(loaded_data, 'min') else 'unknown',
                    'max_value': float(loaded_data.max()) if hasattr(loaded_data, 'max') else 'unknown'
                },
                'loaded_data_stats',
                'Statistics of loaded data'
            )
            
            # Step 3: Preprocessing
            preprocessed = preprocess_data(loaded_data)
            self.save_debug_data(
                {
                    'preprocessing_applied': True,
                    'output_shape': preprocessed.shape,
                    'processing_summary': 'Data preprocessed successfully'
                },
                'preprocessing_result',
                'Results after preprocessing'
            )
            
            # Continue with analysis...
            result = analyze_preprocessed_data(preprocessed)
            
            # Step 4: Final results
            self.save_debug_data(
                result,
                'final_analysis_result',
                'Final analysis results'
            )
            
            return result
            
        except Exception as e:
            # Save error information
            self.save_debug_data(
                {
                    'error': str(e),
                    'traceback': traceback.format_exc(),
                    'failed_at_step': self.step_counter
                },
                'error_info',
                f'Error occurred during analysis: {e}'
            )
            raise

# Usage
debugger = PipelineDebugger('./debug/analysis_run_001')
result = debugger.debug_analysis_pipeline(scene_data)
```

## Prevention Strategies

### Automated Testing

```python
# tests/integration_test.py
import pytest
import tempfile
from pathlib import Path

class TestPipelineIntegration:
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_dir = Path(self.temp_dir) / 'test_data'
        self.test_data_dir.mkdir()
    
    def test_complete_pipeline(self):
        """Test complete pipeline with mock data."""
        # Create mock scene data
        mock_scene = self._create_mock_scene()
        
        # Test data acquisition
        scene_data = acquire_test_data(mock_scene)
        assert scene_data is not None
        assert len(scene_data) > 0
        
        # Test analysis
        analysis_results = analyze_test_scenes(scene_data)
        assert 'success' in analysis_results
        assert analysis_results['success'] is True
        
        # Test scoring
        scoring_results = score_test_zones(analysis_results)
        assert isinstance(scoring_results, dict)
        
        # Test output generation
        outputs = generate_test_outputs(analysis_results, scoring_results)
        assert 'report' in outputs
    
    def test_error_recovery(self):
        """Test pipeline error recovery."""
        # Test with corrupted data
        corrupted_scene = self._create_corrupted_scene()
        
        # Pipeline should handle gracefully
        result = run_pipeline_with_error_handling(corrupted_scene)
        assert result['status'] in ['partial_success', 'failed']
        assert 'error_details' in result
    
    def _create_mock_scene(self):
        """Create mock scene data for testing."""
        # Implementation for creating test data
        pass
    
    def _create_corrupted_scene(self):
        """Create corrupted scene data for error testing."""
        # Implementation for creating corrupted test data
        pass

# Run tests
pytest.main(['-v', 'tests/integration_test.py'])
```

### Health Monitoring

```python
# monitoring/health_monitor.py
import time
import psutil
import logging
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class HealthMetric:
    name: str
    value: float
    threshold: float
    unit: str
    status: str  # 'ok', 'warning', 'critical'

class SystemHealthMonitor:
    def __init__(self, check_interval=60):
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
        self.running = False
    
    def check_system_health(self) -> List[HealthMetric]:
        """Check overall system health."""
        metrics = []
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_metric = HealthMetric(
            name='memory_usage',
            value=memory.percent,
            threshold=85.0,
            unit='%',
            status='ok' if memory.percent < 85 else 'warning' if memory.percent < 95 else 'critical'
        )
        metrics.append(memory_metric)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_metric = HealthMetric(
            name='disk_usage',
            value=disk.percent,
            threshold=90.0,
            unit='%',
            status='ok' if disk.percent < 90 else 'warning' if disk.percent < 95 else 'critical'
        )
        metrics.append(disk_metric)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_metric = HealthMetric(
            name='cpu_usage',
            value=cpu_percent,
            threshold=90.0,
            unit='%',
            status='ok' if cpu_percent < 90 else 'warning' if cpu_percent < 95 else 'critical'
        )
        metrics.append(cpu_metric)
        
        return metrics
    
    def check_pipeline_health(self) -> Dict[str, Any]:
        """Check pipeline-specific health."""
        health_status = {
            'api_connectivity': self._check_api_connectivity(),
            'data_directories': self._check_data_directories(),
            'recent_errors': self._check_recent_errors()
        }
        
        return health_status
    
    def _check_api_connectivity(self) -> Dict[str, bool]:
        """Check external API connectivity."""
        import requests
        
        apis = {
            'nasa_earthdata': 'https://urs.earthdata.nasa.gov/',
            'openai': 'https://api.openai.com/v1/models'
        }
        
        connectivity = {}
        for name, url in apis.items():
            try:
                response = requests.head(url, timeout=10)
                connectivity[name] = response.status_code < 400
            except:
                connectivity[name] = False
        
        return connectivity
    
    def _check_data_directories(self) -> Dict[str, bool]:
        """Check data directory accessibility."""
        import os
        
        required_dirs = [
            os.getenv('DATA_DIR', '/data'),
            os.getenv('RESULTS_DIR', '/results'),
            os.getenv('CACHE_DIR', '/cache')
        ]
        
        dir_status = {}
        for dir_path in required_dirs:
            path = Path(dir_path)
            dir_status[dir_path] = path.exists() and os.access(dir_path, os.W_OK)
        
        return dir_status
    
    def _check_recent_errors(self) -> int:
        """Check for recent errors in logs."""
        # Implementation to parse recent log files for errors
        # Return count of recent errors
        return 0
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        self.running = True
        
        while self.running:
            try:
                # System health check
                system_metrics = self.check_system_health()
                for metric in system_metrics:
                    if metric.status != 'ok':
                        self.logger.warning(
                            f"Health issue: {metric.name} = {metric.value}{metric.unit} "
                            f"(threshold: {metric.threshold}{metric.unit})"
                        )
                
                # Pipeline health check
                pipeline_health = self.check_pipeline_health()
                
                # Log overall status
                critical_issues = [m for m in system_metrics if m.status == 'critical']
                if critical_issues:
                    self.logger.critical(f"Critical health issues detected: {len(critical_issues)}")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.running = False

# Usage
monitor = SystemHealthMonitor(check_interval=300)  # Check every 5 minutes
monitor.start_monitoring()
```

This comprehensive troubleshooting manual provides systematic approaches to diagnosing and resolving common issues with the Amazon Archaeological Discovery Pipeline, enabling reliable operation across diverse computational environments.
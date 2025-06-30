# Performance Optimization Guide

## Overview

This guide provides comprehensive strategies for optimizing the performance of the Amazon Archaeological Discovery Pipeline across different computational environments. The pipeline's performance can be significantly enhanced through proper configuration, hardware optimization, and algorithmic tuning.

## Performance Architecture

### Pipeline Performance Model

```mermaid
graph TB
    subgraph "Performance Bottlenecks"
        IO[I/O Operations]
        CPU[CPU Processing]
        MEM[Memory Usage]
        NET[Network Bandwidth]
        GPU[GPU Utilization]
    end
    
    subgraph "Optimization Targets"
        DATA[Data Acquisition]
        PROC[Processing Speed]
        CACHE[Caching Strategy]
        PAR[Parallelization]
        OPT[Algorithmic Optimization]
    end
    
    subgraph "Performance Metrics"
        THRU[Throughput (scenes/hour)]
        LAT[Latency (time per analysis)]
        UTIL[Resource Utilization]
        SCALE[Scaling Efficiency]
    end
    
    IO --> DATA
    CPU --> PROC
    MEM --> CACHE
    NET --> DATA
    GPU --> OPT
    
    DATA --> THRU
    PROC --> LAT
    CACHE --> UTIL
    PAR --> SCALE
    OPT --> THRU
```

## Hardware Optimization

### CPU Configuration

#### Multi-Core Processing

```python
# config/performance.py
import multiprocessing
import os

# Optimize worker count based on CPU cores and workload
def get_optimal_workers():
    cpu_count = multiprocessing.cpu_count()
    
    # For I/O bound tasks (data download)
    io_workers = min(cpu_count * 2, 16)
    
    # For CPU bound tasks (analysis)
    cpu_workers = max(cpu_count - 1, 1)
    
    return {
        'download_workers': io_workers,
        'analysis_workers': cpu_workers,
        'export_workers': min(cpu_count, 8)
    }

# Set environment variables
workers = get_optimal_workers()
os.environ['DOWNLOAD_WORKERS'] = str(workers['download_workers'])
os.environ['ANALYSIS_WORKERS'] = str(workers['analysis_workers'])
os.environ['EXPORT_WORKERS'] = str(workers['export_workers'])
```

#### CPU Affinity Optimization

```bash
#!/bin/bash
# scripts/cpu_optimization.sh

# Set CPU affinity for pipeline processes
taskset -c 0-7 python main.py --pipeline --zone upper_napo_micro &
PID=$!

# Set CPU governor to performance mode
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU frequency scaling for consistent performance
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Set process priority
renice -10 $PID
```

### Memory Optimization

#### Memory Pool Management

```python
# src/core/memory_optimization.py
import psutil
import gc
from typing import Dict, Any
import numpy as np

class MemoryManager:
    def __init__(self, max_memory_percent: float = 80.0):
        """Initialize memory manager with usage limits."""
        self.max_memory_percent = max_memory_percent
        self.total_memory = psutil.virtual_memory().total
        self.max_memory_bytes = int(self.total_memory * max_memory_percent / 100)
        
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage."""
        memory = psutil.virtual_memory()
        return {
            'used_percent': memory.percent,
            'available_mb': memory.available // (1024**2),
            'used_mb': memory.used // (1024**2),
            'can_allocate': memory.percent < self.max_memory_percent
        }
    
    def optimize_memory_usage(self):
        """Optimize memory usage through garbage collection."""
        # Force garbage collection
        gc.collect()
        
        # Clear numpy cache
        if hasattr(np, 'core'):
            if hasattr(np.core, '_methods'):
                np.core._methods._clip_dep_invoke_with_casting = None
        
        # Clear matplotlib cache if imported
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
        except ImportError:
            pass
    
    def estimate_scene_memory(self, scene_size_mb: float, bands: int = 4) -> float:
        """Estimate memory required for scene processing."""
        # Base memory for scene data
        base_memory = scene_size_mb * bands * 2  # Factor for processing overhead
        
        # Additional memory for intermediate results
        analysis_memory = scene_size_mb * 0.5
        
        # Buffer for system operations
        system_buffer = 512  # MB
        
        return base_memory + analysis_memory + system_buffer

# Usage in pipeline
memory_manager = MemoryManager(max_memory_percent=85.0)

def process_scene_with_memory_check(scene_data):
    """Process scene with memory monitoring."""
    memory_status = memory_manager.check_memory_usage()
    
    if not memory_status['can_allocate']:
        logger.warning(f"Memory usage high: {memory_status['used_percent']:.1f}%")
        memory_manager.optimize_memory_usage()
    
    # Estimate memory needed
    estimated_memory = memory_manager.estimate_scene_memory(
        scene_size_mb=scene_data.estimated_size_mb,
        bands=len(scene_data.available_bands)
    )
    
    if estimated_memory > memory_status['available_mb']:
        raise MemoryError(f"Insufficient memory: need {estimated_memory}MB, have {memory_status['available_mb']}MB")
    
    # Process scene
    return analyze_scene(scene_data)
```

#### Memory-Mapped File Operations

```python
# src/core/memory_mapped_io.py
import numpy as np
import mmap
from pathlib import Path

class MemoryMappedRaster:
    def __init__(self, file_path: Path, mode: str = 'r'):
        """Initialize memory-mapped raster for efficient I/O."""
        self.file_path = file_path
        self.mode = mode
        self._mmap = None
        self._array = None
    
    def __enter__(self):
        """Context manager entry."""
        self.file = open(self.file_path, 'rb' if 'r' in self.mode else 'r+b')
        self._mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Create numpy array view
        self._array = np.frombuffer(self._mmap, dtype=np.float32)
        return self._array
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._mmap:
            self._mmap.close()
        if self.file:
            self.file.close()

# Usage for large raster processing
def process_large_raster(raster_path: Path):
    """Process large raster files without loading entirely into memory."""
    with MemoryMappedRaster(raster_path) as data:
        # Process data in chunks
        chunk_size = 1024 * 1024  # 1M elements
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            # Process chunk
            result = analyze_chunk(chunk)
            yield result
```

### Storage Optimization

#### SSD Configuration

```bash
#!/bin/bash
# scripts/storage_optimization.sh

# Optimize SSD performance
echo mq-deadline | sudo tee /sys/block/nvme0n1/queue/scheduler

# Set readahead for large sequential reads
sudo blockdev --setra 4096 /dev/nvme0n1

# Configure temporary directories on fastest storage
export TMPDIR=/fast/ssd/tmp
export DATA_CACHE_DIR=/fast/ssd/cache
export PROCESSING_TEMP_DIR=/fast/ssd/processing

# Create optimized temp directories
mkdir -p $TMPDIR $DATA_CACHE_DIR $PROCESSING_TEMP_DIR
```

#### Intelligent Caching Strategy

```python
# src/core/intelligent_cache.py
import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Optional
import lru_cache

class IntelligentCache:
    def __init__(self, cache_dir: Path, max_size_gb: float = 50.0):
        """Initialize intelligent caching system."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.access_times = {}
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cache_size(self) -> int:
        """Calculate total cache size."""
        total_size = 0
        for file_path in self.cache_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def _cleanup_cache(self):
        """Remove least recently used cache entries."""
        cache_files = list(self.cache_dir.rglob('*.cache'))
        
        # Sort by access time (least recent first)
        cache_files.sort(key=lambda p: self.access_times.get(str(p), 0))
        
        current_size = self._get_cache_size()
        target_size = int(self.max_size_bytes * 0.8)  # Clean to 80% of max
        
        for cache_file in cache_files:
            if current_size <= target_size:
                break
            
            file_size = cache_file.stat().st_size
            cache_file.unlink()
            current_size -= file_size
            
            if str(cache_file) in self.access_times:
                del self.access_times[str(cache_file)]
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        cache_file = self.cache_dir / f"{key}.cache"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Update access time
                self.access_times[str(cache_file)] = time.time()
                return data
            except Exception:
                # Remove corrupted cache file
                cache_file.unlink()
        
        return None
    
    def set(self, key: str, value: Any):
        """Set cached value."""
        # Check cache size and cleanup if needed
        if self._get_cache_size() > self.max_size_bytes:
            self._cleanup_cache()
        
        cache_file = self.cache_dir / f"{key}.cache"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
            
            self.access_times[str(cache_file)] = time.time()
        except Exception as e:
            logger.error(f"Failed to cache data: {e}")

# Decorator for automatic caching
def cached_analysis(cache_dir: Path):
    """Decorator for caching analysis results."""
    cache = IntelligentCache(cache_dir)
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache._get_cache_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                logger.info(f"Cache hit for {func.__name__}")
                return result
            
            # Compute result
            logger.info(f"Cache miss for {func.__name__}, computing...")
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.set(cache_key, result)
            return result
        
        return wrapper
    return decorator
```

## GPU Acceleration

### CUDA Optimization

#### GPU Memory Management

```python
# src/core/gpu_optimization.py
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Check for CuPy availability
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    logger.warning("CuPy not available, falling back to CPU processing")

class GPUManager:
    def __init__(self, memory_pool_fraction: float = 0.8):
        """Initialize GPU memory management."""
        if not GPU_AVAILABLE:
            self.enabled = False
            return
        
        self.enabled = True
        self.memory_pool_fraction = memory_pool_fraction
        
        # Set memory pool size
        mempool = cp.get_default_memory_pool()
        mempool.set_limit(size=int(self._get_gpu_memory() * memory_pool_fraction))
        
        # Set pinned memory pool for faster transfers
        pinned_mempool = cp.get_default_pinned_memory_pool()
        pinned_mempool.set_limit(size=2**30)  # 1GB pinned memory
    
    def _get_gpu_memory(self) -> int:
        """Get total GPU memory."""
        if not self.enabled:
            return 0
        return cp.cuda.runtime.memGetInfo()[1]
    
    def transfer_to_gpu(self, array: np.ndarray) -> 'cp.ndarray':
        """Transfer array to GPU with memory check."""
        if not self.enabled:
            return array
        
        # Check available GPU memory
        free_memory = cp.cuda.runtime.memGetInfo()[0]
        array_memory = array.nbytes
        
        if array_memory > free_memory * 0.8:
            logger.warning(f"Large array ({array_memory/1e9:.2f}GB) may cause GPU memory issues")
        
        return cp.asarray(array)
    
    def transfer_to_cpu(self, gpu_array) -> np.ndarray:
        """Transfer array from GPU to CPU."""
        if not self.enabled or not hasattr(gpu_array, 'get'):
            return gpu_array
        return gpu_array.get()
    
    def cleanup_gpu_memory(self):
        """Cleanup GPU memory."""
        if self.enabled:
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()

# GPU-accelerated processing functions
def gpu_accelerated_analysis(data: np.ndarray, gpu_manager: GPUManager):
    """Perform GPU-accelerated analysis."""
    if not gpu_manager.enabled:
        return cpu_analysis(data)
    
    try:
        # Transfer to GPU
        gpu_data = gpu_manager.transfer_to_gpu(data)
        
        # GPU computations
        # Example: vegetation index calculation
        if gpu_data.shape[-1] >= 4:  # Has NIR band
            red = gpu_data[..., 2]    # Red band
            nir = gpu_data[..., 3]    # NIR band
            
            # NDVI calculation on GPU
            ndvi = (nir - red) / (nir + red + 1e-8)
            
            # Additional GPU operations
            smoothed = cp.convolve(ndvi, cp.ones((3, 3)) / 9, mode='same')
            
            # Transfer result back to CPU
            result = gpu_manager.transfer_to_cpu(smoothed)
        else:
            result = gpu_manager.transfer_to_cpu(gpu_data)
        
        return result
        
    except cp.cuda.memory.OutOfMemoryError:
        logger.warning("GPU out of memory, falling back to CPU")
        gpu_manager.cleanup_gpu_memory()
        return cpu_analysis(data)
    
    except Exception as e:
        logger.error(f"GPU processing error: {e}, falling back to CPU")
        return cpu_analysis(data)

def cpu_analysis(data: np.ndarray):
    """CPU fallback analysis."""
    if data.shape[-1] >= 4:
        red = data[..., 2]
        nir = data[..., 3]
        ndvi = (nir - red) / (nir + red + 1e-8)
        return ndvi
    return data
```

#### CUDA Kernel Optimization

```python
# src/core/cuda_kernels.py
import cupy as cp

# Custom CUDA kernels for archaeological analysis
terra_preta_kernel = cp.RawKernel(r'''
extern "C" __global__
void terra_preta_detection(
    const float* nir, const float* swir, const float* red,
    float* output, int n, float threshold) {
    
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        // Terra preta spectral signature detection
        float tp_index = (nir[tid] - swir[tid]) / (nir[tid] + swir[tid] + 1e-8);
        float ndvi = (nir[tid] - red[tid]) / (nir[tid] + red[tid] + 1e-8);
        
        // Combined index for terra preta detection
        output[tid] = (tp_index > threshold && ndvi > 0.3) ? 1.0 : 0.0;
    }
}
''', 'terra_preta_detection')

def gpu_terra_preta_detection(nir: cp.ndarray, swir: cp.ndarray, red: cp.ndarray, threshold: float = 0.1):
    """GPU-accelerated terra preta detection."""
    n = nir.size
    output = cp.zeros_like(nir)
    
    # Configure CUDA kernel
    block_size = 256
    grid_size = (n + block_size - 1) // block_size
    
    # Launch kernel
    terra_preta_kernel(
        (grid_size,), (block_size,),
        (nir, swir, red, output, n, threshold)
    )
    
    return output
```

## Network Optimization

### Data Download Optimization

#### Concurrent Download Strategy

```python
# src/core/download_optimization.py
import asyncio
import aiohttp
import concurrent.futures
from typing import List, Dict, Any
import time

class OptimizedDownloader:
    def __init__(self, max_concurrent: int = 8, chunk_size: int = 8192):
        """Initialize optimized downloader."""
        self.max_concurrent = max_concurrent
        self.chunk_size = chunk_size
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=self.max_concurrent // 2,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=1800,  # 30 minutes total
            connect=60,  # 1 minute connection timeout
            sock_read=300  # 5 minutes read timeout
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'Archaeological-Pipeline/2.0'}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def download_file(self, url: str, file_path: Path, 
                          progress_callback=None) -> Dict[str, Any]:
        """Download single file with progress tracking."""
        start_time = time.time()
        
        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                
                with open(file_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(self.chunk_size):
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback:
                            progress_callback(downloaded, total_size)
                
                download_time = time.time() - start_time
                download_speed = total_size / download_time / 1024 / 1024  # MB/s
                
                return {
                    'success': True,
                    'file_path': file_path,
                    'size_mb': total_size / 1024 / 1024,
                    'time_seconds': download_time,
                    'speed_mbps': download_speed
                }
                
        except Exception as e:
            logger.error(f"Download failed for {url}: {e}")
            return {
                'success': False,
                'error': str(e),
                'file_path': file_path
            }
    
    async def download_batch(self, download_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Download multiple files concurrently."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def bounded_download(task):
            async with semaphore:
                return await self.download_file(
                    task['url'], 
                    task['file_path'],
                    task.get('progress_callback')
                )
        
        # Execute downloads concurrently
        results = await asyncio.gather(*[
            bounded_download(task) for task in download_tasks
        ], return_exceptions=True)
        
        return results

# Usage in provider
async def optimized_data_acquisition(scenes_to_download: List[Dict]):
    """Optimized data acquisition with concurrent downloads."""
    async with OptimizedDownloader(max_concurrent=8) as downloader:
        download_tasks = [
            {
                'url': scene['download_url'],
                'file_path': scene['local_path'],
                'progress_callback': lambda d, t: logger.info(f"Downloaded {d/t*100:.1f}%")
            }
            for scene in scenes_to_download
        ]
        
        results = await downloader.download_batch(download_tasks)
        
        # Process results
        successful = [r for r in results if r.get('success')]
        failed = [r for r in results if not r.get('success')]
        
        logger.info(f"Download complete: {len(successful)} successful, {len(failed)} failed")
        
        return successful, failed
```

#### Bandwidth Optimization

```python
# src/core/bandwidth_optimization.py
import time
import statistics
from collections import deque

class BandwidthMonitor:
    def __init__(self, window_size: int = 10):
        """Initialize bandwidth monitoring."""
        self.window_size = window_size
        self.speed_history = deque(maxlen=window_size)
        self.adaptive_chunk_size = 8192
        
    def record_transfer(self, bytes_transferred: int, time_taken: float):
        """Record transfer speed."""
        speed_mbps = (bytes_transferred / time_taken) / (1024 * 1024)
        self.speed_history.append(speed_mbps)
        
        # Adapt chunk size based on performance
        self._adapt_chunk_size()
    
    def _adapt_chunk_size(self):
        """Adapt chunk size based on network performance."""
        if len(self.speed_history) < 3:
            return
        
        avg_speed = statistics.mean(self.speed_history)
        
        if avg_speed > 50:  # High speed connection
            self.adaptive_chunk_size = min(65536, self.adaptive_chunk_size * 2)
        elif avg_speed < 5:  # Slow connection
            self.adaptive_chunk_size = max(4096, self.adaptive_chunk_size // 2)
    
    def get_optimal_concurrent_downloads(self) -> int:
        """Get optimal number of concurrent downloads."""
        if not self.speed_history:
            return 4
        
        avg_speed = statistics.mean(self.speed_history)
        
        if avg_speed > 100:  # Very high speed
            return 12
        elif avg_speed > 50:  # High speed
            return 8
        elif avg_speed > 20:  # Medium speed
            return 4
        else:  # Low speed
            return 2
```

## Algorithm Optimization

### Processing Pipeline Optimization

#### Vectorized Operations

```python
# src/core/vectorized_processing.py
import numpy as np
from numba import jit, prange
import scipy.ndimage as ndi

@jit(nopython=True, parallel=True)
def fast_ndvi_calculation(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Fast NDVI calculation using Numba JIT compilation."""
    output = np.empty_like(red)
    
    for i in prange(red.size):
        r = red.flat[i]
        n = nir.flat[i]
        if n + r > 0:
            output.flat[i] = (n - r) / (n + r)
        else:
            output.flat[i] = 0.0
    
    return output

@jit(nopython=True, parallel=True)
def fast_terra_preta_index(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Fast terra preta index calculation."""
    output = np.empty_like(nir)
    
    for i in prange(nir.size):
        n = nir.flat[i]
        s = swir.flat[i]
        if n + s > 0:
            output.flat[i] = (n - s) / (n + s)
        else:
            output.flat[i] = 0.0
    
    return output

def optimized_feature_detection(scene_data: np.ndarray) -> Dict[str, np.ndarray]:
    """Optimized feature detection using vectorized operations."""
    # Extract bands efficiently
    red = scene_data[..., 2]
    nir = scene_data[..., 3]
    swir = scene_data[..., 4] if scene_data.shape[-1] > 4 else nir
    
    # Vectorized index calculations
    ndvi = fast_ndvi_calculation(red, nir)
    tp_index = fast_terra_preta_index(nir, swir)
    
    # Vectorized thresholding
    vegetation_mask = ndvi > 0.3
    tp_mask = tp_index > 0.1
    
    # Combined archaeological indicator
    archaeological_mask = vegetation_mask & tp_mask
    
    return {
        'ndvi': ndvi,
        'terra_preta_index': tp_index,
        'vegetation_mask': vegetation_mask,
        'terra_preta_mask': tp_mask,
        'archaeological_mask': archaeological_mask
    }
```

#### Spatial Processing Optimization

```python
# src/core/spatial_optimization.py
import numpy as np
from scipy import ndimage
from sklearn.cluster import DBSCAN
from numba import jit

@jit(nopython=True)
def fast_connected_components(binary_image: np.ndarray) -> np.ndarray:
    """Fast connected component labeling."""
    height, width = binary_image.shape
    labels = np.zeros_like(binary_image, dtype=np.int32)
    current_label = 1
    
    # Simple connected component implementation
    for i in range(height):
        for j in range(width):
            if binary_image[i, j] and labels[i, j] == 0:
                # Flood fill for connected component
                stack = [(i, j)]
                while stack:
                    y, x = stack.pop()
                    if (0 <= y < height and 0 <= x < width and 
                        binary_image[y, x] and labels[y, x] == 0):
                        labels[y, x] = current_label
                        stack.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
                current_label += 1
    
    return labels

def optimized_clustering(coordinates: np.ndarray, eps: float = 100.0, min_samples: int = 3):
    """Optimized spatial clustering."""
    if len(coordinates) < min_samples:
        return np.array([-1] * len(coordinates))
    
    # Use DBSCAN with optimized parameters
    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        algorithm='ball_tree',  # Faster for spatial data
        leaf_size=30,
        n_jobs=-1  # Use all available cores
    )
    
    return clustering.fit_predict(coordinates)

def parallel_morphological_operations(binary_image: np.ndarray) -> np.ndarray:
    """Parallel morphological operations for feature cleanup."""
    # Use separable kernels for efficiency
    kernel = np.ones((3, 3), dtype=np.uint8)
    
    # Parallel morphological operations
    opened = ndimage.binary_opening(binary_image, kernel)
    closed = ndimage.binary_closing(opened, kernel)
    
    return closed
```

## Monitoring and Profiling

### Performance Monitoring

```python
# src/core/performance_monitor.py
import time
import psutil
import functools
import logging
from contextlib import contextmanager
from typing import Dict, Any, Callable

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        """Initialize performance monitoring."""
        self.metrics = {}
        self.start_time = time.time()
        
    def record_metric(self, name: str, value: float, unit: str = ""):
        """Record performance metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            'value': value,
            'unit': unit,
            'timestamp': time.time()
        })
    
    @contextmanager
    def timer(self, operation_name: str):
        """Context manager for timing operations."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            self.record_metric(f"{operation_name}_duration", duration, "seconds")
            self.record_metric(f"{operation_name}_memory_delta", memory_delta, "MB")
            
            logger.info(f"{operation_name}: {duration:.2f}s, {memory_delta:+.1f}MB")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        report = {
            'total_runtime': time.time() - self.start_time,
            'system_info': {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent
            },
            'operation_metrics': {}
        }
        
        # Aggregate metrics
        for metric_name, values in self.metrics.items():
            if values:
                report['operation_metrics'][metric_name] = {
                    'count': len(values),
                    'total': sum(v['value'] for v in values),
                    'average': sum(v['value'] for v in values) / len(values),
                    'min': min(v['value'] for v in values),
                    'max': max(v['value'] for v in values)
                }
        
        return report

# Global performance monitor
performance_monitor = PerformanceMonitor()

def monitor_performance(operation_name: str):
    """Decorator for monitoring function performance."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with performance_monitor.timer(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Usage examples
@monitor_performance("scene_analysis")
def analyze_scene(scene_data):
    """Monitored scene analysis."""
    # Analysis implementation
    pass

@monitor_performance("data_download")
def download_scene_data(url, file_path):
    """Monitored data download."""
    # Download implementation
    pass
```

### Profiling Tools

```python
# src/core/profiling_tools.py
import cProfile
import pstats
import io
from pathlib import Path
import tracemalloc

class DetailedProfiler:
    def __init__(self, output_dir: Path):
        """Initialize detailed profiler."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def profile_function(self, func, *args, **kwargs):
        """Profile a specific function."""
        profiler = cProfile.Profile()
        
        # Start memory tracing
        tracemalloc.start()
        
        try:
            # Profile execution
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            
            # Get memory statistics
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Save profiling results
            stats_file = self.output_dir / f"{func.__name__}_profile.stats"
            profiler.dump_stats(str(stats_file))
            
            # Generate human-readable report
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(50)  # Top 50 functions
            
            report_file = self.output_dir / f"{func.__name__}_report.txt"
            with open(report_file, 'w') as f:
                f.write(f"Memory Usage:\n")
                f.write(f"  Current: {current / 1024 / 1024:.2f} MB\n")
                f.write(f"  Peak: {peak / 1024 / 1024:.2f} MB\n\n")
                f.write("Performance Profile:\n")
                f.write(s.getvalue())
            
            logger.info(f"Profile saved: {report_file}")
            return result
            
        except Exception as e:
            tracemalloc.stop()
            raise e

# Memory profiling decorator
def memory_profile(output_dir: str = "./profiles"):
    """Decorator for memory profiling."""
    def decorator(func):
        profiler = DetailedProfiler(Path(output_dir))
        
        def wrapper(*args, **kwargs):
            return profiler.profile_function(func, *args, **kwargs)
        
        return wrapper
    return decorator
```

## Configuration Templates

### High-Performance Configuration

```yaml
# config/high_performance.yaml
performance:
  # CPU Configuration
  max_workers: 16
  cpu_affinity: [0, 1, 2, 3, 4, 5, 6, 7]
  process_priority: -10
  
  # Memory Configuration
  memory_limit_gb: 128
  memory_pool_fraction: 0.85
  gc_threshold: [700, 10, 10]
  
  # I/O Configuration
  io_workers: 8
  read_ahead_kb: 4096
  write_buffer_kb: 1024
  
  # GPU Configuration
  enable_gpu: true
  gpu_memory_fraction: 0.8
  gpu_memory_growth: true
  
  # Cache Configuration
  cache_size_gb: 200
  cache_compression: true
  cache_levels: 3
  
  # Network Configuration
  download_workers: 12
  connection_pool_size: 20
  timeout_seconds: 1800
  chunk_size_kb: 64

analysis:
  # Algorithm Optimization
  use_vectorized_ops: true
  parallel_processing: true
  jit_compilation: true
  
  # Quality vs Speed Trade-offs
  detection_accuracy: "high"  # high, medium, fast
  clustering_algorithm: "dbscan_optimized"
  morphological_ops: "parallel"
  
  # Memory Management
  process_in_chunks: true
  chunk_size_mb: 256
  overlap_percent: 10

data:
  # Storage Optimization
  compression_level: 6
  use_memory_mapping: true
  prefetch_enabled: true
  
  # Caching Strategy
  cache_raw_data: true
  cache_processed_data: true
  cache_results: true
  
  # Cleanup
  auto_cleanup: true
  cleanup_threshold_gb: 500
```

### Resource-Constrained Configuration

```yaml
# config/resource_constrained.yaml
performance:
  # Conservative CPU usage
  max_workers: 2
  process_priority: 0
  
  # Limited memory usage
  memory_limit_gb: 8
  memory_pool_fraction: 0.6
  gc_aggressive: true
  
  # I/O throttling
  io_workers: 2
  concurrent_downloads: 2
  
  # Disable GPU to save memory
  enable_gpu: false
  
  # Small cache
  cache_size_gb: 5
  cache_compression: true

analysis:
  # Speed optimizations
  detection_accuracy: "fast"
  use_approximations: true
  skip_optional_analysis: true
  
  # Memory conservation
  process_in_chunks: true
  chunk_size_mb: 64
  free_memory_aggressively: true

data:
  # Minimal caching
  cache_raw_data: false
  cache_processed_data: false
  cache_results: true
  
  # Aggressive cleanup
  auto_cleanup: true
  cleanup_threshold_gb: 10
```

This performance optimization guide provides comprehensive strategies for maximizing the efficiency of the Amazon Archaeological Discovery Pipeline across different computational environments and resource constraints.
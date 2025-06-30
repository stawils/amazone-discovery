"""
Performance Optimization Module for Archaeological Detection Pipeline
Provides GPU acceleration and parallel processing capabilities
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing as mp
from pathlib import Path
import time

# Optional GPU acceleration imports with __name__ workaround
def _safe_cupy_import():
    """Safely import CuPy with workaround for __name__ attribute issues"""
    try:
        # Temporarily patch functools.update_wrapper to avoid __name__ issues
        import functools
        original_update_wrapper = functools.update_wrapper
        
        def safe_update_wrapper(wrapper, wrapped, *args, **kwargs):
            try:
                return original_update_wrapper(wrapper, wrapped, *args, **kwargs)
            except AttributeError as e:
                if "__name__" in str(e):
                    # Skip the problematic attribute update
                    for attr in ('__module__', '__qualname__', '__doc__', '__annotations__'):
                        try:
                            setattr(wrapper, attr, getattr(wrapped, attr))
                        except (AttributeError, TypeError):
                            pass
                    return wrapper
                raise
        
        # Temporarily replace update_wrapper
        functools.update_wrapper = safe_update_wrapper
        
        try:
            import cupy as cp
            import cupyx.scipy.ndimage as cp_ndimage
            return cp, cp_ndimage, True
        finally:
            # Restore original function
            functools.update_wrapper = original_update_wrapper
            
    except ImportError:
        return None, None, False
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"CuPy import failed with workaround: {e}")
        return None, None, False

cp, cp_ndimage, GPU_AVAILABLE = _safe_cupy_import()
logger = logging.getLogger(__name__)
if GPU_AVAILABLE:
    logger.info("🚀 GPU acceleration available with CuPy")
else:
    logger.info("⚠️ GPU acceleration not available")

# Parallel processing imports
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Joblib parallel processing available")
except ImportError:
    JOBLIB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.info("⚠️ Joblib not available - using ThreadPoolExecutor fallback")

from .config import ProcessingConfig

class PerformanceOptimizer:
    """Main optimization coordinator"""
    
    def __init__(self, use_gpu: bool = True, max_workers: Optional[int] = None):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.max_workers = max_workers or min(ProcessingConfig.MAX_WORKERS, mp.cpu_count())
        self.gpu_memory_pool = None
        
        if self.use_gpu and GPU_AVAILABLE:
            try:
                # Initialize GPU memory pool for efficient memory management
                self.gpu_memory_pool = cp.get_default_memory_pool()
                logger.info(f"🎯 GPU optimization enabled")
            except Exception as e:
                logger.warning(f"GPU initialization failed: {e}. Falling back to CPU.")
                self.use_gpu = False
        
        logger.info(f"🔧 Performance optimizer initialized: GPU={self.use_gpu}, Workers={self.max_workers}")
    
    @property
    def gpu_available(self) -> bool:
        """Check if GPU is available and enabled"""
        return self.use_gpu and GPU_AVAILABLE

class BandLoader:
    """Optimized parallel band loading"""
    
    def __init__(self, optimizer: PerformanceOptimizer):
        self.optimizer = optimizer
    
    def load_bands_parallel(self, band_mapping: Dict[str, int], 
                          scene_path: Path) -> Dict[str, np.ndarray]:
        """Load multiple bands in parallel from a composite file"""
        
        def load_single_band(band_name: str, band_idx: int) -> Tuple[str, Optional[np.ndarray]]:
            """Load a single band - for parallel execution"""
            try:
                import rasterio
                start_time = time.time()
                
                with rasterio.open(scene_path) as src:
                    band_data = src.read(band_idx).astype(np.float32)
                
                load_time = time.time() - start_time
                logger.debug(f"Loaded {band_name} (band {band_idx}) in {load_time:.2f}s")
                return band_name, band_data
                
            except Exception as e:
                logger.error(f"Failed to load {band_name} (band {band_idx}): {e}")
                return band_name, None
        
        logger.info(f"📁 PARALLEL BAND LOADING: Processing {len(band_mapping)} bands")
        logger.info(f"⚡ Using {self.optimizer.max_workers} worker threads for I/O optimization")
        start_time = time.time()
        
        # Use ThreadPoolExecutor for I/O bound operations (file reading)
        with ThreadPoolExecutor(max_workers=self.optimizer.max_workers) as executor:
            futures = [
                executor.submit(load_single_band, band_name, band_idx)
                for band_name, band_idx in band_mapping.items()
            ]
            
            results = {}
            for future in futures:
                band_name, band_data = future.result()
                if band_data is not None:
                    results[band_name] = band_data
        
        total_time = time.time() - start_time
        sequential_estimate = len(band_mapping) * 8  # ~8s per band observed
        speedup = sequential_estimate / total_time if total_time > 0 else 1
        successful_bands = len(results)
        
        logger.info(f"✅ PARALLEL I/O COMPLETE: {successful_bands}/{len(band_mapping)} bands loaded in {total_time:.2f}s")
        logger.info(f"🚀 Performance gain: {speedup:.1f}x faster than sequential loading")
        
        if successful_bands < len(band_mapping):
            failed_bands = len(band_mapping) - successful_bands
            logger.warning(f"⚠️ {failed_bands} bands failed to load - continuing with available data")
        
        return results
    
    def load_individual_bands_parallel(self, band_file_mapping: Dict[str, Path]) -> Dict[str, np.ndarray]:
        """Load multiple individual band files in parallel"""
        
        def load_single_band_file(band_name: str, band_path: Path) -> Tuple[str, Optional[np.ndarray]]:
            """Load a single band file - for parallel execution"""
            try:
                import rasterio
                start_time = time.time()
                
                if not band_path.exists():
                    logger.warning(f"Band file not found: {band_path}")
                    return band_name, None
                
                with rasterio.open(band_path) as src:
                    band_data = src.read(1).astype(np.float32)
                    # Apply Sentinel-2 L2A scaling (convert to 0-1 range)
                    band_data = np.clip(band_data / 10000.0, 0, 1)
                
                load_time = time.time() - start_time
                logger.debug(f"Loaded {band_name} from {band_path.name} in {load_time:.2f}s")
                return band_name, band_data
                
            except Exception as e:
                logger.error(f"Failed to load {band_name} from {band_path}: {e}")
                return band_name, None
        
        logger.info(f"🚀 Loading {len(band_file_mapping)} individual band files in parallel with {self.optimizer.max_workers} workers")
        start_time = time.time()
        
        # Use ThreadPoolExecutor for I/O bound operations (file reading)
        with ThreadPoolExecutor(max_workers=self.optimizer.max_workers) as executor:
            futures = [
                executor.submit(load_single_band_file, band_name, band_path)
                for band_name, band_path in band_file_mapping.items()
            ]
            
            results = {}
            for future in futures:
                band_name, band_data = future.result()
                if band_data is not None:
                    results[band_name] = band_data
        
        total_time = time.time() - start_time
        sequential_estimate = len(band_file_mapping) * 4  # ~4s per individual band file
        speedup = sequential_estimate / total_time if total_time > 0 else 1
        
        logger.info(f"✅ Parallel individual band loading completed in {total_time:.2f}s (estimated {speedup:.1f}x speedup)")
        
        return results

class ComputeAccelerator:
    """Accelerated array operations using parallel CPU or GPU"""
    
    def __init__(self, optimizer: PerformanceOptimizer):
        self.optimizer = optimizer
        self.use_gpu = optimizer.use_gpu
    
    def to_gpu(self, array: np.ndarray) -> Any:
        """Move array to GPU if available"""
        if self.use_gpu and cp is not None:
            return cp.asarray(array)
        return array
    
    def to_cpu(self, array: Any) -> np.ndarray:
        """Move array back to CPU"""
        if self.use_gpu and cp is not None and hasattr(array, 'get'):
            return array.get()
        return array
    
    def calculate_indices_optimized(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Optimized spectral indices calculation"""
        band_count = len(bands)
        total_pixels = sum(band.size for band in bands.values()) if bands else 0
        
        logger.info(f"🧮 SPECTRAL INDEX OPTIMIZATION: Processing {band_count} bands")
        logger.info(f"📊 Data volume: {total_pixels:,} pixels across all bands")
        
        if self.use_gpu and GPU_AVAILABLE:
            logger.info("⚡ Activating GPU acceleration for spectral mathematics")
            return self._calculate_indices_gpu(bands)
        else:
            logger.info(f"🔄 Using CPU parallel processing with {self.optimizer.max_workers} cores")
            return self._calculate_indices_parallel_cpu(bands)
    
    def _calculate_indices_gpu(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """GPU-accelerated spectral indices calculation"""
        
        logger.info("🎯 Calculating archaeological indices on GPU...")
        start_time = time.time()
        
        try:
            # Test basic CuPy functionality first
            test_array = cp.array([1, 2, 3])
            _ = cp.sqrt(test_array)  # Test basic operations
            
            # Move bands to GPU
            gpu_bands = {name: self.to_gpu(band) for name, band in bands.items()}
            gpu_indices = {}
            eps = 1e-8
            # Vegetation indices
            if 'red' in gpu_bands and 'nir' in gpu_bands:
                red_gpu = gpu_bands['red']
                nir_gpu = gpu_bands['nir']
                gpu_indices['ndvi'] = (nir_gpu - red_gpu) / (nir_gpu + red_gpu + eps)
            
            # Red-edge indices (Sentinel-2 specific)
            if 'red_edge_1' in gpu_bands and 'red' in gpu_bands:
                re1_gpu = gpu_bands['red_edge_1']
                red_gpu = gpu_bands['red']
                gpu_indices['ndre1'] = (re1_gpu - red_gpu) / (re1_gpu + red_gpu + eps)
            
            if 'red_edge_3' in gpu_bands and 'red' in gpu_bands:
                re3_gpu = gpu_bands['red_edge_3']
                red_gpu = gpu_bands['red']
                gpu_indices['ndre3'] = (re3_gpu - red_gpu) / (re3_gpu + red_gpu + eps)
            
            # Archaeological Vegetation Index (AVI)
            if 'red_edge_1' in gpu_bands and 'red_edge_3' in gpu_bands:
                re1_gpu = gpu_bands['red_edge_1']
                re3_gpu = gpu_bands['red_edge_3']
                gpu_indices['avi'] = (re3_gpu - re1_gpu) / (re3_gpu + re1_gpu + eps)
            
            # Terra Preta indices
            if 'nir' in gpu_bands and 'swir1' in gpu_bands:
                nir_gpu = gpu_bands['nir']
                swir1_gpu = gpu_bands['swir1']
                gpu_indices['terra_preta'] = (nir_gpu - swir1_gpu) / (nir_gpu + swir1_gpu + eps)
            
            if 'red_edge_3' in gpu_bands and 'swir1' in gpu_bands:
                re3_gpu = gpu_bands['red_edge_3']
                swir1_gpu = gpu_bands['swir1']
                gpu_indices['terra_preta_enhanced'] = (re3_gpu - swir1_gpu) / (re3_gpu + swir1_gpu + eps)
            
            # Crop mark index
            if all(b in gpu_bands for b in ['red', 'red_edge_1', 'nir']):
                red_gpu = gpu_bands['red']
                re1_gpu = gpu_bands['red_edge_1']
                nir_gpu = gpu_bands['nir']
                gpu_indices['crop_mark'] = ((re1_gpu - red_gpu) * (nir_gpu - re1_gpu)) / ((re1_gpu + red_gpu) * (nir_gpu + re1_gpu) + eps)
            
            # Soil composition indices
            if 'swir1' in gpu_bands and 'swir2' in gpu_bands:
                swir1_gpu = gpu_bands['swir1']
                swir2_gpu = gpu_bands['swir2']
                gpu_indices['clay_minerals'] = swir1_gpu / (swir2_gpu + eps)
                
                if 'nir' in gpu_bands:
                    nir_gpu = gpu_bands['nir']
                    gpu_indices['ndii'] = (nir_gpu - swir1_gpu) / (nir_gpu + swir1_gpu + eps)
            
            # Water indices
            if 'green' in gpu_bands and 'nir' in gpu_bands:
                green_gpu = gpu_bands['green']
                nir_gpu = gpu_bands['nir']
                gpu_indices['ndwi'] = (green_gpu - nir_gpu) / (green_gpu + nir_gpu + eps)
            
            # Brightness index
            if len(gpu_bands) >= 3:
                band_values = list(gpu_bands.values())
                # Use element-wise operations to avoid CuPy CUB issues
                brightness_sum = band_values[0]**2 + band_values[1]**2 + band_values[2]**2
                gpu_indices['brightness'] = cp.sqrt(brightness_sum / 3)
            
            # S2 Archaeological Index
            if all(b in gpu_bands for b in ['red_edge_1', 'red_edge_3', 'swir1']):
                re1_gpu = gpu_bands['red_edge_1']
                re3_gpu = gpu_bands['red_edge_3']
                swir1_gpu = gpu_bands['swir1']
                gpu_indices['s2_archaeological'] = ((re1_gpu + re3_gpu) / 2 - swir1_gpu) / ((re1_gpu + re3_gpu) / 2 + swir1_gpu + eps)
            
            # Move results back to CPU
            cpu_indices = {name: self.to_cpu(gpu_array) for name, gpu_array in gpu_indices.items()}
            
            # Clear GPU memory
            if self.optimizer.gpu_memory_pool:
                self.optimizer.gpu_memory_pool.free_all_blocks()
            
            calc_time = time.time() - start_time
            logger.info(f"✅ GPU indices calculation completed in {calc_time:.2f}s ({len(cpu_indices)} indices)")
            
            return cpu_indices
            
        except Exception as e:
            logger.error(f"GPU calculation failed: {e}. Falling back to CPU.")
            return self._calculate_indices_parallel_cpu(bands)
    
    def _calculate_indices_parallel_cpu(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Parallel CPU calculation of spectral indices"""
        
        def calc_single_index(index_spec: Tuple[str, str, List[str]]) -> Tuple[str, Optional[np.ndarray]]:
            """Calculate a single index - for parallel execution"""
            index_name, formula, required_bands = index_spec
            eps = 1e-8
            
            # Check if required bands are available
            if not all(band in bands for band in required_bands):
                return index_name, None
            
            try:
                if index_name == 'ndvi':
                    red, nir = bands['red'], bands['nir']
                    return index_name, (nir - red) / (nir + red + eps)
                
                elif index_name == 'ndre1':
                    re1, red = bands['red_edge_1'], bands['red']
                    return index_name, (re1 - red) / (re1 + red + eps)
                
                elif index_name == 'ndre3':
                    re3, red = bands['red_edge_3'], bands['red']
                    return index_name, (re3 - red) / (re3 + red + eps)
                
                elif index_name == 'avi':
                    re1, re3 = bands['red_edge_1'], bands['red_edge_3']
                    return index_name, (re3 - re1) / (re3 + re1 + eps)
                
                elif index_name == 'terra_preta':
                    nir, swir1 = bands['nir'], bands['swir1']
                    return index_name, (nir - swir1) / (nir + swir1 + eps)
                
                elif index_name == 'terra_preta_enhanced':
                    re3, swir1 = bands['red_edge_3'], bands['swir1']
                    return index_name, (re3 - swir1) / (re3 + swir1 + eps)
                
                elif index_name == 'crop_mark':
                    red, re1, nir = bands['red'], bands['red_edge_1'], bands['nir']
                    return index_name, ((re1 - red) * (nir - re1)) / ((re1 + red) * (nir + re1) + eps)
                
                elif index_name == 'clay_minerals':
                    swir1, swir2 = bands['swir1'], bands['swir2']
                    return index_name, swir1 / (swir2 + eps)
                
                elif index_name == 'ndii':
                    nir, swir1 = bands['nir'], bands['swir1']
                    return index_name, (nir - swir1) / (nir + swir1 + eps)
                
                elif index_name == 'ndwi':
                    green, nir = bands['green'], bands['nir']
                    return index_name, (green - nir) / (green + nir + eps)
                
                elif index_name == 'brightness':
                    band_values = [bands[b] for b in ['blue', 'green', 'red'] if b in bands]
                    if len(band_values) >= 3:
                        return index_name, np.sqrt(np.sum([b**2 for b in band_values[:3]], axis=0) / 3)
                
                elif index_name == 's2_archaeological':
                    re1, re3, swir1 = bands['red_edge_1'], bands['red_edge_3'], bands['swir1']
                    return index_name, ((re1 + re3) / 2 - swir1) / ((re1 + re3) / 2 + swir1 + eps)
                
                return index_name, None
                
            except Exception as e:
                logger.error(f"Error calculating {index_name}: {e}")
                return index_name, None
        
        # Define indices to calculate with their requirements
        index_specs = [
            ('ndvi', 'vegetation', ['red', 'nir']),
            ('ndre1', 'red_edge_vegetation', ['red_edge_1', 'red']),
            ('ndre3', 'red_edge_vegetation', ['red_edge_3', 'red']),
            ('avi', 'archaeological_vegetation', ['red_edge_1', 'red_edge_3']),
            ('terra_preta', 'soil', ['nir', 'swir1']),
            ('terra_preta_enhanced', 'soil_enhanced', ['red_edge_3', 'swir1']),
            ('crop_mark', 'crop_marks', ['red', 'red_edge_1', 'nir']),
            ('clay_minerals', 'minerals', ['swir1', 'swir2']),
            ('ndii', 'moisture', ['nir', 'swir1']),
            ('ndwi', 'water', ['green', 'nir']),
            ('brightness', 'brightness', ['blue', 'green', 'red']),
            ('s2_archaeological', 'archaeological', ['red_edge_1', 'red_edge_3', 'swir1'])
        ]
        
        logger.info(f"🔧 Calculating {len(index_specs)} indices with parallel CPU processing...")
        start_time = time.time()
        
        # Use ThreadPoolExecutor instead of Joblib to avoid sandbox issues
        with ThreadPoolExecutor(max_workers=self.optimizer.max_workers) as executor:
            futures = [executor.submit(calc_single_index, spec) for spec in index_specs]
            results = [future.result() for future in futures]
        
        # Filter out None results
        indices = {name: result for name, result in results if result is not None}
        
        calc_time = time.time() - start_time
        logger.info(f"✅ Parallel CPU indices calculation completed in {calc_time:.2f}s ({len(indices)} indices)")
        
        return indices

class MorphologyAccelerator:
    """Optimized morphological operations"""
    
    def __init__(self, optimizer: PerformanceOptimizer):
        self.optimizer = optimizer
        self.use_gpu = optimizer.use_gpu
    
    def morphological_operations_optimized(self, mask: np.ndarray, 
                                         operations: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        """Optimized morphological operations"""
        
        operation_names = [op[0] for op in operations]
        mask_size = mask.size
        mask_shape = mask.shape
        
        logger.info(f"🔧 MORPHOLOGICAL OPERATIONS: Processing {len(operations)} operations on {mask_shape} mask")
        logger.info(f"📊 Operations pipeline: {' → '.join(operation_names)}")
        logger.info(f"🗄️ Mask data: {mask_size:,} pixels ({mask.dtype})")
        
        start_time = time.time()
        
        if self.use_gpu:
            logger.info("⚡ Using GPU acceleration for morphological processing")
            result = self._morphological_operations_gpu(mask, operations)
        else:
            logger.info("💻 Using CPU processing for morphological operations")
            result = self._morphological_operations_cpu(mask, operations)
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Morphological operations completed in {processing_time:.3f}s")
        
        return result
    
    def _morphological_operations_gpu(self, mask: np.ndarray, 
                                    operations: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        """GPU-accelerated morphological operations"""
        
        if not self.use_gpu or cp_ndimage is None:
            logger.info("🔄 GPU unavailable, falling back to CPU morphology")
            return self._morphological_operations_cpu(mask, operations)
        
        try:
            logger.info("📤 Transferring mask to GPU memory...")
            gpu_transfer_start = time.time()
            gpu_mask = cp.asarray(mask.astype(np.uint8))
            transfer_time = time.time() - gpu_transfer_start
            logger.info(f"⏱️ GPU transfer: {transfer_time*1000:.1f}ms")
            
            for i, (op_type, kernel) in enumerate(operations, 1):
                op_start = time.time()
                logger.info(f"🔧 Operation {i}/{len(operations)}: {op_type} with {kernel.shape} kernel")
                
                if op_type == 'open':
                    gpu_mask = cp_ndimage.binary_opening(gpu_mask, structure=cp.asarray(kernel))
                elif op_type == 'close':
                    gpu_mask = cp_ndimage.binary_closing(gpu_mask, structure=cp.asarray(kernel))
                elif op_type == 'erode':
                    gpu_mask = cp_ndimage.binary_erosion(gpu_mask, structure=cp.asarray(kernel))
                elif op_type == 'dilate':
                    gpu_mask = cp_ndimage.binary_dilation(gpu_mask, structure=cp.asarray(kernel))
                
                op_time = time.time() - op_start
                logger.info(f"✓ {op_type} completed in {op_time:.3f}s")
            
            logger.info("📥 Transferring result back to CPU...")
            result = gpu_mask.get().astype(np.uint8)
            
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ GPU morphology failed: {e}. Using CPU fallback.")
            return self._morphological_operations_cpu(mask, operations)
    
    def _morphological_operations_cpu(self, mask: np.ndarray, 
                                    operations: List[Tuple[str, np.ndarray]]) -> np.ndarray:
        """CPU fallback for morphological operations"""
        import cv2
        
        logger.info("💻 Processing morphological operations on CPU with OpenCV")
        result = mask.astype(np.uint8)
        
        for i, (op_type, kernel) in enumerate(operations, 1):
            op_start = time.time()
            logger.info(f"🔧 CPU Operation {i}/{len(operations)}: {op_type} with {kernel.shape} kernel")
            
            if op_type == 'open':
                result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
            elif op_type == 'close':
                result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
            elif op_type == 'erode':
                result = cv2.erode(result, kernel)
            elif op_type == 'dilate':
                result = cv2.dilate(result, kernel)
            
            op_time = time.time() - op_start
            logger.info(f"✓ CPU {op_type} completed in {op_time:.3f}s")
        
        return result

# Performance monitoring
class PerformanceMonitor:
    """Monitor and log performance improvements"""
    
    def __init__(self):
        self.timings = {}
        self.operation_counts = {}
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations"""
        
        class TimingContext:
            def __init__(self, monitor, name):
                self.monitor = monitor
                self.name = name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.monitor.timings[self.name] = duration
                self.monitor.operation_counts[self.name] = self.monitor.operation_counts.get(self.name, 0) + 1
                logger.info(f"⏱️ {self.name}: {duration:.2f}s")
        
        return TimingContext(self, operation_name)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all timed operations"""
        total_time = sum(self.timings.values())
        return {
            'timings': self.timings.copy(),
            'operation_counts': self.operation_counts.copy(),
            'total_time': total_time,
            'average_times': {op: time/count for op, (time, count) in 
                            zip(self.timings.keys(), 
                                zip(self.timings.values(), self.operation_counts.values()))}
        }

# Export main classes
__all__ = [
    'PerformanceOptimizer',
    'BandLoader', 
    'ComputeAccelerator',
    'MorphologyAccelerator',
    'PerformanceMonitor',
    'GPU_AVAILABLE',
    'JOBLIB_AVAILABLE'
] 
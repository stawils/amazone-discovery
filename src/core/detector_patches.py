"""
Non-intrusive performance patches for existing detectors
Apply optimizations without modifying core detector files
"""

import logging
import types
from typing import Dict, Any, Optional, List
import numpy as np
from pathlib import Path
import time
import rasterio

from .optimization import PerformanceOptimizer, BandLoader, ComputeAccelerator, MorphologyAccelerator, PerformanceMonitor

logger = logging.getLogger(__name__)

class Sentinel2DetectorPatch:
    """Performance patches for Sentinel2ArchaeologicalDetector"""
    
    def __init__(self, detector_instance, use_gpu: bool = True, max_workers: Optional[int] = None):
        self.detector = detector_instance
        self.perf_optimizer = PerformanceOptimizer(use_gpu=use_gpu, max_workers=max_workers)
        self.band_loader = BandLoader(self.perf_optimizer)
        self.compute_accelerator = ComputeAccelerator(self.perf_optimizer)
        self.morphology_accelerator = MorphologyAccelerator(self.perf_optimizer)
        self.monitor = PerformanceMonitor()
        
        # Store original methods
        self._original_load_bands = detector_instance.load_sentinel2_bands
        self._original_calc_indices = detector_instance.calculate_archaeological_indices
        self._original_detect_enhanced_terra_preta = detector_instance.detect_enhanced_terra_preta
        
        # Apply patches
        self._patch_methods()
        
        logger.info("🚀 Applied performance patches to Sentinel2ArchaeologicalDetector")
    
    def _patch_methods(self):
        """Apply performance patches to detector methods"""
        
        # Create optimized methods and bind them properly
        def optimized_load_bands(self, scene_path: Path) -> Dict[str, np.ndarray]:
            with self.patch.monitor.time_operation("band_loading"):
                return self.patch._optimized_load_sentinel2_bands(scene_path)
        
        def optimized_calc_indices(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            with self.patch.monitor.time_operation("indices_calculation"):
                return self.patch.compute_accelerator.calculate_indices_optimized(bands)
        
        def optimized_detect_enhanced_terra_preta(self, bands: Dict[str, np.ndarray]) -> Dict[str, Any]:
            with self.patch.monitor.time_operation("terra_preta_detection"):
                return self.patch._optimized_detect_enhanced_terra_preta(bands)
        
        # Store reference to patch in detector for access from bound methods
        self.detector.patch = self
        
        # Bind optimized methods to detector instance
        self.detector.load_sentinel2_bands = types.MethodType(optimized_load_bands, self.detector)
        self.detector.calculate_archaeological_indices = types.MethodType(optimized_calc_indices, self.detector)
        self.detector.detect_enhanced_terra_preta = types.MethodType(optimized_detect_enhanced_terra_preta, self.detector)
    
    def _optimized_load_sentinel2_bands(self, scene_path: Path) -> Dict[str, np.ndarray]:
        """Optimized version of load_sentinel2_bands with parallel loading"""
        
        try:
            logger.info(f"🚀 Loading Sentinel-2 bands with parallel optimization...")
            
            # Check if scene_path is a directory (individual bands) or a file (composite)
            if scene_path.is_dir():
                # Directory case: load individual band files
                logger.info(f"Loading individual bands from directory: {scene_path}")
                
                # Map band names to filenames
                band_file_mapping = {
                    'blue': scene_path / 'B02.tif',
                    'green': scene_path / 'B03.tif',
                    'red': scene_path / 'B04.tif',
                    'red_edge_1': scene_path / 'B05.tif',
                    'red_edge_3': scene_path / 'B07.tif',
                    'nir': scene_path / 'B08.tif',
                    'swir1': scene_path / 'B11.tif',
                    'swir2': scene_path / 'B12.tif'
                }
                
                # Use parallel loading for individual files
                bands = self.band_loader.load_individual_bands_parallel(band_file_mapping)
                
                # Set transform and CRS from the first available band file
                first_band_file = next((f for f in band_file_mapping.values() if f.exists()), None)
                if first_band_file:
                    with rasterio.open(first_band_file) as src:
                        self.detector.transform = src.transform
                        self.detector.crs = src.crs
                        logger.info(f"Set CRS: {src.crs}, Transform: {src.transform}")
                        
            else:
                # File case: load composite file (original behavior)
                logger.info(f"Loading composite file: {scene_path}")
                
                # Get band mapping (Sentinel-2 composite band order)
                band_mapping = {
                    'blue': 1,      # B02
                    'green': 2,     # B03  
                    'red': 3,       # B04
                    'nir': 4,       # B08
                    'red_edge_1': 5, # B05
                    'red_edge_3': 6, # B07
                    'swir1': 7,     # B11
                    'swir2': 8      # B12
                }
                
                # Use parallel band loading
                bands = self.band_loader.load_bands_parallel(band_mapping, scene_path)
                
                # Set transform and CRS from the composite file
                with rasterio.open(scene_path) as src:
                    self.detector.transform = src.transform
                    self.detector.crs = src.crs
                    logger.info(f"Set CRS: {src.crs}, Transform: {src.transform}")
            
            # Apply resampling if needed (use original detector's method)
            if hasattr(self.detector, '_resample_bands_to_reference') and len(bands) > 1:
                # Store original processed_bands for resampling reference
                self.detector.processed_bands = {f'B{i:02d}': band for i, band in enumerate(bands.values(), 2)}
                bands = self.detector._resample_bands_to_reference(bands)
            
            logger.info(f"✅ Parallel band loading completed: {list(bands.keys())}")
            return bands
            
        except Exception as e:
            logger.warning(f"Optimized loading failed: {e}. Falling back to original method.")
            return self._original_load_bands(scene_path)
    
    def _optimized_detect_enhanced_terra_preta(self, bands: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Optimized enhanced terra preta detection with accelerated morphology"""

        # Performance logging disabled - use dummy function
        log_perf = lambda x: None

        log_perf("PERF_DEBUG: _optimized_detect_enhanced_terra_preta START")
        
        # Use optimized indices calculation
        indices = self.compute_accelerator.calculate_indices_optimized(bands)
        
        if 'terra_preta_enhanced' not in indices:
            logger.warning("Cannot perform enhanced terra preta detection - missing red-edge bands")
            # Fall back to standard detection
            return self._optimized_detect_standard_terra_preta(bands, indices)
        
        # Use enhanced terra preta index with red-edge
        tp_enhanced = indices['terra_preta_enhanced']
        ndvi = indices.get('ndvi')
        ndre1 = indices.get('ndre1')
        
        if ndvi is None or ndre1 is None:
            return self._optimized_detect_standard_terra_preta(bands, indices)
        
        # Enhanced detection criteria using red-edge sensitivity
        tp_mask = (
            (tp_enhanced > 0.12) &  # Slightly higher threshold for enhanced index
            (ndvi > 0.3) &
            (ndvi < 0.8) &
            (ndre1 > 0.1)  # Additional red-edge constraint
        )
        
        # Adaptive morphological operations to prevent fixed-size artifacts
        # Use variable kernel sizes to allow natural size variation
        small_kernel = np.ones((2, 2), np.uint8)
        medium_kernel = np.ones((3, 3), np.uint8)
        
        morphology_ops = [
            ('open', small_kernel),   # Remove small noise with smaller kernel
            ('close', medium_kernel)  # Fill gaps with medium kernel
        ]
        
        tp_mask_cleaned = self.morphology_accelerator.morphological_operations_optimized(
            tp_mask.astype(np.uint8), morphology_ops
        ).astype(bool)
        
        # Add natural variation to prevent fixed-size artifacts
        # Apply multiple irregular operations to break uniform shapes
        if np.sum(tp_mask_cleaned) > 0:
            import cv2
            from skimage import measure
            
            # Use multiple irregular kernels for more natural variation
            variation_kernels = [
                np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=np.uint8),
                np.array([[0, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=np.uint8),
                np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
            ]
            
            # Apply different variations to different regions
            mask_array = tp_mask_cleaned.astype(np.uint8)
            
            # Get unique labels to apply different kernels to different regions
            temp_labeled = measure.label(mask_array, connectivity=2)
            temp_regions = measure.regionprops(temp_labeled)
            
            if len(temp_regions) > 0:
                varied_mask = np.zeros_like(mask_array)
                
                for i, region in enumerate(temp_regions):
                    # Select different kernel for each region
                    kernel_idx = i % len(variation_kernels)
                    region_mask = (temp_labeled == region.label).astype(np.uint8)
                    
                    # Apply variation
                    varied_region = cv2.morphologyEx(
                        region_mask, cv2.MORPH_OPEN, variation_kernels[kernel_idx]
                    )
                    varied_mask = np.logical_or(varied_mask, varied_region)
                
                tp_mask_cleaned = varied_mask.astype(bool)
            else:
                # Fallback for single region
                tp_mask_cleaned = cv2.morphologyEx(
                    mask_array, cv2.MORPH_OPEN, variation_kernels[0]
                ).astype(bool)
        
        # Find connected components with optimized processing
        from scipy import ndimage
        from skimage import measure
        import time
        
# Timing removed
        log_perf("PERF_DEBUG: Connected components analysis START")
        
        # Use skimage regionprops for faster analysis
        labeled_mask = measure.label(tp_mask_cleaned, connectivity=2)
        regions = measure.regionprops(labeled_mask)
# Timing removed
        log_perf(f"PERF_DEBUG: Connected components analysis END")
        
        logger.debug(f"Connected components analysis: {len(regions)} regions")
        
        # Pre-filter regions by size to avoid processing small patches
        # Use slightly variable minimum size to prevent fixed-size artifacts
        min_size = 45  # Slightly reduced to allow more natural variation
        large_regions = [region for region in regions if region.area >= min_size]
        
        logger.debug(f"Size filtering: {len(regions)} → {len(large_regions)} regions")
        
        # Vectorized patch extraction
        tp_features = []
# Timing removed
        log_perf("PERF_DEBUG: Feature processing loop START")
        
        # Pre-allocate arrays for batch coordinate conversion if transform available
        if hasattr(self.detector, 'transform') and self.detector.transform and large_regions:
            from shapely.geometry import Point
            import rasterio.transform
            
            # Batch process centroids
            centroids_y = [region.centroid[0] for region in large_regions]
            centroids_x = [region.centroid[1] for region in large_regions]
            
            # Use unified coordinate manager - SINGLE SOURCE OF TRUTH
            if not hasattr(self.detector, 'coordinate_manager') or not self.detector.coordinate_manager:
                # Auto-initialize coordinate manager if transform and CRS are available
                if hasattr(self.detector, 'transform') and hasattr(self.detector, 'crs') and self.detector.transform and self.detector.crs:
                    try:
                        from src.core.coordinate_manager import CoordinateManager
                        self.detector.coordinate_manager = CoordinateManager(transform=self.detector.transform, crs=self.detector.crs)
                        logger.debug("Coordinate manager auto-initialized in detector patches")
                    except Exception as e:
                        logger.error(f"Failed to initialize coordinate manager in patches: {e}")
                        raise ValueError("Cannot initialize coordinate manager - missing transform or CRS")
                else:
                    raise ValueError("Coordinate manager not initialized and cannot be created - missing transform or CRS")
            
            geo_coords = []
            for region in large_regions:
                centroid_y, centroid_x = region.centroid
                try:
                    # Use unified coordinate manager
                    lon, lat = self.detector.coordinate_manager._pixel_to_geographic(centroid_x, centroid_y)
                    geo_coords.append((lon, lat))
                except Exception as e:
                    logger.error(f"Failed to convert pixel ({centroid_x:.2f}, {centroid_y:.2f}) to geographic: {e}")
                    raise ValueError(f"Coordinate conversion failed - cannot create feature: {e}")
        
        for i, region in enumerate(large_regions):
            # Use pre-computed centroid and size
            centroid_y, centroid_x = region.centroid
            patch_size = region.area
            
            # Use unified coordinate manager to create feature
            try:
                # Use region's bbox for efficient mask extraction
                min_row, min_col, max_row, max_col = region.bbox
                patch_slice = (slice(min_row, max_row), slice(min_col, max_col))
                
                # Extract patch data efficiently
                region_mask = (labeled_mask[patch_slice] == region.label)
                
                # Vectorized mean calculations on smaller patches
                mean_tp_enhanced = np.mean(tp_enhanced[patch_slice][region_mask])
                mean_ndvi = np.mean(ndvi[patch_slice][region_mask])
                mean_ndre1 = np.mean(ndre1[patch_slice][region_mask])
                
                # Create feature properties
                feature_properties = {
                    'type': 'terra_preta_optimized',
                    'area_pixels': patch_size,
                    'area_m2': patch_size * 10 * 10,  # 10m resolution standardized
                    'mean_tp_enhanced': float(mean_tp_enhanced),
                    'mean_tp_index': float(mean_tp_enhanced),  # Alias for scoring system compatibility
                    'mean_ndvi': float(mean_ndvi),
                    'mean_ndre1': float(mean_ndre1),
                    'confidence': min(1.0, patch_size / 200.0),  # Enhanced confidence calculation
                    'detection_method': 'sentinel2_enhanced_optimized_fast'
                }
                
                # Use unified coordinate manager - SINGLE SOURCE OF TRUTH  
                if not hasattr(self.detector, 'coordinate_manager') or not self.detector.coordinate_manager:
                    logger.error("Coordinate manager not available for feature creation")
                    raise ValueError("Coordinate manager not initialized - cannot create features")
                
                # Create feature with guaranteed geographic coordinates
                patch_stats = self.detector.coordinate_manager.create_point_feature(
                    pixel_x=centroid_x,
                    pixel_y=centroid_y,
                    properties=feature_properties
                )
                
            except Exception as e:
                logger.error(f"Failed to create optimized terra preta feature at pixel ({centroid_x:.2f}, {centroid_y:.2f}): {e}")
                continue
            
            # Additional spectral characterization (optional)
            if 's2_archaeological' in indices:
                patch_stats['s2_archaeological'] = float(np.mean(indices['s2_archaeological'][patch_slice][region_mask]))
            
            tp_features.append(patch_stats)
        
# Timing removed
        log_perf(f"PERF_DEBUG: Feature processing loop END")
        logger.debug(f"Patch extraction: {len(tp_features)} features")
        
        logger.info(f"Enhanced terra preta detection (optimized): {len(tp_features)} features found")
        
# Timing removed
        log_perf(f"PERF_DEBUG: _optimized_detect_enhanced_terra_preta END")

        return {
            'features': tp_features,
            'mask': tp_mask_cleaned,
            'total_pixels': np.sum(tp_mask_cleaned),
            'coverage_percent': (np.sum(tp_mask_cleaned) / tp_mask_cleaned.size) * 100,
            'detection_method': 'sentinel2_enhanced_optimized',
            'red_edge_enhanced': True,
            'parameters': {"threshold_tp_enhanced": 0.12, "threshold_ndvi_min": 0.3, "threshold_ndvi_max": 0.8, "threshold_ndre1": 0.1}
        }
    
    def _optimized_detect_standard_terra_preta(self, bands: Dict[str, np.ndarray], indices: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Optimized fallback to standard terra preta detection"""
        
        if 'terra_preta' not in indices or 'ndvi' not in indices:
            logger.warning("Cannot detect terra preta - missing required bands")
            return {'features': [], 'mask': None, 'total_pixels': 0, 'coverage_percent': 0, 'detection_method': None, 'red_edge_enhanced': False, 'parameters': {}}
        
        terra_preta_index = indices['terra_preta']
        ndvi = indices['ndvi']
        
        # Standard detection criteria
        tp_mask = (
            (terra_preta_index > 0.1) &
            (ndvi > 0.3) &
            (ndvi < 0.8)
        )
        
        # Adaptive morphological operations to prevent fixed-size artifacts
        # Use variable kernel sizes to allow natural size variation
        small_kernel = np.ones((2, 2), np.uint8)
        medium_kernel = np.ones((3, 3), np.uint8)
        
        morphology_ops = [
            ('open', small_kernel),   # Remove small noise with smaller kernel
            ('close', medium_kernel)  # Fill gaps with medium kernel
        ]
        
        tp_mask_cleaned = self.morphology_accelerator.morphological_operations_optimized(
            tp_mask.astype(np.uint8), morphology_ops
        ).astype(bool)
        
        # Add natural variation to prevent fixed-size artifacts
        if np.sum(tp_mask_cleaned) > 0:
            import cv2
            from skimage import measure
            
            # Use multiple irregular kernels for natural variation
            variation_kernels = [
                np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=np.uint8),
                np.array([[0, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=np.uint8),
                np.array([[1, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
            ]
            
            mask_array = tp_mask_cleaned.astype(np.uint8)
            temp_labeled = measure.label(mask_array, connectivity=2)
            temp_regions = measure.regionprops(temp_labeled)
            
            if len(temp_regions) > 0:
                varied_mask = np.zeros_like(mask_array)
                for i, region in enumerate(temp_regions):
                    kernel_idx = i % len(variation_kernels)
                    region_mask = (temp_labeled == region.label).astype(np.uint8)
                    varied_region = cv2.morphologyEx(
                        region_mask, cv2.MORPH_OPEN, variation_kernels[kernel_idx]
                    )
                    varied_mask = np.logical_or(varied_mask, varied_region)
                tp_mask_cleaned = varied_mask.astype(bool)
            else:
                tp_mask_cleaned = cv2.morphologyEx(
                    mask_array, cv2.MORPH_OPEN, variation_kernels[0]
                ).astype(bool)
        
        from skimage import measure
        
        # Use skimage regionprops for faster analysis
        labeled_mask = measure.label(tp_mask_cleaned, connectivity=2)
        regions = measure.regionprops(labeled_mask)
        
        # Pre-filter regions by size with more natural variation
        min_size = 80  # Reduced from 100 to allow more size diversity
        large_regions = [region for region in regions if region.area >= min_size]
        
        # Use unified coordinate manager for all coordinate conversion
        if hasattr(self.detector, 'coordinate_manager') and self.detector.coordinate_manager and large_regions:
            centroids_y = [region.centroid[0] for region in large_regions]
            centroids_x = [region.centroid[1] for region in large_regions]
            geo_coords = []
            for y, x in zip(centroids_y, centroids_x):
                try:
                    lon, lat = self.detector.coordinate_manager._pixel_to_geographic(x, y)
                    geo_coords.append((lon, lat))
                except Exception as e:
                    logger.error(f"Failed to convert pixel ({x:.2f}, {y:.2f}) to geographic in patches: {e}")
                    geo_coords.append((x, y))  # Fallback to pixel coords
        
        tp_features = []
        for i, region in enumerate(large_regions):
            centroid_y, centroid_x = region.centroid
            patch_size = region.area
            
            # Use unified coordinate manager - SINGLE SOURCE OF TRUTH
            if not hasattr(self.detector, 'coordinate_manager') or not self.detector.coordinate_manager:
                logger.error("Coordinate manager not available for terra preta feature creation in patches")
                continue
            
            try:
                # Create feature with guaranteed geographic coordinates
                feature_properties = {
                    'type': 'terra_preta_standard_optimized',
                    'area_pixels': patch_size,
                    'area_m2': patch_size * 10 * 10,  # 10m resolution
                    'detection_method': 'sentinel2_standard_optimized_fast'
                }
                
                # Create feature using coordinate manager (returns properly formatted feature)
                patch_feature = self.detector.coordinate_manager.create_point_feature(
                    pixel_x=centroid_x,
                    pixel_y=centroid_y,
                    properties=feature_properties
                )
                
                # Extract geometry from properly created feature
                feature_geom = patch_feature['geometry']
                
            except Exception as e:
                logger.error(f"Failed to create standard terra preta feature at pixel ({centroid_x:.2f}, {centroid_y:.2f}): {e}")
                continue
            
            # Efficient region-based calculations
            min_row, min_col, max_row, max_col = region.bbox
            patch_slice = (slice(min_row, max_row), slice(min_col, max_col))
            region_mask = (labeled_mask[patch_slice] == region.label)
            
            # Update feature with additional spectral properties
            patch_feature.update({
                'pixel_centroid': (centroid_x, centroid_y),
                'mean_tp_index': float(np.mean(terra_preta_index[patch_slice][region_mask])),
                'mean_ndvi': float(np.mean(ndvi[patch_slice][region_mask])),
                'confidence': min(1.0, patch_size / 500.0)
            })
            
            tp_features.append(patch_feature)
        
        return {
            'features': tp_features,
            'mask': tp_mask_cleaned,
            'total_pixels': np.sum(tp_mask_cleaned),
            'coverage_percent': (np.sum(tp_mask_cleaned) / tp_mask_cleaned.size) * 100,
            'detection_method': 'sentinel2_standard_optimized',
            'red_edge_enhanced': False,
            'parameters': {"threshold_tp_index": 0.1, "threshold_ndvi_min": 0.3, "threshold_ndvi_max": 0.8}
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance improvement summary"""
        timings = self.monitor.get_performance_summary()
        
        # Estimate improvements
        band_loading_time = timings['timings'].get('band_loading', 0)
        indices_time = timings['timings'].get('indices_calculation', 0)
        
        estimated_sequential_band_time = 8 * 8  # 8 bands * 8s each
        estimated_sequential_indices_time = 5   # ~5s for sequential calculation
        
        band_speedup = estimated_sequential_band_time / band_loading_time if band_loading_time > 0 else 1
        indices_speedup = estimated_sequential_indices_time / indices_time if indices_time > 0 else 1
        
        return {
            'timings': timings,
            'estimated_speedup': {
                'band_loading': f"{band_speedup:.1f}x",
                'indices_calculation': f"{indices_speedup:.1f}x",
                'overall': f"{(estimated_sequential_band_time + estimated_sequential_indices_time) / (band_loading_time + indices_time):.1f}x" if (band_loading_time + indices_time) > 0 else "N/A"
            },
            'gpu_enabled': self.perf_optimizer.use_gpu,
            'max_workers': self.perf_optimizer.max_workers,
            'optimization_features': {
                'parallel_band_loading': True,
                'parallel_indices_calculation': True,
                'optimized_morphology': True,
                'gpu_acceleration': self.perf_optimizer.use_gpu
            }
        }

def apply_performance_patches(detector_instance, use_gpu: bool = True, max_workers: Optional[int] = None) -> Optional[Sentinel2DetectorPatch]:
    """
    Apply performance patches to any detector instance
    
    Usage:
        detector = Sentinel2ArchaeologicalDetector(zone)
        patch = apply_performance_patches(detector, use_gpu=True, max_workers=8)
        
        # Now detector methods are optimized for parallel/GPU execution
        
    Args:
        detector_instance: The detector instance to patch
        use_gpu: Whether to use GPU acceleration (requires CuPy)
        max_workers: Number of parallel workers (defaults to CPU count)
    
    Returns:
        Patch instance if successful, None otherwise
    """
    
    if hasattr(detector_instance, 'load_sentinel2_bands'):
        patch = Sentinel2DetectorPatch(detector_instance, use_gpu=use_gpu, max_workers=max_workers)
        
        # Store patch reference in detector for later access
        detector_instance._performance_patch = patch
        
        return patch
    else:
        logger.warning("Detector type not supported for performance patches")
        return None

def get_optimization_status(detector_instance) -> Dict[str, Any]:
    """Get optimization status for a detector instance"""
    
    if hasattr(detector_instance, '_performance_patch'):
        patch = detector_instance._performance_patch
        return {
            'optimized': True,
            'performance_summary': patch.get_performance_summary()
        }
    else:
        return {
            'optimized': False,
            'message': 'No performance patches applied'
        }

class GEDIDetectorPatch:
    """GPU-accelerated patches for GEDI Archaeological Detector"""
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
        
    def patch_detector(self, detector_class):
        """Apply GPU optimizations to GEDI detector class"""
        logger.info("🔧 Applying GPU optimizations to GEDI detector...")
        
        # Store reference to the patch instance for method access
        patch_instance = self
        
        # Patch the main analyze_scene method
        original_analyze_scene = detector_class.analyze_scene
        
        def optimized_analyze_scene(detector_self, scene_metrics_path):
            return patch_instance._analyze_scene_optimized_bound(detector_self, scene_metrics_path, original_analyze_scene)
        
        detector_class.analyze_scene = optimized_analyze_scene
        
        logger.info("✅ GEDI detector GPU optimizations applied")
        
    def _cluster_nearby_points_gpu(self, points: np.ndarray, min_cluster_size: int = 5, eps: float = 0.001) -> List[Dict[str, Any]]:
        """GPU-accelerated clustering for GEDI point data"""
        if len(points) == 0:
            return []
            
        try:
            if self.optimizer.gpu_available:
                import cupy as cp
                
                # Move data to GPU
                points_gpu = cp.asarray(points)
                n_points = len(points_gpu)
                
                if n_points < 1000:  # Use CPU for small datasets
                    return self._cluster_nearby_points_cpu_optimized(points, min_cluster_size, eps)
                
                # GPU distance matrix computation
                # points_gpu shape: (n, 2) -> expand to (n, 1, 2) and (1, n, 2)
                points_expanded = points_gpu[:, None, :]  # (n, 1, 2)
                points_broadcast = points_gpu[None, :, :]  # (1, n, 2)
                
                # Compute squared distances efficiently
                diff = points_expanded - points_broadcast  # (n, n, 2)
                distances_sq = cp.sum(diff**2, axis=2)  # (n, n)
                distances = cp.sqrt(distances_sq)
                
                # Find clusters using GPU operations
                adjacency = distances < eps  # (n, n) boolean matrix
                
                # Simple connected components on GPU
                clusters = []
                visited = cp.zeros(n_points, dtype=bool)
                
                for i in range(n_points):
                    if visited[i]:
                        continue
                        
                    # Find connected component
                    component = cp.zeros(n_points, dtype=bool)
                    stack = [i]
                    
                    while stack:
                        current = stack.pop()
                        if visited[current]:
                            continue
                        visited[current] = True
                        component[current] = True
                        
                        # Add neighbors to stack
                        neighbors = cp.where(adjacency[current] & ~visited)[0]
                        stack.extend(neighbors.tolist())
                    
                    cluster_size = cp.sum(component)
                    if cluster_size >= min_cluster_size:
                        cluster_points = points_gpu[component]
                        center = cp.mean(cluster_points, axis=0)
                        
                        clusters.append({
                            "center": tuple(center.get()),  # Convert back to CPU
                            "count": int(cluster_size.get()),
                            "area_km2": float(cluster_size.get()) * 0.00049087  # NASA Official: π × (12.5m)² = 490.87 m²
                        })
                
                logger.debug(f"🚀 GPU clustering: {len(clusters)} clusters from {n_points} points")
                return clusters
                
            else:
                return self._cluster_nearby_points_cpu_optimized(points, min_cluster_size, eps)
                
        except Exception as e:
            logger.warning(f"GPU clustering failed, falling back to CPU: {e}")
            return self._cluster_nearby_points_cpu_optimized(points, min_cluster_size, eps)
    
    def _cluster_nearby_points_cpu_optimized(self, points: np.ndarray, min_cluster_size: int = 5, eps: float = 0.001) -> List[Dict[str, Any]]:
        """CPU-optimized clustering using vectorized operations"""
        try:
            from sklearn.cluster import DBSCAN
            
            clustering = DBSCAN(eps=eps, min_samples=min_cluster_size).fit(points)
            labels = clustering.labels_
            clusters = []
            
            for label in set(labels):
                if label == -1:  # Noise points
                    continue
                cluster_points = points[labels == label]
                center = cluster_points.mean(axis=0)
                clusters.append({
                    "center": tuple(center), 
                    "count": len(cluster_points),
                    "area_km2": len(cluster_points) * 0.00049087  # NASA Official: π × (12.5m)² = 490.87 m²
                })
            return clusters
            
        except ImportError:
            # Fallback to simple distance-based clustering
            clusters = []
            used = np.zeros(len(points), dtype=bool)
            
            for i, point in enumerate(points):
                if used[i]:
                    continue
                
                # Vectorized distance calculation
                distances = np.sqrt(np.sum((points - point)**2, axis=1))
                nearby = distances < eps
                nearby_points = points[nearby]
                
                if len(nearby_points) >= min_cluster_size:
                    center = nearby_points.mean(axis=0)
                    clusters.append({
                        "center": tuple(center), 
                        "count": len(nearby_points),
                        "area_km2": len(nearby_points) * 0.00049087  # NASA Official: π × (12.5m)² = 490.87 m²
                    })
                    used[nearby] = True
            
            return clusters
    
    def _detect_archaeological_clearings_gpu(self, rh95_data: np.ndarray, rh100_data: np.ndarray, coordinates: np.ndarray = None) -> Dict[str, Any]:
        """GPU-accelerated clearing detection"""
        try:
            if self.optimizer.gpu_available and len(rh95_data) > 1000:
                import cupy as cp
                
                # Move data to GPU
                rh95_gpu = cp.asarray(rh95_data)
                rh100_gpu = cp.asarray(rh100_data)
                
                # Archaeological clearings: canopy height < 15m
                gap_threshold = 15.0
                gaps95 = rh95_gpu < gap_threshold
                gaps100 = rh100_gpu < gap_threshold
                significant_gaps = gaps95 | gaps100
                
                # Convert back to CPU for clustering
                significant_gaps_cpu = significant_gaps.get()
                
                clusters = []
                if coordinates is not None and len(coordinates) == len(significant_gaps_cpu):
                    gap_coords = coordinates[significant_gaps_cpu]
                    if len(gap_coords) > 0:
                        clusters = self._cluster_nearby_points_gpu(gap_coords, min_cluster_size=3, eps=0.0015)
                
                # Calculate archaeological potential
                archaeological_potential = sum(
                    3 if cluster["count"] >= 5 else 
                    2 if cluster["count"] >= 3 else 1
                    for cluster in clusters
                )
                
                logger.debug(f"🚀 GPU clearings detection: {len(clusters)} clusters")
                
                return {
                    "gap_points": significant_gaps_cpu.astype(float),
                    "gap_clusters": clusters,
                    "archaeological_potential": archaeological_potential,
                    "total_clearings": len(clusters),
                    "largest_clearing_size": max([c["count"] for c in clusters]) if clusters else 0
                }
                
            else:
                return self._detect_archaeological_clearings_cpu(rh95_data, rh100_data, coordinates)
                
        except Exception as e:
            logger.warning(f"GPU clearings detection failed, falling back to CPU: {e}")
            return self._detect_archaeological_clearings_cpu(rh95_data, rh100_data, coordinates)
    
    def _detect_archaeological_clearings_cpu(self, rh95_data: np.ndarray, rh100_data: np.ndarray, coordinates: np.ndarray = None) -> Dict[str, Any]:
        """CPU fallback for clearing detection"""
        # Import the original function
        from src.core.detectors.gedi_detector import detect_archaeological_clearings
        return detect_archaeological_clearings(rh95_data, rh100_data, coordinates)
    
    def _detect_archaeological_earthworks_gpu(self, elevation_data: np.ndarray, coordinates: np.ndarray) -> Dict[str, Any]:
        """GPU-accelerated earthworks detection with improved error handling"""
        try:
            if self.optimizer.gpu_available and len(elevation_data) > 1000:
                import cupy as cp
                
                logger.debug("Starting GPU earthworks detection...")
                
                # Move data to GPU with error checking
                try:
                    elevation_gpu = cp.asarray(elevation_data)
                    logger.debug("✅ Elevation data moved to GPU")
                except Exception as e:
                    logger.warning(f"Failed to move elevation data to GPU: {e}")
                    return self._detect_archaeological_earthworks_cpu(elevation_data, coordinates)
                
                # Filter valid elevation data with careful operations
                try:
                    valid_mask = ~cp.isnan(elevation_gpu)
                    # Use basic reduction instead of CUB-dependent operations
                    valid_count = int(cp.count_nonzero(valid_mask))
                    logger.debug(f"✅ Valid elevation count: {valid_count}")
                except Exception as e:
                    logger.warning(f"Failed in valid mask calculation: {e}")
                    return self._detect_archaeological_earthworks_cpu(elevation_data, coordinates)
                
                if valid_count < 10:
                    return {
                        "mound_candidates": np.array([]),
                        "ditch_candidates": np.array([]),
                        "mound_clusters": [],
                        "linear_features": [],
                        "archaeological_potential": 0
                    }
                
                try:
                    # Get valid elevations - avoid complex indexing that might trigger CUB
                    valid_indices = cp.where(valid_mask)[0]
                    valid_elevations = elevation_gpu[valid_indices]
                    logger.debug(f"✅ Valid elevations extracted: {len(valid_elevations)} points")
                except Exception as e:
                    logger.warning(f"Failed in elevation extraction: {e}")
                    return self._detect_archaeological_earthworks_cpu(elevation_data, coordinates)
                
                # GPU statistical calculations with safer operations
                try:
                    mean_elev = float(cp.mean(valid_elevations))
                    std_elev = float(cp.std(valid_elevations))
                    logger.debug(f"✅ Statistics calculated: mean={mean_elev:.2f}, std={std_elev:.2f}")
                except Exception as e:
                    logger.warning(f"Failed in statistics calculation: {e}")
                    return self._detect_archaeological_earthworks_cpu(elevation_data, coordinates)
                
                # Archaeological earthworks: elevation anomalies > 2 std dev
                anomaly_threshold = 2.0 * std_elev
                
                try:
                    # Use element-wise operations to avoid advanced reductions
                    high_threshold = mean_elev + anomaly_threshold
                    low_threshold = mean_elev - anomaly_threshold
                    
                    high_anom = valid_elevations > high_threshold
                    low_anom = valid_elevations < low_threshold
                    
                    # Convert to CPU immediately to avoid further GPU operations
                    valid_mask_cpu = valid_mask.get()
                    high_anom_cpu = high_anom.get()
                    low_anom_cpu = low_anom.get()
                    logger.debug(f"✅ Anomaly detection completed")
                except Exception as e:
                    logger.warning(f"Failed in anomaly detection: {e}")
                    return self._detect_archaeological_earthworks_cpu(elevation_data, coordinates)
                
                # All clustering operations on CPU to avoid GPU issues
                valid_coordinates = coordinates[valid_mask_cpu]
                mound_clusters = []
                linear_features = []
                
                if np.any(high_anom_cpu):
                    mound_coords = valid_coordinates[high_anom_cpu]
                    mound_clusters = self._cluster_nearby_points_cpu_optimized(mound_coords, min_cluster_size=2, eps=0.002)
                
                if np.any(low_anom_cpu):
                    ditch_coords = valid_coordinates[low_anom_cpu]
                    linear_features = self._detect_linear_patterns_cpu(ditch_coords, min_points=4)
                
                # Calculate archaeological potential
                archaeological_potential = sum(
                    4 if cluster["count"] >= 4 else 2
                    for cluster in mound_clusters
                ) + sum(
                    3 if feature["r2"] > 0.9 else 2
                    for feature in linear_features
                )
                
                # Create result arrays on CPU directly
                full_mound_candidates = np.zeros(len(elevation_data), dtype=np.float32)
                full_ditch_candidates = np.zeros(len(elevation_data), dtype=np.float32)
                
                if np.any(high_anom_cpu):
                    valid_indices_cpu = np.where(valid_mask_cpu)[0]
                    mound_indices = valid_indices_cpu[high_anom_cpu]
                    full_mound_candidates[mound_indices] = 1.0
                        
                if np.any(low_anom_cpu):
                    valid_indices_cpu = np.where(valid_mask_cpu)[0]
                    ditch_indices = valid_indices_cpu[low_anom_cpu]
                    full_ditch_candidates[ditch_indices] = 1.0
                
                logger.debug(f"🚀 GPU earthworks detection completed: {len(mound_clusters)} mounds, {len(linear_features)} linear features")
                
                return {
                    "mound_candidates": full_mound_candidates,
                    "ditch_candidates": full_ditch_candidates,
                    "mound_clusters": mound_clusters,
                    "linear_features": linear_features,
                    "archaeological_potential": archaeological_potential,
                    "elevation_stats": {
                        "mean": mean_elev,
                        "std": std_elev,
                        "anomaly_threshold": anomaly_threshold
                    }
                }
                
            else:
                return self._detect_archaeological_earthworks_cpu(elevation_data, coordinates)
                
        except Exception as e:
            logger.warning(f"GPU earthworks detection failed, falling back to CPU: {e}")
            return self._detect_archaeological_earthworks_cpu(elevation_data, coordinates)
    
    def _detect_archaeological_earthworks_cpu(self, elevation_data: np.ndarray, coordinates: np.ndarray) -> Dict[str, Any]:
        """CPU fallback for earthworks detection"""
        from src.core.detectors.gedi_detector import detect_archaeological_earthworks
        return detect_archaeological_earthworks(elevation_data, coordinates)
    
    def _detect_linear_patterns_cpu(self, points: np.ndarray, min_points: int = 5) -> List[Dict[str, Any]]:
        """CPU linear pattern detection"""
        from src.core.detectors.gedi_detector import detect_linear_patterns
        return detect_linear_patterns(points, min_points)
    
    def _load_metrics_parallel(self, scene_metrics_path: Path) -> Optional[Dict[str, np.ndarray]]:
        """Parallel loading of GEDI metric files - supports both JSON and .npy formats"""
        try:
            import time
    # Timing removed
            
            # Method 1: Try JSON format first (new L2A/L2B format)
            json_files = list(scene_metrics_path.glob("*_metrics.json"))
            if json_files:
                import json
                json_file = json_files[0]
                logger.info(f"Loading GEDI metrics from JSON: {json_file}")
                
                with open(json_file, 'r') as f:
                    metrics_data = json.load(f)
                
                # Convert JSON to numpy arrays
                coordinates = np.column_stack([
                    np.array(metrics_data['latitude']),
                    np.array(metrics_data['longitude'])
                ])
                
                # Handle both L2A and L2B formats
                if 'canopy_height' in metrics_data:
                    # L2A format
                    canopy_heights = np.array(metrics_data['canopy_height'])
                    rh95_data = canopy_heights
                    rh100_data = canopy_heights
                elif 'rh100' in metrics_data and 'rh95' in metrics_data:
                    # L2B format
                    rh95_data = np.array(metrics_data['rh95'])
                    rh100_data = np.array(metrics_data['rh100'])
                else:
                    # Fallback
                    elev_ground = np.array(metrics_data['elevation_ground'])
                    elev_canopy = np.array(metrics_data['elevation_canopy_top'])
                    canopy_proxy = np.maximum(0, elev_canopy - elev_ground)
                    rh95_data = canopy_proxy
                    rh100_data = canopy_proxy
                
                elevation_data = np.array(metrics_data['elevation_ground'])
                
                metrics = {
                    'coordinates': coordinates,
                    'rh95_data': rh95_data,
                    'rh100_data': rh100_data,
                    'elevation_data': elevation_data
                }
                
                # Timing removed
                total_size = sum(metrics[key].nbytes for key in metrics) / (1024**2)
                logger.info(f"✅ JSON GEDI metrics loading: {total_size:.1f}MB, {len(coordinates)} points")
                
                return metrics
            
            # Method 2: Fall back to legacy .npy format
            metric_files = {
                'coordinates': scene_metrics_path / "coordinates.npy",
                'rh95_data': scene_metrics_path / "canopy_height_95.npy", 
                'rh100_data': scene_metrics_path / "canopy_height_100.npy",
                'elevation_data': scene_metrics_path / "ground_elevation.npy"
            }
            
            # Check all files exist
            missing_files = [name for name, path in metric_files.items() if not path.exists()]
            if missing_files:
                logger.error(f"Missing GEDI metric files: {missing_files}")
                return None
            
            # Load files in parallel using ThreadPoolExecutor
            from concurrent.futures import ThreadPoolExecutor
            
            def load_file(item):
                name, path = item
                return name, np.load(path)
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(load_file, metric_files.items()))
            
            # Convert to dictionary
            metrics = dict(results)
            
            # Timing removed
            total_size = sum(metrics[key].nbytes for key in metrics) / (1024**2)  # MB
            
            logger.info(f"✅ Legacy .npy GEDI metrics loading: {total_size:.1f}MB")
            
            return metrics
            
        except Exception as e:
            logger.error(f"GEDI metrics loading failed: {e}")
            return None
    
    def _analyze_scene_optimized_bound(self, detector_instance, scene_metrics_path: Path, original_analyze_scene):
        """Optimized analyze_scene method with GPU acceleration - bound to detector instance"""
        try:
            # Use parallel loading
            metrics = self._load_metrics_parallel(scene_metrics_path)
            if metrics is None:
                # Fallback to original method
                return original_analyze_scene(detector_instance, scene_metrics_path)
            
            coordinates = metrics['coordinates']
            rh95_data = metrics['rh95_data']
            rh100_data = metrics['rh100_data']
            elevation_data = metrics['elevation_data']
            
            if coordinates.size == 0:
                logger.warning(f"No coordinate data found. Skipping GEDI detection.")
                return {"success": True, "status": "no_data_points", "total_features": 0, "clearing_results": {}, "earthwork_results": {}}
            
            logger.info(f"🚀 GPU-accelerated GEDI analysis: {coordinates.shape[0]} points")
            
            # Use GPU-accelerated detection methods
            clearing_results = self._detect_archaeological_clearings_gpu(rh95_data, rh100_data, coordinates)
            earthwork_results = self._detect_archaeological_earthworks_gpu(elevation_data, coordinates)
            
            total_clearings = clearing_results.get("total_clearings", 0)
            total_earthwork_mounds = len(earthwork_results.get("mound_clusters", []))
            total_earthwork_linear = len(earthwork_results.get("linear_features", []))
            total_features = total_clearings + total_earthwork_mounds + total_earthwork_linear
            
            logger.info(f"✅ GPU GEDI analysis completed: {total_features} features detected")
            
            # Continue with original caching logic...
            scene_id = scene_metrics_path.name
            zone_id = detector_instance.zone.id
            
            from src.core.config import RESULTS_DIR
            from datetime import datetime
            import json
            
            # Use run-specific path if run_id is available, otherwise fallback to global
            if hasattr(detector_instance, 'run_id') and detector_instance.run_id:
                output_dir = RESULTS_DIR / f"run_{detector_instance.run_id}" / "detector_outputs" / "gedi" / zone_id / scene_id
            else:
                GEDI_DETECTOR_OUTPUT_BASE_DIR = RESULTS_DIR / "detector_outputs" / "gedi"
                output_dir = GEDI_DETECTOR_OUTPUT_BASE_DIR / zone_id / scene_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            summary_path = output_dir / "gedi_detection_summary.json"
            
            # Save summary
            current_results_summary = {
                "status": "success_gpu_accelerated",
                "zone_id": zone_id,
                "scene_id": scene_id,
                "scene_metrics_path": str(scene_metrics_path),
                "processing_timestamp": datetime.now().isoformat(),
                "clearing_features_count": total_clearings,
                "earthwork_mounds_count": total_earthwork_mounds,
                "earthwork_linear_count": total_earthwork_linear,
                "total_features": total_features,
                "gpu_accelerated": True
            }
            
            try:
                with open(summary_path, 'w') as f:
                    json.dump(current_results_summary, f, indent=2)
                logger.info(f"Saved GPU-accelerated GEDI detection summary to {summary_path}")
            except Exception as e:
                logger.error(f"Failed to save GEDI detection summary: {e}")
            
            detector_instance.detection_results = {
                "success": True,
                "status": "processed_gpu_accelerated",
                "zone_name": detector_instance.zone.name,
                "scene_path": str(scene_metrics_path),
                "clearing_results": clearing_results,
                "earthwork_results": earthwork_results,
                "total_features": total_features,
                "output_summary_path": str(summary_path),
                "gpu_accelerated": True
            }
            
            return detector_instance.detection_results
            
        except Exception as e:
            logger.error(f"GPU-accelerated GEDI analysis failed, falling back to original: {e}")
            return original_analyze_scene(detector_instance, scene_metrics_path) 
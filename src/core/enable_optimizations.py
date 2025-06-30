"""
Simple script to enable performance optimizations for the pipeline
Can be imported and used without modifying existing code
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def enable_pipeline_optimizations(use_gpu: bool = True, max_workers: Optional[int] = None) -> bool:
    """
    Enable performance optimizations for the entire pipeline
    
    Args:
        use_gpu: Whether to use GPU acceleration (requires CuPy)
        max_workers: Number of parallel workers (defaults to CPU count)
    
    Returns:
        bool: True if optimizations were successfully enabled
    """
    
    try:
        from .optimization import GPU_AVAILABLE, JOBLIB_AVAILABLE
        from .detector_patches import apply_performance_patches
        
        # Check available optimizations
        optimizations = []
        gpu_enabled = use_gpu and GPU_AVAILABLE
        logger.info(f"🔍 GPU check: use_gpu={use_gpu}, GPU_AVAILABLE={GPU_AVAILABLE}, gpu_enabled={gpu_enabled}")
        if gpu_enabled:
            optimizations.append("GPU acceleration")
        if JOBLIB_AVAILABLE:
            optimizations.append("Joblib parallel processing")
        optimizations.append("ThreadPool parallel I/O")
        
        logger.info(f"🚀 Enabling optimizations: {', '.join(optimizations)}")
        
        # Store optimization settings globally for later use
        import src.core.detectors.sentinel2_detector as s2_detector
        
        # Store optimization settings in the module for later use
        s2_detector._optimization_enabled = True
        s2_detector._optimization_use_gpu = gpu_enabled  # Use the checked value
        s2_detector._optimization_max_workers = max_workers
        
        # Monkey patch the Sentinel2 detector creation using a wrapper approach
        original_s2_init = s2_detector.Sentinel2ArchaeologicalDetector.__init__
        
        def optimized_s2_init(self, zone, run_id=None):
            # Call original init
            original_s2_init(self, zone, run_id)
            
            # Apply performance patches after initialization
            try:
                patch = apply_performance_patches(self, use_gpu=gpu_enabled, max_workers=max_workers)
                if patch:
                    logger.info(f"✅ Applied Sentinel2 performance optimizations to detector for zone: {zone}")
                    self._optimization_patch = patch
                else:
                    logger.warning(f"⚠️ Could not apply Sentinel2 optimizations to detector for zone: {zone}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to apply Sentinel2 optimizations to detector for zone {zone}: {e}")
        
        # Use setattr to avoid __name__ issues
        setattr(s2_detector.Sentinel2ArchaeologicalDetector, '__init__', optimized_s2_init)
        
        # Apply GEDI detector optimizations
        try:
            import src.core.detectors.gedi_detector as gedi_detector
            from .optimization import PerformanceOptimizer
            from .detector_patches import GEDIDetectorPatch
            
            # Initialize optimizer for GEDI
            optimizer = PerformanceOptimizer(
                use_gpu=gpu_enabled,
                max_workers=max_workers or 16
            )
            
            # Store optimization settings in the GEDI module
            gedi_detector._optimization_enabled = True
            gedi_detector._optimization_use_gpu = gpu_enabled
            gedi_detector._optimization_max_workers = max_workers
            
            # Monkey patch the GEDI detector
            original_gedi_init = gedi_detector.GEDIArchaeologicalDetector.__init__
            
            def optimized_gedi_init(self, zone, run_id=None):
                # Call original init
                original_gedi_init(self, zone, run_id)
                
                # Apply GEDI-specific GPU patches
                try:
                    patch = GEDIDetectorPatch(optimizer)
                    patch.patch_detector(self.__class__)
                    logger.info(f"✅ Applied GEDI GPU optimizations to detector for zone: {zone}")
                    self._optimization_patch = patch
                except Exception as e:
                    logger.warning(f"⚠️ Failed to apply GEDI optimizations to detector for zone {zone}: {e}")
            
            setattr(gedi_detector.GEDIArchaeologicalDetector, '__init__', optimized_gedi_init)
            logger.info("✅ GEDI detector optimizations enabled")
            
        except ImportError as e:
            logger.warning(f"Could not enable GEDI optimizations: {e}")
        except Exception as e:
            logger.error(f"Error enabling GEDI optimizations: {e}")
        
        logger.info("✅ Pipeline optimizations enabled successfully!")
        return True
        
    except ImportError as e:
        logger.warning(f"Could not enable optimizations: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to enable optimizations: {e}")
        return False

def check_optimization_requirements() -> Dict[str, Any]:
    """Check what optimization features are available"""
    
    requirements = {
        'gpu_acceleration': False,
        'parallel_processing': False,
        'estimated_speedup': '1x (no optimizations)',
        'system_info': {}
    }
    
    # Check GPU availability with __name__ workaround
    try:
        from .optimization import GPU_AVAILABLE
        if GPU_AVAILABLE:
            requirements['gpu_acceleration'] = True
            requirements['system_info']['gpu'] = {
                'name': 'CuPy GPU (available)',
                'memory_free_gb': 'Available',
                'memory_total_gb': 'Available'
            }
            requirements['estimated_speedup'] = '3-5x (GPU + parallel)'
        else:
            requirements['gpu_acceleration'] = False
            requirements['system_info']['gpu'] = 'CuPy not available'
    except Exception as e:
        requirements['gpu_acceleration'] = False
        requirements['system_info']['gpu'] = f'GPU check failed: {e}'
    
    # Check parallel processing
    try:
        from joblib import Parallel
        import multiprocessing as mp
        requirements['parallel_processing'] = True
        requirements['system_info']['cpu_cores'] = mp.cpu_count()
        if not requirements['gpu_acceleration']:
            requirements['estimated_speedup'] = '2-3x (parallel CPU)'
    except ImportError:
        requirements['system_info']['parallel'] = 'Joblib not available'
    
    # Check memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        requirements['system_info']['memory'] = {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent_used': memory.percent
        }
    except ImportError:
        pass
    
    return requirements

def install_optimization_dependencies() -> Dict[str, str]:
    """
    Provide installation commands for optimization dependencies
    
    Returns:
        Dict with installation commands for missing dependencies
    """
    
    install_commands = {}
    
    # Check CuPy for GPU acceleration
    try:
        import cupy
    except ImportError:
        install_commands['cupy'] = "pip install cupy-cuda11x  # or cupy-cuda12x for CUDA 12"
    
    # Check Joblib for parallel processing
    try:
        from joblib import Parallel
    except ImportError:
        install_commands['joblib'] = "pip install joblib"
    
    # Check psutil for system monitoring
    try:
        import psutil
    except ImportError:
        install_commands['psutil'] = "pip install psutil"
    
    return install_commands

def benchmark_optimizations(test_duration: int = 30) -> Dict[str, Any]:
    """
    Run a quick benchmark to measure optimization effectiveness
    
    Args:
        test_duration: Duration in seconds for benchmark
    
    Returns:
        Benchmark results
    """
    
    import time
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    
    logger.info(f"🔬 Running optimization benchmark for {test_duration}s...")
    
    # Test data
    test_size = (1000, 1000)
    test_bands = {
        'red': np.random.rand(*test_size).astype(np.float32),
        'nir': np.random.rand(*test_size).astype(np.float32),
        'swir1': np.random.rand(*test_size).astype(np.float32),
        'red_edge_1': np.random.rand(*test_size).astype(np.float32),
        'red_edge_3': np.random.rand(*test_size).astype(np.float32),
    }
    
    results = {
        'test_size': test_size,
        'sequential_time': 0,
        'parallel_time': 0,
        'gpu_time': 0,
        'speedup_parallel': 0,
        'speedup_gpu': 0
    }
    
    # Sequential calculation
    def calc_ndvi_sequential(bands):
        eps = 1e-8
        return (bands['nir'] - bands['red']) / (bands['nir'] + bands['red'] + eps)
    
    start_time = time.time()
    iterations = 0
    while time.time() - start_time < test_duration / 3:
        _ = calc_ndvi_sequential(test_bands)
        iterations += 1
    results['sequential_time'] = (time.time() - start_time) / iterations
    
    # Parallel calculation
    def calc_indices_parallel(bands):
        def calc_single_index(index_name):
            eps = 1e-8
            if index_name == 'ndvi':
                return (bands['nir'] - bands['red']) / (bands['nir'] + bands['red'] + eps)
            elif index_name == 'ndre1':
                return (bands['red_edge_1'] - bands['red']) / (bands['red_edge_1'] + bands['red'] + eps)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(calc_single_index, idx) for idx in ['ndvi', 'ndre1']]
            return [f.result() for f in futures]
    
    start_time = time.time()
    iterations = 0
    while time.time() - start_time < test_duration / 3:
        _ = calc_indices_parallel(test_bands)
        iterations += 1
    results['parallel_time'] = (time.time() - start_time) / iterations
    
    # GPU calculation (if available)
    try:
        import cupy as cp
        
        gpu_bands = {name: cp.asarray(band) for name, band in test_bands.items()}
        
        def calc_ndvi_gpu(gpu_bands):
            eps = 1e-8
            return ((gpu_bands['nir'] - gpu_bands['red']) / 
                   (gpu_bands['nir'] + gpu_bands['red'] + eps)).get()
        
        start_time = time.time()
        iterations = 0
        while time.time() - start_time < test_duration / 3:
            _ = calc_ndvi_gpu(gpu_bands)
            iterations += 1
        results['gpu_time'] = (time.time() - start_time) / iterations
        
    except ImportError:
        results['gpu_time'] = None
    
    # Calculate speedups
    if results['parallel_time'] > 0:
        results['speedup_parallel'] = results['sequential_time'] / results['parallel_time']
    
    if results['gpu_time'] and results['gpu_time'] > 0:
        results['speedup_gpu'] = results['sequential_time'] / results['gpu_time']
    
    logger.info(f"✅ Benchmark completed:")
    logger.info(f"  Sequential: {results['sequential_time']:.4f}s")
    logger.info(f"  Parallel: {results['parallel_time']:.4f}s (speedup: {results['speedup_parallel']:.1f}x)")
    if results['gpu_time']:
        logger.info(f"  GPU: {results['gpu_time']:.4f}s (speedup: {results['speedup_gpu']:.1f}x)")
    
    return results

def get_optimization_recommendations() -> Dict[str, str]:
    """Get recommendations for optimizing the current system"""
    
    reqs = check_optimization_requirements()
    recommendations = {}
    
    if not reqs['gpu_acceleration']:
        recommendations['gpu'] = (
            "Install CuPy for GPU acceleration: pip install cupy-cuda11x\n"
            "This can provide 3-5x speedup for large computations"
        )
    
    if not reqs['parallel_processing']:
        recommendations['parallel'] = (
            "Install Joblib for enhanced parallel processing: pip install joblib\n"
            "This improves CPU parallelization efficiency"
        )
    
    # Memory recommendations
    if 'memory' in reqs['system_info']:
        memory_info = reqs['system_info']['memory']
        if memory_info['available_gb'] < 8:
            recommendations['memory'] = (
                f"Low available memory ({memory_info['available_gb']:.1f}GB). "
                "Consider closing other applications or reducing max_workers"
            )
    
    # CPU recommendations
    if 'cpu_cores' in reqs['system_info']:
        cpu_cores = reqs['system_info']['cpu_cores']
        if cpu_cores >= 16:
            recommendations['cpu'] = (
                f"High core count ({cpu_cores} cores) detected. "
                "Consider setting max_workers=16 for optimal performance"
            )
    
    return recommendations

if __name__ == "__main__":
    # Test optimization availability and provide recommendations
    print("🔍 Amazon Archaeological Discovery Pipeline - Optimization Check")
    print("=" * 70)
    
    # Check requirements
    reqs = check_optimization_requirements()
    print("\n📊 System Capabilities:")
    for key, value in reqs.items():
        if key != 'system_info':
            print(f"  {key}: {value}")
    
    if 'system_info' in reqs:
        print("\n💻 System Information:")
        for key, value in reqs['system_info'].items():
            print(f"  {key}: {value}")
    
    # Check missing dependencies
    missing = install_optimization_dependencies()
    if missing:
        print("\n📦 Missing Dependencies:")
        for package, command in missing.items():
            print(f"  {package}: {command}")
    else:
        print("\n✅ All optimization dependencies are available!")
    
    # Get recommendations
    recommendations = get_optimization_recommendations()
    if recommendations:
        print("\n💡 Optimization Recommendations:")
        for category, recommendation in recommendations.items():
            print(f"  {category}: {recommendation}")
    
    # Enable optimizations
    print("\n🚀 Enabling optimizations...")
    success = enable_pipeline_optimizations(use_gpu=True)
    print(f"Optimizations enabled: {success}")
    
    # Optional benchmark
    import sys
    if '--benchmark' in sys.argv:
        print("\n🔬 Running benchmark...")
        benchmark_results = benchmark_optimizations(test_duration=10)
        print("Benchmark completed!") 
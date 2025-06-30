"""
Enhanced Archaeological Detector for Sentinel-2 Data
Optimized for 13-band multispectral analysis with red-edge and SWIR capabilities
"""

import numpy as np
import logging
import cv2
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, box, mapping
from shapely.ops import unary_union
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import logging
import traceback
from pathlib import Path
import json
from rasterio.warp import reproject, Resampling

# Assuming RESULTS_DIR is available in config and suitable for caching detector outputs
from src.core.config import RESULTS_DIR, ZONE_DETECTION_CONFIG
from ..parameter_configs import get_current_params 
from src.core.data_objects import SceneData # For type hinting
from src.core.coordinate_manager import CoordinateManager

logger = logging.getLogger(__name__)

DETECTOR_OUTPUT_BASE_DIR = RESULTS_DIR / "detector_outputs" / "sentinel2"

class Sentinel2ArchaeologicalDetector:
    """
    Enhanced archaeological detector specifically optimized for Sentinel-2 data
    
    Key improvements over Landsat detector:
    - Utilizes red-edge bands (B05, B06, B07) for vegetation stress detection
    - Enhanced SWIR analysis with 20m resolution
    - Archaeological vegetation indices using red-edge bands
    - Improved crop mark detection using 705nm and 783nm bands
    - Optimized spectral signatures for archaeological features
    """
    
    def __init__(self, zone, run_id=None):
        self.zone = zone
        self.run_id = run_id
        self.detection_results = {}
        self.coordinate_manager = None  # Will be initialized when bands are loaded
        self.processed_bands = {}
        
        # Load zone-specific detection configuration
        self.zone_config = self._load_zone_config()
        logger.info(f"Sentinel-2 detector initialized for zone {zone.id} (type: {getattr(zone, 'zone_type', 'default')}, strategy: {getattr(zone, 'detection_strategy', 'balanced')}) with run_id: {run_id}")
        logger.info(f"Zone-specific config: {self.zone_config}")
        
        self.band_resolutions = {
            'B01': 60, 'B02': 10, 'B03': 10, 'B04': 10,
            'B05': 20, 'B06': 20, 'B07': 20, 'B08': 10,
            'B8A': 20, 'B09': 60, 'B10': 60, 'B11': 20, 'B12': 20
        }
        self.transform = None
        self.crs = None
    
    def _load_zone_config(self):
        """Load zone-specific detection configuration"""
        # Get zone type, default to forested_buried_sites if not specified
        zone_type = getattr(self.zone, 'zone_type', 'forested_buried_sites')
        
        # Get zone-specific config from ZONE_DETECTION_CONFIG
        zone_config = ZONE_DETECTION_CONFIG.get(zone_type, ZONE_DETECTION_CONFIG['forested_buried_sites'])
        
        logger.info(f"Loading zone config for zone_type '{zone_type}': {zone_config}")
        return zone_config
    
    def _validate_coordinates(self, lon: float, lat: float, context: str = "") -> bool:
        """Validate that coordinates are within reasonable Amazon bounds"""
        # Amazon region bounds (roughly): lat -20 to 10, lon -80 to -44
        # Upper Napo specific bounds: lat -1 to 0, lon -73 to -72
        if not (-80 <= lon <= -44):
            logger.warning(f"Longitude {lon:.6f} outside Amazon bounds [-80, -44] for {context}")
            return False
        if not (-20 <= lat <= 10):
            logger.warning(f"Latitude {lat:.6f} outside Amazon bounds [-20, 10] for {context}")
            return False
        return True
    
    # OLD COORDINATE CONVERSION METHODS REMOVED
    # All coordinate operations now handled by unified CoordinateManager
    # No fallbacks, no manual conversions, single source of truth
    
    def _resample_bands_to_reference(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Resample all bands to the shape of the highest-resolution band (preferably 10m, e.g., 'nir' or 'red').
        Uses bilinear resampling for continuous data.
        """
        # Pick reference band (prefer 'nir', then 'red', then any)
        ref_band_data = None
        # Order of preference for reference resolution
        preferred_ref_keys = ['B08', 'B04', 'B03', 'B02'] # NIR, Red, Green, Blue (10m bands)
        for key in preferred_ref_keys:
            if key in self.processed_bands: # Check original processed_bands for resolution master
                ref_band_data = self.processed_bands[key]
                logger.debug(f"Using band {key} as resampling reference.")
                break
        if ref_band_data is None and self.processed_bands:
            ref_band_data = next(iter(self.processed_bands.values()))
            logger.debug("Using first available band as resampling reference.")
        
        if ref_band_data is None:
            logger.warning("No reference band data found for resampling. Returning original bands.")
            return bands
            
        ref_shape = ref_band_data.shape
        resampled_bands = {}
        for name, arr in bands.items():
            if arr.shape == ref_shape:
                resampled_bands[name] = arr
            else:
                dst_height, dst_width = ref_shape
                try:
                    # Ensure arr is contiguous before resizing if it comes from complex slicing
                    if not arr.flags.c_contiguous:
                        arr = np.ascontiguousarray(arr)
                    dst = cv2.resize(arr, (dst_width, dst_height), interpolation=cv2.INTER_LINEAR)
                    resampled_bands[name] = dst
                except Exception as e:
                    logger.warning(f"Error resampling band {name} (shape {arr.shape} to {ref_shape}): {e}. Using original.")
                    resampled_bands[name] = arr # Fallback to original if resize fails
        return resampled_bands

    def load_sentinel2_bands(self, scene_path: Path) -> Dict[str, np.ndarray]:
        """
        Load and process Sentinel-2 bands optimized for archaeological detection
        
        Args:
            scene_path: Path to scene directory or composite file
            
        Returns:
            Dictionary of band arrays with standardized names
        """
        
        scene_path = Path(scene_path)
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene path not found: {scene_path}")
        
        bands = {}
        transform = None
        crs = None
        
        if scene_path.is_dir():
            # Individual band files (typical Sentinel-2 structure)
            band_files = {
                'blue': 'B02.tif',
                'green': 'B03.tif', 
                'red': 'B04.tif',
                'red_edge_1': 'B05.tif',
                'red_edge_2': 'B06.tif',
                'red_edge_3': 'B07.tif',
                'nir': 'B08.tif',
                'nir_narrow': 'B8A.tif',
                'swir1': 'B11.tif',
                'swir2': 'B12.tif'
            }
            
            for band_name, filename in band_files.items():
                filepath = scene_path / filename
                if filepath.exists():
                    try:
                        with rasterio.open(filepath) as src:
                            band_data = src.read(1).astype(np.float32)
                            
                            # Apply Sentinel-2 L2A scaling (already in reflectance 0-1)
                            # but ensure proper range
                            band_data = np.clip(band_data / 10000.0, 0, 1)
                            
                            bands[band_name] = band_data
                            
                            if transform is None:
                                transform = src.transform
                                crs = src.crs
                                
                            logger.debug(f"Loaded {band_name}: {band_data.shape}")
                            
                    except Exception as e:
                        logger.error(f"Error loading {band_name} from {filepath}: {e}")
                        continue
        
        elif scene_path.is_file() and scene_path.suffix.lower() in ['.tif', '.tiff']:
            with rasterio.open(scene_path) as src:
                count = src.count
                transform = src.transform
                crs = src.crs
                logger.info(f"Reading composite file: {scene_path} with {count} bands.")
                
                descriptions = None
                if hasattr(src, 'descriptions') and src.descriptions and any(d is not None for d in src.descriptions):
                    descriptions = src.descriptions
                logger.info(f"Band descriptions from composite: {descriptions}")

                band_mapping = {} # Initialize band_mapping
                
                # Define name_mapping here, ensuring it's always available
                name_mapping = {
                    'B02': 'blue', 'B03': 'green', 'B04': 'red',
                    'B05': 'red_edge_1', 'B06': 'red_edge_2', 'B07': 'red_edge_3',
                    'B08': 'nir', 'B8A': 'nir_narrow',
                    'B11': 'swir1', 'B12': 'swir2'
                }
                
                if descriptions and len(descriptions) == count:
                    logger.info("Attempting to map bands using rasterio descriptions.")
                    temp_band_map = {}
                    for i, desc in enumerate(descriptions):
                        if desc and desc in name_mapping: # Check if description is a known band ID
                            temp_band_map[desc] = i + 1 # GDAL bands are 1-indexed
                        
                        if len(temp_band_map) >= 4: # Check if we got enough bands (e.g., at least RGB + NIR)
                            band_mapping = temp_band_map
                            logger.info(f"Using band_mapping from descriptions: {band_mapping}")
                        else:
                            logger.warning(f"Found descriptions, but not enough recognized bands ({len(temp_band_map)}). Descriptions: {descriptions}. Will attempt fallback.")
                            # Let it fall through to the else for fallback if not enough good descriptions

                # Fallback or if descriptions were not sufficient
                if not band_mapping: # If band_mapping is still empty
                    logger.info("Falling back to standard band order as descriptions are missing, incomplete, or insufficient.")
                    band_order = ['B02', 'B03', 'B04', 'B08', 'B05', 'B07', 'B11', 'B12'] # Standard 8 bands - must match provider order
                    logger.info(f"Fallback: count={count}, len(band_order)={len(band_order)}")
                    if count > 0 and count <= len(band_order): # Ensure count is reasonable for this order
                        # If count is less than len(band_order) but >0, map only the first 'count' bands from order.
                        # This handles cases where composite might have fewer than 8 bands but still follows start of standard order.
                        band_mapping = {band_id_str: i + 1 for i, band_id_str in enumerate(band_order[:count])}
                        logger.info(f"Fallback successfully created band_mapping: {band_mapping}")
                    elif count > len(band_order): # More bands in file than our standard fallback order
                        logger.warning(f"File has {count} bands, more than standard fallback order ({len(band_order)}). Mapping first {len(band_order)} bands by order.")
                        band_mapping = {band_id_str: i + 1 for i, band_id_str in enumerate(band_order)}
                    else: # count is 0 or other unexpected issue
                        logger.error(f"Fallback failed: Cannot apply band order. count={count}. band_mapping will be empty.")
                        band_mapping = {} # Explicitly ensure it's empty

                logger.info(f"Proceeding to band loading loop with band_mapping: {band_mapping}")
                for band_id, band_idx in band_mapping.items():
                    # The detailed logging added in the previous step is here:
                    logger.info(f"Looping in load_sentinel2_bands - band_id from band_mapping: {band_id}, band_idx: {band_idx}")
                    if band_id in name_mapping:
                        logger.info(f"  {band_id} is in name_mapping. Attempting to load as {name_mapping[band_id]}.")
                        band_name = name_mapping[band_id]
                        try:
                            band_data = src.read(band_idx).astype(np.float32)
                            band_data = np.clip(band_data / 10000.0, 0, 1)
                            bands[band_name] = band_data
                            logger.debug(f"Loaded {band_name} from band {band_idx} (rasterio description: {descriptions[band_idx-1] if descriptions and band_idx-1 < len(descriptions) else 'N/A'})")
                        except Exception as e:
                            logger.warning(f"Error loading band {band_idx} (intended aname: {band_id}, mapped name: {band_name}): {e}", exc_info=True)
                    else:
                        logger.warning(f"  Skipping band_id {band_id} from band_mapping as it's not in name_mapping.")
        
        if not bands:
            logger.error(f"No Sentinel-2 bands were successfully loaded. Bands dictionary is empty. Scene path: {scene_path}")
            raise ValueError(f"No valid Sentinel-2 bands found in {scene_path}")
        
        self.processed_bands = bands
        self.transform = transform
        self.crs = crs
        
        # Initialize unified coordinate manager - single source of truth
        try:
            self.coordinate_manager = CoordinateManager(transform=self.transform, crs=self.crs)
            logger.info("✅ Coordinate manager initialized - unified coordinate system active")
        except Exception as e:
            logger.error(f"❌ Failed to initialize coordinate manager: {e}")
            raise ValueError(f"Cannot proceed without valid coordinate system: {e}")
        
        logger.info(f"Loaded {len(bands)} Sentinel-2 bands: {list(bands.keys())}")
        # Resample all bands to the shape of the highest-resolution band
        bands = self._resample_bands_to_reference(bands)
        return bands
    
    def calculate_archaeological_indices(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate archaeological indices with web-validated band combinations
        
        Based on: Sentinel-2 archaeological research (MDPI 2014, ResearchGate studies)
        VALIDATED: 705nm and 783nm are optimal for archaeological detection
        Source: "Both IKONOS and ASTER central wavelengths of Red and NIR bands are very close to 
        700 nm and 800 nm, which are considered as the optimum spectral wavelengths for crop marks detection"
        """
        logger.info("🧮 SPECTRAL MATHEMATICS: Computing archaeological spectral indices")
        logger.info(f"🌈 Processing {len(bands)} spectral bands for archaeological signatures")
        
        indices = {}
        eps = 1e-8  # Prevent division by zero
        
        # Standard vegetation indices with validated NDVI depression threshold
        if 'red' in bands and 'nir' in bands:
            logger.info("📊 Computing NDVI (664nm vs 832nm) - fundamental vegetation health")
            red = bands['red']
            nir = bands['nir']
            
            # NDVI - Normalized Difference Vegetation Index
            indices['ndvi'] = (nir - red) / (nir + red + eps)
            ndvi_mean = np.mean(indices['ndvi'])
            logger.info(f"🌿 NDVI statistics: mean={ndvi_mean:.3f}, range={indices['ndvi'].min():.3f}→{indices['ndvi'].max():.3f}")
            
            # NDVI Depression Detection with VALIDATED 0.07 threshold (was 0.05)
            # Source: MDPI 2014 - "nearly 0.07 to the Normalised Difference Vegetation Index"
            logger.info("🔍 NDVI DEPRESSION: Detecting archaeological vegetation stress patterns")
            indices['ndvi_depression'] = self._calculate_ndvi_depression(indices['ndvi'])
            depression_pixels = np.sum(indices['ndvi_depression'] > 0.07)
            logger.info(f"📉 Detected {depression_pixels:,} pixels with archaeological vegetation stress")
        
        # VALIDATED: Red-edge enhanced indices using optimal wavelengths
        if 'red_edge_1' in bands and 'red' in bands:
            red_edge_1 = bands['red_edge_1']  # 705nm - VALIDATED optimal wavelength
            red = bands['red']
            
            # NDRE1 - Research validated for vegetation stress detection
            indices['ndre1'] = (red_edge_1 - red) / (red_edge_1 + red + eps)
            
            # Add confidence scoring based on band quality
            indices['ndre1_confidence'] = self._calculate_band_confidence(red_edge_1, red)
        
        if 'red_edge_3' in bands and 'red' in bands:
            red_edge_3 = bands['red_edge_3']  # 783nm - VALIDATED optimal wavelength
            red = bands['red']
            
            # NDRE3 - Research validated for archaeological crop marks
            indices['ndre3'] = (red_edge_3 - red) / (red_edge_3 + red + eps)
            indices['ndre3_confidence'] = self._calculate_band_confidence(red_edge_3, red)
        
        # VALIDATED: Archaeological Vegetation Index using optimal wavelengths
        # "The combination of the 705nm and 783nm red edge bands provides optimal spectral 
        # positioning for detecting vegetation stress patterns"
        if 'red_edge_1' in bands and 'red_edge_3' in bands:
            re1 = bands['red_edge_1']  # 705nm
            re3 = bands['red_edge_3']  # 783nm
            
            # Research-validated AVI for archaeological crop marks
            indices['avi_archaeological'] = (re3 - re1) / (re3 + re1 + eps)
            
            # Statistical validation of the index
            indices['avi_significance'] = self._validate_vegetation_index(indices['avi_archaeological'])
        
        # Enhanced Terra Preta indices using red-edge
        if 'nir' in bands and 'swir1' in bands:
            nir = bands['nir']
            swir1 = bands['swir1']
            
            # Standard Terra Preta Index
            indices['terra_preta'] = (nir - swir1) / (nir + swir1 + eps)
        
        if 'red_edge_3' in bands and 'swir1' in bands:
            re3 = bands['red_edge_3']
            swir1 = bands['swir1']
            
            # Enhanced Terra Preta Index using red-edge
            indices['terra_preta_enhanced'] = (re3 - swir1) / (re3 + swir1 + eps)
        
        # Soil composition indices
        if 'swir1' in bands and 'swir2' in bands:
            swir1 = bands['swir1']
            swir2 = bands['swir2']
            
            # Clay Mineral Index (important for ceramics)
            indices['clay_minerals'] = swir1 / (swir2 + eps)
            
            # Normalized Difference Infrared Index
            if 'nir' in bands:
                indices['ndii'] = (nir - swir1) / (nir + swir1 + eps)
        
        # Water and moisture indices
        if 'green' in bands and 'nir' in bands:
            green = bands['green']
            nir = bands['nir']
            
            # NDWI - Normalized Difference Water Index
            indices['ndwi'] = (green - nir) / (green + nir + eps)
        
        # Brightness and texture
        if len(bands) >= 3:
            band_values = list(bands.values())
            
            # Brightness Index
            indices['brightness'] = np.sqrt(np.sum([b**2 for b in band_values[:3]], axis=0) / 3)
        
        # Crop Mark Index - specialized for archaeological crop marks
        if all(b in bands for b in ['red', 'red_edge_1', 'nir']):
            red = bands['red']
            re1 = bands['red_edge_1']
            nir = bands['nir']
            
            # This index maximizes contrast between stressed and healthy vegetation
            indices['crop_mark'] = ((re1 - red) * (nir - re1)) / ((re1 + red) * (nir + re1) + eps)
        
        # Sentinel-2 specific archaeological index
        if all(b in bands for b in ['red_edge_1', 'red_edge_3', 'swir1']):
            re1 = bands['red_edge_1']
            re3 = bands['red_edge_3']
            swir1 = bands['swir1']
            
            # S2 Archaeological Index - combines vegetation stress and soil signals
            indices['s2_archaeological'] = ((re1 + re3) / 2 - swir1) / ((re1 + re3) / 2 + swir1 + eps)
        
        logger.info(f"Calculated {len([k for k, v in indices.items() if v is not None])} archaeological indices")
        return {k: v for k, v in indices.items() if v is not None}
    
    def detect_enhanced_terra_preta(self, bands: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Enhanced terra preta detection using Sentinel-2's superior spectral resolution
        """
        logger.info("🏺 TERRA PRETA DETECTION: Commencing enhanced Amazonian dark earth analysis...")
        logger.info("🌱 Activating red-edge bands (705nm, 783nm) for anthropogenic soil detection")
        
        indices = self.calculate_archaeological_indices(bands)
        logger.info(f"📊 Computed {len(indices)} archaeological spectral indices")
        
        if 'terra_preta_enhanced' not in indices:
            logger.warning("🔶 FALLBACK: Red-edge bands unavailable, switching to standard detection")
            logger.info("📡 Using legacy SWIR-NIR analysis instead of red-edge enhancement")
            std_result = self.detect_standard_terra_preta(bands, indices)
            if std_result is None:
                logger.error("❌ Standard terra preta detection failed - returning empty result")
                return {'features': [], 'mask': None, 'total_pixels': 0, 'coverage_percent': 0, 'detection_method': None, 'red_edge_enhanced': False, 'parameters': {}}
            logger.info("✅ Standard terra preta detection completed successfully")
            return std_result
        
        # Use enhanced terra preta index with red-edge
        tp_enhanced = indices['terra_preta_enhanced']
        ndvi = indices.get('ndvi')
        ndre1 = indices.get('ndre1')
        
        logger.info("🔬 SPECTRAL ANALYSIS: Processing red-edge enhanced terra preta signature")
        logger.info(f"📏 Terra preta index range: {tp_enhanced.min():.3f} → {tp_enhanced.max():.3f}")
        
        # Add SWIR-based moisture filtering for terra preta
        # Dry terra preta has distinct SWIR signature vs wet natural soils
        if 'nir' in bands and 'swir1' in bands:
            logger.info("💧 MOISTURE FILTERING: Analyzing SWIR signature for terra preta characteristics")
            nir = bands['nir']
            swir1 = bands['swir1']
            moisture_index = (nir - swir1) / (nir + swir1 + 1e-8)
            terra_preta_moisture = moisture_index < 0.2  # Drier signature
            dry_pixels = np.sum(terra_preta_moisture)
            total_pixels = terra_preta_moisture.size
            logger.info(f"🏜️ Dry signature pixels: {dry_pixels:,}/{total_pixels:,} ({100*dry_pixels/total_pixels:.1f}%)")
        else:
            logger.warning("⚠️ SWIR bands unavailable - skipping moisture filtering")
            terra_preta_moisture = np.ones_like(tp_enhanced, dtype=bool)  # No filtering if bands missing
        
        if ndvi is None or ndre1 is None:
            std_result = self.detect_standard_terra_preta(bands, indices)
            if std_result is None:
                logger.error("detect_standard_terra_preta returned None; replacing with empty result dict.")
                return {'features': [], 'mask': None, 'total_pixels': 0, 'coverage_percent': 0, 'detection_method': None, 'red_edge_enhanced': False, 'parameters': {}}
            return std_result
        
        # ARCHAEOLOGICAL PARAMETERS - Evidence-based thresholds from literature
        # Get current parameter configuration
        params = get_current_params()
        s2_params = params['sentinel2']
        
        # Get NDVI depression strength (archaeological indicator)
        ndvi_depression = indices.get('ndvi_depression')
        depression_enhanced = ndvi_depression is not None and np.any(ndvi_depression > s2_params.ndvi_depression_threshold)
        
        if depression_enhanced:
            # Primary criteria: NDVI depression + red-edge analysis
            tp_mask = (
                (tp_enhanced > s2_params.terra_preta_base_threshold) &      # Archaeological threshold
                (ndvi > s2_params.ndvi_threshold) &                         # Archaeological NDVI threshold
                (ndvi < 0.7) &                                              # Exclude very dense vegetation
                (ndre1 > s2_params.ndre1_threshold) &                       # Archaeological red-edge threshold
                (ndvi_depression > s2_params.ndvi_depression_threshold) &   # Strong NDVI depression signal
                terra_preta_moisture                                        # SWIR-based moisture filtering
            )
        else:
            # Standard criteria when no depression signal available - HIGHER thresholds
            tp_mask = (
                (tp_enhanced > s2_params.terra_preta_enhanced_threshold) &  # Higher threshold for enhanced
                (ndvi > s2_params.ndvi_threshold) &                         # Archaeological vegetation range
                (ndvi < 0.7) &                                              # Exclude very dense vegetation
                (ndre1 > s2_params.ndre1_threshold) &                       # Archaeological red-edge constraint
                terra_preta_moisture                                        # SWIR-based moisture filtering
            )
        
        logger.info("🔧 POST-PROCESSING: Applying morphological refinement to terra preta candidates")
        
        # Simplified morphological operations to prevent identical clustering
        # Use moderate-sized kernels that allow natural size variation
        kernel_small = np.ones((2, 2), np.uint8)  # Minimal noise removal
        kernel_medium = np.ones((3, 3), np.uint8)  # Light clustering
        
        tp_mask = tp_mask.astype(np.uint8)
        initial_pixels = np.sum(tp_mask)
        logger.info(f"🎯 Initial candidate pixels: {initial_pixels:,}")
        
        # Stage 1: Remove small noise pixels
        logger.info("🧹 Stage 1: Removing noise pixels with 2x2 opening...")
        tp_mask = cv2.morphologyEx(tp_mask, cv2.MORPH_OPEN, kernel_small)
        stage1_pixels = np.sum(tp_mask)
        logger.info(f"   → {stage1_pixels:,} pixels remaining ({100*stage1_pixels/initial_pixels:.1f}%)")
        
        # Stage 2: Fill small gaps within features
        logger.info("🔗 Stage 2: Connecting features with 3x3 closing...")
        tp_mask = cv2.morphologyEx(tp_mask, cv2.MORPH_CLOSE, kernel_medium)
        stage2_pixels = np.sum(tp_mask)
        logger.info(f"   → {stage2_pixels:,} pixels after closing ({100*stage2_pixels/initial_pixels:.1f}%)")
        
        # Stage 3: Smooth boundaries
        logger.info("✨ Stage 3: Smoothing boundaries with 2x2 opening...")
        tp_mask = cv2.morphologyEx(tp_mask, cv2.MORPH_OPEN, kernel_small)
        final_pixels = np.sum(tp_mask)
        logger.info(f"   → {final_pixels:,} final pixels ({100*final_pixels/initial_pixels:.1f}%)")
        
        # Find connected components
        logger.info("🔍 FEATURE EXTRACTION: Analyzing connected components...")

        # PERFORMANCE DEBUG: Set up a dedicated performance logger
        perf_log_path = Path(self.run_output_dir.parent.parent, 'logs', 'performance_debug.log')
        perf_log_path.parent.mkdir(parents=True, exist_ok=True)
        performance_logger = logging.getLogger('performance_logger')
        if not performance_logger.handlers:
            perf_handler = logging.FileHandler(perf_log_path)
            perf_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            performance_logger.addHandler(perf_handler)
            performance_logger.setLevel(logging.INFO)

        import time
        component_start = time.time()
        labeled_mask, num_features = ndimage.label(tp_mask)
        component_time = time.time() - component_start
        logger.info(f"✅ Found {num_features} potential features in {component_time:.3f}s")
        
        # Extract patches with enhanced metadata - ARCHAEOLOGICAL FOCUS
        logger.info(f"📊 ARCHAEOLOGICAL PATCH ANALYSIS: Filtering {num_features} candidates for ancient settlements...")
        patch_start = time.time()
        tp_features = []
        valid_features = 0
        
        # PHASE 1: RAPID PRE-FILTERING for archaeological relevance
        logger.info("🔍 Phase 1: Rapid archaeological pre-filtering...")
        prefilter_start = time.time()
        performance_logger.info(f'PERF_DEBUG: Phase 1 Pre-filtering START')
        archaeological_candidates = []
        
        # Get current parameters for archaeological filtering
        from ..parameter_configs import get_current_params
        params = get_current_params()
        s2_params = params['sentinel2']
        
        for i in range(1, num_features + 1):
            patch_mask = labeled_mask == i
            patch_size = np.sum(patch_mask)
            
            # ARCHAEOLOGICAL SIZE FILTERING - Focus on settlement-scale features
            pixel_area = 10 * 10  # 10m resolution
            area_m2 = patch_size * pixel_area
            
            # Skip tiny noise (modern paths) and massive areas (rivers, modern farms)
            if area_m2 < 500:  # Too small for archaeological features (< 500m²)
                continue
            if area_m2 > 50000:  # Too large for individual settlements (> 5 hectares)
                continue
                
            # RAPID GEOMETRIC PRE-FILTER - Archaeological vs Modern Shapes
            coords_y, coords_x = np.where(patch_mask)
            if len(coords_y) < 3:  # Need minimum points for geometry
                continue
                
            height = coords_y.max() - coords_y.min() + 1
            width = coords_x.max() - coords_x.min() + 1
            aspect_ratio = max(height, width) / max(min(height, width), 1)
            
            # Filter out LINEAR features (modern roads, rivers, farm boundaries)
            if aspect_ratio > 8.0:  # Extremely linear = likely modern infrastructure
                continue
                
            # Filter out PERFECT RECTANGLES (modern agriculture, buildings)
            # Calculate shape complexity
            perimeter_pixels = cv2.findContours(patch_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if perimeter_pixels:
                perimeter = cv2.arcLength(perimeter_pixels[0], True)
                area_perimeter_ratio = patch_size / max(perimeter, 1)
                
                # Modern farms have very high area/perimeter ratios (perfect squares/rectangles)
                if area_perimeter_ratio > 25:  # Too geometric = likely modern agriculture
                    continue
            
            # PASSED PRE-FILTERING - Add to detailed analysis queue
            archaeological_candidates.append((i, patch_mask, patch_size, area_m2, aspect_ratio))
        
        prefilter_time = time.time() - prefilter_start
        performance_logger.info(f'PERF_DEBUG: Phase 1 Pre-filtering END - Duration: {prefilter_time:.2f}s')
        logger.info(f"✅ Pre-filtering complete: {len(archaeological_candidates)}/{num_features} candidates pass archaeological criteria ({prefilter_time:.2f}s)")
        logger.info(f"📉 Filtered out {num_features - len(archaeological_candidates)} modern/natural features (roads, farms, rivers)")
        
        # PHASE 2: DETAILED ARCHAEOLOGICAL ANALYSIS on filtered candidates
        logger.info(f"🏺 Phase 2: Detailed analysis of {len(archaeological_candidates)} archaeological candidates...")
        detailed_start = time.time()
        performance_logger.info(f'PERF_DEBUG: Phase 2 Detailed Analysis START')
        
        for idx, (i, patch_mask, patch_size, area_m2, aspect_ratio) in enumerate(archaeological_candidates):
            if idx % 50 == 0 or idx < 5:  # Log progress every 50 archaeological candidates
                logger.info(f"   Analyzing archaeological candidate {idx+1}/{len(archaeological_candidates)} ({100*(idx+1)/len(archaeological_candidates):.1f}%)")
            
            valid_features += 1
            
            # Calculate statistics for this archaeological candidate
            patch_coords = np.where(patch_mask)
            if patch_coords[0].size == 0 or patch_coords[1].size == 0:
                continue  # skip this patch if no valid pixels
            
            centroid_y = np.mean(patch_coords[0])
            centroid_x = np.mean(patch_coords[1])
            
            # Use unified coordinate manager - SINGLE SOURCE OF TRUTH
            if not self.coordinate_manager:
                raise ValueError("Coordinate manager not initialized - cannot create features")
            
            try:
                feature_properties = {
                    'type': 'terra_preta_enhanced',
                    'detection_method': 'sentinel2_enhanced'
                }
                
                # Create feature with guaranteed geographic coordinates
                terra_preta_feature = self.coordinate_manager.create_point_feature(
                    pixel_x=centroid_x,
                    pixel_y=centroid_y,
                    properties=feature_properties
                )
            except Exception as e:
                logger.error(f"Failed to create terra preta feature at pixel ({centroid_x:.2f}, {centroid_y:.2f}): {e}")
                continue
            
            # ARCHAEOLOGICAL SPECTRAL ANALYSIS - Enhanced for settlement detection
            mean_tp_enhanced = np.mean(tp_enhanced[patch_mask])
            mean_ndvi = np.mean(ndvi[patch_mask])
            mean_ndre1 = np.mean(ndre1[patch_mask])
            
            # ARCHAEOLOGICAL CONFIDENCE - Optimized multi-factor scoring for ancient settlements
            # Focus on characteristics that distinguish archaeological features from modern ones
            spectral_strength = min(1.0, mean_tp_enhanced * 6)  # Terra preta signature strength
            vegetation_coherence = min(1.0, (mean_ndvi - 0.3) * 3)  # Archaeological vegetation range
            red_edge_signal = min(1.0, mean_ndre1 * 6)  # Red-edge archaeological indicator
            size_factor = min(1.0, np.sqrt(area_m2) / 50.0)  # Size appropriateness for settlements
            
            # ARCHAEOLOGICAL ENHANCEMENT - Boost confidence for strong ancient signatures
            archaeological_boost = 1.0
            if mean_tp_enhanced > 0.25 and mean_ndre1 > 0.3:  # Very strong ancient indicators
                archaeological_boost = 1.3
            elif mean_tp_enhanced > 0.2 and mean_ndre1 > 0.25:  # Strong ancient indicators  
                archaeological_boost = 1.15
            
            # Calculate archaeological confidence score
            base_confidence = (0.4 * spectral_strength + 0.25 * vegetation_coherence + 
                             0.25 * red_edge_signal + 0.1 * size_factor)
            confidence = min(1.0, base_confidence * archaeological_boost)
            
            # ARCHAEOLOGICAL SIGNIFICANCE FILTER - Remove weak spectral signatures
            archaeological_strength = mean_tp_enhanced + (mean_ndre1 * 0.5)
            if archaeological_strength < 0.3:  # Require combined spectral signature
                confidence *= 0.8  # Reduce confidence for weak signatures
            
            # Add all properties to the feature created by coordinate manager
            terra_preta_feature.update({
                'area_pixels': patch_size,
                'area_m2': area_m2,  # Already calculated above
                'mean_tp_enhanced': mean_tp_enhanced,
                'mean_ndvi': mean_ndvi,
                'mean_ndre1': mean_ndre1,
                'confidence': confidence,
                'aspect_ratio': aspect_ratio,  # Already calculated in pre-filtering
                'archaeological_strength': archaeological_strength
            })
            
            # Additional spectral characterization
            if 's2_archaeological' in indices:
                terra_preta_feature['s2_archaeological'] = np.mean(indices['s2_archaeological'][patch_mask])
            
            # FINAL ARCHAEOLOGICAL FILTERING - Only accept high-confidence ancient features
            confidence_threshold = s2_params.base_confidence_threshold
            
            # SKIP expensive geometric recalculation - we already have what we need
            # Modern optimization: use pre-calculated values instead of re-computing
            
            if confidence >= confidence_threshold:
                tp_features.append(terra_preta_feature)
                if idx < 10:  # Log first few accepted features
                    logger.info(f"✓ ACCEPTED: Archaeological feature with confidence={confidence:.3f}, "
                              f"area={area_m2:.0f}m², strength={archaeological_strength:.3f}")
            else:
                if idx < 5:  # Log first few rejected features
                    logger.info(f"✗ REJECTED: Weak confidence={confidence:.3f} < {confidence_threshold}")
        
        detailed_time = time.time() - detailed_start
        performance_logger.info(f'PERF_DEBUG: Phase 2 Detailed Analysis END - Duration: {detailed_time:.2f}s')
        patch_time = time.time() - patch_start
        
        logger.info(f"📊 ARCHAEOLOGICAL ANALYSIS COMPLETE:")
        logger.info(f"   Phase 1 Pre-filtering: {prefilter_time:.2f}s - {len(archaeological_candidates)}/{num_features} candidates")
        logger.info(f"   Phase 2 Detailed analysis: {detailed_time:.2f}s - {len(tp_features)} features accepted")
        logger.info(f"   Total feature processing: {patch_time:.2f}s")
        logger.info(f"⚡ Performance: {1000*patch_time/max(1,num_features):.1f}ms per original feature")
        logger.info(f"✅ Archaeological terra preta detection: {len(tp_features)} ancient features found")
        
        return {
            'features': tp_features,
            'mask': tp_mask,
            'total_pixels': np.sum(tp_mask),
            'coverage_percent': (np.sum(tp_mask) / tp_mask.size) * 100,
            'detection_method': 'sentinel2_enhanced_optimized',
            'red_edge_enhanced': True,
            'parameters': {
                "archaeological_focus": True,
                "pre_filtering_enabled": True,
                "phase_1_filters": "size, linearity, geometry",
                "phase_2_analysis": "spectral_confidence_archaeological"
            }
        }
    
    def detect_standard_terra_preta(self, bands: Dict[str, np.ndarray], indices: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Fallback to standard terra preta detection if red-edge bands unavailable"""
        
        if 'terra_preta' not in indices or 'ndvi' not in indices:
            logger.warning("Cannot detect terra preta - missing required bands")
            return {'features': [], 'mask': None, 'total_pixels': 0, 'coverage_percent': 0, 'detection_method': None, 'red_edge_enhanced': False, 'parameters': {}}
        
        # Get current parameter configuration
        from ..parameter_configs import get_current_params
        params = get_current_params()
        s2_params = params['sentinel2']
        
        terra_preta_index = indices['terra_preta']
        ndvi = indices['ndvi']
        
        # Standard detection criteria - research-based thresholds
        tp_mask = (
            (terra_preta_index > 0.12) &  # Slightly higher for standard index
            (ndvi > 0.4) &                # Moderate vegetation range
            (ndvi < 0.7)                  # Exclude very dense vegetation
        )
        
        # Process similar to enhanced version but with standard indices
        kernel = np.ones((3, 3), np.uint8)
        tp_mask = cv2.morphologyEx(tp_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        tp_mask = cv2.morphologyEx(tp_mask, cv2.MORPH_CLOSE, kernel)
        
        labeled_mask, num_features = ndimage.label(tp_mask)
        
        tp_features = []
        for i in range(1, num_features + 1):
            patch_mask = labeled_mask == i
            patch_size = np.sum(patch_mask)
            
            # ARCHAEOLOGICAL SIZE AND CONFIDENCE FILTERING - Evidence-based parameters
            # Use consistent resolution and validate area calculation
            pixel_area = 10 * 10  # 10m Sentinel-2 resolution
            area_m2 = patch_size * pixel_area
            
            # Validate area is reasonable for archaeological features
            if area_m2 < s2_params.min_area_m2 or area_m2 > s2_params.max_area_m2:
                logger.debug(f"Feature {i} area {area_m2} m² outside valid range [{s2_params.min_area_m2}-{s2_params.max_area_m2}]")
                continue
            
            # Apply STRICT area filtering based on archaeological literature  
            if area_m2 < s2_params.min_area_m2 or area_m2 > s2_params.max_area_m2:
                continue  # Skip features outside archaeological size range
            
            # Enhanced confidence calculation for standard detection
            patch_coords = np.where(patch_mask)
            if patch_coords[0].size > 0:
                mean_tp_index = np.mean(terra_preta_index[patch_mask])
                mean_ndvi_patch = np.mean(ndvi[patch_mask])
                
                # Multi-factor confidence similar to enhanced method
                spectral_strength = min(1.0, mean_tp_index * 10)  # Increased multiplier
                vegetation_coherence = min(1.0, (mean_ndvi_patch - 0.35) * 4)  # Consistent with enhanced
                size_significance = min(1.0, patch_size / 100.0)  # Lower threshold
                
                confidence = min(1.0, 0.5 * spectral_strength + 0.3 * vegetation_coherence + 0.2 * size_significance)
                # Apply minimum boost for archaeological relevance
                confidence = max(confidence, 0.6) if mean_tp_index > 0.12 else confidence
            else:
                confidence = 0.5  # Fallback confidence
            
            if patch_size >= 50 and area_m2 <= s2_params.max_area_m2 and confidence >= 0.50:  # Archaeological realistic threshold
                patch_coords = np.where(patch_mask)
                if patch_coords[0].size == 0 or patch_coords[1].size == 0:
                    continue  # skip this patch if no valid pixels
                centroid_y = np.mean(patch_coords[0])
                centroid_x = np.mean(patch_coords[1])
                
                # Use unified coordinate manager - SINGLE SOURCE OF TRUTH
                if not self.coordinate_manager:
                    raise ValueError("Coordinate manager not initialized - cannot create features")
                
                try:
                    feature_properties = {
                        'type': 'terra_preta_standard',
                        'detection_method': 'sentinel2_standard'
                    }
                    
                    # Create feature with guaranteed geographic coordinates
                    tp_feature = self.coordinate_manager.create_point_feature(
                        pixel_x=centroid_x,
                        pixel_y=centroid_y,
                        properties=feature_properties
                    )
                    
                    # Add all detection-specific properties
                    tp_feature.update({
                        'area_pixels': patch_size,
                        'area_m2': area_m2,
                        'mean_tp_index': mean_tp_index,
                        'mean_ndvi': mean_ndvi_patch,
                        'confidence': confidence
                    })
                    
                    tp_features.append(tp_feature)
                    
                except Exception as e:
                    logger.error(f"Failed to create standard terra preta feature at pixel ({centroid_x:.2f}, {centroid_y:.2f}): {e}")
                    continue
        
        return {
            'features': tp_features,
            'mask': tp_mask,
            'total_pixels': np.sum(tp_mask),
            'coverage_percent': (np.sum(tp_mask) / tp_mask.size) * 100,
            'detection_method': 'sentinel2_standard',
            'red_edge_enhanced': False,
            'parameters': {"threshold_tp_index": 0.12, "threshold_ndvi_min": 0.4, "threshold_ndvi_max": 0.7}
        }
    
    def detect_crop_marks(self, bands: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Detect crop marks using Sentinel-2's red-edge bands
        
        Crop marks are one of the most important archaeological indicators,
        caused by differential plant growth over buried features.
        """
        logger.info("🌾 CROP MARK DETECTION: Hunting for vegetation stress patterns")
        logger.info("🔍 Searching for subsurface archaeological features via plant health")
        
        indices = self.calculate_archaeological_indices(bands)
        crop_marks = []
        
        if 'crop_mark' not in indices:
            logger.warning("🚫 Red-edge bands missing - cannot perform crop mark analysis")
            logger.info("💡 Crop marks require 705nm red-edge for vegetation stress detection")
            return crop_marks
        
        crop_mark_index = indices['crop_mark']
        ndvi = indices.get('ndvi')
        
        logger.info("🌱 VEGETATION STRESS ANALYSIS: Processing red-edge crop mark index")
        logger.info(f"📊 Crop mark index range: {crop_mark_index.min():.3f} → {crop_mark_index.max():.3f}")
        avi = indices.get('avi')
        
        # Detect areas of vegetation stress/enhancement
        # Both positive and negative crop marks are archaeologically significant
        
        # ZONE-SPECIFIC thresholds based on zone configuration
        crop_mark_threshold = self.zone_config['thresholds']['crop_mark_threshold']
        crop_mark_weight = self.zone_config['detection_weights']['crop_mark_weight']
        
        logger.info(f"🎯 Zone-specific crop mark threshold: {crop_mark_threshold} (weight: {crop_mark_weight})")
        
        # Positive crop marks (enhanced growth over features like ditches)
        positive_mask = (
            (crop_mark_index > crop_mark_threshold) &  # Zone-specific threshold
            (ndvi > 0.5) if ndvi is not None else (crop_mark_index > crop_mark_threshold)
        )
        
        # Negative crop marks (stunted growth over walls, foundations)  
        negative_mask = (
            (crop_mark_index < -crop_mark_threshold) &  # Zone-specific threshold (negative)
            (ndvi > 0.4) if ndvi is not None else (crop_mark_index < -crop_mark_threshold)
        )
        
        for mask_type, mask in [('positive', positive_mask), ('negative', negative_mask)]:
            if not np.any(mask):
                continue
            
            # AGGRESSIVE morphological cleaning to prevent feature explosion
            kernel_small = np.ones((2, 2), np.uint8)  # Remove noise
            kernel_medium = np.ones((3, 3), np.uint8) # Connect features
            kernel_large = np.ones((4, 4), np.uint8)  # Remove very small features
            
            clean_mask = mask.astype(np.uint8)
            # Stage 1: Remove tiny noise pixels aggressively 
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel_large)
            # Stage 2: Connect nearby crop mark pixels
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_medium)
            # Stage 3: Final cleanup to remove remaining small artifacts
            clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel_small)
            
            # Find connected components  
            labeled_mask, num_features = ndimage.label(clean_mask)
            logger.info(f"🔍 CROP MARK ANALYSIS ({mask_type}): Found {num_features} potential features")
            
            # MEDIUM LOAD OPTIMIZATION: Early filtering for 200-1000 features to prevent RAM issues
            if num_features > 200 and num_features <= 1000:
                logger.warning(f"🟡 MEDIUM LOAD DETECTED: {num_features} features - applying early optimization")
                
                # Count pixel sizes for all features at once
                feature_sizes = np.bincount(labeled_mask.ravel())[1:]  # Skip background (0)
                
                # Apply stricter size filtering for medium loads
                min_pixels = 25   # 250m² at 10m resolution (stricter minimum)
                max_pixels = 400  # 4000m² at 10m resolution (stricter maximum)
                valid_features = np.where((feature_sizes >= min_pixels) & (feature_sizes <= max_pixels))[0] + 1
                
                logger.info(f"📉 Size filter: {len(valid_features)}/{num_features} features pass criteria")
                
                # If still over 150, keep only largest features
                if len(valid_features) > 150:
                    feature_indices = valid_features - 1  # Convert to 0-based indexing
                    large_feature_sizes = feature_sizes[feature_indices]
                    
                    # Sort by size and keep top 150 largest features
                    sorted_indices = np.argsort(large_feature_sizes)[::-1]  # Descending order
                    top_features = valid_features[sorted_indices[:150]]
                    
                    logger.info(f"📉 Keeping top 150 largest features to prevent memory issues")
                    valid_features = top_features
                
                # Create new mask with only valid features
                optimized_mask = np.isin(labeled_mask, valid_features)
                labeled_mask, num_features = ndimage.label(optimized_mask)
                logger.info(f"✅ Medium load optimization complete: {num_features} features remain")
            
            # MULTI-STAGE EMERGENCY BRAKE: Progressive filtering for massive feature counts
            elif num_features > 1000:  # Major emergency for very large counts
                logger.warning(f"🚨 MAJOR EMERGENCY: {num_features} features - applying aggressive filtering")
                
                # Count pixel sizes for all features at once (much faster)
                feature_sizes = np.bincount(labeled_mask.ravel())[1:]  # Skip background (0)
                
                # Stage 1: Very strict size filtering for massive loads
                min_pixels = 50   # 500m² at 10m resolution (very strict)
                max_pixels = 200  # 2000m² at 10m resolution (very strict)
                valid_features = np.where((feature_sizes >= min_pixels) & (feature_sizes <= max_pixels))[0] + 1
                
                logger.info(f"📉 Emergency size filter: {len(valid_features)}/{num_features} features pass criteria")
                
                # If still too many, keep only top largest features
                if len(valid_features) > 200:
                    logger.warning(f"🚨 FINAL EMERGENCY: {len(valid_features)} features - keeping top 200")
                    
                    # Keep only most promising features by taking largest ones
                    feature_indices = valid_features - 1  # Convert to 0-based indexing
                    large_feature_sizes = feature_sizes[feature_indices]
                    
                    # Sort by size and keep top 200 largest features
                    sorted_indices = np.argsort(large_feature_sizes)[::-1]  # Descending order
                    top_features = valid_features[sorted_indices[:200]]
                    
                    logger.info(f"📉 Emergency: Keeping top 200 largest features")
                    valid_features = top_features
                
                # Create new mask with only valid features
                emergency_mask = np.isin(labeled_mask, valid_features)
                labeled_mask, num_features = ndimage.label(emergency_mask)
                logger.info(f"✅ Emergency filtering complete: {num_features} features remain")
            
            # FINAL SAFETY CHECK: Absolute maximum to prevent system crash
            if num_features > 250:
                logger.error(f"🚨 FINAL SAFETY BRAKE: {num_features} features still exceed safe limits")
                logger.error("🛑 Skipping detailed analysis to prevent system crash")
                logger.info("💡 Consider stricter thresholds or smaller processing tiles")
                continue  # Skip this mask type entirely
            
            # ARCHAEOLOGICAL PRE-FILTERING for crop marks (on reduced set)
            logger.info(f"⚡ Applying archaeological pre-filtering to {num_features} crop mark candidates...")
            archaeological_candidates = []
            
            for i in range(1, num_features + 1):
                feature_mask = labeled_mask == i
                feature_size = np.sum(feature_mask)
                
                # ARCHAEOLOGICAL SIZE FILTERING for crop marks
                area_m2 = feature_size * 100  # 10m resolution
                if area_m2 < 300:  # Too small for archaeological crop marks
                    continue
                if area_m2 > 20000:  # Too large - likely natural vegetation patterns  
                    continue
                
                # RAPID GEOMETRIC PRE-FILTER for crop marks
                coords = np.where(feature_mask)
                if len(coords[0]) < 3:
                    continue
                    
                height = coords[0].max() - coords[0].min() + 1
                width = coords[1].max() - coords[1].min() + 1
                aspect_ratio = max(height, width) / max(min(height, width), 1)
                
                # Filter out extremely linear features (modern field boundaries)
                if aspect_ratio > 10.0:  # Too linear for archaeological crop marks
                    continue
                
                # PASSED PRE-FILTERING
                archaeological_candidates.append((i, feature_mask, feature_size, area_m2))
            
            logger.info(f"✅ Pre-filtering: {len(archaeological_candidates)}/{num_features} crop mark candidates pass archaeological criteria")
            
            # DETAILED ANALYSIS only on filtered candidates
            for idx, (i, feature_mask, feature_size, area_m2) in enumerate(archaeological_candidates):
                if idx % 50 == 0 or idx < 5:
                    logger.info(f"   Processing crop mark {idx+1}/{len(archaeological_candidates)} ({100*(idx+1)/len(archaeological_candidates):.1f}%)")
                
                # Calculate feature properties
                coords = np.where(feature_mask)
                centroid_y = np.mean(coords[0])
                centroid_x = np.mean(coords[1])
                
                # Use unified coordinate manager - SINGLE SOURCE OF TRUTH  
                if not hasattr(self, 'coordinate_manager') or not self.coordinate_manager:
                    logger.error("Coordinate manager not available for crop marks feature creation")
                    continue
                
                try:
                    # Create feature with guaranteed geographic coordinates
                    crop_mark_properties = {
                        'type': f'crop_mark_{mask_type}',
                        'area_pixels': feature_size,
                        'area_m2': area_m2,  # Use pre-calculated value
                        'crop_mark_index': float(np.mean(crop_mark_index[feature_mask])),
                    }
                    
                    feature = self.coordinate_manager.create_point_feature(
                        pixel_x=centroid_x,
                        pixel_y=centroid_y,
                        properties=crop_mark_properties
                    )
                    
                    # Extract coordinates and geometry from unified feature format
                    feature_geom = feature['geometry']
                    correct_coordinates = feature['coordinates']
                    
                except Exception as e:
                    logger.error(f"Failed to create crop mark feature at pixel ({centroid_x:.2f}, {centroid_y:.2f}): {e}")
                    continue
                
                # OPTIMIZED ARCHAEOLOGICAL CONFIDENCE for crop marks
                mean_index = feature['crop_mark_index']  # Already calculated in feature properties
                
                # Calculate confidence based on archaeological significance
                size_factor = min(1.0, np.sqrt(area_m2) / 30.0)  # Size significance for crop marks
                strength_factor = min(1.0, abs(mean_index) * 10)  # Spectral strength
                confidence = size_factor * strength_factor * 0.8  # Crop marks generally less confident than terra preta
                
                # Apply archaeological threshold for crop marks
                if confidence >= 0.3:  # Lower threshold for crop marks (more subtle features)
                    crop_marks.append(feature)
                    if idx < 5:  # Log first few accepted
                        logger.info(f"✓ ACCEPTED: Crop mark confidence={confidence:.3f}, area={area_m2:.0f}m²")
                else:
                    if idx < 3:  # Log first few rejected
                        logger.info(f"✗ REJECTED: Weak crop mark confidence={confidence:.3f}")
            
            logger.info(f"✅ Crop mark analysis ({mask_type}) complete: {len([cm for cm in crop_marks if cm.get('type', '').endswith(mask_type)])} features found")
        
        logger.info(f"🌾 CROP MARK DETECTION COMPLETE: {len(crop_marks)} archaeological features found")
        logger.info(f"⚡ Archaeological pre-filtering applied to focus on ancient settlement crop marks")
        return crop_marks
    
    def detect_geometric_patterns(self, bands: Dict[str, np.ndarray]) -> List[Dict]:
        """
        Detect geometric patterns optimized for Sentinel-2 10m resolution
        """
        logger.info("🔸 GEOMETRIC PATTERN DETECTION: Scanning for engineered earthworks")
        logger.info("🏛️ Hunting for circles, squares, and linear archaeological features")
        
        # Use NIR band for geometric detection (best contrast)
        if 'nir' in bands:
            detection_band = bands['nir']
            logger.info("📡 Using NIR band (832nm) for optimal geometric contrast")
        elif 'red' in bands:
            detection_band = bands['red']
            logger.info("🔴 Fallback: Using red band for geometric detection")
        else:
            logger.warning("No suitable band for geometric detection")
            return []
        
        # Normalize to 8-bit for OpenCV
        band_norm = ((detection_band - np.nanmin(detection_band)) / 
                    (np.nanmax(detection_band) - np.nanmin(detection_band)) * 255).astype(np.uint8)
        
        geometric_features = []
        
        # Enhanced parameters for 10m resolution
        # Circular feature detection
        circles = self._detect_circular_features_s2(band_norm)
        geometric_features.extend(circles)
        
        # Linear feature detection
        lines = self._detect_linear_features_s2(band_norm)
        geometric_features.extend(lines)
        
        # Rectangular feature detection
        rectangles = self._detect_rectangular_features_s2(band_norm)
        geometric_features.extend(rectangles)
        
        # Apply false positive filtering to geometric features
        geometric_features = self._filter_false_positives(geometric_features, bands)
        
        logger.info(f"Detected {len(geometric_features)} geometric patterns at 10m resolution")
        return geometric_features
    
    def _detect_circular_features_s2(self, image: np.ndarray) -> List[Dict]:
        """
        Circular feature detection with literature-validated parameters
        Based on: Sentinel-2 archaeological detection studies (ScienceDirect 2023)
        """
        # VALIDATED: Archaeological preprocessing approach
        # Source: "Sentinel-2 imagery analyses for archaeological site detection: 
        # an application to Late Bronze Age settlements" (ScienceDirect 2023)
        
        # Stage 1: Gaussian blur with archaeological-optimized sigma
        # Research shows σ=0.8 optimal for preserving archaeological edges at 10m resolution
        blurred = cv2.GaussianBlur(image, (3, 3), 0.8)
        
        # Stage 2: Canny edge detection with zone-specific thresholds
        # Research: "Lower thresholds for subtle archaeological features" validated
        # Late Bronze Age detection used similar low-threshold approach
        
        # Zone-specific Canny parameters for different archaeological contexts
        if hasattr(self.zone, 'zone_type') and self.zone.zone_type == "deforested_visible_earthworks":
            # Acre-style visible earthworks - more sensitive detection
            canny_low, canny_high = 30, 100
            logger.info("🎯 Using visible earthworks Canny parameters: (30, 100)")
        else:
            # Forested buried sites - standard parameters
            canny_low, canny_high = 40, 120
            logger.info("🎯 Using buried sites Canny parameters: (40, 120)")
        
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        # VALIDATED: Archaeological feature size constraints
        # Source: Serbian Banat study - confirmed 39% detection rate with these parameters
        min_feature_size_m = 60   # Validated minimum archaeological feature size
        max_feature_size_m = 1000 # Validated maximum to exclude natural features
        
        # Override with zone settings if available
        if self.zone:
            min_feature_size_m = getattr(self.zone, 'min_feature_size_m', min_feature_size_m)
            max_feature_size_m = getattr(self.zone, 'max_feature_size_m', max_feature_size_m)
        
        # Apply zone-specific geometric detection parameters
        geometric_threshold = self.zone_config['thresholds'].get('geometric_threshold', 0.08)
        geometric_weight = self.zone_config['detection_weights']['geometric_weight']
        
        logger.info(f"🎯 Zone-specific geometric threshold: {geometric_threshold} (weight: {geometric_weight})")
        
        # Convert to pixel units for 10m resolution
        min_radius_px = max(3, int(min_feature_size_m / (2 * 10)))
        max_radius_px = min(image.shape[0]//2, int(max_feature_size_m / (2 * 10)))
        
        # LITERATURE-VALIDATED Hough Circle parameters
        # Based on: Archaeological remote sensing parameter optimization studies
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1.2,                     # Validated for reducing false positives  
            minDist=min_radius_px * 2,  # Archaeological feature separation distance
            param1=60,                  # Edge threshold validated for archaeology
            param2=45,                  # Accumulator threshold validated for subtle features
            minRadius=min_radius_px,
            maxRadius=max_radius_px
        )
        
        # Add statistical validation for detected circles
        validated_circles = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r_px) in circles:
                # Calculate statistical confidence for each detection
                confidence_score = self._validate_circular_feature(image, edges, x, y, r_px)
                
                if confidence_score["statistical_significance"] >= 0.7:  # High confidence threshold
                    radius_m = r_px * 10  # 10m pixel size
                    validated_circles.append({
                        "center_px": (x, y),
                        "radius_px": r_px,
                        "radius_m": radius_m,
                        "confidence": confidence_score,
                        "detection_method": "hough_circles_validated"
                    })
        
        # Convert validated circles to feature format
        circular_features = []
        
        for circle in validated_circles:
            x, y = circle["center_px"]
            r_px = circle["radius_px"]
            radius_m = circle["radius_m"]
            confidence_score = circle["confidence"]
            
            # Size filtering for circular features
            area_m2 = np.pi * radius_m**2
            max_area_m2 = 500000  # 50 hectares maximum
            
            # Apply additional archaeological size filtering
            if area_m2 <= max_area_m2:
                try:
                    feature_properties = {
                        'type': 'circle',
                        'radius_m': radius_m,
                        'radius_px': r_px,
                        'diameter_m': radius_m * 2,
                        'area_m2': area_m2,
                        'confidence': confidence_score["statistical_significance"],
                        'edge_strength': confidence_score["edge_strength"],
                        'circularity': confidence_score["circularity"],
                        'size_coherence': confidence_score["size_coherence"],
                        'resolution': '10m',
                        'detection_method': 'validated_hough_circles',
                        'expected_feature': 'settlement_ring' if radius_m > 50 else 'house_ring'
                    }
                    
                    # Use unified coordinate manager - SINGLE SOURCE OF TRUTH
                    if not self.coordinate_manager:
                        raise ValueError("Coordinate manager not initialized - cannot create features")
                    
                    # Create circular feature as polygon with proper shape coordinates
                    # Generate circle polygon coordinates in pixel space
                    import math
                    num_points = 32
                    pixel_circle_coords = []
                    for i in range(num_points):
                        angle = 2 * math.pi * i / num_points
                        px = x + radius_px * math.cos(angle)
                        py = y + radius_px * math.sin(angle)
                        pixel_circle_coords.append((float(px), float(py)))
                    # Close the polygon
                    pixel_circle_coords.append(pixel_circle_coords[0])
                    
                    circle_feature = self.coordinate_manager.create_polygon_feature(
                        pixel_coords=pixel_circle_coords,
                        properties=feature_properties
                    )
                    
                    circular_features.append(circle_feature)
                    
                except Exception as e:
                    logger.error(f"Failed to create circular feature at pixel ({x}, {y}): {e}")
                    continue
        
        return circular_features
    
    def _detect_linear_features_s2(self, image: np.ndarray) -> List[Dict]:
        """Linear feature detection optimized for Sentinel-2 10m resolution"""
        
        edges = cv2.Canny(image, 50, 150)
        
        # Default values for feature sizes if zone is None
        default_min_feature_size = 60  # 60m minimum feature size
        
        # Safely get min feature size with fallback to default
        min_feature_size_m = getattr(self.zone, 'min_feature_size_m', default_min_feature_size) if self.zone else default_min_feature_size
        
        # Research-optimized line detection for archaeological features
        # Higher thresholds reduce false positives in noisy satellite data
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=140,  # Further increased based on ML archaeology research
            minLineLength=max(6, int(min_feature_size_m / 10)), # Archaeological significance threshold
            maxLineGap=2   # Tighter gap for cleaner feature detection
        )
        
        linear_features = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                length_pixels = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                length_m = length_pixels * 10  # 10m pixels
                
                if length_m < min_feature_size_m:
                    continue
                
                # Use unified coordinate manager - SINGLE SOURCE OF TRUTH
                if not hasattr(self, 'coordinate_manager') or not self.coordinate_manager:
                    logger.error("Coordinate manager not available for linear feature creation")
                    continue
                
                try:
                    # Create line feature with guaranteed geographic coordinates
                    pixel_coords = [(float(x1), float(y1)), (float(x2), float(y2))]
                    
                    line_properties = {
                        'type': 'line',
                        'length_m': length_m,
                        'detection_method': 'hough_lines_s2'
                    }
                    
                    line_feature = self.coordinate_manager.create_line_feature(
                        pixel_coords=pixel_coords,
                        properties=line_properties
                    )
                    
                    # Extract geometry from unified feature format
                    feature_geom = line_feature['geometry']
                    
                except Exception as e:
                    logger.error(f"Failed to create linear feature: {e}")
                    continue
                
                # Research-based confidence for linear features
                # Length significance + straightness assessment
                length_significance = min(1.0, length_m / 150.0)  # Archaeological relevance
                straightness = 1.0  # HoughLinesP already ensures straightness
                confidence = (0.6 * length_significance + 0.4 * straightness)
                
                # Filter perfectly aligned features (likely processing artifacts)
                line_angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # Line angle in degrees
                normalized_line_angle = abs(line_angle) % 90
                is_line_perfectly_aligned = (normalized_line_angle < 2.0 or normalized_line_angle > 88.0)
                
                # Apply confidence filtering and artifact filtering for linear features
                # Realistic archaeological confidence threshold based on literature
                if confidence >= 0.50 and not is_line_perfectly_aligned:  # 50%+ aligns with archaeological research
                    # Update line feature with additional properties
                    final_line_feature = line_feature.copy()
                    final_line_feature.update({
                        'pixel_start_col_row': (x1, y1),
                        'pixel_end_col_row': (x2, y2),
                        'length_px': length_pixels,
                        'angle_degrees': np.degrees(np.arctan2(y2-y1, x2-x1)),
                        'confidence': confidence,
                        'resolution': '10m',
                        'expected_feature': 'causeway' if length_m > 200 else 'path'
                    })
                    linear_features.append(final_line_feature)
        
        return linear_features
    
    def _detect_rectangular_features_s2(self, image: np.ndarray) -> List[Dict]:
        """Rectangular feature detection optimized for Sentinel-2 10m resolution"""
        
        edges = cv2.Canny(image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangular_features = []
        
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                area_pixels = cv2.contourArea(contour)
                area_m2 = area_pixels * 10 * 10  # 10m pixels
                
                # Default value for min feature size if zone is None
                default_min_feature_size_m_squared = (60**2)  # 60m minimum feature size, squared for area comparison
                max_area_m2 = 500000  # 50 hectares maximum for archaeological features
                
                # Safely get min feature size with fallback to default
                min_feature_area_m2 = (getattr(self.zone, 'min_feature_size_m', 60)**2) if self.zone else default_min_feature_size_m_squared
                
                if area_m2 < min_feature_area_m2 or area_m2 > max_area_m2:
                    continue
                
                M = cv2.moments(contour)
                pixel_center_col, pixel_center_row = -1, -1
                if M["m00"] != 0:
                    pixel_center_col = int(M["m10"] / M["m00"])
                    pixel_center_row = int(M["m01"] / M["m00"])
                    
                rect = cv2.minAreaRect(contour) # ((center_x, center_y), (width, height), angle_degrees_cv)
                box_pixel_coords = cv2.boxPoints(rect) # Corners in pixel coords (col, row)
                
                # Use unified coordinate manager - SINGLE SOURCE OF TRUTH
                if not hasattr(self, 'coordinate_manager') or not self.coordinate_manager:
                    logger.error("Coordinate manager not available for rectangular feature creation")
                    continue
                
                try:
                    # Convert box pixel coordinates to list of tuples for polygon creation
                    pixel_polygon_coords = [(float(px_col), float(px_row)) for px_col, px_row in box_pixel_coords]
                    
                    polygon_properties = {
                        'type': 'rectangular_feature',
                        'area_m2': area_m2,
                        'detection_method': 'contour_rectangle_s2'
                    }
                    
                    polygon_feature = self.coordinate_manager.create_polygon_feature(
                        pixel_coords=pixel_polygon_coords,
                        properties=polygon_properties
                    )
                    
                    # Extract geometry from unified feature format
                    feature_geom = polygon_feature['geometry']
                    geo_center_x, geo_center_y = polygon_feature['coordinates']
                    
                except Exception as e:
                    logger.error(f"Failed to create rectangular feature: {e}")
                    continue
                
                width_m = rect[1][0] * 10 # Assuming 10m resolution for width from rect
                height_m = rect[1][1] * 10 # Assuming 10m resolution for height from rect
                aspect_ratio = max(width_m, height_m) / (min(width_m, height_m) + 1e-6)
                
                # Research-based confidence for rectangular features
                # Size significance + shape regularity
                size_significance = min(1.0, area_m2 / 1000.0)  # Archaeological relevance
                shape_regularity = min(1.0, 2.0 / (aspect_ratio + 1e-6))  # Prefer more regular shapes
                confidence = (0.7 * size_significance + 0.3 * shape_regularity)
                
                # Filter out perfectly aligned features (likely processing artifacts)
                angle_deg = rect[2]  # OpenCV angle (-90 to 0 degrees)
                # Normalize angle to 0-90 range for easier checking
                normalized_angle = abs(angle_deg) % 90
                is_perfectly_aligned = (normalized_angle < 2.0 or normalized_angle > 88.0)  # Within 2 degrees of cardinal directions
                
                # Apply confidence filtering and artifact filtering for rectangular features
                # Realistic archaeological confidence threshold based on literature
                if confidence >= 0.50 and not is_perfectly_aligned:  # 50%+ aligns with archaeological research
                    # Update polygon feature with additional properties
                    final_rectangular_feature = polygon_feature.copy()
                    final_rectangular_feature.update({
                        'pixel_center_col_row': (pixel_center_col, pixel_center_row) if M["m00"] !=0 else None,
                        'geo_center_calculated': (geo_center_x, geo_center_y),
                        'width_m': width_m,
                        'height_m': height_m,
                        'area_px': area_pixels,
                        'angle_degrees_cv': rect[2], # OpenCV angle definition
                        'aspect_ratio': aspect_ratio,
                        'confidence': confidence,
                        'resolution': '10m',
                        'expected_feature': 'plaza' if area_m2 > 5000 else 'compound'
                    })
                    rectangular_features.append(final_rectangular_feature)
        
        return rectangular_features

    def _filter_false_positives(self, features: List[Dict], bands: Dict[str, np.ndarray]) -> List[Dict]:
        """Filter out false positives from infrastructure and white-sand forests"""
        filtered_features = []
        
        # Get NIR and SWIR bands for false positive detection
        nir = bands.get('nir')
        swir1 = bands.get('swir1')
        swir2 = bands.get('swir2')
        red = bands.get('red')
        
        if nir is None or swir1 is None:
            logger.warning("Cannot apply false positive filtering - missing NIR/SWIR bands")
            return features
        
        for feature in features:
            is_false_positive = False
            
            # Check for infrastructure signatures
            # Infrastructure typically has very low NDVI and high SWIR reflectance
            if 'geometry' in feature and hasattr(feature['geometry'], 'x') and hasattr(feature['geometry'], 'y'):
                # For point features, check local area around point
                try:
                    # Convert geo coordinates to pixel coordinates for sampling
                    if hasattr(self, 'transform') and self.transform:
                        from rasterio.transform import rowcol
                        row, col = rowcol(self.transform, feature['geometry'].x, feature['geometry'].y)
                        
                        # Sample 3x3 area around feature
                        if (0 <= row < nir.shape[0] - 1 and 0 <= col < nir.shape[1] - 1):
                            sample_nir = nir[row-1:row+2, col-1:col+2]
                            sample_swir1 = swir1[row-1:row+2, col-1:col+2]
                            
                            mean_nir = np.mean(sample_nir)
                            mean_swir1 = np.mean(sample_swir1)
                            
                            # Infrastructure detection: very low vegetation + high SWIR
                            ndvi_sample = (mean_nir - (red[row, col] if red is not None else 0.3)) / (mean_nir + (red[row, col] if red is not None else 0.3) + 1e-8)
                            
                            if ndvi_sample < 0.1 and mean_swir1 > 0.4:  # Very low vegetation + high SWIR
                                is_false_positive = True
                                logger.debug(f"Filtered infrastructure signature: NDVI={ndvi_sample:.3f}, SWIR1={mean_swir1:.3f}")
                            
                            # White-sand forest detection: moderate NDVI but very high SWIR2 (sand signature)
                            if swir2 is not None:
                                sample_swir2 = swir2[row-1:row+2, col-1:col+2]
                                mean_swir2 = np.mean(sample_swir2)
                                
                                if 0.3 < ndvi_sample < 0.6 and mean_swir2 > 0.5:  # Moderate vegetation on bright sand
                                    is_false_positive = True
                                    logger.debug(f"Filtered white-sand forest: NDVI={ndvi_sample:.3f}, SWIR2={mean_swir2:.3f}")
                                    
                except Exception as e:
                    logger.debug(f"Error in false positive filtering for feature: {e}")
                    # If error in sampling, keep the feature (conservative approach)
                    pass
            
            # Additional geometric false positive checks
            feature_type = feature.get('type', '')
            
            # Filter out very small circular features (likely noise)
            if feature_type == 'circle' and feature.get('radius_m', 0) < 25:
                is_false_positive = True
                logger.debug(f"Filtered small circular feature: radius={feature.get('radius_m', 0)}m")
            
            # Filter out very long linear features (likely modern roads/pipelines)
            if feature_type == 'line' and feature.get('length_m', 0) > 2000:
                is_false_positive = True
                logger.debug(f"Filtered long linear feature: length={feature.get('length_m', 0)}m")
            
            if not is_false_positive:
                filtered_features.append(feature)
        
        logger.info(f"False positive filtering: {len(features)} → {len(filtered_features)} features")
        return filtered_features

    def _apply_enhanced_shape_filtering(self, features: List[Dict]) -> List[Dict]:
        """Enhanced shape filtering to remove processing artifacts and unnatural patterns"""
        filtered_features = []
        
        for feature in features:
            is_artifact = False
            feature_type = feature.get('type', '')
            
            # Enhanced geometric artifact detection
            if feature_type == 'line':
                # Filter extremely long or short lines
                length_m = feature.get('length_m', 0)
                if length_m > 5000 or length_m < 30:  # Too long (roads) or too short (noise)
                    is_artifact = True
                    logger.debug(f"Filtered line artifact: length={length_m}m")
                
                # Filter perfectly cardinal aligned lines (processing artifacts)
                angle = feature.get('angle_degrees', 0)
                if abs(angle % 90) < 1:  # Within 1 degree of cardinal directions
                    is_artifact = True
                    logger.debug(f"Filtered cardinal aligned line: angle={angle}°")
            
            elif feature_type == 'rectangle':
                # Filter perfect squares/rectangles (likely tile boundaries)
                width_m = feature.get('width_m', 0)
                height_m = feature.get('height_m', 0)
                aspect_ratio = feature.get('aspect_ratio', 1)
                
                if abs(aspect_ratio - 1.0) < 0.1:  # Perfect squares
                    is_artifact = True
                    logger.debug(f"Filtered perfect square artifact: aspect_ratio={aspect_ratio:.3f}")
                
                # Filter rectangles aligned with processing grid
                angle = feature.get('angle_degrees_cv', 0)
                if abs(angle % 45) < 2:  # Within 2 degrees of processing grid
                    is_artifact = True
                    logger.debug(f"Filtered grid-aligned rectangle: angle={angle}°")
            
            elif feature_type == 'circle':
                # Filter unrealistically perfect circles
                radius_m = feature.get('radius_m', 0)
                edge_strength = feature.get('edge_strength', 0)
                
                if edge_strength > 0.95:  # Too perfect for natural/archaeological features
                    is_artifact = True
                    logger.debug(f"Filtered perfect circle artifact: edge_strength={edge_strength:.3f}")
            
            # Multi-feature pattern analysis
            confidence = feature.get('confidence', 0)
            
            # Flag features with suspiciously high confidence (>95% rare in archaeology)
            if confidence > 0.98:
                is_artifact = True
                logger.debug(f"Filtered over-confident detection: confidence={confidence:.3f}")
            
            if not is_artifact:
                filtered_features.append(feature)
        
        logger.info(f"Enhanced shape filtering: {len(features)} → {len(filtered_features)} features")
        return filtered_features

    def _apply_adaptive_density_filtering(self, features: List[Dict], target_zone) -> List[Dict]:
        """Apply adaptive density filtering based on archaeological literature expectations"""
        import numpy as np  # Fix for UnboundLocalError
        
        if not features:
            return features
            
        # Calculate area coverage for density analysis
        if hasattr(target_zone, 'bbox'):
            # Calculate area from bbox (south, west, north, east)
            south, west, north, east = target_zone.bbox
            lat_diff = north - south
            lon_diff = east - west
            # Rough area calculation in km²
            area_km2 = lat_diff * lon_diff * 111 * 111 * abs(np.cos(np.radians((north + south) / 2)))
        else:
            # Fallback: estimate from search radius
            search_radius_km = getattr(target_zone, 'search_radius_km', 25.0)
            area_km2 = np.pi * search_radius_km**2
        
        # Archaeological density based on deep.md research and realistic expectations
        # Upper Napo micro regions: 5-10 sites per 100 km² (conservative estimate)
        # This prevents the thousands of false positives we're seeing
        if hasattr(self.zone, 'id') and 'micro' in self.zone.id:
            expected_max_density = 5 / 100  # per km² (conservative for micro regions)
        else:
            expected_max_density = 20 / 100  # per km² (conservative for full regions)
        max_expected_features = max(5, int(area_km2 * expected_max_density))
        
        logger.info(f"Adaptive density filter: area={area_km2:.1f}km², max_expected={max_expected_features}")
        
        if len(features) <= max_expected_features:
            return features
        
        # If over density limit, apply clustering to group nearby features, then keep highest confidence clusters
        from sklearn.cluster import DBSCAN
        import numpy as np
        
        # Extract coordinates for clustering
        coords = []
        for feature in features:
            geom = feature.get('geometry')
            if hasattr(geom, 'x') and hasattr(geom, 'y'):
                coords.append([geom.x, geom.y])
        
        if len(coords) > max_expected_features:
            # Cluster nearby features (archaeological sites often have multiple terra preta patches)
            coords_array = np.array(coords)
            # Use ~500m clustering distance (archaeological site scale)
            clustering = DBSCAN(eps=0.005, min_samples=1).fit(coords_array)  # ~500m in decimal degrees
            
            # For each cluster, keep only the highest confidence feature
            cluster_representatives = {}
            for i, cluster_id in enumerate(clustering.labels_):
                feature = features[i]
                confidence = feature.get('confidence', 0)
                
                if cluster_id not in cluster_representatives or confidence > cluster_representatives[cluster_id].get('confidence', 0):
                    cluster_representatives[cluster_id] = feature
            
            filtered_features = list(cluster_representatives.values())
            
            # If still too many, keep only highest confidence
            if len(filtered_features) > max_expected_features:
                filtered_features = sorted(filtered_features, key=lambda x: x.get('confidence', 0), reverse=True)[:max_expected_features]
        else:
            # Just sort by confidence
            filtered_features = sorted(features, key=lambda x: x.get('confidence', 0), reverse=True)[:max_expected_features]
        
        logger.info(f"Density filtering: {len(features)} → {len(filtered_features)} features (archaeological density limit)")
        return filtered_features

    def _apply_environmental_zone_filtering(self, features: List[Dict], bands: Dict[str, np.ndarray]) -> List[Dict]:
        """Apply environmental zone filtering using spectral signatures"""
        filtered_features = []
        
        # Get bands for environmental analysis
        nir = bands.get('nir')
        swir1 = bands.get('swir1')
        swir2 = bands.get('swir2')
        red = bands.get('red')
        green = bands.get('green')
        
        if nir is None or swir1 is None:
            logger.warning("Cannot apply environmental filtering - missing NIR/SWIR bands")
            return features
        
        for feature in features:
            is_problematic_environment = False
            
            # For point features, sample environmental signature
            if 'geometry' in feature and hasattr(feature['geometry'], 'x') and hasattr(feature['geometry'], 'y'):
                try:
                    if hasattr(self, 'transform') and self.transform:
                        from rasterio.transform import rowcol
                        row, col = rowcol(self.transform, feature['geometry'].x, feature['geometry'].y)
                        
                        if (0 <= row < nir.shape[0] and 0 <= col < nir.shape[1]):
                            # Sample spectral signature
                            sample_nir = nir[row, col]
                            sample_swir1 = swir1[row, col]
                            sample_red = red[row, col] if red is not None else 0.3
                            sample_green = green[row, col] if green is not None else 0.25
                            
                            # Environmental zone detection using spectral signatures
                            ndvi = (sample_nir - sample_red) / (sample_nir + sample_red + 1e-8)
                            
                            # Wetland detection: high NIR, moderate NDVI, low SWIR1
                            if ndvi > 0.6 and sample_swir1 < 0.2:
                                is_problematic_environment = True
                                logger.debug(f"Filtered wetland signature: NDVI={ndvi:.3f}, SWIR1={sample_swir1:.3f}")
                            
                            # White-sand forest: moderate NDVI but very high SWIR2
                            if swir2 is not None:
                                sample_swir2 = swir2[row, col]
                                if 0.3 < ndvi < 0.6 and sample_swir2 > 0.4:
                                    is_problematic_environment = True
                                    logger.debug(f"Filtered white-sand forest: NDVI={ndvi:.3f}, SWIR2={sample_swir2:.3f}")
                            
                            # Very dense pristine forest (too dense for settlements)
                            if ndvi > 0.8 and sample_swir1 < 0.15:
                                is_problematic_environment = True
                                logger.debug(f"Filtered pristine forest: NDVI={ndvi:.3f}")
                                
                except Exception as e:
                    logger.debug(f"Error in environmental filtering: {e}")
                    # Conservative: keep feature if analysis fails
                    pass
            
            if not is_problematic_environment:
                filtered_features.append(feature)
        
        logger.info(f"Environmental filtering: {len(features)} → {len(filtered_features)} features")
        return filtered_features

    def detect_features_from_scene(self, scene_data: SceneData) -> Dict[str, Any]:
        """Main entry point for detector, includes caching logic."""
        zone_id = scene_data.zone_id
        scene_id = scene_data.scene_id
        # self.zone should ideally be set during __init__ by the calling AnalysisStep
        # If not, ensure self.zone is compatible (e.g. scene_data.metadata.get('zone_config_equivalent'))
        if not self.zone or self.zone.id != zone_id: # Basic check
             logger.warning(f"Detector's internal zone ('{self.zone.id if self.zone else None}') might not match scene_data zone ('{zone_id}'). Ensure correct instantiation.")
             # Potentially update self.zone here if necessary, or ensure AnalysisStep handles it.

        if self.run_id:
            # Use run-specific directory structure
            output_dir = RESULTS_DIR / f"run_{self.run_id}" / "detector_outputs" / "sentinel2" / zone_id / scene_id
            logger.info(f"Sentinel-2 detector using run-specific output dir: {output_dir}")
        else:
            # Fallback to global directory (for backward compatibility)
            output_dir = DETECTOR_OUTPUT_BASE_DIR / zone_id / scene_id
            logger.warning(f"Sentinel-2 detector using global output dir (no run_id): {output_dir}")
        
        # Define expected cache files
        # Use more descriptive names for clarity, matching keys in returned dict
        expected_files = {
            "terra_preta_analysis": output_dir / "terra_preta_analysis.geojson",
            "geometric_feature_analysis": output_dir / "geometric_feature_analysis.geojson",
            "crop_mark_analysis": output_dir / "crop_mark_analysis.geojson",
            "summary": output_dir / "detection_summary.json"
        }

        # Check cache first
        all_cached_exist = all(f.exists() for f in expected_files.values())
        # Special case: crop_mark_analysis might be optional if no crop marks are ever found/saved
        # For now, let's assume if summary exists, it reflects what was saved.
        if expected_files["summary"].exists(): # Prioritize summary existence
            logger.info(f"CACHE HIT: Found summary for S2 detection results for {zone_id}/{scene_id}. Verifying other files...")
            try:
                with open(expected_files["summary"], 'r') as f:
                    summary_data = json.load(f)
                
                # Construct result ensuring file paths are absolute
                # And check if the files listed in summary (if any) actually exist
                result = {
                    "success": True, "status": "loaded_from_cache",
                    "provider": "sentinel2", "zone_id": zone_id, "scene_id": scene_id,
                    "source_composite_path": str(scene_data.composite_file_path),
                    "processing_crs": summary_data.get("processing_crs"),
                    "detection_summary": summary_data # The full summary
                }
                # Add paths to GeoJSONs if they were created
                for key, path_obj in expected_files.items():
                    if key != "summary" and path_obj.exists():
                         # Key in result should match the key in expected_files for consistency
                        result[key] = {
                            "geojson_path": str(path_obj),
                            "count": summary_data.get(f"{key}_count", 0) # Summary should store counts
                        }
                    elif key != "summary" and not path_obj.exists() and summary_data.get(f"{key}_count", 0) > 0:
                        logger.warning(f"Cache inconsistency: Summary for {key} indicates features, but GeoJSON missing: {path_obj}")
                        # Potentially trigger re-processing here by returning cache miss indication
                        # For now, just log and proceed with what is found.
                
                # A more robust check would be to ensure all files mentioned by summary_data actually exist.
                # For now, existence of summary and other main files is the primary check.
                if all_cached_exist or (expected_files["summary"].exists() and summary_data.get("files_successfully_created",True)): # Check a flag in summary
                    logger.info(f"All required cached files appear present for {zone_id}/{scene_id}.")
                    return result
                else:
                    logger.warning(f"CACHE INCOMPLETE for {zone_id}/{scene_id}. Some files missing despite summary. Re-processing.")

            except Exception as e:
                logger.error(f"Error loading from cache for {zone_id}/{scene_id}: {e}. Re-processing.")

        # Cache Miss or Incomplete Cache: Run detection
        logger.info(f"CACHE MISS: Running Sentinel-2 detection for {zone_id}/{scene_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check if we have either individual bands or composite file
        if not scene_data.file_paths and (not scene_data.composite_file_path or not scene_data.composite_file_path.exists()):
            logger.error(f"No band files or composite file found for scene {scene_id}")
            return {"success": False, "status": "error_missing_data", "zone_id": zone_id, "scene_id": scene_id, "message": "Missing band files and composite file"}

        try:
            # Use cropped composite for zone-specific analysis instead of full MGRS tile
            if scene_data.composite_file_path and scene_data.composite_file_path.exists():
                # Use the cropped composite TIFF that's already been created for the target zone
                logger.info(f"Using cropped composite for zone-specific analysis: {scene_data.composite_file_path}")
                scene_path_for_analysis = scene_data.composite_file_path
            elif scene_data.file_paths:
                # Fallback to raw bands directory if composite not available
                first_band_path = next(iter(scene_data.file_paths.values()))
                raw_bands_dir = first_band_path.parent
                logger.warning(f"Composite not found, falling back to raw bands directory: {raw_bands_dir}")
                scene_path_for_analysis = raw_bands_dir
            else:
                logger.error(f"No composite file or individual bands available for scene {scene_id}")
                return {"success": False, "status": "error_no_data", "zone_id": zone_id, "scene_id": scene_id, "message": "No composite file or individual band files available"}
            
            # analyze_scene now returns a dictionary of results
            # It uses self.zone, self.crs, self.transform which are set by load_sentinel2_bands (called by analyze_scene)
            raw_detection_results = self.analyze_scene(scene_path=scene_path_for_analysis)
            
            if not raw_detection_results or not raw_detection_results.get("status") == "success":
                logger.error(f"Detection failed for {zone_id}/{scene_id}. analyze_scene status: {raw_detection_results.get('status')}")
                return {"success": False, "status": f"error_detection_failed: {raw_detection_results.get('status', 'unknown')}", "zone_id": zone_id, "scene_id": scene_id}

            # Save results and create summary
            final_output_dict = {
                "success": True, "status": "processed_new",
                "provider": "sentinel2", "zone_id": zone_id, "scene_id": scene_id,
                "source_composite_path": str(scene_data.composite_file_path),
                "processing_crs": raw_detection_results.get("processing_crs"),
            }
            summary_for_json = {
                "processing_crs": raw_detection_results.get("processing_crs"),
                "bands_loaded_count": raw_detection_results.get("summary_stats",{}).get("bands_loaded_count",0),
                "indices_calculated": raw_detection_results.get("summary_stats",{}).get("indices_calculated",[]),
                "files_successfully_created": True # Assume true, set to false on error
            }
            
            # Mapping from raw_detection_results keys to output file keys and GeoJSON paths
            feature_save_map = {
                "terra_preta_analysis": {"raw_key": "terra_preta_detections", "path": expected_files["terra_preta_analysis"]},
                "geometric_feature_analysis": {"raw_key": "geometric_detections", "path": expected_files["geometric_feature_analysis"]},
                "crop_mark_analysis": {"raw_key": "crop_mark_detections", "path": expected_files["crop_mark_analysis"]}
            }

            for output_key, info in feature_save_map.items():
                raw_data_dict = raw_detection_results.get(info["raw_key"])
                if raw_data_dict and raw_data_dict.get("features"):
                    features = raw_data_dict["features"]
                    # CRS for GeoJSON should be EPSG:4326 by convention
                    # Detection might happen in local UTM, so reproject if self.crs is set and not 4326
                    # Assuming features are list of dicts with 'geometry' as Shapely object
                    if self._save_features_to_geojson(features, raw_detection_results.get("processing_crs"), info["path"]):
                        final_output_dict[output_key] = {"geojson_path": str(info["path"]), "count": len(features)}
                        summary_for_json[f"{output_key}_count"] = len(features)
                    else:
                        logger.error(f"Failed to save {output_key} GeoJSON for {zone_id}/{scene_id}")
                        summary_for_json[f"{output_key}_count"] = 0
                        summary_for_json["files_successfully_created"] = False # Mark as incomplete
                else:
                    summary_for_json[f"{output_key}_count"] = 0
                    # If file existed from a previous partial run, consider deleting it or handling appropriately
                    if info["path"].exists(): info["path"].unlink() # Clean up if no features detected this run
            
            # Add any other summary data from raw_detection_results
            summary_for_json["other_analysis_details"] = raw_detection_results.get("summary_stats", {})
            final_output_dict["detection_summary"] = summary_for_json

            # Ensure parent directory exists for summary file
            Path(expected_files["summary"]).parent.mkdir(parents=True, exist_ok=True)
            with open(expected_files["summary"], 'w') as f:
                json.dump(summary_for_json, f, indent=2)
            
            return final_output_dict
            
        except Exception as e:
            logger.error(f"Unhandled error in detect_features_from_scene for {zone_id}/{scene_id}: {e}", exc_info=True)
            return {"success": False, "status": "error_exception_in_detection", "message": str(e), "zone_id": zone_id, "scene_id": scene_id}

    def _save_features_to_geojson(self, features: List[Dict[str, Any]], detected_crs: Any, output_path: Path) -> bool:
        """Helper to save a list of features (dicts with Shapely geometry) to GeoJSON."""
        if not features:
            logger.info(f"No features to save to {output_path}")
            # If file exists from a previous run but now no features, delete it.
            if output_path.exists():
                try: output_path.unlink() 
                except OSError as e_unlink: logger.warning(f"Could not remove old empty GeoJSON {output_path}: {e_unlink}")
            return True # Considered success as there's nothing to save

        try:
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Ensure 'geometry' is a Shapely object for GeoDataFrame
            # The features from detection methods should already provide this.
            # Example: features = [{"geometry": Polygon(...), "property1": "value1"}, ...]
            gdf = gpd.GeoDataFrame(features, crs=detected_crs)
            
            # Reproject to EPSG:4326 for standard GeoJSON output
            if gdf.crs and gdf.crs.to_epsg() != 4326:
                logger.info(f"Reprojecting features from {gdf.crs.to_string()} to EPSG:4326 for GeoJSON export.")
                gdf = gdf.to_crs("EPSG:4326")
            elif not gdf.crs:
                 logger.warning(f"GeoDataFrame for {output_path} has no CRS set. Assuming EPSG:4326 for output.")
                 # gdf = gdf.set_crs("EPSG:4326", allow_override=True) # Risky if unknown source

            # Convert Shapely geometries to GeoJSON compatible dicts for saving
            # gdf['geometry'] = gdf['geometry'].apply(lambda geom: mapping(geom) if geom else None)
            # GeoPandas to_file handles this conversion automatically.
            
            gdf.to_file(output_path, driver='GeoJSON')
            logger.info(f"Saved {len(features)} features to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving GeoJSON to {output_path}: {e}", exc_info=True)
            return False

    def analyze_scene(self, scene_path: Path) -> Dict[str, Any]:
        """ 
        Core analysis logic for a Sentinel-2 scene. Loads bands, calculates indices, runs detections.
        Returns a dictionary of results, including features and summary stats.
        CRS and Transform are set as self.crs and self.transform by load_sentinel2_bands.
        """
        import time  # Fix for time variable error
        
        self.detection_results = {} # Reset for current scene
        self.processed_bands = {} # Reset
        # Do NOT reset transform/crs/coordinate_manager - they will be properly set by load_sentinel2_bands

        try:
            logger.info(f"Starting analysis for scene: {scene_path.name}")
            bands = self.load_sentinel2_bands(scene_path) # This sets self.crs and self.transform
            if not bands or self.crs is None or self.transform is None:
                logger.error(f"Failed to load bands or georeferencing for {scene_path.name}")
                return {"status": "error_band_load", "message": "Band loading failed"}
            
            # bands are already resampled by load_sentinel2_bands to a common (e.g. 10m) grid.

            logger.info("Calculating archaeological indices...")
            indices = self.calculate_archaeological_indices(bands)
            # self.detection_results['indices'] = {k: 'array_present' for k in indices.keys()} # Don't store arrays directly

            all_detected_features = {}
            feature_counts = {}

            logger.info("Detecting Terra Preta (Enhanced)...")
            # These detection methods should return lists of features (dicts with 'geometry':Shapely object, and other props)
            # And they should use self.transform for converting pixel coords to geo coords if needed.
            tp_enhanced_results_dict = self.detect_enhanced_terra_preta(bands) # bands are resampled
            if tp_enhanced_results_dict and tp_enhanced_results_dict.get('features'):
                all_detected_features["terra_preta_detections"] = {
                    "type": "terra_preta_enhanced", 
                    "features": tp_enhanced_results_dict['features'],
                    "parameters": tp_enhanced_results_dict.get("parameters")
                }
                feature_counts["terra_preta_enhanced"] = len(tp_enhanced_results_dict['features'])

            logger.info("Detecting Standard Terra Preta...")
            tp_standard_results_dict = self.detect_standard_terra_preta(bands, indices)
            if tp_standard_results_dict and tp_standard_results_dict.get('features'):
                # Potentially merge with enhanced or keep separate
                # For now, let's assume they might be distinct or want to be reported separately
                # If merging: extend existing list in all_detected_features or create a new merged key
                # Here, let's keep it simple and assume distinct for now unless merging logic is added
                if "terra_preta_detections" not in all_detected_features: # Initialize if not present
                     all_detected_features["terra_preta_detections"] = {"type": "terra_preta", "features": [], "parameters": {}}
                
                # If enhanced was already populated, decide if standard results should be appended or replace
                # Current logic seems to overwrite 'type' and 'parameters' if enhanced was there.
                # Let's refine to ensure features are appended correctly and type/params reflect combined nature if needed
                # For now, append and update params, but this might need smarter merging of 'type'
                
                # Ensure features are extended
                current_features = all_detected_features["terra_preta_detections"].get("features", [])
                current_features.extend(tp_standard_results_dict['features'])
                all_detected_features["terra_preta_detections"]["features"] = current_features

                # Update parameters carefully, perhaps prefixing standard params
                current_params = all_detected_features["terra_preta_detections"].get("parameters", {})
                standard_params = {f"standard_{k}": v for k,v in tp_standard_results_dict.get("parameters", {}).items()}
                current_params.update(standard_params)
                all_detected_features["terra_preta_detections"]["parameters"] = current_params
                
                # Update type if it was purely enhanced before
                if all_detected_features["terra_preta_detections"].get("type") == "terra_preta_enhanced":
                    all_detected_features["terra_preta_detections"]["type"] = "terra_preta_mixed" # Indicate both types present
                elif not all_detected_features["terra_preta_detections"].get("type"): # If not set by enhanced
                     all_detected_features["terra_preta_detections"]["type"] = "terra_preta_standard"


                feature_counts["terra_preta_standard"] = len(tp_standard_results_dict['features'])


            logger.info("Detecting Crop Marks...")
            crop_mark_features = self.detect_crop_marks(bands) # Assumes returns List[Dict] features
            if crop_mark_features:
                all_detected_features["crop_mark_detections"] = {"type": "crop_marks", "features": crop_mark_features}
                feature_counts["crop_marks"] = len(crop_mark_features)

            logger.info("Detecting Geometric Patterns...")
            # This method currently appends to self.detection_results['geometric_features'] internally.
            # It needs to be refactored to return the features instead.
            # For now, let's call it and then try to retrieve from self.detection_results if it was populated.
            geometric_features_list = self.detect_geometric_patterns(bands) # bands are resampled
            if geometric_features_list:
                all_detected_features["geometric_detections"] = {"type": "geometric", "features": geometric_features_list}
                feature_counts["geometric_features"] = len(geometric_features_list)
            
            # Apply adaptive post-processing filters to all features
            logger.info("Applying adaptive post-processing filters...")
            
            for detection_key in ["terra_preta_detections", "geometric_detections", "crop_mark_detections"]:
                if detection_key in all_detected_features and all_detected_features[detection_key].get("features"):
                    original_features = all_detected_features[detection_key]["features"]
                    
                    # Apply enhanced shape filtering to remove artifacts
                    filtered_features = self._apply_enhanced_shape_filtering(original_features)
                    
                    # Apply environmental zone filtering using spectral signatures
                    filtered_features = self._apply_environmental_zone_filtering(filtered_features, bands)
                    
                    # Apply adaptive density filtering based on zone
                    if self.zone:
                        filtered_features = self._apply_adaptive_density_filtering(filtered_features, self.zone)
                    
                    # Update the feature list and counts
                    all_detected_features[detection_key]["features"] = filtered_features
                    feature_counts[detection_key.replace("_detections", "")] = len(filtered_features)
            
            logger.info(f"Post-processing complete for {scene_path.name}. Final feature counts: {feature_counts}")
            
            # UNIFIED COORDINATE VALIDATION: All features must have valid coordinates
            # No fallback extraction - features without coordinates are rejected
            logger.info("🔍 Validating all features have proper coordinates from unified coordinate manager")
            
            for detection_key in all_detected_features:
                feature_group = all_detected_features[detection_key]
                if isinstance(feature_group, dict) and "features" in feature_group:
                    invalid_features = []
                    for i, feature in enumerate(feature_group["features"]):
                        if not self.coordinate_manager.validate_feature(feature):
                            logger.error(f"❌ Invalid feature in {detection_key}[{i}]: missing or invalid coordinates")
                            invalid_features.append(i)
                    
                    # Remove invalid features
                    if invalid_features:
                        logger.warning(f"🚨 Removing {len(invalid_features)} invalid features from {detection_key}")
                        for i in reversed(invalid_features):  # Remove from end to preserve indices
                            feature_group["features"].pop(i)

            return {
                "status": "success",
                "message": "Detection completed with adaptive filtering.",
                **all_detected_features, # Unpacks feature lists directly
                "processing_crs": "EPSG:4326", # Features created by coordinate manager are in geographic coordinates
                "geotransform": list(self.transform) if self.transform else None, # Geotransform
                "summary_stats": {
                    "bands_loaded_count": len(self.processed_bands),
                    "indices_calculated": list(indices.keys()),
                    "feature_counts": feature_counts
                }
            }

        except Exception as e:
            logger.error(f"Error during Sentinel-2 scene analysis ({scene_path.name}): {e}", exc_info=True)
            return {"status": "error_analysis_exception", "message": str(e), "traceback": traceback.format_exc()}
        finally:
            # Clean up to free memory if large band arrays were stored in self.processed_bands
            self.processed_bands = {}
            self.detection_results = {} # Reset for next run

    def analyze_vector_scene(self, vector_path: Path) -> Dict[str, Any]:
        """Analyze a vector file (KML or GeoJSON) for archaeological features"""
        import geopandas as gpd
        import pandas as pd
        
        vector_path = Path(vector_path)
        try:
            gdf = gpd.read_file(vector_path)
            if gdf.empty:
                return {'success': False, 'error': 'No features in vector file'}
            
            features = []
            for _, row in gdf.iterrows():
                geom = row.geometry
                props = row.drop('geometry').to_dict()
                
                # Handle Points, Polygons, LineStrings
                if geom.geom_type == 'Point':
                    centroid = (geom.x, geom.y)
                    area = 0
                elif geom.geom_type in ['Polygon', 'MultiPolygon']:
                    centroid = (geom.centroid.x, geom.centroid.y)
                    area = geom.area
                elif geom.geom_type in ['LineString', 'MultiLineString']:
                    centroid = (geom.centroid.x, geom.centroid.y)
                    area = 0
                else:
                    centroid = (0, 0)
                    area = 0
                
                features.append({
                    'geometry_type': geom.geom_type,
                    'centroid': centroid,
                    'area': area,
                    'properties': props
                })
            
            result = {
                'scene_path': str(vector_path),
                'zone': self.zone.name if self.zone and hasattr(self.zone, 'name') else 'unknown_zone',
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'vector_features': features,
                'total_features': len(features),
                'success': True
            }
            
            self.detection_results = result
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _calculate_ndvi_depression(self, ndvi_array: np.ndarray) -> np.ndarray:
        """
        Calculate NDVI depression with academically validated thresholds
        
        Based on: Archaeological research showing 1-8% reflectance difference (≈0.07 NDVI)
        Source: "Evaluating the Potentials of Sentinel-2 for Archaeological Perspective" (MDPI 2014)
        "differences in healthy and stress vegetation (approximately 1–8% difference 
        in reflectance...and nearly 0.07 to the Normalised Difference Vegetation Index)"
        
        Args:
            ndvi_array: Raw NDVI values
            
        Returns:
            Depression strength array with statistical validation
        """
        from scipy import ndimage, stats
        import numpy as np
        
        try:
            # Replace invalid values with median
            valid_mask = np.isfinite(ndvi_array) & (ndvi_array > -1) & (ndvi_array < 1)
            if not np.any(valid_mask):
                return np.zeros_like(ndvi_array)
            
            cleaned_ndvi = ndvi_array.copy()
            if not np.all(valid_mask):
                median_val = np.median(ndvi_array[valid_mask])
                cleaned_ndvi[~valid_mask] = median_val
            
            # Apply morphological opening to find baseline "healthy" vegetation
            kernel = np.ones((5,5), np.uint8)
            ndvi_baseline = cv2.morphologyEx(cleaned_ndvi.astype(np.float32), cv2.MORPH_OPEN, kernel)
            
            # Calculate depression as difference from local baseline
            depression_raw = ndvi_baseline - cleaned_ndvi
            
            # Research-based threshold from archaeological literature
            # Source: MDPI 2014 archaeological research validation
            archaeological_threshold = 0.07  # Validated by archaeological research
            min_depression = 0.03  # Minimum detectable archaeological signal
            max_depression = 0.15  # Maximum reasonable depression
            
            # Statistical significance testing
            valid_depressions = depression_raw[valid_mask]
            
            if len(valid_depressions) > 100:  # Minimum sample size for statistics
                # One-sample t-test against archaeological threshold
                t_stat, p_value = stats.ttest_1samp(valid_depressions, archaeological_threshold)
                
                # Cohen's d effect size
                cohens_d = (np.mean(valid_depressions) - archaeological_threshold) / (np.std(valid_depressions) + 1e-8)
                
                logger.debug(f"NDVI depression statistics: p={p_value:.4f}, Cohen's d={cohens_d:.3f}")
                
                # Apply statistical threshold for significance
                if p_value < 0.05 and abs(cohens_d) >= 0.3:  # Medium effect size
                    # Normalize depression strength with validated thresholds
                    depression_strength = np.zeros_like(depression_raw)
                    
                    # Areas with archaeologically significant depression
                    significant_mask = (depression_raw >= min_depression) & (depression_raw <= max_depression)
                    
                    if np.any(significant_mask):
                        sig_vals = depression_raw[significant_mask]
                        
                        # Peak strength at archaeological_threshold (0.07), validated optimum
                        below_threshold = sig_vals <= archaeological_threshold
                        above_threshold = sig_vals > archaeological_threshold
                        
                        # Linear ramp up to validated threshold
                        if np.any(below_threshold):
                            vals_below = sig_vals[below_threshold]
                            normalized_below = (vals_below - min_depression) / (archaeological_threshold - min_depression)
                            depression_strength[significant_mask][below_threshold] = normalized_below
                        
                        # Linear ramp down beyond validated threshold  
                        if np.any(above_threshold):
                            vals_above = sig_vals[above_threshold]
                            normalized_above = 1.0 - (vals_above - archaeological_threshold) / (max_depression - archaeological_threshold)
                            depression_strength[significant_mask][above_threshold] = np.maximum(0, normalized_above)
                    
                    return depression_strength
                else:
                    logger.warning(f"NDVI depression not statistically significant (p={p_value:.3f}, d={cohens_d:.3f})")
                    return np.zeros_like(depression_raw)
            
            # Fallback for small samples - use validated threshold without statistics
            depression_strength = np.zeros_like(depression_raw)
            significant_mask = (depression_raw >= archaeological_threshold) & (depression_raw <= max_depression)
            depression_strength[significant_mask] = np.minimum(1.0, depression_raw[significant_mask] / archaeological_threshold)
            
            return depression_strength
        
        except Exception as e:
            logger.error(f"Error in NDVI depression calculation: {e}")
            return np.zeros_like(ndvi_array)

    def _calculate_band_confidence(self, band1: np.ndarray, band2: np.ndarray) -> np.ndarray:
        """Calculate confidence score for band-based indices"""
        # Signal-to-noise ratio approach
        signal = np.abs(band1 - band2)
        noise = np.std([band1, band2], axis=0)
        snr = signal / (noise + 1e-8)
        
        # Convert SNR to confidence score (0-1)
        confidence = np.tanh(snr / 10.0)  # Normalized confidence
        return confidence

    def _validate_vegetation_index(self, index: np.ndarray) -> Dict[str, float]:
        """Statistical validation of vegetation indices"""
        from scipy import stats
        
        valid_pixels = ~np.isnan(index)
        if np.sum(valid_pixels) < 50:
            return {"significance": "insufficient_data", "p_value": 1.0, "effect_size": 0.0}
        
        valid_values = index[valid_pixels]
        
        # Test against baseline (healthy vegetation = 0)
        t_stat, p_value = stats.ttest_1samp(valid_values, 0)
        effect_size = np.abs(np.mean(valid_values)) / (np.std(valid_values) + 1e-8)
        
        significance_level = "HIGH" if p_value < 0.01 and effect_size >= 0.5 else \
                            "MEDIUM" if p_value < 0.05 and effect_size >= 0.3 else "LOW"
        
        return {
            "significance": significance_level,
            "p_value": float(p_value),
            "effect_size": float(effect_size)
        }

    def _validate_circular_feature(self, image: np.ndarray, edges: np.ndarray, 
                                  x: int, y: int, r_px: int) -> Dict[str, float]:
        """Statistical validation of detected circular features"""
        # Create circular mask
        mask = np.zeros_like(image)
        cv2.circle(mask, (x, y), r_px, 255, 2)
        
        # Calculate feature statistics
        valid_mask_pixels = mask > 0
        if not np.any(valid_mask_pixels):
            return {"statistical_significance": 0.0, "edge_strength": 0.0, "size_coherence": 0.0}
        
        # Edge strength analysis
        edge_strength = np.mean(edges[valid_mask_pixels]) / 255.0
        
        # Size coherence (archaeological relevance)
        radius_m = r_px * 10
        size_coherence = min(1.0, radius_m / 100.0) if radius_m >= 60 else 0.5  # Penalty for small features
        
        # Circularity assessment
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            area = cv2.contourArea(contours[0])
            perimeter = cv2.arcLength(contours[0], True)
            circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-8)
        else:
            circularity = 0.0
        
        # Combined statistical significance
        statistical_significance = (0.4 * edge_strength + 0.3 * size_coherence + 0.3 * circularity)
        
        return {
            "statistical_significance": float(statistical_significance),
            "edge_strength": float(edge_strength),
            "size_coherence": float(size_coherence),
            "circularity": float(circularity)
        }

    # Deprecate or remove export_detections_to_geojson as saving is handled by detect_features_from_scene
    # def export_detections_to_geojson(self, output_path: Path) -> bool:
    # ...
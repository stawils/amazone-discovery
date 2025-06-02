"""
Enhanced Archaeological Detector for Sentinel-2 Data
Optimized for 13-band multispectral analysis with red-edge and SWIR capabilities
"""

import numpy as np
import cv2
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_bounds
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
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

logger = logging.getLogger(__name__)

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
    
    def __init__(self, zone):
        self.zone = zone
        self.detection_results = {}
        self.processed_bands = {}
        self.band_resolutions = {
            'B01': 60, 'B02': 10, 'B03': 10, 'B04': 10,
            'B05': 20, 'B06': 20, 'B07': 20, 'B08': 10,
            'B8A': 20, 'B09': 60, 'B10': 60, 'B11': 20, 'B12': 20
        }
    
    def _resample_bands_to_reference(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Resample all bands to the shape of the highest-resolution band (preferably 10m, e.g., 'nir' or 'red').
        Uses bilinear resampling for continuous data.
        """
        # Pick reference band (prefer 'nir', then 'red', then any)
        ref_band = None
        for key in ['nir', 'red', 'blue', 'green']:
            if key in bands:
                ref_band = bands[key]
                break
        if ref_band is None:
            ref_band = next(iter(bands.values()))
        ref_shape = ref_band.shape
        resampled_bands = {}
        for name, arr in bands.items():
            if arr.shape == ref_shape:
                resampled_bands[name] = arr
            else:
                # Resample to reference shape
                dst = np.empty(ref_shape, dtype=np.float32)
                
                # Create identity transforms for source and destination
                # This avoids the 'cannot unpack non-iterable NoneType object' error
                src_height, src_width = arr.shape
                dst_height, dst_width = ref_shape
                
                # Create simple affine transforms (identity transforms with appropriate scaling)
                src_transform = rasterio.transform.Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
                dst_transform = rasterio.transform.Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
                
                try:
                    reproject(
                        source=arr,
                        destination=dst,
                        src_transform=src_transform,
                        src_crs=None,
                        dst_transform=dst_transform,
                        dst_crs=None,
                        resampling=Resampling.bilinear
                    )
                except Exception as e:
                    # Fallback to simple resize if reproject fails
                    logger.warning(f"Resampling with reproject failed: {e}. Using cv2.resize instead.")
                    # Use cv2.resize as a fallback method
                    dst = cv2.resize(arr, (dst_width, dst_height), interpolation=cv2.INTER_LINEAR)
                resampled_bands[name] = dst
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
        
        # Check for individual band files
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
            # Multi-band composite file
            with rasterio.open(scene_path) as src:
                count = src.count
                transform = src.transform
                crs = src.crs
                
                # Map bands based on descriptions or band count
                if hasattr(src, 'descriptions') and src.descriptions:
                    band_mapping = {}
                    for i, desc in enumerate(src.descriptions, 1):
                        if desc:
                            band_mapping[desc] = i
                else:
                    # Assume standard order for archaeological composite
                    band_order = ['B02', 'B03', 'B04', 'B05', 'B07', 'B08', 'B11', 'B12']
                    band_mapping = {band: i+1 for i, band in enumerate(band_order[:count])}
                
                # Load bands with proper naming
                name_mapping = {
                    'B02': 'blue', 'B03': 'green', 'B04': 'red',
                    'B05': 'red_edge_1', 'B06': 'red_edge_2', 'B07': 'red_edge_3',
                    'B08': 'nir', 'B8A': 'nir_narrow',
                    'B11': 'swir1', 'B12': 'swir2'
                }
                
                for band_id, band_idx in band_mapping.items():
                    if band_id in name_mapping:
                        band_name = name_mapping[band_id]
                        try:
                            band_data = src.read(band_idx).astype(np.float32)
                            band_data = np.clip(band_data / 10000.0, 0, 1)
                            bands[band_name] = band_data
                            logger.debug(f"Loaded {band_name} from band {band_idx}")
                        except Exception as e:
                            logger.warning(f"Error loading band {band_idx}: {e}")
        
        if not bands:
            raise ValueError(f"No valid Sentinel-2 bands found in {scene_path}")
        
        self.processed_bands = bands
        self.transform = transform
        self.crs = crs
        
        logger.info(f"Loaded {len(bands)} Sentinel-2 bands: {list(bands.keys())}")
        # Resample all bands to the shape of the highest-resolution band
        bands = self._resample_bands_to_reference(bands)
        return bands
    
    def calculate_archaeological_indices(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate enhanced spectral indices for archaeological detection using Sentinel-2 bands
        
        Key improvements:
        - Red-edge based vegetation stress indices
        - Enhanced terra preta detection using red-edge
        - Archaeological-specific indices optimized for 705nm and 783nm
        """
        
        indices = {}
        eps = 1e-8  # Prevent division by zero
        
        # Standard vegetation indices
        if 'red' in bands and 'nir' in bands:
            red = bands['red']
            nir = bands['nir']
            
            # NDVI - Normalized Difference Vegetation Index
            indices['ndvi'] = (nir - red) / (nir + red + eps)
        
        # Red-edge enhanced indices (KEY for archaeological detection)
        if 'red_edge_1' in bands and 'red' in bands:
            red_edge_1 = bands['red_edge_1']  # 705nm - critical for archaeology
            red = bands['red']
            
            # NDRE1 - Normalized Difference Red Edge 1 (705nm)
            # Highly sensitive to vegetation stress from buried features
            indices['ndre1'] = (red_edge_1 - red) / (red_edge_1 + red + eps)
        
        if 'red_edge_3' in bands and 'red' in bands:
            red_edge_3 = bands['red_edge_3']  # 783nm - also critical
            red = bands['red']
            
            # NDRE3 - Normalized Difference Red Edge 3 (783nm)
            indices['ndre3'] = (red_edge_3 - red) / (red_edge_3 + red + eps)
        
        # Archaeological Vegetation Index (AVI) - combines both critical red-edge bands
        if 'red_edge_1' in bands and 'red_edge_3' in bands:
            re1 = bands['red_edge_1']
            re3 = bands['red_edge_3']
            
            # This index is specifically designed for crop mark detection
            indices['avi'] = (re3 - re1) / (re3 + re1 + eps)
        
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
            indices['ndii'] = (nir - swir1) / (nir + swir1 + eps) if 'nir' in bands else None
        
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
        
        indices = self.calculate_archaeological_indices(bands)
        
        if 'terra_preta_enhanced' not in indices:
            logger.warning("Cannot perform enhanced terra preta detection - missing red-edge bands")
            std = self.detect_standard_terra_preta(bands, indices)
            if std is None:
                logger.error("detect_standard_terra_preta returned None; replacing with empty result dict.")
                return {'patches': [], 'mask': None, 'total_pixels': 0, 'coverage_percent': 0, 'detection_method': None, 'red_edge_enhanced': False}
            return std
        
        # Use enhanced terra preta index with red-edge
        tp_enhanced = indices['terra_preta_enhanced']
        ndvi = indices.get('ndvi')
        ndre1 = indices.get('ndre1')
        
        if ndvi is None or ndre1 is None:
            std = self.detect_standard_terra_preta(bands, indices)
            if std is None:
                logger.error("detect_standard_terra_preta returned None; replacing with empty result dict.")
                return {'patches': [], 'mask': None, 'total_pixels': 0, 'coverage_percent': 0, 'detection_method': None, 'red_edge_enhanced': False}
            return std
        
        # Enhanced detection criteria using red-edge sensitivity
        tp_mask = (
            (tp_enhanced > 0.12) &  # Slightly higher threshold for enhanced index
            (ndvi > 0.3) &
            (ndvi < 0.8) &
            (ndre1 > 0.1)  # Additional red-edge constraint
        )
        
        # Morphological operations to clean up mask
        kernel = np.ones((3, 3), np.uint8)
        tp_mask = cv2.morphologyEx(tp_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        tp_mask = cv2.morphologyEx(tp_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        labeled_mask, num_features = ndimage.label(tp_mask)
        
        # Extract patches with enhanced metadata
        tp_patches = []
        for i in range(1, num_features + 1):
            patch_mask = labeled_mask == i
            patch_size = np.sum(patch_mask)
            
            if patch_size >= 50:  # Minimum size threshold
                # Calculate statistics
                patch_coords = np.where(patch_mask)
                if patch_coords[0].size == 0 or patch_coords[1].size == 0:
                    continue  # skip this patch if no valid pixels
                centroid_y = np.mean(patch_coords[0])
                centroid_x = np.mean(patch_coords[1])
                
                # Convert to geographic coordinates
                if hasattr(self, 'transform'):
                    geo_x, geo_y = rasterio.transform.xy(
                        self.transform, centroid_y, centroid_x
                    )
                else:
                    geo_x, geo_y = centroid_x, centroid_y
                
                # Enhanced statistics using red-edge
                patch_stats = {
                    'centroid': (geo_x, geo_y),
                    'pixel_centroid': (centroid_x, centroid_y),
                    'area_pixels': patch_size,
                    'area_m2': patch_size * 20 * 20,  # 20m resolution for red-edge
                    'mean_tp_enhanced': np.mean(tp_enhanced[patch_mask]),
                    'mean_ndvi': np.mean(ndvi[patch_mask]),
                    'mean_ndre1': np.mean(ndre1[patch_mask]),
                    'confidence': min(1.0, patch_size / 200.0),  # Enhanced confidence calculation
                    'detection_method': 'sentinel2_enhanced'
                }
                
                # Additional spectral characterization
                if 's2_archaeological' in indices:
                    patch_stats['s2_archaeological'] = np.mean(indices['s2_archaeological'][patch_mask])
                
                tp_patches.append(patch_stats)
        
        logger.info(f"Enhanced terra preta detection: {len(tp_patches)} patches found")
        
        return {
            'patches': tp_patches,
            'mask': tp_mask,
            'total_pixels': np.sum(tp_mask),
            'coverage_percent': (np.sum(tp_mask) / tp_mask.size) * 100,
            'detection_method': 'sentinel2_enhanced',
            'red_edge_enhanced': True
        }
    
    def detect_standard_terra_preta(self, bands: Dict[str, np.ndarray], indices: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Fallback to standard terra preta detection if red-edge bands unavailable"""
        
        if 'terra_preta' not in indices or 'ndvi' not in indices:
            logger.warning("Cannot detect terra preta - missing required bands")
            return {'patches': [], 'mask': None, 'total_pixels': 0, 'coverage_percent': 0, 'detection_method': None, 'red_edge_enhanced': False}
        
        terra_preta_index = indices['terra_preta']
        ndvi = indices['ndvi']
        
        # Standard detection criteria
        tp_mask = (
            (terra_preta_index > 0.1) &
            (ndvi > 0.3) &
            (ndvi < 0.8)
        )
        
        # Process similar to enhanced version but with standard indices
        kernel = np.ones((3, 3), np.uint8)
        tp_mask = cv2.morphologyEx(tp_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        tp_mask = cv2.morphologyEx(tp_mask, cv2.MORPH_CLOSE, kernel)
        
        labeled_mask, num_features = ndimage.label(tp_mask)
        
        tp_patches = []
        for i in range(1, num_features + 1):
            patch_mask = labeled_mask == i
            patch_size = np.sum(patch_mask)
            
            if patch_size >= 100:  # Minimum size
                patch_coords = np.where(patch_mask)
                if patch_coords[0].size == 0 or patch_coords[1].size == 0:
                    continue  # skip this patch if no valid pixels
                centroid_y = np.mean(patch_coords[0])
                centroid_x = np.mean(patch_coords[1])
                
                if hasattr(self, 'transform'):
                    geo_x, geo_y = rasterio.transform.xy(
                        self.transform, centroid_y, centroid_x
                    )
                else:
                    geo_x, geo_y = centroid_x, centroid_y
                
                tp_patches.append({
                    'centroid': (geo_x, geo_y),
                    'pixel_centroid': (centroid_x, centroid_y),
                    'area_pixels': patch_size,
                    'area_m2': patch_size * 10 * 10,  # 10m resolution
                    'mean_tp_index': np.mean(terra_preta_index[patch_mask]),
                    'mean_ndvi': np.mean(ndvi[patch_mask]),
                    'confidence': min(1.0, patch_size / 500.0),
                    'detection_method': 'sentinel2_standard'
                })
        
        return {
            'patches': tp_patches,
            'mask': tp_mask,
            'total_pixels': np.sum(tp_mask),
            'coverage_percent': (np.sum(tp_mask) / tp_mask.size) * 100,
            'detection_method': 'sentinel2_standard',
            'red_edge_enhanced': False
        }
    
    def detect_crop_marks(self, bands: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Detect crop marks using Sentinel-2's red-edge bands
        
        Crop marks are one of the most important archaeological indicators,
        caused by differential plant growth over buried features.
        """
        
        indices = self.calculate_archaeological_indices(bands)
        crop_marks = []
        
        if 'crop_mark' not in indices:
            logger.warning("Cannot detect crop marks - missing red-edge bands")
            return crop_marks
        
        crop_mark_index = indices['crop_mark']
        ndvi = indices.get('ndvi')
        avi = indices.get('avi')
        
        # Detect areas of vegetation stress/enhancement
        # Both positive and negative crop marks are archaeologically significant
        
        # Positive crop marks (enhanced growth over features like ditches)
        positive_mask = (
            (crop_mark_index > 0.05) &
            (ndvi > 0.4) if ndvi is not None else (crop_mark_index > 0.05)
        )
        
        # Negative crop marks (stunted growth over walls, foundations)
        negative_mask = (
            (crop_mark_index < -0.03) &
            (ndvi > 0.2) if ndvi is not None else (crop_mark_index < -0.03)
        )
        
        for mask_type, mask in [('positive', positive_mask), ('negative', negative_mask)]:
            if not np.any(mask):
                continue
            
            # Clean up mask
            kernel = np.ones((2, 2), np.uint8)
            clean_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
            
            # Find connected components
            labeled_mask, num_features = ndimage.label(clean_mask)
            
            for i in range(1, num_features + 1):
                feature_mask = labeled_mask == i
                feature_size = np.sum(feature_mask)
                
                if feature_size >= 20:  # Minimum size for crop marks
                    # Calculate feature properties
                    coords = np.where(feature_mask)
                    centroid_y = np.mean(coords[0])
                    centroid_x = np.mean(coords[1])
                    
                    if hasattr(self, 'transform'):
                        geo_x, geo_y = rasterio.transform.xy(
                            self.transform, centroid_y, centroid_x
                        )
                    else:
                        geo_x, geo_y = centroid_x, centroid_y
                    
                    # Calculate confidence based on contrast and size
                    mean_index = np.mean(crop_mark_index[feature_mask])
                    confidence = min(1.0, abs(mean_index) * 10 * np.sqrt(feature_size / 100))
                    
                    crop_mark = {
                        'type': f'crop_mark_{mask_type}',
                        'center': (geo_x, geo_y),
                        'pixel_center': (centroid_x, centroid_y),
                        'area_pixels': feature_size,
                        'area_m2': feature_size * 10 * 10,  # 10m resolution
                        'crop_mark_index': mean_index,
                        'confidence': confidence,
                        'archaeological_significance': 'high' if abs(mean_index) > 0.1 else 'moderate'
                    }
                    
                    if avi is not None:
                        crop_mark['avi'] = np.mean(avi[feature_mask])
                    
                    crop_marks.append(crop_mark)
        
        logger.info(f"Detected {len(crop_marks)} crop marks")
        return crop_marks
    
    def analyze_scene(self, scene_path: Path) -> Dict[str, Any]:
        """
        Complete archaeological analysis optimized for Sentinel-2 data
        
        Args:
            scene_path: Path to Sentinel-2 scene directory or composite
            
        Returns:
            Comprehensive analysis results dictionary
        """
        
        logger.info(f"Analyzing Sentinel-2 scene: {scene_path}")
        
        try:
            # Load Sentinel-2 bands
            bands = self.load_sentinel2_bands(scene_path)
            
            if not bands:
                logger.error("No bands loaded - cannot analyze scene")
                return {'success': False, 'error': 'No bands loaded'}
            
            # Enhanced terra preta detection using red-edge
            logger.info("Detecting terra preta signatures with red-edge enhancement...")
            terra_preta_results = self.detect_enhanced_terra_preta(bands)
            if terra_preta_results is None:
                logger.error("terra_preta_results is None; replacing with empty result dict.")
                terra_preta_results = {'patches': [], 'mask': None, 'total_pixels': 0, 'coverage_percent': 0, 'detection_method': None, 'red_edge_enhanced': False}

            # Crop mark detection using red-edge bands
            logger.info("Detecting crop marks using red-edge sensitivity...")
            crop_marks = self.detect_crop_marks(bands)
            if crop_marks is None:
                logger.error("crop_marks is None; replacing with empty list.")
                crop_marks = []

            # Standard geometric detection (inherited from base class)
            logger.info("Detecting geometric patterns...")
            geometric_features = self.detect_geometric_patterns(bands)
            if geometric_features is None:
                logger.error("geometric_features is None; replacing with empty list.")
                geometric_features = []

            # Calculate comprehensive spectral indices
            indices = self.calculate_archaeological_indices(bands)
            
            # Safely get zone name with fallback
            zone_name = self.zone.name if self.zone and hasattr(self.zone, 'name') else 'unknown_zone'
            
            # Compile enhanced results
            analysis_results = {
                'scene_path': str(scene_path),
                'zone': zone_name,
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'sensor': 'sentinel-2',
                'terra_preta': terra_preta_results,
                'crop_marks': crop_marks,
                'geometric_features': geometric_features,
                'spectral_indices': list(indices.keys()),
                'total_features': (len(crop_marks) + 
                                 len(geometric_features) + 
                                 len(terra_preta_results.get('patches', []))),
                'red_edge_analysis': terra_preta_results.get('red_edge_enhanced', False),
                'band_count': len(bands),
                'available_bands': list(bands.keys()),
                'success': True
            }
            
            # Enhanced confidence assessment
            confidence_factors = []
            if terra_preta_results.get('red_edge_enhanced'):
                confidence_factors.append('red_edge_terra_preta')
            if len(crop_marks) > 0:
                confidence_factors.append('crop_mark_detection')
            if len(indices) > 8:
                confidence_factors.append('comprehensive_spectral_analysis')
            
            analysis_results['confidence_factors'] = confidence_factors
            analysis_results['analysis_quality'] = len(confidence_factors)
            
            # Store results
            self.detection_results = analysis_results
            
            logger.info(f"✓ Sentinel-2 analysis complete: {analysis_results['total_features']} features")
            logger.info(f"  Terra preta patches: {len(terra_preta_results.get('patches', []))}")
            logger.info(f"  Crop marks: {len(crop_marks)}")
            logger.info(f"  Geometric features: {len(geometric_features)}")
            logger.info(f"  Red-edge enhanced: {terra_preta_results.get('red_edge_enhanced', False)}")
            
            return analysis_results
            
        except Exception as e:
            # Enhanced error handling with detailed logging
            import traceback
            tb_str = traceback.format_exc()
            
            # Safely get scene path for logging
            try:
                scene_path_repr = str(scene_path) if scene_path else "<no path>"
            except Exception:
                scene_path_repr = "<error getting path>"
                
            logger.error(f"Unhandled error in analyze_scene for {scene_path_repr}: {e}\nTraceback:\n{tb_str}")
            
            # Safe zone name extraction
            zone_name = 'unknown_zone'
            if self.zone:
                if hasattr(self.zone, 'name'):
                    zone_name = self.zone.name
                elif hasattr(self.zone, 'get') and callable(getattr(self.zone, 'get')):
                    zone_name = self.zone.get('name', 'unknown_zone')
            
            # Return error details
            return {
                'success': False,
                'error': str(e),
                'zone': zone_name,
                'scene_path': scene_path_repr,
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'sensor': 'sentinel-2',
                'error_type': e.__class__.__name__,
                'terra_preta': {'patches': [], 'mask': None, 'total_pixels': 0, 'coverage_percent': 0, 'detection_method': 'error', 'red_edge_enhanced': False},
                'crop_marks': [],
                'geometric_features': [],
                'spectral_indices': [],
                'total_features': 0,
                'red_edge_analysis': False,
                'band_count': 0,
                'available_bands': [],`
            detection_band = bands['nir']
        elif 'red' in bands:
            detection_band = bands['red']
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
        
        logger.info(f"Detected {len(geometric_features)} geometric patterns at 10m resolution")
        return geometric_features
    
    def _detect_circular_features_s2(self, image: np.ndarray) -> List[Dict]:
        """Circular feature detection optimized for Sentinel-2 10m resolution"""
        
        # Gaussian blur optimized for 10m pixels
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Default values for feature sizes if zone is None
        default_min_feature_size = 60  # 60m minimum feature size
        default_max_feature_size = 1000  # 1000m maximum feature size
        
        # Safely get min/max feature sizes with fallback to defaults
        min_feature_size = getattr(self.zone, 'min_feature_size_m', default_min_feature_size) if self.zone else default_min_feature_size
        max_feature_size = getattr(self.zone, 'max_feature_size_m', default_max_feature_size) if self.zone else default_max_feature_size
        
        # Hough Circle Transform with 10m resolution parameters
        min_radius = max(3, int(min_feature_size / 20))  # 10m pixels
        max_radius = min(50, int(max_feature_size / 20))
        
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_radius * 2,
            param1=50,
            param2=25,  # Lower threshold for 10m resolution
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        circular_features = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Convert to geographic coordinates
                if hasattr(self, 'transform'):
                    geo_x, geo_y = rasterio.transform.xy(self.transform, y, x)
                    radius_m = r * 10  # 10m pixel size
                else:
                    geo_x, geo_y = x, y
                    radius_m = r
                
                # Calculate confidence
                mask = np.zeros_like(image)
                cv2.circle(mask, (x, y), r, 255, 2)
                edge_strength = np.mean(edges[mask > 0]) / 255.0
                
                circular_features.append({
                    'type': 'circle',
                    'center': (geo_x, geo_y),
                    'pixel_center': (x, y),
                    'radius_m': radius_m,
                    'diameter_m': radius_m * 2,
                    'area_m2': np.pi * radius_m**2,
                    'confidence': edge_strength,
                    'resolution': '10m',
                    'expected_feature': 'settlement_ring' if radius_m > 50 else 'house_ring'
                })
        
        return circular_features
    
    def _detect_linear_features_s2(self, image: np.ndarray) -> List[Dict]:
        """Linear feature detection optimized for Sentinel-2 10m resolution"""
        
        edges = cv2.Canny(image, 50, 150)
        
        # Default values for feature sizes if zone is None
        default_min_feature_size = 60  # 60m minimum feature size
        
        # Safely get min feature size with fallback to default
        min_feature_size = getattr(self.zone, 'min_feature_size_m', default_min_feature_size) if self.zone else default_min_feature_size
        
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=80,  # Adjusted for 10m resolution
            minLineLength=int(min_feature_size / 10),
            maxLineGap=5   # Smaller gap for higher resolution
        )
        
        linear_features = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                length_pixels = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                length_m = length_pixels * 10  # 10m pixels
                
                # Default value for min feature size if zone is None
                default_min_feature_size = 60  # 60m minimum feature size
                
                # Safely get min feature size with fallback to default
                min_feature_size = getattr(self.zone, 'min_feature_size_m', default_min_feature_size) if self.zone else default_min_feature_size
                
                if length_m < min_feature_size:
                    continue
                
                if hasattr(self, 'transform'):
                    geo_x1, geo_y1 = rasterio.transform.xy(self.transform, y1, x1)
                    geo_x2, geo_y2 = rasterio.transform.xy(self.transform, y2, x2)
                else:
                    geo_x1, geo_y1 = x1, y1
                    geo_x2, geo_y2 = x2, y2
                
                linear_features.append({
                    'type': 'line',
                    'start': (geo_x1, geo_y1),
                    'end': (geo_x2, geo_y2),
                    'pixel_start': (x1, y1),
                    'pixel_end': (x2, y2),
                    'length_m': length_m,
                    'angle_degrees': np.degrees(np.arctan2(y2-y1, x2-x1)),
                    'resolution': '10m',
                    'expected_feature': 'causeway' if length_m > 200 else 'path'
                })
        
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
                default_min_feature_size = 60  # 60m minimum feature size
                
                # Safely get min feature size with fallback to default
                min_feature_size = getattr(self.zone, 'min_feature_size_m', default_min_feature_size) if self.zone else default_min_feature_size
                
                if area_m2 < (min_feature_size ** 2):
                    continue
                
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    if hasattr(self, 'transform'):
                        geo_x, geo_y = rasterio.transform.xy(self.transform, cy, cx)
                    else:
                        geo_x, geo_y = cx, cy
                    
                    rect = cv2.minAreaRect(contour)
                    width_m = rect[1][0] * 10
                    height_m = rect[1][1] * 10
                    
                    rectangular_features.append({
                        'type': 'rectangle',
                        'center': (geo_x, geo_y),
                        'pixel_center': (cx, cy),
                        'width_m': width_m,
                        'height_m': height_m,
                        'area_m2': area_m2,
                        'angle_degrees': rect[2],
                        'aspect_ratio': max(width_m, height_m) / min(width_m, height_m),
                        'resolution': '10m',
                        'expected_feature': 'plaza' if area_m2 > 5000 else 'compound'
                    })
        
        return rectangular_features
"""
Archaeological Detection Engine
Advanced algorithms for detecting archaeological features in Amazon satellite imagery
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
from pathlib import Path
import json

from ..config import DetectionConfig, ScoringConfig, TargetZone

logger = logging.getLogger(__name__)

class ArchaeologicalDetector:
    """Advanced archaeological feature detection using multi-modal analysis"""
    
    def __init__(self, zone: TargetZone):
        self.zone = zone
        self.detection_results = {}
        self.processed_bands = {}
        
    def _gee_bandmap(self, count):
        # Standard 6-band GEE Landsat export: [Blue, Green, Red, NIR, SWIR1, SWIR2]
        return {1: 'blue', 2: 'green', 3: 'red', 4: 'nir', 5: 'swir1', 6: 'swir2'}

    def load_landsat_bands(self, scene_path: Path) -> Dict[str, np.ndarray]:
        """Load and process Landsat bands optimized for archaeological detection (supports multi-band GeoTIFFs)"""
        scene_path = Path(scene_path)
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene path not found: {scene_path}")
        # Find band files (Landsat Collection 2 naming)
        band_files = {}
        # Key bands for archaeological analysis
        target_bands = {
            'blue': '*_SR_B2.TIF',      # Blue (0.45-0.51 μm)
            'green': '*_SR_B3.TIF',     # Green (0.53-0.59 μm) 
            'red': '*_SR_B4.TIF',       # Red (0.64-0.67 μm)
            'nir': '*_SR_B5.TIF',       # NIR (0.85-0.88 μm)
            'swir1': '*_SR_B6.TIF',     # SWIR1 (1.57-1.65 μm)
            'swir2': '*_SR_B7.TIF',     # SWIR2 (2.11-2.29 μm)
        }
        # Find files
        for band_name, pattern in target_bands.items():
            files = list(scene_path.glob(pattern))
            if files:
                band_files[band_name] = files[0]
                logger.debug(f"Found {band_name} band: {files[0].name}")
            else:
                logger.warning(f"Missing {band_name} band in {scene_path}")
        # If no individual band files, check for a single .tif (multi-band) file
        if not band_files:
            tifs = list(scene_path.glob('*.tif')) + list(scene_path.glob('*.TIF'))
            if len(tifs) == 1:
                tif_path = tifs[0]
                with rasterio.open(tif_path) as src:
                    count = src.count
                    bands = {}
                    gee_export = False
                    if count == 6:
                        bandmap = self._gee_bandmap(count)
                        gee_export = True
                        for i in range(1, 7):
                            band_name = bandmap[i]
                            arr = src.read(i).astype(np.float32)
                            bands[band_name] = arr
                        self.processed_bands = bands
                        self.transform = src.transform
                        self.crs = src.crs
                        logger.info(f"Loaded 6 bands (GEE-mapped by index) from {tif_path.name}")
                        return bands
                    elif count > 6:
                        logger.warning(f"Multi-band file has {count} bands; expected 6. Please export only SR_B2 to SR_B7.")
                        raise ValueError(f"Too many bands in {tif_path.name}. Re-export with only 6 reflectance bands.")
                    else:
                        logger.warning(f"Multi-band file has {count} bands; expected 6.")
                        raise ValueError(f"Not enough bands in {tif_path.name}. Expected 6 reflectance bands.")
            else:
                logger.warning(f"No valid Landsat bands or multi-band GeoTIFF found in {scene_path}")
                raise ValueError(f"No valid Landsat bands found in {scene_path}")
        # Load bands into arrays (individual files)
        bands = {}
        transform = None
        crs = None
        for band_name, filepath in band_files.items():
            try:
                with rasterio.open(filepath) as src:
                    bands[band_name] = src.read(1).astype(np.float32)
                    if transform is None:
                        transform = src.transform
                        crs = src.crs
                    bands[band_name] = bands[band_name] * 0.0000275 - 0.2
                logger.debug(f"Loaded {band_name}: {bands[band_name].shape}")
            except Exception as e:
                logger.error(f"Error loading {band_name} from {filepath}: {e}")
                continue
        self.processed_bands = bands
        self.transform = transform
        self.crs = crs
        return bands
    
    def calculate_spectral_indices(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate spectral indices useful for archaeological detection"""
        
        indices = {}
        
        # Ensure we have required bands
        required = ['red', 'nir', 'swir1', 'swir2']
        if not all(band in bands for band in required):
            logger.warning("Missing bands for spectral index calculation")
            return indices
        
        red = bands['red']
        nir = bands['nir']
        swir1 = bands['swir1']
        swir2 = bands['swir2']
        
        # Add small epsilon to prevent division by zero
        eps = 1e-8
        
        # NDVI - Normalized Difference Vegetation Index
        indices['ndvi'] = (nir - red) / (nir + red + eps)
        
        # NDWI - Normalized Difference Water Index
        if 'green' in bands:
            green = bands['green']
            indices['ndwi'] = (green - nir) / (green + nir + eps)
        
        # Terra Preta Index (custom for archaeological soils)
        # Terra preta has higher NIR and lower SWIR1 reflectance
        indices['terra_preta'] = (nir - swir1) / (nir + swir1 + eps)
        
        # Clay Mineral Index (archaeological ceramics)
        indices['clay_minerals'] = swir1 / swir2
        
        # Brightness Index (settlement areas often have different brightness)
        if 'blue' in bands and 'green' in bands:
            blue = bands['blue']
            green = bands['green']
            indices['brightness'] = np.sqrt((blue**2 + green**2 + red**2) / 3)
        
        # Modified Soil Adjusted Vegetation Index
        indices['msavi'] = (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2
        
        logger.info(f"Calculated {len(indices)} spectral indices")
        return indices
    
    def detect_terra_preta_signatures(self, bands: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Detect terra preta (anthropogenic soil) spectral signatures"""
        
        indices = self.calculate_spectral_indices(bands)
        
        if 'terra_preta' not in indices or 'ndvi' not in indices:
            logger.warning("Cannot detect terra preta - missing spectral indices")
            return {}
        
        terra_preta_index = indices['terra_preta']
        ndvi = indices['ndvi']
        
        # Terra preta detection criteria
        tp_mask = (
            (terra_preta_index > DetectionConfig.TERRA_PRETA_INDEX_MIN) &
            (ndvi > DetectionConfig.TERRA_PRETA_NDVI_MIN) &
            (ndvi < 0.8)  # Exclude dense forest
        )
        
        # Remove noise with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        tp_mask = cv2.morphologyEx(tp_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        tp_mask = cv2.morphologyEx(tp_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        labeled_mask, num_features = ndimage.label(tp_mask)
        
        # Filter by size
        tp_patches = []
        for i in range(1, num_features + 1):
            patch_mask = labeled_mask == i
            patch_size = np.sum(patch_mask)
            
            if patch_size >= DetectionConfig.MIN_ANOMALY_PIXELS:
                # Calculate patch statistics
                patch_coords = np.where(patch_mask)
                centroid_y = np.mean(patch_coords[0])
                centroid_x = np.mean(patch_coords[1])
                
                # Convert to geographic coordinates
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
                    'area_m2': patch_size * 30 * 30,  # Landsat pixel size
                    'mean_tp_index': np.mean(terra_preta_index[patch_mask]),
                    'mean_ndvi': np.mean(ndvi[patch_mask]),
                    'confidence': min(1.0, patch_size / (DetectionConfig.MIN_ANOMALY_PIXELS * 5))
                })
        
        logger.info(f"Detected {len(tp_patches)} terra preta patches")
        
        return {
            'patches': tp_patches,
            'mask': tp_mask,
            'total_pixels': np.sum(tp_mask),
            'coverage_percent': (np.sum(tp_mask) / tp_mask.size) * 100
        }
    
    def detect_geometric_patterns(self, bands: Dict[str, np.ndarray]) -> List[Dict]:
        """Detect geometric patterns indicative of archaeological features"""
        
        # Use NIR band for geometric detection (best contrast)
        if 'nir' not in bands:
            logger.warning("NIR band not available for geometric detection")
            return []
        
        nir_band = bands['nir']
        
        # Normalize to 8-bit for OpenCV
        nir_norm = ((nir_band - np.min(nir_band)) / 
                   (np.max(nir_band) - np.min(nir_band)) * 255).astype(np.uint8)
        
        geometric_features = []
        
        # Circular feature detection
        circles = self._detect_circular_features(nir_norm)
        geometric_features.extend(circles)
        
        # Linear feature detection (roads, causeways)
        lines = self._detect_linear_features(nir_norm)
        geometric_features.extend(lines)
        
        # Rectangular feature detection (compounds, plazas)
        rectangles = self._detect_rectangular_features(nir_norm)
        geometric_features.extend(rectangles)
        
        logger.info(f"Detected {len(geometric_features)} geometric patterns")
        return geometric_features
    
    def _detect_circular_features(self, image: np.ndarray) -> List[Dict]:
        """Detect circular earthworks and settlements"""
        
        # Apply Gaussian blur to reduce noise
        kernel_size = DetectionConfig.CIRCLE_DETECTION_PARAMS['blur_kernel']
        blurred = cv2.GaussianBlur(image, kernel_size, 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Hough Circle Transform
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=DetectionConfig.CIRCLE_DETECTION_PARAMS['dp'],
            minDist=int(self.zone.min_feature_size_m / 30),  # Convert to pixels
            param1=DetectionConfig.CIRCLE_DETECTION_PARAMS['param1'],
            param2=DetectionConfig.CIRCLE_DETECTION_PARAMS['param2'],
            minRadius=int(self.zone.min_feature_size_m / 60),  # min diameter/2 in pixels
            maxRadius=int(self.zone.max_feature_size_m / 60)   # max diameter/2 in pixels
        )
        
        circular_features = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Convert to geographic coordinates
                if hasattr(self, 'transform'):
                    geo_x, geo_y = rasterio.transform.xy(self.transform, y, x)
                    radius_m = r * 30  # Convert pixel radius to meters
                else:
                    geo_x, geo_y = x, y
                    radius_m = r
                
                # Calculate confidence based on edge strength
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
                    'expected_feature': 'settlement_ring' if radius_m > 100 else 'house_ring'
                })
        
        return circular_features
    
    def _detect_linear_features(self, image: np.ndarray) -> List[Dict]:
        """Detect linear features like causeways and roads"""
        
        # Edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=int(self.zone.min_feature_size_m / 30),
            maxLineGap=10
        )
        
        linear_features = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line properties
                length_pixels = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                length_m = length_pixels * 30  # Convert to meters
                
                if length_m < self.zone.min_feature_size_m:
                    continue
                
                # Convert endpoints to geographic coordinates
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
                    'expected_feature': 'causeway' if length_m > 500 else 'path'
                })
        
        return linear_features
    
    def _detect_rectangular_features(self, image: np.ndarray) -> List[Dict]:
        """Detect rectangular features like plazas and compounds"""
        
        # Edge detection
        edges = cv2.Canny(image, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangular_features = []
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular (4 corners)
            if len(approx) == 4:
                # Calculate area
                area_pixels = cv2.contourArea(contour)
                area_m2 = area_pixels * 30 * 30
                
                if area_m2 < (self.zone.min_feature_size_m ** 2):
                    continue
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Convert to geographic coordinates
                    if hasattr(self, 'transform'):
                        geo_x, geo_y = rasterio.transform.xy(self.transform, cy, cx)
                    else:
                        geo_x, geo_y = cx, cy
                    
                    # Calculate rectangle properties
                    rect = cv2.minAreaRect(contour)
                    width_m = rect[1][0] * 30
                    height_m = rect[1][1] * 30
                    
                    rectangular_features.append({
                        'type': 'rectangle',
                        'center': (geo_x, geo_y),
                        'pixel_center': (cx, cy),
                        'width_m': width_m,
                        'height_m': height_m,
                        'area_m2': area_m2,
                        'angle_degrees': rect[2],
                        'aspect_ratio': max(width_m, height_m) / min(width_m, height_m),
                        'expected_feature': 'plaza' if area_m2 > 10000 else 'compound'
                    })
        
        return rectangular_features
    
    def analyze_scene(self, scene_path: Path) -> Dict[str, Any]:
        """Complete archaeological analysis of a satellite scene"""
        
        logger.info(f"Analyzing scene: {scene_path}")
        
        try:
            # Resolve scene directory
            scene_path = Path(scene_path)
            if scene_path.is_file():
                scene_path = scene_path.parent
            
            if not scene_path.exists():
                raise FileNotFoundError(f"Scene path not found: {scene_path}")
            
            # Load satellite bands with better error handling
            try:
                bands = self.load_landsat_bands(scene_path)
            except Exception as e:
                logger.error(f"Failed to load bands from {scene_path}: {e}")
                # Try to find any .tif files in the directory
                tif_files = list(scene_path.glob("*.tif")) + list(scene_path.glob("*.TIF"))
                if tif_files:
                    logger.info(f"Found {len(tif_files)} .tif files, attempting analysis")
                    bands = self.load_landsat_bands(scene_path)
                else:
                    raise
            
            if not bands:
                logger.error("No bands loaded - cannot analyze scene")
                return {}
            
            # Terra preta detection
            logger.info("Detecting terra preta signatures...")
            terra_preta_results = self.detect_terra_preta_signatures(bands)
            
            # Geometric pattern detection
            logger.info("Detecting geometric patterns...")
            geometric_features = self.detect_geometric_patterns(bands)
            
            # Compile results
            analysis_results = {
                'scene_path': str(scene_path),
                'zone': self.zone.name,
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'terra_preta': terra_preta_results,
                'geometric_features': geometric_features,
                'total_features': len(geometric_features) + len(terra_preta_results.get('patches', [])),
                'success': True
            }
            
            # Store results
            self.detection_results = analysis_results
            
            logger.info(f"Analysis complete: {analysis_results['total_features']} features detected")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error analyzing scene {scene_path}: {e}")
            return {
                'scene_path': str(scene_path), 
                'zone': self.zone.name,
                'error': str(e),
                'success': False
            }
    
    def export_detections_to_geojson(self, output_path: Path) -> bool:
        """Export detected features to GeoJSON format"""
        
        if not self.detection_results or not self.detection_results.get('success'):
            logger.warning("No successful detection results to export")
            return False
        
        features = []
        
        # Export terra preta patches
        tp_patches = self.detection_results.get('terra_preta', {}).get('patches', [])
        for patch in tp_patches:
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [patch['centroid'][0], patch['centroid'][1]]
                },
                'properties': {
                    'feature_type': 'terra_preta',
                    'area_m2': patch['area_m2'],
                    'confidence': patch['confidence'],
                    'tp_index': patch['mean_tp_index'],
                    'ndvi': patch['mean_ndvi']
                }
            }
            features.append(feature)
        
        # Export geometric features
        geometric_features = self.detection_results.get('geometric_features', [])
        for geom in geometric_features:
            if geom['type'] == 'circle':
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [geom['center'][0], geom['center'][1]]
                    },
                    'properties': {
                        'feature_type': f"geometric_{geom['type']}",
                        'diameter_m': geom['diameter_m'],
                        'area_m2': geom['area_m2'],
                        'confidence': geom['confidence'],
                        'expected_feature': geom['expected_feature']
                    }
                }
            elif geom['type'] == 'line':
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': [
                            [geom['start'][0], geom['start'][1]],
                            [geom['end'][0], geom['end'][1]]
                        ]
                    },
                    'properties': {
                        'feature_type': f"geometric_{geom['type']}",
                        'length_m': geom['length_m'],
                        'angle_degrees': geom['angle_degrees'],
                        'expected_feature': geom['expected_feature']
                    }
                }
            else:  # rectangle
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [geom['center'][0], geom['center'][1]]
                    },
                    'properties': {
                        'feature_type': f"geometric_{geom['type']}",
                        'area_m2': geom['area_m2'],
                        'width_m': geom['width_m'],
                        'height_m': geom['height_m'],
                        'expected_feature': geom['expected_feature']
                    }
                }
            
            features.append(feature)
        
        # Create GeoJSON
        geojson = {
            'type': 'FeatureCollection',
            'features': features,
            'metadata': {
                'zone': self.zone.name,
                'scene_path': self.detection_results['scene_path'],
                'analysis_timestamp': self.detection_results['analysis_timestamp'],
                'total_features': len(features)
            }
        }
        
        # Write to file
        try:
            def convert_np(obj):
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                if isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                return obj

            with open(output_path, 'w') as f:
                json.dump(geojson, f, indent=2, default=convert_np)
            
            logger.info(f"Exported {len(features)} features to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting GeoJSON: {e}")
            return False

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
                'zone': self.zone.name,
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'vector_features': features,
                'total_features': len(features),
                'success': True
            }
            self.detection_results = result
            return result
        except Exception as e:
            return {'success': False, 'error': str(e)}
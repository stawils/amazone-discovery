"""
Image Processing Utilities for Archaeological Analysis
Advanced preprocessing and enhancement techniques for satellite imagery
"""

import numpy as np
import cv2
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box, Point
from scipy import ndimage
from scipy.signal import medfilt2d
from skimage import exposure, restoration, morphology, filters
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

from .config import ProcessingConfig, DetectionConfig

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Advanced image processing for satellite imagery analysis"""
    
    def __init__(self):
        self.processing_params = ProcessingConfig()
        
    def enhance_contrast(self, image: np.ndarray, method: str = 'adaptive_histogram') -> np.ndarray:
        """Enhance image contrast using various methods"""
        
        if method == 'adaptive_histogram':
            # Contrast Limited Adaptive Histogram Equalization
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            
            if len(image.shape) == 2:
                return clahe.apply(image.astype(np.uint8))
            else:
                # Apply to each channel
                enhanced = np.zeros_like(image)
                for i in range(image.shape[2]):
                    enhanced[:, :, i] = clahe.apply(image[:, :, i].astype(np.uint8))
                return enhanced
                
        elif method == 'histogram_equalization':
            if len(image.shape) == 2:
                return cv2.equalizeHist(image.astype(np.uint8))
            else:
                enhanced = np.zeros_like(image)
                for i in range(image.shape[2]):
                    enhanced[:, :, i] = cv2.equalizeHist(image[:, :, i].astype(np.uint8))
                return enhanced
                
        elif method == 'gamma_correction':
            # Gamma correction for brightness adjustment
            gamma = 1.2  # Adjust as needed
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)])
            return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))
            
        else:
            logger.warning(f"Unknown contrast enhancement method: {method}")
            return image
    
    def remove_noise(self, image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """Remove noise while preserving edges"""
        
        if method == 'bilateral':
            # Bilateral filter - good for preserving edges
            if len(image.shape) == 2:
                return cv2.bilateralFilter(image.astype(np.uint8), 9, 75, 75)
            else:
                denoised = np.zeros_like(image)
                for i in range(image.shape[2]):
                    denoised[:, :, i] = cv2.bilateralFilter(
                        image[:, :, i].astype(np.uint8), 9, 75, 75
                    )
                return denoised
                
        elif method == 'median':
            # Median filter - good for salt and pepper noise
            return medfilt2d(image.astype(np.float32), kernel_size=3).astype(image.dtype)
            
        elif method == 'gaussian':
            # Gaussian blur
            return cv2.GaussianBlur(image, (5, 5), 0)
            
        elif method == 'morphological':
            # Morphological operations
            kernel = np.ones((3, 3), np.uint8)
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            
        else:
            logger.warning(f"Unknown noise removal method: {method}")
            return image
    
    def sharpen_image(self, image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """Sharpen image to enhance feature detection"""
        
        # Unsharp masking
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)
        
        return unsharp_mask
    
    def enhance_edges(self, image: np.ndarray, method: str = 'sobel') -> np.ndarray:
        """Enhance edges for geometric feature detection"""
        
        if len(image.shape) > 2:
            # Convert to grayscale if needed
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if method == 'sobel':
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            return np.sqrt(grad_x**2 + grad_y**2)
            
        elif method == 'canny':
            return cv2.Canny(image.astype(np.uint8), 50, 150)
            
        elif method == 'laplacian':
            return cv2.Laplacian(image, cv2.CV_64F)
            
        elif method == 'scharr':
            grad_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
            grad_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
            return np.sqrt(grad_x**2 + grad_y**2)
            
        else:
            logger.warning(f"Unknown edge enhancement method: {method}")
            return image
    
    def normalize_bands(self, bands: Dict[str, np.ndarray], method: str = 'minmax') -> Dict[str, np.ndarray]:
        """Normalize spectral bands for consistent analysis"""
        
        normalized = {}
        
        for band_name, band_data in bands.items():
            if method == 'minmax':
                # Min-max normalization to 0-1 range
                band_min = np.nanmin(band_data)
                band_max = np.nanmax(band_data)
                if band_max > band_min:
                    normalized[band_name] = (band_data - band_min) / (band_max - band_min)
                else:
                    normalized[band_name] = band_data
                    
            elif method == 'zscore':
                # Z-score normalization
                band_mean = np.nanmean(band_data)
                band_std = np.nanstd(band_data)
                if band_std > 0:
                    normalized[band_name] = (band_data - band_mean) / band_std
                else:
                    normalized[band_name] = band_data - band_mean
                    
            elif method == 'percentile':
                # Percentile normalization (2nd to 98th percentile)
                p2 = np.nanpercentile(band_data, 2)
                p98 = np.nanpercentile(band_data, 98)
                if p98 > p2:
                    normalized[band_name] = np.clip((band_data - p2) / (p98 - p2), 0, 1)
                else:
                    normalized[band_name] = band_data
                    
            else:
                logger.warning(f"Unknown normalization method: {method}")
                normalized[band_name] = band_data
        
        return normalized
    
    def create_composite(self, bands: Dict[str, np.ndarray], 
                        composite_type: str = 'true_color') -> np.ndarray:
        """Create RGB composite from spectral bands"""
        
        if composite_type == 'true_color':
            # Standard RGB composite
            required_bands = ['red', 'green', 'blue']
            
        elif composite_type == 'false_color':
            # NIR-Red-Green (vegetation enhanced)
            required_bands = ['nir', 'red', 'green']
            
        elif composite_type == 'archaeology':
            # Custom composite optimized for archaeological features
            # SWIR1-NIR-Red (enhances soil and vegetation differences)
            required_bands = ['swir1', 'nir', 'red']
            
        elif composite_type == 'urban':
            # SWIR2-SWIR1-Red (urban and bare soil enhancement)
            required_bands = ['swir2', 'swir1', 'red']
            
        else:
            logger.warning(f"Unknown composite type: {composite_type}")
            required_bands = ['red', 'green', 'blue']
        
        # Check if required bands are available
        available_bands = [band for band in required_bands if band in bands]
        
        if len(available_bands) < 3:
            logger.error(f"Insufficient bands for {composite_type} composite")
            return None
        
        # Stack bands
        composite = np.stack([bands[band] for band in available_bands[:3]], axis=-1)
        
        # Normalize to 0-255 range
        composite_norm = np.zeros_like(composite, dtype=np.uint8)
        for i in range(composite.shape[2]):
            band = composite[:, :, i]
            band_min, band_max = np.nanpercentile(band, [2, 98])
            if band_max > band_min:
                band_norm = np.clip((band - band_min) / (band_max - band_min) * 255, 0, 255)
            else:
                band_norm = np.zeros_like(band)
            composite_norm[:, :, i] = band_norm.astype(np.uint8)
        
        return composite_norm
    
    def apply_cloud_mask(self, bands: Dict[str, np.ndarray], 
                        cloud_mask: np.ndarray = None) -> Dict[str, np.ndarray]:
        """Apply cloud mask to remove cloudy pixels"""
        
        if cloud_mask is None:
            # Create simple cloud mask using brightness threshold
            if 'blue' in bands and 'green' in bands and 'red' in bands:
                brightness = (bands['blue'] + bands['green'] + bands['red']) / 3
                cloud_mask = brightness > np.percentile(brightness, 95)
            else:
                logger.warning("Cannot create cloud mask - insufficient bands")
                return bands
        
        # Apply mask to all bands
        masked_bands = {}
        for band_name, band_data in bands.items():
            masked_data = band_data.copy()
            masked_data[cloud_mask] = np.nan
            masked_bands[band_name] = masked_data
        
        return masked_bands

class BandProcessor:
    """Specialized processor for individual spectral bands"""
    
    def __init__(self):
        self.processor = ImageProcessor()
    
    def load_and_preprocess_band(self, filepath: Path, 
                               enhance: bool = True) -> Tuple[np.ndarray, Dict]:
        """Load and preprocess a single spectral band"""
        
        try:
            with rasterio.open(filepath) as src:
                # Read band data
                band_data = src.read(1).astype(np.float32)
                
                # Get metadata
                metadata = {
                    'transform': src.transform,
                    'crs': src.crs,
                    'bounds': src.bounds,
                    'shape': band_data.shape,
                    'nodata': src.nodata
                }
                
                # Handle Landsat scaling (if needed)
                if 'SR_B' in filepath.name:  # Landsat Surface Reflectance
                    # Convert from scaled integers to reflectance
                    band_data = band_data * 0.0000275 - 0.2
                    # Clip to valid reflectance range
                    band_data = np.clip(band_data, 0, 1)
                
                # Handle no-data values
                if src.nodata is not None:
                    band_data[band_data == src.nodata] = np.nan
                
                # Optional enhancement
                if enhance:
                    # Remove extreme outliers
                    p1, p99 = np.nanpercentile(band_data, [1, 99])
                    band_data = np.clip(band_data, p1, p99)
                    
                    # Fill small gaps with interpolation
                    if np.any(np.isnan(band_data)):
                        band_data = self._fill_small_gaps(band_data)
                
                logger.debug(f"Loaded band: {filepath.name}, Shape: {band_data.shape}")
                return band_data, metadata
                
        except Exception as e:
            logger.error(f"Error loading band {filepath}: {e}")
            return None, {}
    
    def _fill_small_gaps(self, band_data: np.ndarray, max_gap_size: int = 5) -> np.ndarray:
        """Fill small gaps in band data using interpolation"""
        
        # Identify NaN pixels
        nan_mask = np.isnan(band_data)
        
        if not np.any(nan_mask):
            return band_data
        
        # Use morphological operations to identify small gaps
        kernel = np.ones((max_gap_size, max_gap_size), np.uint8)
        small_gaps = cv2.morphologyEx(
            nan_mask.astype(np.uint8), 
            cv2.MORPH_CLOSE, 
            kernel
        ) != nan_mask.astype(np.uint8)
        
        # Fill small gaps only
        filled_data = band_data.copy()
        if np.any(small_gaps):
            # Simple interpolation using nearby valid pixels
            from scipy.ndimage import generic_filter
            
            def local_mean(values):
                valid_values = values[~np.isnan(values)]
                return np.mean(valid_values) if len(valid_values) > 0 else np.nan
            
            # Apply local mean filter to small gaps
            gap_filled = generic_filter(
                filled_data, 
                local_mean, 
                size=3, 
                mode='constant', 
                cval=np.nan
            )
            
            filled_data[small_gaps] = gap_filled[small_gaps]
        
        return filled_data
    
    def calculate_spectral_indices(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate comprehensive set of spectral indices"""
        
        indices = {}
        eps = 1e-8  # Small value to prevent division by zero
        
        # Vegetation indices
        if 'red' in bands and 'nir' in bands:
            # NDVI - Normalized Difference Vegetation Index
            indices['ndvi'] = (bands['nir'] - bands['red']) / (bands['nir'] + bands['red'] + eps)
            
            # EVI - Enhanced Vegetation Index
            if 'blue' in bands:
                indices['evi'] = 2.5 * (bands['nir'] - bands['red']) / (
                    bands['nir'] + 6 * bands['red'] - 7.5 * bands['blue'] + 1 + eps
                )
        
        # Water indices
        if 'green' in bands and 'nir' in bands:
            # NDWI - Normalized Difference Water Index
            indices['ndwi'] = (bands['green'] - bands['nir']) / (bands['green'] + bands['nir'] + eps)
        
        if 'nir' in bands and 'swir1' in bands:
            # MNDWI - Modified Normalized Difference Water Index
            indices['mndwi'] = (bands['green'] - bands['swir1']) / (bands['green'] + bands['swir1'] + eps)
        
        # Soil indices
        if 'red' in bands and 'swir1' in bands:
            # NDSI - Normalized Difference Soil Index
            indices['ndsi'] = (bands['swir1'] - bands['red']) / (bands['swir1'] + bands['red'] + eps)
        
        # Archaeological indices
        if 'nir' in bands and 'swir1' in bands:
            # Terra Preta Index (custom for anthropogenic soils)
            indices['terra_preta'] = (bands['nir'] - bands['swir1']) / (bands['nir'] + bands['swir1'] + eps)
        
        if 'swir1' in bands and 'swir2' in bands:
            # Clay Mineral Index
            indices['clay_minerals'] = bands['swir1'] / (bands['swir2'] + eps)
        
        # Brightness and texture indices
        if len(bands) >= 3:
            # Brightness Index
            band_values = list(bands.values())
            indices['brightness'] = np.sqrt(np.sum([b**2 for b in band_values[:3]], axis=0) / 3)
            
            # Greenness (Tasseled Cap-like)
            if all(b in bands for b in ['blue', 'green', 'red', 'nir']):
                indices['greenness'] = (
                    -0.2848 * bands['blue'] - 0.2435 * bands['green'] - 
                    0.5436 * bands['red'] + 0.7243 * bands['nir']
                )
        
        logger.info(f"Calculated {len(indices)} spectral indices")
        return indices
    
    def detect_anomalies(self, band_data: np.ndarray, 
                        method: str = 'statistical') -> np.ndarray:
        """Detect anomalous pixels in spectral bands"""
        
        if method == 'statistical':
            # Statistical outlier detection
            mean_val = np.nanmean(band_data)
            std_val = np.nanstd(band_data)
            threshold = 2.5  # Standard deviations
            
            anomaly_mask = (
                (band_data > mean_val + threshold * std_val) |
                (band_data < mean_val - threshold * std_val)
            )
            
        elif method == 'percentile':
            # Percentile-based anomaly detection
            p5 = np.nanpercentile(band_data, 5)
            p95 = np.nanpercentile(band_data, 95)
            
            anomaly_mask = (band_data < p5) | (band_data > p95)
            
        elif method == 'iqr':
            # Interquartile range method
            q1 = np.nanpercentile(band_data, 25)
            q3 = np.nanpercentile(band_data, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            anomaly_mask = (band_data < lower_bound) | (band_data > upper_bound)
            
        else:
            logger.warning(f"Unknown anomaly detection method: {method}")
            return np.zeros_like(band_data, dtype=bool)
        
        return anomaly_mask

def preprocess_landsat_scene(scene_path: Path, 
                           target_bands: List[str] = None,
                           enhance_contrast: bool = True,
                           remove_clouds: bool = True) -> Dict[str, Any]:
    """Complete preprocessing pipeline for Landsat scenes"""
    
    if target_bands is None:
        target_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    
    processor = BandProcessor()
    img_processor = ImageProcessor()
    
    # Load all bands
    bands = {}
    metadata = {}
    
    band_mapping = {
        'blue': '*_SR_B2.TIF',
        'green': '*_SR_B3.TIF', 
        'red': '*_SR_B4.TIF',
        'nir': '*_SR_B5.TIF',
        'swir1': '*_SR_B6.TIF',
        'swir2': '*_SR_B7.TIF'
    }
    
    logger.info(f"Preprocessing Landsat scene: {scene_path}")
    
    for band_name in target_bands:
        if band_name in band_mapping:
            pattern = band_mapping[band_name]
            band_files = list(scene_path.glob(pattern))
            
            if band_files:
                band_data, band_metadata = processor.load_and_preprocess_band(
                    band_files[0], enhance=True
                )
                
                if band_data is not None:
                    bands[band_name] = band_data
                    if not metadata:  # Store metadata from first successful band
                        metadata = band_metadata
    
    if not bands:
        logger.error(f"No bands loaded from {scene_path}")
        return {}
    
    logger.info(f"Loaded {len(bands)} bands: {list(bands.keys())}")
    
    # Apply cloud masking if requested
    if remove_clouds:
        bands = img_processor.apply_cloud_mask(bands)
    
    # Normalize bands
    bands = img_processor.normalize_bands(bands, method='percentile')
    
    # Calculate spectral indices
    indices = processor.calculate_spectral_indices(bands)
    
    # Create useful composites
    composites = {}
    for comp_type in ['true_color', 'false_color', 'archaeology']:
        composite = img_processor.create_composite(bands, comp_type)
        if composite is not None:
            composites[comp_type] = composite
    
    # Enhance contrast if requested
    if enhance_contrast:
        enhanced_bands = {}
        for band_name, band_data in bands.items():
            # Convert to 8-bit for enhancement
            band_8bit = ((band_data - np.nanmin(band_data)) / 
                        (np.nanmax(band_data) - np.nanmin(band_data)) * 255).astype(np.uint8)
            
            enhanced = img_processor.enhance_contrast(band_8bit, method='adaptive_histogram')
            enhanced_bands[f"{band_name}_enhanced"] = enhanced
        
        bands.update(enhanced_bands)
    
    result = {
        'bands': bands,
        'indices': indices,
        'composites': composites,
        'metadata': metadata,
        'scene_path': str(scene_path),
        'preprocessing_successful': True,
        'bands_loaded': list(bands.keys()),
        'indices_calculated': list(indices.keys())
    }
    
    logger.info(f"✓ Preprocessing complete: {len(bands)} bands, {len(indices)} indices")
    
    return result

if __name__ == "__main__":
    # Test preprocessing
    print("Testing image processing utilities...")
    
    # Create synthetic test data
    test_bands = {
        'red': np.random.rand(100, 100) * 0.3,
        'nir': np.random.rand(100, 100) * 0.5 + 0.2,
        'swir1': np.random.rand(100, 100) * 0.4
    }
    
    processor = BandProcessor()
    indices = processor.calculate_spectral_indices(test_bands)
    
    print(f"✓ Calculated indices: {list(indices.keys())}")
    print(f"✓ NDVI range: {np.min(indices['ndvi']):.3f} to {np.max(indices['ndvi']):.3f}")
    
    print("Image processing utilities test completed")
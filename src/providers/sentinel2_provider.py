"""
Clean Sentinel-2 Provider for Archaeological Discovery
Refactored for simplicity, performance, and maintainability.
"""

import os
import json
import logging
import requests
import rasterio
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Core dependencies
from pystac_client import Client
import pystac
import geopandas as gpd
from shapely.geometry import box, Polygon
import rioxarray
import xarray as xr
import pandas as pd

# Your existing core modules
from src.core.data_objects import SceneData, BaseProvider

logger = logging.getLogger(__name__)

@dataclass
class Sentinel2Config:
    """Clean configuration for Sentinel-2 processing"""
    
    # API Configuration
    STAC_API_URL: str = "https://earth-search.aws.element84.com/v1"
    COLLECTION: str = "sentinel-2-l2a"
    
    # Archaeological bands (only what we need)
    BANDS: Dict[str, str] = field(default_factory=lambda: {
        'blue': 'B02',        # 10m - Blue (492.4nm) - Official ESA specs
        'green': 'B03',       # 10m - Green (559.8nm) - Official ESA specs
        'red': 'B04',         # 10m - Red (664.6nm) - Official ESA specs
        'nir': 'B08',         # 10m - NIR (832.8nm) - Official ESA specs
        'red_edge_1': 'B05',  # 20m - Red Edge 1 (704.1nm) - Critical for archaeology
        'red_edge_3': 'B07',  # 20m - Red Edge 3 (782.8nm) - Critical for archaeology
        'swir1': 'B11',       # 20m - SWIR1 (1613.7nm) - Terra preta detection
        'swir2': 'B12'        # 20m - SWIR2 (2202.4nm) - Official ESA specs
    })
    
    # Asset name mappings for STAC API
    ASSET_MAPPINGS: Dict[str, List[str]] = field(default_factory=lambda: {
        'B02': ['blue'],
        'B03': ['green'],
        'B04': ['red'],
        'B05': ['rededge1'],
        'B07': ['rededge3'],
        'B08': ['nir08', 'nir'],
        'B11': ['swir16'],
        'B12': ['swir22']
    })
    
    # Quality filters (relaxed for Amazon tropical conditions)
    MAX_CLOUD_COVER: float = 60.0  # Increased for tropical regions
    MIN_OVERLAP: float = 50.0       # Reduced minimum overlap requirement
    MAX_CENTER_DISTANCE_KM: float = 75.0  # Increased search radius
    
    # Processing
    TARGET_RESOLUTION: tuple = (10.0, -10.0)  # 10m resolution
    PREFERRED_MONTHS: List[int] = field(default_factory=lambda: [6, 7, 8, 9])  # Dry season


class Sentinel2Provider(BaseProvider):
    """
    Clean Sentinel-2 provider focused on archaeological analysis.
    
    Responsibilities:
    1. Search and download Sentinel-2 scenes
    2. Create multi-band composites for analysis
    3. Cache processed data
    """
    
    def __init__(self, config: Optional[Sentinel2Config] = None):
        self.config = config or Sentinel2Config()
        self.stac_client = Client.open(self.config.STAC_API_URL)
        self.session = requests.Session()
        
        # Setup cache directories
        from src.core.config import SATELLITE_DIR
        self.cache_dir = SATELLITE_DIR / "sentinel2"
        self.raw_cache = self.cache_dir / "raw_bands"
        self.processed_cache = self.cache_dir / "processed_scenes"
        
        self.raw_cache.mkdir(parents=True, exist_ok=True)
        self.processed_cache.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🛰️ Sentinel-2 Provider initialized")
        logger.info(f"Cache directory: {self.cache_dir}")

    def download_data(self, zones: Optional[List[str]] = None, max_scenes: int = 3) -> List[SceneData]:
        """Download Sentinel-2 data for specified zones."""
        from src.core.config import TARGET_ZONES
        
        if not zones:
            zones = [zone_id for zone_id, config in TARGET_ZONES.items() if config.priority == 1]
            
        if isinstance(zones, str):
            zones = [zones]
        
        all_scenes = []
        
        for zone_id in zones:
            if zone_id not in TARGET_ZONES:
                logger.warning(f"Unknown zone: {zone_id}")
                continue
                
            zone_config = TARGET_ZONES[zone_id]
            logger.info(f"🎯 Processing {zone_config.name}")
            
            try:
                scenes = self._process_zone(zone_config, max_scenes)
                all_scenes.extend(scenes)
                logger.info(f"✓ Completed {zone_config.name}: {len(scenes)} scenes")
            except Exception as e:
                logger.error(f"Error processing zone {zone_config.name}: {e}", exc_info=True)
                
        logger.info(f"🎯 Total scenes processed: {len(all_scenes)}")
        return all_scenes

    def _process_zone(self, zone_config, max_scenes: int) -> List[SceneData]:
        """Process a single zone: search, filter, and download scenes."""
        # Search for scenes
        candidate_scenes = self._search_scenes(zone_config, max_scenes * 3)
        if not candidate_scenes:
            logger.warning(f"No suitable scenes found for {zone_config.name}")
            return []
        
        # Process scenes (download + composite creation) with automatic fallback
        processed_scenes = []
        scenes_attempted = 0
        
        # Try up to 2x max_scenes to account for corrupted data
        max_attempts = min(len(candidate_scenes), max_scenes * 2)
        
        for i, scene_data in enumerate(candidate_scenes[:max_attempts]):
            if len(processed_scenes) >= max_scenes:
                break  # We have enough scenes
                
            scenes_attempted += 1
            scene_id = scene_data['item'].id
            
            try:
                scene = self._process_scene(scene_data['item'], zone_config)
                if scene:
                    processed_scenes.append(scene)
                    logger.info(f"✓ Scene {len(processed_scenes)}/{max_scenes}: {scene_id}")
                else:
                    logger.warning(f"⚠️ Scene {scene_id} processing returned None, trying next scene...")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to process scene {scene_id}: {str(e)[:100]}... Trying next scene.")
                continue
        
        if not processed_scenes and scenes_attempted > 0:
            logger.error(f"Failed to process any of {scenes_attempted} candidate scenes due to data corruption or errors")
        elif len(processed_scenes) < max_scenes:
            logger.warning(f"Only processed {len(processed_scenes)}/{max_scenes} scenes ({scenes_attempted} attempted)")
            
        return processed_scenes

    def _search_scenes(self, zone_config, max_results: int) -> List[Dict]:
        """Search and filter Sentinel-2 scenes with cascading cloud cover strategy."""
        logger.info(f"🔍 Searching scenes for {zone_config.name}")
        
        # Create search geometry
        bbox = zone_config.bbox  # (south, west, north, east)
        stac_bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]  # (west, south, east, north)
        aoi_polygon = box(bbox[1], bbox[0], bbox[3], bbox[2])
        
        # Search with expanded area to get more candidates
        expanded_bbox = self._expand_bbox(stac_bbox, 1.5)
        
        # Cascading cloud cover thresholds - start strict, progressively relax
        cloud_thresholds = [10, 20, 30, 40, 50, 60, 70, 80]
        
        for threshold in cloud_thresholds:
            logger.info(f"🌥️ Trying cloud cover threshold: {threshold}%")
            
            try:
                search = self.stac_client.search(
                    collections=[self.config.COLLECTION],
                    bbox=expanded_bbox,
                    datetime="2023-01-01/2024-12-31",
                    limit=max_results * 3,  # Get more candidates for better selection
                    query={"eo:cloud_cover": {"lt": threshold}}
                )
                
                items = list(search.items())
                logger.info(f"Found {len(items)} candidate scenes")
                
                if not items:
                    continue  # Try next threshold
                
                # Filter and score scenes with current threshold
                filtered_scenes = []
                for item in items:
                    score_data = self._score_scene(item, zone_config, aoi_polygon)
                    
                    if (score_data['overlap_percentage'] >= self.config.MIN_OVERLAP and
                        score_data['center_distance_km'] <= self.config.MAX_CENTER_DISTANCE_KM and
                        score_data['cloud_cover'] <= threshold):  # Use current threshold
                        
                        filtered_scenes.append({
                            'item': item,
                            'score': score_data['quality_score'],
                            'overlap': score_data['overlap_percentage'],
                            'clouds': score_data['cloud_cover'],
                            'threshold_used': threshold
                        })
                
                # Sort by quality score
                filtered_scenes.sort(key=lambda x: x['score'], reverse=True)
                
                logger.info(f"✓ {len(filtered_scenes)} scenes passed filters at {threshold}% threshold")
                
                # If we found good scenes, use them
                if filtered_scenes:
                    best = filtered_scenes[0]
                    logger.info(f"🎯 Success with {threshold}% threshold!")
                    logger.info(f"Best: {best['item'].id} (score: {best['score']:.1f}, "
                              f"overlap: {best['overlap']:.1f}%, clouds: {best['clouds']:.1f}%)")
                    return filtered_scenes[:max_results]
                
            except Exception as e:
                logger.warning(f"Error at {threshold}% threshold: {e}")
                continue
        
        # If we get here, no scenes found at any threshold
        logger.warning(f"❌ No suitable scenes found for {zone_config.name} at any cloud threshold")
        return []

    def _expand_bbox(self, bbox: List[float], factor: float) -> List[float]:
        """Expand bounding box by factor."""
        west, south, east, north = bbox
        center_lon, center_lat = (west + east) / 2, (south + north) / 2
        width, height = (east - west) * factor, (north - south) * factor
        
        return [
            center_lon - width/2,  # west
            center_lat - height/2, # south
            center_lon + width/2,  # east
            center_lat + height/2  # north
        ]

    def _score_scene(self, item: pystac.Item, zone_config, aoi_polygon: Polygon) -> Dict[str, Any]:
        """Score scene quality for archaeological analysis."""
        props = item.properties
        
        # Basic metadata
        cloud_cover = props.get('eo:cloud_cover', 100)
        data_coverage = props.get('s2:data_coverage', 0)
        acquisition_date = datetime.fromisoformat(props['datetime'].replace('Z', '+00:00'))
        
        # Spatial metrics
        item_geometry = Polygon(item.geometry['coordinates'][0])
        
        try:
            # Overlap calculation
            intersection = item_geometry.intersection(aoi_polygon)
            overlap_percentage = (intersection.area / aoi_polygon.area) * 100
            
            # Distance calculation  
            aoi_center = aoi_polygon.centroid
            item_center = item_geometry.centroid
            center_distance_km = aoi_center.distance(item_center) * 111.32  # deg to km
            
        except Exception as e:
            logger.warning(f"Error calculating spatial metrics for {item.id}: {e}")
            overlap_percentage = 0
            center_distance_km = float('inf')
        
        # Quality scoring
        score = 0
        
        # Cloud cover (0-30 points)
        if cloud_cover <= 5:
            score += 30
        elif cloud_cover <= 10:
            score += 25
        elif cloud_cover <= 15:
            score += 15
        elif cloud_cover <= 20:
            score += 8
        
        # Overlap (0-40 points) - High weight for spatial fit
        if overlap_percentage >= 95:
            score += 40
        elif overlap_percentage >= 85:
            score += 32
        elif overlap_percentage >= 75:
            score += 24
        elif overlap_percentage >= 65:
            score += 16
        
        # Distance penalty (0-20 points)
        if center_distance_km <= 10:
            score += 20
        elif center_distance_km <= 25:
            score += 15
        elif center_distance_km <= 50:
            score += 10
        
        # Seasonal bonus (0-10 points)
        if acquisition_date.month in self.config.PREFERRED_MONTHS:
            score += 10
        
        return {
            'quality_score': score,
            'cloud_cover': cloud_cover,
            'data_coverage': data_coverage,
            'overlap_percentage': overlap_percentage,
            'center_distance_km': center_distance_km,
            'acquisition_date': acquisition_date.strftime('%Y-%m-%d')
        }

    def _process_scene(self, item: pystac.Item, zone_config) -> Optional[SceneData]:
        """Process a single scene: download bands only (no composite creation)."""
        scene_id = item.id
        
        # Check cache first
        composite_path = self.processed_cache / zone_config.name / f"{scene_id}_composite_cropped.tif"
        metadata_path = self.processed_cache / zone_config.name / f"{scene_id}_metadata.json"
        
        if composite_path.exists() and metadata_path.exists():
            logger.info(f"CACHE HIT: {scene_id}")
            return self._load_cached_scene(metadata_path, composite_path)
        
        logger.info(f"CACHE MISS: Processing {scene_id}")
        
        # Download bands
        band_paths = self._download_bands(item, scene_id, zone_config.name)
        if not band_paths:
            logger.error(f"No bands downloaded for {scene_id}")
            return None
        
        # Create cropped composite TIFF with all bands
        composite_path = self.processed_cache / zone_config.name / f"{scene_id}_composite_cropped.tif"
        composite_path = self._create_cropped_composite(
            band_paths, composite_path, zone_config.bbox
        )
        
        if not composite_path:
            logger.error(f"Failed to create cropped composite for {scene_id}")
            return None
        
        # Create high-quality RGB preview for OpenAI analysis
        rgb_preview_path = self.create_rgb_preview(composite_path)
        
        logger.info(f"✅ Created cropped composite and preview for {scene_id}")
        
        # Create scene data with individual bands AND composite
        scene_data = SceneData(
            zone_id=zone_config.id,
            provider='sentinel2',
            scene_id=scene_id,
            file_paths=band_paths,  # Keep original band references
            available_bands=list(band_paths.keys()),
            metadata=item.properties,
            features={'rgb_preview_path': str(rgb_preview_path) if rgb_preview_path else None},
            composite_file_path=composite_path  # Use cropped composite
        )
        
        self._cache_scene_metadata(scene_data, metadata_path)
        return scene_data

    def _download_bands(self, item: pystac.Item, scene_id: str, zone_name: str) -> Dict[str, Path]:
        """Download required bands for archaeological analysis."""
        band_paths = {}
        
        for desc_name, band_name in self.config.BANDS.items():
            # Find asset
            asset = item.assets.get(band_name)
            if not asset:
                # Try fallback names
                for fallback in self.config.ASSET_MAPPINGS.get(band_name, []):
                    asset = item.assets.get(fallback)
                    if asset:
                        logger.debug(f"Using fallback '{fallback}' for {band_name}")
                        break
            
            if asset:
                file_path = self._download_band(asset, scene_id, band_name, zone_name)
                if file_path:
                    band_paths[band_name] = file_path
            else:
                logger.warning(f"Asset {band_name} not found for {scene_id}")
        
        logger.info(f"Downloaded {len(band_paths)}/{len(self.config.BANDS)} bands")
        return band_paths

    def _download_band(self, asset: pystac.Asset, scene_id: str, band_name: str, zone_name: str) -> Optional[Path]:
        """Download a single band with caching."""
        cache_dir = self.raw_cache / zone_name / scene_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        band_file = cache_dir / f"{band_name}.tif"
        
        if band_file.exists() and band_file.stat().st_size > 0:
            logger.debug(f"CACHE HIT: {band_name}")
            return band_file
        
        logger.debug(f"CACHE MISS: Downloading {band_name}")
        
        try:
            response = self.session.get(asset.href, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(band_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            if band_file.stat().st_size > 0:
                return band_file
            else:
                band_file.unlink()
                return None
                
        except Exception as e:
            logger.error(f"Download failed for {band_name}: {e}")
            return None

    def _crop_bands_to_zone(self, band_paths: Dict[str, Path], zone_config) -> Dict[str, Path]:
        """Crop individual band files to zone bbox"""
        import rioxarray
        import geopandas as gpd
        from shapely.geometry import box
        
        cropped_paths = {}
        
        # Create AOI geometry from zone bbox
        bbox = zone_config.bbox  # (south, west, north, east)
        aoi_polygon = box(bbox[1], bbox[0], bbox[3], bbox[2])  # (west, south, east, north)
        aoi_gdf = gpd.GeoDataFrame([{'geometry': aoi_polygon}], crs="EPSG:4326")
        
        logger.info(f"Cropping bands to zone {zone_config.name}: {bbox}")
        
        for band_name, band_path in band_paths.items():
            try:
                # Create cropped filename
                cropped_filename = f"{band_path.stem}_cropped{band_path.suffix}"
                cropped_path = band_path.parent / cropped_filename
                
                # Skip if already exists
                if cropped_path.exists():
                    logger.debug(f"Using existing cropped band: {cropped_path}")
                    cropped_paths[band_name] = cropped_path
                    continue
                
                # Load and crop the band
                with rioxarray.open_rasterio(band_path, masked=True) as band_data:
                    # Reproject AOI to band CRS
                    aoi_reprojected = aoi_gdf.to_crs(band_data.rio.crs)
                    
                    # Crop to AOI
                    band_cropped = band_data.rio.clip(aoi_reprojected.geometry, all_touched=True, drop=False)
                    
                    # Check if cropped area is meaningful
                    if band_cropped.sizes.get('x', 0) < 3 or band_cropped.sizes.get('y', 0) < 3:
                        logger.warning(f"Cropped {band_name} area very small, using small buffer")
                        # Add small buffer for micro regions
                        buffered_polygon = aoi_polygon.buffer(0.005)  # ~500m buffer
                        buffered_gdf = gpd.GeoDataFrame([{'geometry': buffered_polygon}], crs="EPSG:4326")
                        buffered_reprojected = buffered_gdf.to_crs(band_data.rio.crs)
                        band_cropped = band_data.rio.clip(buffered_reprojected.geometry, all_touched=True, drop=False)
                    
                    # Save cropped band
                    band_cropped.rio.to_raster(cropped_path, compress='lzw')
                    cropped_paths[band_name] = cropped_path
                    
                    logger.debug(f"Cropped {band_name}: {band_data.sizes} -> {band_cropped.sizes}")
                    
            except Exception as e:
                logger.error(f"Failed to crop {band_name}: {e}")
                # Fallback to original band if cropping fails
                cropped_paths[band_name] = band_path
        
        logger.info(f"Successfully cropped {len(cropped_paths)} bands to zone area")
        return cropped_paths

    def _create_cropped_composite(self, band_paths: Dict[str, Path], output_path: Path, bbox: tuple) -> Optional[Path]:
        """Create cropped multi-band composite from downloaded bands for target zone area"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating cropped composite: {output_path.name}")
        
        try:
            # Band order optimized for archaeological analysis
            band_order = ['B02', 'B03', 'B04', 'B08', 'B05', 'B07', 'B11', 'B12']
            
            bands_for_composite = []
            band_names_for_coordinate = []
            target_crs = None
            
            for band_name_str in band_order:
                if band_name_str not in band_paths:
                    logger.warning(f"Band {band_name_str} missing from band_paths for composite. Skipping.")
                    continue
                
                try:
                    band_data = rioxarray.open_rasterio(band_paths[band_name_str], masked=True).squeeze()
                    
                    if target_crs is None:
                        target_crs = band_data.rio.crs
                    
                    # Resample to 10m if needed
                    if band_data.rio.resolution() != self.config.TARGET_RESOLUTION:
                        logger.debug(f"Resampling {band_name_str} to 10m resolution")
                        band_data = band_data.rio.reproject(
                            target_crs, 
                            resolution=self.config.TARGET_RESOLUTION,
                            resampling=rasterio.enums.Resampling.bilinear
                        )
                    
                    bands_for_composite.append(band_data)
                    band_names_for_coordinate.append(band_name_str)
                    
                except (rasterio.errors.RasterioIOError, Exception) as e:
                    logger.warning(f"Corrupted band {band_name_str} detected: {str(e)[:100]}... Skipping this band.")
                    continue
            
            if not bands_for_composite:
                logger.error("No bands available for composite")
                return None
            
            # Check if we have essential bands for analysis
            essential_bands = ['B02', 'B03', 'B04', 'B08']  # Blue, Green, Red, NIR
            available_essential = [b for b in band_names_for_coordinate if b in essential_bands]
            
            if len(available_essential) < 3:
                logger.error(f"Insufficient essential bands for analysis. Available: {available_essential}, Need at least 3 of: {essential_bands}")
                return None
            
            logger.info(f"Collected {len(bands_for_composite)}/{len(band_order)} bands for composite: {band_names_for_coordinate}")
            if len(bands_for_composite) < len(band_order):
                missing_bands = [b for b in band_order if b not in band_names_for_coordinate]
                logger.warning(f"Missing/corrupted bands: {missing_bands}. Proceeding with available bands.")

            # Concatenate along a new 'band' dimension
            try:
                composite = xr.concat(bands_for_composite, dim=pd.Index(band_names_for_coordinate, name='band'))
            except Exception as e: 
                logger.error(f"xr.concat failed: {e}. Attempting fallback construction.", exc_info=True)
                # Fallback construction
                try:
                    np_arrays = [b.to_numpy() for b in bands_for_composite]
                    stacked_np = np.stack(np_arrays, axis=0)
                    
                    # Get spatial coords from the first band
                    y_coords = bands_for_composite[0].y
                    x_coords = bands_for_composite[0].x
                    current_crs = bands_for_composite[0].rio.crs
                    current_transform = bands_for_composite[0].rio.transform

                    composite = xr.DataArray(
                        stacked_np,
                        coords={'band': band_names_for_coordinate, 'y': y_coords, 'x': x_coords},
                        dims=['band', 'y', 'x']
                    )
                    composite = composite.rio.write_crs(current_crs, inplace=True)
                    composite = composite.rio.write_transform(current_transform, inplace=True)
                except Exception as e_stack:
                    logger.error(f"Fallback DataArray construction also failed: {e_stack}", exc_info=True)
                    return None

            # Ensure CRS is set on the final composite
            if composite.rio.crs is None and bands_for_composite:
                logger.info(f"Manually setting CRS on composite from first band: {bands_for_composite[0].rio.crs}")
                composite = composite.rio.write_crs(bands_for_composite[0].rio.crs, inplace=True)
            if composite.rio.transform().is_identity and bands_for_composite:
                 logger.info("Manually setting transform on composite from first band.")
                 composite = composite.rio.write_transform(bands_for_composite[0].rio.transform(), inplace=True)

            logger.info(f"Composite band coordinate values before cropping: {list(composite.band.values)}")
            
            # Create AOI polygon for clipping to target zone
            aoi_polygon = box(bbox[1], bbox[0], bbox[3], bbox[2])  # west, south, east, north
            aoi_gdf = gpd.GeoDataFrame([{'geometry': aoi_polygon}], crs="EPSG:4326")
            
            # Ensure composite has CRS before reprojecting AOI to it
            if composite.rio.crs is None:
                logger.error("Composite has no CRS before clipping. Aborting composite creation.")
                return None
            aoi_reprojected = aoi_gdf.to_crs(composite.rio.crs)
            
            # Log clipping information for debugging
            logger.info(f"Original composite shape: {composite.sizes}")
            logger.info(f"AOI bbox: {bbox}")
            
            # Clip to target zone area 
            composite_clipped = composite.rio.clip(aoi_reprojected.geometry, all_touched=True, drop=False)
            
            # Check if clipped area is reasonable
            if composite_clipped.sizes.get('x', 0) < 10 or composite_clipped.sizes.get('y', 0) < 10:
                logger.warning(f"Clipped area too small ({composite_clipped.sizes}), using larger buffer")
                # Create expanded AOI for very small micro regions
                expanded_polygon = aoi_polygon.buffer(0.01)  # ~1km buffer
                expanded_gdf = gpd.GeoDataFrame([{'geometry': expanded_polygon}], crs="EPSG:4326")
                expanded_reprojected = expanded_gdf.to_crs(composite.rio.crs)
                composite_clipped = composite.rio.clip(expanded_reprojected.geometry, all_touched=True, drop=False)
                logger.info(f"Expanded clipped shape: {composite_clipped.sizes}")
            
            composite = composite_clipped
            
            # Save cropped composite as multi-band GeoTIFF
            composite.rio.to_raster(
                output_path,
                compress='lzw',
                tiled=True,
                predictor=2
            )
            
            logger.info(f"✅ Cropped composite created: {output_path}")
            logger.info(f"✅ Cropped resolution: {composite.sizes['x']}x{composite.sizes['y']} pixels")
            logger.info(f"✅ File size: ~{output_path.stat().st_size / (1024*1024):.1f} MB")
            return output_path
            
        except Exception as e:
            logger.error(f"Composite creation failed: {e}", exc_info=True)
            return None

    # DISABLED: Composite creation removed - using individual bands directly
    def _create_composite_DISABLED(self, band_paths: Dict[str, Path], output_path: Path, bbox: tuple) -> Optional[Path]:
        """Create multi-band composite from downloaded bands with proper scaling for visualization."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating composite: {output_path.name}")
        
        # Temporarily enable debug logging for this function
        import logging
        original_level = logger.level
        logger.setLevel(logging.DEBUG)
        
        try:
            # Band order optimized for RGB visualization and archaeological analysis
            # B02=Blue, B03=Green, B04=Red for natural color (RGB = B04, B03, B02)
            band_order = ['B02', 'B03', 'B04', 'B08', 'B05', 'B07', 'B11', 'B12']
            
            bands_for_composite = [] # List to hold the 2D DataArray objects
            band_names_for_coordinate = [] # List to hold the string names
            target_crs = None
            
            for band_name_str in band_order:
                if band_name_str not in band_paths:
                    logger.warning(f"Band {band_name_str} missing from band_paths for composite. Skipping.")
                    continue
                    
                band_data = rioxarray.open_rasterio(band_paths[band_name_str], masked=True).squeeze()
                
                if target_crs is None:
                    target_crs = band_data.rio.crs
                
                # Resample to 10m if needed
                if band_data.rio.resolution() != self.config.TARGET_RESOLUTION:
                    logger.debug(f"Resampling {band_name_str} to 10m resolution")
                    band_data = band_data.rio.reproject(
                        target_crs, 
                        resolution=self.config.TARGET_RESOLUTION,
                        resampling=rasterio.enums.Resampling.bilinear
                    )
                
                # Convert to numpy array for processing
                band_values = band_data.values
                
                # Debug: Check actual data range
                valid_mask = ~np.isnan(band_values)
                valid_data = band_values[valid_mask]
                if len(valid_data) > 0:
                    data_min, data_max = np.min(valid_data), np.max(valid_data)
                    logger.debug(f"Band {band_name_str} data range: {data_min:.2f} to {data_max:.2f}")
                
                # Convert to float32 for processing
                band_values = band_values.astype(np.float32)
                
                # Handle different potential scaling scenarios
                valid_mask = ~np.isnan(band_values) & (band_values > 0)
                valid_data = band_values[valid_mask]
                if len(valid_data) > 0:
                    max_val = np.max(valid_data)
                    min_val = np.min(valid_data)
                    
                    logger.debug(f"Band {band_name_str} original range: {min_val:.2f} to {max_val:.2f}")
                    
                    # Determine scaling based on data range
                    if max_val <= 1.0:
                        # Data is in 0-1 range (already normalized), scale to 0-10000
                        logger.debug(f"Band {band_name_str} is 0-1 normalized, scaling to reflectance")
                        band_values = band_values * 10000
                    elif max_val <= 10000:
                        # Data is already in reflectance range (0-10000)
                        logger.debug(f"Band {band_name_str} already in reflectance range")
                        pass  # Keep as is
                    else:
                        # Data is in uint16 range, convert to reflectance
                        logger.debug(f"Band {band_name_str} in uint16 range, converting to reflectance")
                        band_values = band_values * 10000.0 / max_val  # Scale to 10000 max
                else:
                    logger.warning(f"Band {band_name_str} has no valid data!")
                    # Fill with small values instead of zeros to avoid completely black image
                    band_values = np.full_like(band_values, 100.0, dtype=np.float32)
                
                # Ensure minimum values for visibility (avoid pure black)
                band_values = np.maximum(band_values, 1.0)
                
                # Clip to expected reflectance range  
                band_values = np.clip(band_values, 1, 10000)
                
                # Create new DataArray with processed values
                band_data = band_data.copy(data=band_values)
                
                bands_for_composite.append(band_data)
                band_names_for_coordinate.append(band_name_str)
            
            if not bands_for_composite:
                logger.error("No bands available for composite")
                return None
            
            logger.info(f"Collected {len(bands_for_composite)} bands for composite: {band_names_for_coordinate}")

            # Concatenate along a new 'band' dimension, providing the names for its coordinate
            try:
                composite = xr.concat(bands_for_composite, dim=pd.Index(band_names_for_coordinate, name='band'))
            except Exception as e: 
                logger.error(f"xr.concat with pd.Index failed: {e}. Attempting fallback construction.", exc_info=True)
                # Fallback construction (less ideal, might lose some per-band CRS/transform if not careful)
                try:
                    np_arrays = [b.to_numpy() for b in bands_for_composite]
                    stacked_np = np.stack(np_arrays, axis=0)
                    
                    # Get spatial coords from the first band (assuming all are aligned)
                    y_coords = bands_for_composite[0].y
                    x_coords = bands_for_composite[0].x
                    current_crs = bands_for_composite[0].rio.crs
                    current_transform = bands_for_composite[0].rio.transform

                    composite = xr.DataArray(
                        stacked_np,
                        coords={'band': band_names_for_coordinate, 'y': y_coords, 'x': x_coords},
                        dims=['band', 'y', 'x']
                    )
                    composite = composite.rio.write_crs(current_crs, inplace=True)
                    composite = composite.rio.write_transform(current_transform, inplace=True)
                except Exception as e_stack:
                    logger.error(f"Fallback DataArray construction also failed: {e_stack}", exc_info=True)
                    return None

            # Ensure CRS is set on the final composite (xr.concat should preserve if consistent)
            if composite.rio.crs is None and bands_for_composite: # Check if bands_for_composite is not empty
                logger.info(f"Manually setting CRS on composite from first band: {bands_for_composite[0].rio.crs}")
                composite = composite.rio.write_crs(bands_for_composite[0].rio.crs, inplace=True)
            if composite.rio.transform().is_identity and bands_for_composite: # Check if transform is identity
                 logger.info("Manually setting transform on composite from first band.")
                 composite = composite.rio.write_transform(bands_for_composite[0].rio.transform(), inplace=True)

            logger.info(f"Composite band coordinate values before saving: {list(composite.band.values)}")
            
            # Create AOI polygon for clipping
            aoi_polygon = box(bbox[1], bbox[0], bbox[3], bbox[2])  # west, south, east, north
            aoi_gdf = gpd.GeoDataFrame([{'geometry': aoi_polygon}], crs="EPSG:4326")
            
            # Ensure composite has CRS before reprojecting AOI to it
            if composite.rio.crs is None:
                logger.error("Composite has no CRS before clipping. Aborting composite creation.")
                return None
            aoi_reprojected = aoi_gdf.to_crs(composite.rio.crs)
            
            # Log clipping information for debugging
            logger.info(f"Original composite shape: {composite.sizes}")
            logger.info(f"AOI bbox: {bbox}")
            
            # Clip to AOI with larger buffer to ensure we get meaningful data
            composite_clipped = composite.rio.clip(aoi_reprojected.geometry, all_touched=True, drop=False)
            
            # Check if clipped area is too small and adjust if needed
            if composite_clipped.sizes.get('x', 0) < 10 or composite_clipped.sizes.get('y', 0) < 10:
                logger.warning(f"Clipped area too small ({composite_clipped.sizes}), using larger buffer")
                # Create expanded AOI for very small micro regions
                expanded_polygon = aoi_polygon.buffer(0.01)  # ~1km buffer
                expanded_gdf = gpd.GeoDataFrame([{'geometry': expanded_polygon}], crs="EPSG:4326")
                expanded_reprojected = expanded_gdf.to_crs(composite.rio.crs)
                composite_clipped = composite.rio.clip(expanded_reprojected.geometry, all_touched=True, drop=False)
                logger.info(f"Expanded clipped shape: {composite_clipped.sizes}")
            
            composite = composite_clipped
            
            # Debug: Check composite data range before scaling
            composite_values = composite.values
            valid_mask = ~np.isnan(composite_values) & (composite_values > 0)
            valid_composite = composite_values[valid_mask]
            if len(valid_composite) > 0:
                comp_min, comp_max = np.min(valid_composite), np.max(valid_composite)
                logger.info(f"Composite data range before scaling: {comp_min:.2f} to {comp_max:.2f}")
            
            # Professional Sentinel-2 processing: properly handle reflectance scaling
            # Sentinel-2 L2A data is in surface reflectance (0-10000 scale)
            
            composite_values = composite.values
            valid_data = composite_values[composite_values > 0]
            
            if len(valid_data) > 0:
                actual_min, actual_max = np.min(valid_data), np.max(valid_data)
                mean_val = np.mean(valid_data)
                logger.info(f"Original reflectance range: {actual_min:.0f} to {actual_max:.0f} (mean: {mean_val:.0f})")
                
                # Handle different data scaling scenarios
                if actual_max <= 1.0:
                    # Data is normalized (0-1), scale to 0-10000
                    logger.info("Data in 0-1 range, scaling to 0-10000 reflectance")
                    composite_reflectance = composite * 10000
                elif actual_max <= 10000:
                    # Data already in proper 0-10000 reflectance range
                    logger.info("Data already in 0-10000 reflectance range")
                    composite_reflectance = composite
                elif actual_max <= 65535:
                    # Data in uint16 range, likely DN values - convert to reflectance
                    logger.info("Converting from DN to reflectance (scaling to 0-10000)")
                    # Use more conservative scaling to prevent data loss
                    percentile_99 = np.percentile(valid_data, 99)
                    scale_factor = 10000.0 / percentile_99
                    composite_reflectance = composite * scale_factor
                    logger.info(f"Applied scale factor: {scale_factor:.3f} (based on 99th percentile)")
                else:
                    # Very high values - apply stronger normalization
                    logger.info("Very high values detected, applying strong normalization")
                    scale_factor = 10000.0 / actual_max
                    composite_reflectance = composite * scale_factor
                    logger.info(f"Applied scale factor: {scale_factor:.6f}")
                
                # Ensure proper reflectance range
                composite_scaled = np.clip(composite_reflectance, 0, 10000)
                
                # Apply nodata mask properly
                composite_final = np.where(composite > 0, composite_scaled, 0)
                
                # Convert to uint16 for storage
                composite_final = composite_final.astype(np.uint16)
                
                # Additional check for black image prevention
                final_valid = composite_final[composite_final > 0]
                if len(final_valid) > 0:
                    final_min, final_max = np.min(final_valid), np.max(final_valid)
                    logger.info(f"Final valid data range: {final_min} to {final_max}")
                    
                    # If still too dark, apply brightness boost
                    if final_max < 500:  # Very dark image
                        logger.warning("Image appears very dark, applying brightness correction")
                        brightness_factor = 2000.0 / final_max
                        composite_final = np.where(
                            composite_final > 0, 
                            np.clip(composite_final * brightness_factor, 1, 10000).astype(np.uint16), 
                            0
                        )
                        corrected_valid = composite_final[composite_final > 0]
                        if len(corrected_valid) > 0:
                            logger.info(f"After brightness correction: {np.min(corrected_valid)} to {np.max(corrected_valid)}")
                else:
                    logger.error("No valid data after scaling!")
                    
            else:
                logger.error("No valid data found in composite!")
                composite_final = composite.astype(np.uint16)
            
            # Verify final data range
            valid_final = composite_final[composite_final > 0]
            if len(valid_final) > 0:
                final_min, final_max = np.min(valid_final), np.max(valid_final)
                logger.info(f"Final reflectance range: {final_min} to {final_max} (standard 0-10000 scale)")
                logger.info(f"Data coverage: {len(valid_final)}/{composite_final.size} pixels ({len(valid_final)/composite_final.size*100:.1f}%)")
            
            # Create final xarray with proper reflectance values
            nodata_value = 0
            composite_final_xr = composite.copy(data=composite_final)
            composite_final_xr = composite_final_xr.where(composite > 0, nodata_value)
            
            # Create professional band descriptions following ESA naming conventions
            band_name_mapping = {
                'B02': 'Blue (490nm)',
                'B03': 'Green (560nm)', 
                'B04': 'Red (665nm)',
                'B08': 'NIR (842nm)',
                'B05': 'Red Edge 1 (705nm)',
                'B07': 'Red Edge 3 (783nm)',
                'B11': 'SWIR 1 (1610nm)',
                'B12': 'SWIR 2 (2190nm)'
            }
            
            # Create high-quality RGB visualization TIFF as the main output
            rgb_output_path = output_path.parent / f"{output_path.stem}_rgb_visual.tif"
            rgb_result = self._create_rgb_visualization_tiff(composite_final_xr, band_names_for_coordinate, rgb_output_path)
            
            if rgb_result:
                logger.info(f"✓ High-quality RGB composite created as main output")
                return rgb_result
            else:
                logger.error(f"Failed to create RGB visualization, falling back to full composite")
                # Fallback: create the full composite if RGB fails
                return self._create_fallback_composite(composite_final_xr, band_names_for_coordinate, output_path, nodata_value, band_name_mapping)
            
        except Exception as e:
            logger.error(f"Composite creation failed: {e}", exc_info=True)
            return None
        finally:
            # Restore original logging level
            logger.setLevel(original_level)

    def _load_cached_scene(self, metadata_path: Path, composite_path: Path) -> Optional[SceneData]:
        """Load scene data from cache."""
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            return SceneData(
                zone_id=metadata['zone_id'],
                provider='sentinel2',
                scene_id=metadata['scene_id'],
                file_paths={k: Path(v) for k, v in metadata['file_paths'].items()},
                available_bands=metadata['available_bands'],
                metadata=metadata['metadata'],
                features=metadata.get('features', {}),
                composite_file_path=composite_path
            )
        except Exception as e:
            logger.error(f"Failed to load cached scene: {e}")
            return None

    def _cache_scene_metadata(self, scene_data: SceneData, metadata_path: Path):
        """Cache scene metadata."""
        try:
            metadata = {
                'zone_id': scene_data.zone_id,
                'scene_id': scene_data.scene_id,
                'provider': 'sentinel2',
                'file_paths': {k: str(v) for k, v in scene_data.file_paths.items()},
                'available_bands': scene_data.available_bands,
                'metadata': scene_data.metadata,
                'features': scene_data.features
                # No composite_file_path since we don't create composites
            }
            
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to cache metadata: {e}")

    def create_rgb_preview(self, composite_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """Create high-quality RGB preview for OpenAI Vision analysis."""
        try:
            if output_path is None:
                output_path = composite_path.parent / f"{composite_path.stem}_rgb_preview.png"
            
            # Open the multi-band composite
            with rasterio.open(composite_path) as src:
                if src.count < 3:
                    logger.error(f"Composite has only {src.count} bands, need at least 3 for RGB")
                    return None
                
                logger.debug(f"Creating high-quality RGB preview from {src.width}x{src.height} composite")
                
                # Read the RGB bands (B02=Blue, B03=Green, B04=Red)
                blue = src.read(1)   # B02 - Blue band
                green = src.read(2)  # B03 - Green band  
                red = src.read(3)    # B04 - Red band
                
                # Find the actual data bounds (crop to remove black borders)
                data_mask = (red > 0) | (green > 0) | (blue > 0)
                
                if not np.any(data_mask):
                    logger.error("No valid data found in any RGB band")
                    return None
                
                # Find bounding box of actual data
                rows_with_data = np.any(data_mask, axis=1)
                cols_with_data = np.any(data_mask, axis=0)
                
                if not np.any(rows_with_data) or not np.any(cols_with_data):
                    logger.error("Could not find data boundaries")
                    return None
                
                # Get crop boundaries with small padding
                top = np.where(rows_with_data)[0][0]
                bottom = np.where(rows_with_data)[0][-1] + 1
                left = np.where(cols_with_data)[0][0]
                right = np.where(cols_with_data)[0][-1] + 1
                
                # Add 5% padding if possible
                height, width = data_mask.shape
                padding_h = max(1, int((bottom - top) * 0.05))
                padding_w = max(1, int((right - left) * 0.05))
                
                top = max(0, top - padding_h)
                bottom = min(height, bottom + padding_h)
                left = max(0, left - padding_w)
                right = min(width, right + padding_w)
                
                logger.info(f"Cropping to data area: {right-left}x{bottom-top} (was {width}x{height})")
                
                # Crop all bands to data area
                red_crop = red[top:bottom, left:right]
                green_crop = green[top:bottom, left:right]
                blue_crop = blue[top:bottom, left:right]
                data_mask_crop = data_mask[top:bottom, left:right]
                
                # High-quality normalization
                def normalize_band_high_quality(band_data, mask, band_name):
                    if not np.any(mask):
                        logger.warning(f"{band_name} has no valid data in crop area")
                        return np.zeros_like(band_data, dtype=np.uint8)
                    
                    valid_data = band_data[mask]
                    
                    # Check data range and apply appropriate scaling
                    data_min, data_max = np.min(valid_data), np.max(valid_data)
                    data_mean = np.mean(valid_data)
                    
                    logger.debug(f"{band_name} raw data: min={data_min:.0f}, max={data_max:.0f}, mean={data_mean:.0f}")
                    
                    # Handle different scaling scenarios
                    if data_max <= 1.0:
                        scaled_data = valid_data * 10000
                        working_data = band_data * 10000
                    elif data_max <= 10000:
                        scaled_data = valid_data
                        working_data = band_data
                    else:
                        scale_factor = 10000.0 / data_max
                        scaled_data = valid_data * scale_factor
                        working_data = band_data * scale_factor
                    
                    # Use high-quality percentiles (1-99%)
                    if len(scaled_data) > 0:
                        p1, p99 = np.percentile(scaled_data, [1, 99])
                        
                        if p99 <= p1:
                            p1, p99 = np.min(scaled_data), np.max(scaled_data)
                            if p99 <= p1:
                                p1, p99 = 0, 10000
                    else:
                        p1, p99 = 0, 10000
                    
                    # Apply high-quality 1-99% stretch
                    normalized = np.clip((working_data.astype(np.float32) - p1) / (p99 - p1), 0, 1)
                    
                    # Apply minimal gamma correction for natural look
                    gamma = 0.9
                    normalized = np.power(normalized, gamma)
                    
                    # Convert to 8-bit with full dynamic range
                    result = (normalized * 255).astype(np.uint8)
                    
                    # Set nodata areas to black
                    result[~mask] = 0
                    
                    return result
                
                # Normalize each band with high-quality processing
                red_norm = normalize_band_high_quality(red_crop, data_mask_crop, "Red")
                green_norm = normalize_band_high_quality(green_crop, data_mask_crop, "Green")
                blue_norm = normalize_band_high_quality(blue_crop, data_mask_crop, "Blue")
                
                # Stack into RGB image
                rgb_image = np.stack([red_norm, green_norm, blue_norm], axis=-1)
                
                # Create high quality PIL image
                from PIL import Image, ImageEnhance
                pil_image = Image.fromarray(rgb_image, mode='RGB')
                
                # Professional enhancement pipeline for satellite imagery
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.02)  # Minimal contrast boost
                
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(1.01)  # Very subtle sharpening
                
                # Save as high quality PNG
                pil_image.save(
                    output_path, 
                    format='PNG', 
                    optimize=False, 
                    compress_level=1  # Minimal compression for max quality
                )
                
                logger.info(f"✅ High quality RGB preview created: {output_path} ({pil_image.size[0]}x{pil_image.size[1]})")
                return output_path
                
        except Exception as e:
            logger.error(f"Failed to create RGB preview: {e}", exc_info=True)
            return None

    # DISABLED: RGB preview creation removed - no composites created
    def create_rgb_preview_DISABLED(self, composite_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """Create high-quality cropped RGB preview for OpenAI Vision analysis."""
        try:
            if output_path is None:
                output_path = composite_path.parent / f"{composite_path.stem}_rgb_preview.png"
            
            # Open the multi-band composite
            with rasterio.open(composite_path) as src:
                if src.count < 3:
                    logger.error(f"Composite has only {src.count} bands, need at least 3 for RGB")
                    return None
                
                logger.debug(f"Creating high-quality RGB preview from {src.width}x{src.height} composite")
                
                # Read the RGB bands and reorder for natural color
                blue = src.read(1)   # B02 - Blue band
                green = src.read(2)  # B03 - Green band  
                red = src.read(3)    # B04 - Red band
                
                # Find the actual data bounds (crop to remove black borders)
                data_mask = (red > 0) | (green > 0) | (blue > 0)
                
                if not np.any(data_mask):
                    logger.error("No valid data found in any RGB band")
                    return None
                
                # Find bounding box of actual data
                rows_with_data = np.any(data_mask, axis=1)
                cols_with_data = np.any(data_mask, axis=0)
                
                if not np.any(rows_with_data) or not np.any(cols_with_data):
                    logger.error("Could not find data boundaries")
                    return None
                
                # Get crop boundaries with small padding
                top = np.where(rows_with_data)[0][0]
                bottom = np.where(rows_with_data)[0][-1] + 1
                left = np.where(cols_with_data)[0][0]
                right = np.where(cols_with_data)[0][-1] + 1
                
                # Add 5% padding if possible
                height, width = data_mask.shape
                padding_h = max(1, int((bottom - top) * 0.05))
                padding_w = max(1, int((right - left) * 0.05))
                
                top = max(0, top - padding_h)
                bottom = min(height, bottom + padding_h)
                left = max(0, left - padding_w)
                right = min(width, right + padding_w)
                
                logger.info(f"Cropping to data area: {right-left}x{bottom-top} (was {width}x{height})")
                
                # Crop all bands to data area
                red_crop = red[top:bottom, left:right]
                green_crop = green[top:bottom, left:right]
                blue_crop = blue[top:bottom, left:right]
                data_mask_crop = data_mask[top:bottom, left:right]
                
                # High-quality normalization matching the RGB TIFF processing
                def normalize_band_high_quality(band_data, mask, band_name):
                    if not np.any(mask):
                        logger.warning(f"{band_name} has no valid data in crop area")
                        return np.zeros_like(band_data, dtype=np.uint8)
                    
                    valid_data = band_data[mask]
                    
                    # Check data range and apply appropriate scaling
                    data_min, data_max = np.min(valid_data), np.max(valid_data)
                    data_mean = np.mean(valid_data)
                    
                    logger.debug(f"{band_name} raw data: min={data_min:.0f}, max={data_max:.0f}, mean={data_mean:.0f}")
                    
                    # Handle different scaling scenarios (same as RGB TIFF)
                    if data_max <= 1.0:
                        scaled_data = valid_data * 10000
                        working_data = band_data * 10000
                    elif data_max <= 10000:
                        scaled_data = valid_data
                        working_data = band_data
                    else:
                        scale_factor = 10000.0 / data_max
                        scaled_data = valid_data * scale_factor
                        working_data = band_data * scale_factor
                    
                    # Use same high-quality percentiles as RGB TIFF (1-99%)
                    if len(scaled_data) > 0:
                        p1, p99 = np.percentile(scaled_data, [1, 99])  # Conservative percentiles
                        
                        if p99 <= p1:
                            p1, p99 = np.min(scaled_data), np.max(scaled_data)
                            if p99 <= p1:
                                p1, p99 = 0, 10000
                    else:
                        p1, p99 = 0, 10000
                    
                    # Apply high-quality 1-99% stretch
                    normalized = np.clip((working_data.astype(np.float32) - p1) / (p99 - p1), 0, 1)
                    
                    # Apply minimal gamma correction for natural look
                    gamma = 0.9  # Very subtle enhancement
                    normalized = np.power(normalized, gamma)
                    
                    # Convert to 8-bit with full dynamic range
                    result = (normalized * 255).astype(np.uint8)
                    
                    # Set nodata areas to black
                    result[~mask] = 0
                    
                    # Verify result
                    result_valid = result[mask]
                    if len(result_valid) > 0:
                        logger.debug(f"{band_name} high-quality: {np.min(result_valid)}-{np.max(result_valid)} "
                                   f"(stretch: {p1:.0f}-{p99:.0f})")
                    
                    return result
                
                # Normalize each band with high-quality processing
                red_norm = normalize_band_high_quality(red_crop, data_mask_crop, "Red")
                green_norm = normalize_band_high_quality(green_crop, data_mask_crop, "Green")
                blue_norm = normalize_band_high_quality(blue_crop, data_mask_crop, "Blue")
                
                # Stack into RGB image
                rgb_image = np.stack([red_norm, green_norm, blue_norm], axis=-1)
                
                # Create maximum quality PIL image
                from PIL import Image, ImageEnhance, ImageFilter
                pil_image = Image.fromarray(rgb_image, mode='RGB')
                
                # Professional enhancement pipeline for satellite imagery
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.02)  # Minimal contrast boost
                
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(1.01)  # Very subtle sharpening
                
                # Preserve completely natural colors (no saturation change)
                enhancer = ImageEnhance.Color(pil_image)
                pil_image = enhancer.enhance(1.0)  # No color enhancement
                
                # High-resolution output - ensure minimum 4K quality
                min_size = 4096  # 4K minimum for maximum detail
                if max(pil_image.size) < min_size:
                    scale_factor = min_size / max(pil_image.size)
                    new_size = (int(pil_image.width * scale_factor), int(pil_image.height * scale_factor))
                    # Use best quality upscaling algorithm
                    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                    logger.info(f"Upscaled to 4K quality: {new_size}")
                
                # Cap at 8K for reasonable file size
                max_size = 8192  # 8K maximum
                if max(pil_image.size) > max_size:
                    ratio = max_size / max(pil_image.size)
                    new_size = (int(pil_image.width * ratio), int(pil_image.height * ratio))
                    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
                    logger.info(f"Scaled to 8K maximum: {new_size}")
                
                # Apply subtle unsharp mask for archaeological detail enhancement
                pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=0.5, percent=110, threshold=2))
                
                # Save as maximum quality PNG with optimal settings
                pil_image.save(
                    output_path, 
                    format='PNG', 
                    optimize=False, 
                    compress_level=1,  # Minimal compression for max quality
                    pnginfo=None  # No metadata for cleaner file
                )
                
                logger.info(f"✅ Maximum quality RGB preview created: {output_path} ({pil_image.size[0]}x{pil_image.size[1]})")
                logger.info(f"✅ Enhanced from {width}x{height} to {pil_image.size[0]}x{pil_image.size[1]} with 4K+ quality")
                logger.info(f"✅ Processing: 1-99% stretch, minimal gamma, archaeological detail enhancement")
                return output_path
                
        except Exception as e:
            logger.error(f"Failed to create RGB preview: {e}", exc_info=True)
            return None

    # DISABLED: RGB TIFF creation removed - no composites created  
    def _create_rgb_visualization_tiff_DISABLED(self, composite_xr, band_names, output_path, target_bbox=None):
        """Create maximum quality RGB GeoTIFF cropped to target area"""
        try:
            logger.info(f"Creating MAXIMUM QUALITY RGB TIFF: {output_path.name}")
            
            # Get RGB bands (B04=Red, B03=Green, B02=Blue)
            rgb_indices = []
            rgb_band_names = ['B04', 'B03', 'B02']  # Red, Green, Blue
            
            for rgb_band in rgb_band_names:
                if rgb_band in band_names:
                    rgb_indices.append(band_names.index(rgb_band))
                else:
                    logger.warning(f"RGB band {rgb_band} not found, using fallback")
                    rgb_indices.append(0)  # Use first available band as fallback
            
            # Extract RGB data from composite
            red_data = composite_xr.values[rgb_indices[0]].astype(np.float64)
            green_data = composite_xr.values[rgb_indices[1]].astype(np.float64)
            blue_data = composite_xr.values[rgb_indices[2]].astype(np.float64)
            
            # Create data mask
            data_mask = (red_data > 0) | (green_data > 0) | (blue_data > 0)
            
            logger.info(f"Original composite size: {red_data.shape[1]}x{red_data.shape[0]} pixels")
            
            # Find the actual data bounds within the composite
            if np.any(data_mask):
                rows, cols = np.where(data_mask)
                data_min_row, data_max_row = rows.min(), rows.max()
                data_min_col, data_max_col = cols.min(), cols.max()
                
                # Add small padding to ensure we get all data
                padding = 50  # Larger padding for safety
                crop_min_row = max(0, data_min_row - padding)
                crop_max_row = min(red_data.shape[0] - 1, data_max_row + padding)
                crop_min_col = max(0, data_min_col - padding)
                crop_max_col = min(red_data.shape[1] - 1, data_max_col + padding)
                
                # Crop to actual data area with padding
                red_crop = red_data[crop_min_row:crop_max_row+1, crop_min_col:crop_max_col+1]
                green_crop = green_data[crop_min_row:crop_max_row+1, crop_min_col:crop_max_col+1]
                blue_crop = blue_data[crop_min_row:crop_max_row+1, crop_min_col:crop_max_col+1]
                data_mask_crop = data_mask[crop_min_row:crop_max_row+1, crop_min_col:crop_max_col+1]
                
                logger.info(f"Cropped to data area: {red_crop.shape[1]}x{red_crop.shape[0]} pixels")
                logger.info(f"Data coverage: {np.sum(data_mask_crop)}/{data_mask_crop.size} pixels ({np.sum(data_mask_crop)/data_mask_crop.size*100:.1f}%)")
                
                # Calculate new geotransform for cropped area
                old_transform = composite_xr.rio.transform()
                west, south = old_transform * (crop_min_col, crop_max_row + 1)
                east, north = old_transform * (crop_max_col + 1, crop_min_row)
                
                new_transform = rasterio.transform.from_bounds(
                    west, south, east, north,
                    red_crop.shape[1], red_crop.shape[0]
                )
                
                # Use cropped data
                final_red = red_crop
                final_green = green_crop
                final_blue = blue_crop
                final_mask = data_mask_crop
                final_transform = new_transform
                final_height, final_width = red_crop.shape
                
            else:
                logger.warning("No valid data found, using full composite")
                final_red = red_data
                final_green = green_data
                final_blue = blue_data
                final_mask = data_mask
                final_transform = composite_xr.rio.transform()
                final_height, final_width = red_data.shape
            
            if not np.any(final_mask):
                logger.warning("No valid RGB data found")
                return None
            
            # PROFESSIONAL MAXIMUM QUALITY scaling to preserve ALL detail
            def scale_band_maximum_quality(band_data, mask):
                if not np.any(mask):
                    return np.zeros_like(band_data, dtype=np.uint16)
                
                valid_data = band_data[mask]
                
                # Use VERY conservative percentiles to preserve ALL detail
                p0_5, p99_5 = np.percentile(valid_data, [0.5, 99.5])  # Ultra-conservative
                
                if p99_5 <= p0_5:
                    p0_5, p99_5 = np.min(valid_data), np.max(valid_data)
                    if p99_5 <= p0_5:
                        return np.full_like(band_data, 32768, dtype=np.uint16)
                
                logger.info(f"Scaling range: {p0_5:.2f} to {p99_5:.2f} -> 0 to 65535")
                
                # MAXIMUM DETAIL stretch to full 16-bit range
                scaled = np.clip((band_data - p0_5) / (p99_5 - p0_5) * 65535, 0, 65535)
                result = scaled.astype(np.uint16)
                result[~mask] = 0
                
                return result
            
            # Process bands with maximum quality scaling
            logger.info("Applying maximum quality scaling to cropped data...")
            red_scaled = scale_band_maximum_quality(final_red, final_mask)
            green_scaled = scale_band_maximum_quality(final_green, final_mask)
            blue_scaled = scale_band_maximum_quality(final_blue, final_mask)
            
            # Verify quality
            for band_name, band_data in [("Red", red_scaled), ("Green", green_scaled), ("Blue", blue_scaled)]:
                valid_pixels = band_data[final_mask]
                if len(valid_pixels) > 0:
                    logger.info(f"{band_name} band: {np.min(valid_pixels)} to {np.max(valid_pixels)} "
                              f"(mean: {np.mean(valid_pixels):.0f})")
            
            # Save MAXIMUM QUALITY RGB GeoTIFF cropped to data area
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=final_height,
                width=final_width,
                count=3,
                dtype='uint16',  # 16-bit for maximum quality
                crs=composite_xr.rio.crs,
                transform=final_transform,
                compress='lzw',
                tiled=True,
                predictor=2,  # Horizontal differencing for better compression
                photometric='RGB',
                bigtiff='yes' if final_width * final_height > 50000000 else 'no'  # BigTIFF only if needed
            ) as dst:
                dst.write(red_scaled, 1)    # Red band
                dst.write(green_scaled, 2)  # Green band
                dst.write(blue_scaled, 3)   # Blue band
                
                # Set color interpretation
                dst.colorinterp = [rasterio.enums.ColorInterp.red, 
                                 rasterio.enums.ColorInterp.green, 
                                 rasterio.enums.ColorInterp.blue]
                
                dst.set_band_description(1, 'Red (B04) - Maximum Quality 16-bit Cropped')
                dst.set_band_description(2, 'Green (B03) - Maximum Quality 16-bit Cropped')
                dst.set_band_description(3, 'Blue (B02) - Maximum Quality 16-bit Cropped')
                
                dst.update_tags(
                    TIFFTAG_SOFTWARE='Amazon Archaeological Discovery Pipeline',
                    PROCESSING_LEVEL='MAXIMUM QUALITY RGB Cropped to Data Area',
                    SCALING='0.5-99.5% percentile stretch to 0-65535',
                    DATA_TYPE='16-bit RGB for absolute maximum quality',
                    QUALITY='Cropped to data area, maximum detail preservation',
                    RESOLUTION=f'{final_width}x{final_height}',
                    PIXEL_SIZE='10m',
                    DATA_COVERAGE=f'{np.sum(final_mask)/final_mask.size*100:.1f}%'
                )
            
            logger.info(f"✅ MAXIMUM QUALITY RGB TIFF created: {output_path}")
            logger.info(f"✅ CROPPED RESOLUTION: {final_width}x{final_height} pixels")
            logger.info(f"✅ 16-bit depth, 0.5-99.5% stretch, LZW compressed")
            logger.info(f"✅ File size: ~{(final_width * final_height * 3 * 2) / (1024*1024):.1f} MB")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create RGB visualization TIFF: {e}", exc_info=True)
            return None

    # DISABLED: Fallback composite creation removed - no composites created
    def _create_fallback_composite_DISABLED(self, composite_xr, band_names, output_path, nodata_value, band_name_mapping):
        """Create fallback full composite if RGB fails"""
        try:
            logger.info("Creating fallback full composite")
            
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=composite_xr.sizes['y'],
                width=composite_xr.sizes['x'],
                count=len(band_names),
                dtype='uint16',
                crs=composite_xr.rio.crs,
                transform=composite_xr.rio.transform(),
                compress='lzw',
                tiled=True,
                nodata=nodata_value,
                photometric='minisblack'
            ) as dst:
                for i, band_name in enumerate(band_names, 1):
                    dst.write(composite_xr.values[i-1], i)
                    band_desc = band_name_mapping.get(band_name, f'Sentinel-2 {band_name}')
                    dst.set_band_description(i, band_desc)
                
                dst.update_tags(
                    TIFFTAG_SOFTWARE='Amazon Archaeological Discovery Pipeline',
                    PROCESSING_LEVEL='Surface Reflectance (L2A)',
                    SCALE_FACTOR='10000'
                )
            
            logger.info(f"✓ Fallback composite created: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Fallback composite creation failed: {e}")
            return None


# Factory function for compatibility
def create_sentinel2_provider() -> Sentinel2Provider:
    """Create Sentinel-2 provider instance."""
    return Sentinel2Provider()


# Test function
def test_sentinel2_provider(zone_id: str = 'upper_napo') -> bool:
    """Test Sentinel-2 provider functionality."""
    try:
        from src.core.config import TARGET_ZONES
        
        if zone_id not in TARGET_ZONES:
            logger.error(f"Zone {zone_id} not found")
            return False
        
        provider = Sentinel2Provider()
        scenes = provider.download_data([zone_id], max_scenes=1)
        
        if scenes:
            logger.info(f"✅ Test successful: {len(scenes)} scene(s) processed")
            return True
        else:
            logger.warning(f"⚠️  No scenes processed for {zone_id}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the clean provider
    success = test_sentinel2_provider('upper_napo')
    print(f"Test result: {'✅ PASSED' if success else '❌ FAILED'}")       # Use RGB preview if available, otherwise fall back to composite

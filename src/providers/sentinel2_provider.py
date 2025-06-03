"""
IMPROVED Sentinel-2 AWS Open Data Provider for Archaeological Discovery
Fixed: Precise location targeting with overlap filtering and center distance
"""

import os
import json
import logging
import requests
import rasterio
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

# Core dependencies for Sentinel-2 STAC
from pystac_client import Client
import pystac
import geopandas as gpd
from shapely.geometry import box, Point, Polygon
from shapely.ops import unary_union
import rioxarray
import xarray as xr

# Your existing core modules
from src.core.config import TARGET_ZONES, DetectionConfig, SATELLITE_DIR
from src.core.data_objects import SceneData, BaseProvider

logger = logging.getLogger(__name__)

@dataclass
class Sentinel2Config:
    """Configuration for Sentinel-2 data access and processing"""
    
    # Earth Search STAC API (Element84 on AWS)
    STAC_API_URL: str = "https://earth-search.aws.element84.com/v1"
    
    # Collection names
    L2A_COLLECTION: str = "sentinel-2-l2a"  # Atmospherically corrected
    L1C_COLLECTION: str = "sentinel-2-l1c"  # Top of atmosphere
    
    # Band mapping optimized for archaeological analysis
    ARCHAEOLOGICAL_BANDS: Dict[str, str] = field(default_factory=lambda: {
        'coastal': 'B01',     # 443nm, 60m - Coastal aerosol
        'blue': 'B02',        # 490nm, 10m - Blue
        'green': 'B03',       # 560nm, 10m - Green  
        'red': 'B04',         # 665nm, 10m - Red
        'red_edge_1': 'B05',  # 705nm, 20m - Red Edge 1 (CRITICAL for archaeology)
        'red_edge_2': 'B06',  # 740nm, 20m - Red Edge 2
        'red_edge_3': 'B07',  # 783nm, 20m - Red Edge 3 (CRITICAL for archaeology)
        'nir': 'B08',         # 842nm, 10m - NIR
        'nir_narrow': 'B8A',  # 865nm, 20m - NIR Narrow
        'water_vapor': 'B09', # 945nm, 60m - Water vapor
        'cirrus': 'B10',      # 1375nm, 60m - Cirrus
        'swir1': 'B11',       # 1610nm, 20m - SWIR1 (terra preta detection)
        'swir2': 'B12'        # 2190nm, 20m - SWIR2
    })
    
    # Priority bands for archaeological analysis (highest resolution + key bands)
    PRIORITY_BANDS: List[str] = field(default_factory=lambda: ['B02', 'B03', 'B04', 'B05', 'B07', 'B08', 'B11', 'B12'])
    
    # Quality filters
    MAX_CLOUD_COVER: float = 20.0
    MIN_DATA_COVERAGE: float = 80.0
    
    # NEW: Precise targeting filters
    MIN_OVERLAP_PERCENTAGE: float = 70.0  # Minimum overlap with AOI
    MAX_CENTER_DISTANCE_KM: float = 50.0  # Maximum distance from AOI center
    
    # Preferred dates (dry season for Amazon)
    PREFERRED_MONTHS: List[int] = field(default_factory=lambda: [6, 7, 8, 9])  # June-September
    
    # Temporal range
    DEFAULT_START_DATE: str = "2023-01-01"
    DEFAULT_END_DATE: str = "2024-12-31"

class Sentinel2Error(Exception):
    """Custom exception for Sentinel-2 related errors"""
    pass

class Sentinel2Provider(BaseProvider):
    """
    IMPROVED Sentinel-2 provider with precise location targeting
    
    Key Improvements:
    1. Overlap percentage filtering - scenes must overlap significantly with AOI
    2. Center distance filtering - scenes must be reasonably close to AOI center
    3. AOI clipping support - option to clip downloaded data to exact AOI
    4. Better scene ranking based on spatial fit, not just cloud cover
    """
    
    def __init__(self, config: Optional[Sentinel2Config] = None):
        self.config = config or Sentinel2Config()
        self.stac_client = Client.open(self.config.STAC_API_URL)
        self.session = requests.Session()
        
        logger.info("🛰️ IMPROVED Sentinel-2 AWS Provider initialized")
        logger.info(f"STAC API: {self.config.STAC_API_URL}")
        logger.info(f"Collection: {self.config.L2A_COLLECTION}")
        logger.info(f"Min Overlap: {self.config.MIN_OVERLAP_PERCENTAGE}%")
        logger.info(f"Max Center Distance: {self.config.MAX_CENTER_DISTANCE_KM}km")

    def download_data(self, zones: Optional[List[str]] = None, max_scenes: int = 3) -> List[SceneData]:
        """
        Download Sentinel-2 data for archaeological analysis with improved targeting
        """
        all_scene_data = []
        
        # Handle None or empty zones list
        if zones is None or len(zones) == 0:
            # Default to priority 1 zones
            zones = [zone_id for zone_id, zone in TARGET_ZONES.items() if zone.priority == 1]
            logger.info(f"No zones specified, using priority 1 zones: {zones}")
        
        # Ensure zones is a list
        if isinstance(zones, str):
            zones = [zones]
            
        for zone_id in zones:
            if zone_id not in TARGET_ZONES:
                logger.warning(f"Unknown zone: {zone_id}")
                continue
            zone = TARGET_ZONES[zone_id]
            logger.info(f"\n🎯 Processing {zone.name} with IMPROVED Sentinel-2 targeting")
            try:
                scenes = self.search_scenes_improved(zone, max_scenes * 3)  # Get more to filter better
                if not scenes:
                    logger.warning(f"No suitable Sentinel-2 scenes found for {zone.name}")
                    continue
                zone_scenes = []
                for i, scene in enumerate(scenes[:max_scenes]):
                    logger.info(f"Processing scene {i+1}/{max_scenes}: {scene['id']} (overlap: {scene.get('overlap_percentage', 0):.1f}%)")
                    scene_data = self.process_scene(scene, zone)
                    if scene_data:
                        zone_scenes.append(scene_data)
                        logger.info(f"✓ Successfully processed scene {scene['id']}")
                    else:
                        logger.warning(f"Failed to process scene {scene['id']}")
                all_scene_data.extend(zone_scenes)
                logger.info(f"✓ Completed {zone.name}: {len(zone_scenes)} scenes processed")
            except Exception as e:
                logger.error(f"Error processing zone {zone.name}: {e}")
                continue
        logger.info(f"🎯 IMPROVED Sentinel-2 download complete: {len(all_scene_data)} scenes total")
        return all_scene_data

    def search_scenes_improved(self, zone, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        IMPROVED scene search with precise location targeting and overlap filtering
        """
        logger.info(f"🔍 IMPROVED Sentinel-2 search for {zone.name} with spatial filtering")
        try:
            # Create AOI geometry for overlap calculations
            config_bbox = zone.bbox  # (south, west, north, east)
            aoi_polygon = box(config_bbox[1], config_bbox[0], config_bbox[3], config_bbox[2])  # (west, south, east, north)
            aoi_center = Point((config_bbox[1] + config_bbox[3])/2, (config_bbox[0] + config_bbox[2])/2)
            
            # Convert to STAC format: (west, south, east, north) 
            stac_bbox = [config_bbox[1], config_bbox[0], config_bbox[3], config_bbox[2]]
            
            # Search with larger bbox to get more candidates for filtering
            expanded_bbox = self._expand_bbox(stac_bbox, factor=1.5)  # 50% larger search area
            
            search = self.stac_client.search(
                collections=[self.config.L2A_COLLECTION],
                bbox=expanded_bbox,  # Use expanded bbox to get more candidates
                datetime=f"{self.config.DEFAULT_START_DATE}/{self.config.DEFAULT_END_DATE}",
                limit=max_results * 5,  # Get many more to filter spatially
                query={
                    "eo:cloud_cover": {"lt": 50.0}  # Initial cloud filter
                }
            )
            items = list(search.items())
            logger.info(f"Found {len(items)} candidate scenes in expanded search area")
            
            if not items:
                return []
            
            # NEW: Apply spatial filtering and scoring
            spatially_filtered_scenes = []
            for item in items:
                try:
                    score_data = self.score_archaeological_suitability_improved(item, zone, aoi_polygon, aoi_center)
                    
                    # Apply spatial filters
                    if (score_data.get('overlap_percentage', 0) >= self.config.MIN_OVERLAP_PERCENTAGE and
                        score_data.get('center_distance_km', float('inf')) <= self.config.MAX_CENTER_DISTANCE_KM):
                        spatially_filtered_scenes.append(score_data)
                        logger.debug(f"✓ Scene {item.id}: overlap={score_data['overlap_percentage']:.1f}%, distance={score_data['center_distance_km']:.1f}km")
                    else:
                        logger.debug(f"✗ Scene {item.id}: overlap={score_data.get('overlap_percentage', 0):.1f}%, distance={score_data.get('center_distance_km', 999):.1f}km (filtered out)")
                        
                except Exception as e:
                    logger.warning(f"Error scoring scene {item.id}: {e}")
                    continue
            
            if not spatially_filtered_scenes:
                logger.warning(f"No scenes passed spatial filtering for {zone.name}")
                return []
            
            # Sort by improved quality score (includes spatial fit)
            spatially_filtered_scenes.sort(key=lambda x: x['quality_score'], reverse=True)
            
            logger.info(f"✓ Selected {len(spatially_filtered_scenes)} spatially-suitable scenes for {zone.name}")
            logger.info(f"Best scene: {spatially_filtered_scenes[0]['id']} (score: {spatially_filtered_scenes[0]['quality_score']:.1f}, overlap: {spatially_filtered_scenes[0]['overlap_percentage']:.1f}%)")
            
            return spatially_filtered_scenes[:max_results]
            
        except Exception as e:
            raise Sentinel2Error(f"IMPROVED scene search failed for {zone.name}: {e}")

    def _expand_bbox(self, bbox: List[float], factor: float = 1.5) -> List[float]:
        """Expand bounding box by a factor to get more search candidates"""
        west, south, east, north = bbox
        center_lon = (west + east) / 2
        center_lat = (south + north) / 2
        
        width = east - west
        height = north - south
        
        new_width = width * factor
        new_height = height * factor
        
        return [
            center_lon - new_width/2,  # west
            center_lat - new_height/2,  # south  
            center_lon + new_width/2,  # east
            center_lat + new_height/2   # north
        ]

    def score_archaeological_suitability_improved(self, item: pystac.Item, zone, aoi_polygon: Polygon, aoi_center: Point) -> Dict[str, Any]:
        """
        IMPROVED scoring with spatial overlap and center distance calculations
        """
        properties = item.properties
        
        # Extract key metadata
        cloud_cover = properties.get('eo:cloud_cover', 100)
        data_coverage = properties.get('s2:data_coverage', 0)
        acquisition_date = datetime.fromisoformat(properties['datetime'].replace('Z', '+00:00'))
        
        # NEW: Calculate spatial metrics
        item_geometry = Polygon(item.geometry['coordinates'][0])
        
        # Calculate overlap percentage
        try:
            intersection = item_geometry.intersection(aoi_polygon)
            aoi_area = aoi_polygon.area
            overlap_area = intersection.area
            overlap_percentage = (overlap_area / aoi_area) * 100 if aoi_area > 0 else 0
        except Exception as e:
            logger.warning(f"Error calculating overlap for {item.id}: {e}")
            overlap_percentage = 0
        
        # Calculate center distance
        try:
            item_center = item_geometry.centroid
            # Convert degrees to km (rough approximation)
            distance_degrees = aoi_center.distance(item_center)
            center_distance_km = distance_degrees * 111.32  # 1 degree ≈ 111.32 km
        except Exception as e:
            logger.warning(f"Error calculating center distance for {item.id}: {e}")
            center_distance_km = float('inf')
        
        # Calculate IMPROVED quality score with spatial weighting
        quality_score = 0
        
        # Cloud cover score (0-25 points, reduced from 30)
        if cloud_cover <= 5:
            quality_score += 25
        elif cloud_cover <= 10:
            quality_score += 20
        elif cloud_cover <= 15:
            quality_score += 12
        elif cloud_cover <= 20:
            quality_score += 5
        
        # Data coverage score (0-15 points, reduced from 20)
        if data_coverage >= 95:
            quality_score += 15
        elif data_coverage >= 90:
            quality_score += 12
        elif data_coverage >= 80:
            quality_score += 8
        
        # NEW: Spatial overlap score (0-30 points) - HIGH WEIGHT
        if overlap_percentage >= 95:
            quality_score += 30
        elif overlap_percentage >= 85:
            quality_score += 25
        elif overlap_percentage >= 75:
            quality_score += 20
        elif overlap_percentage >= 65:
            quality_score += 15
        elif overlap_percentage >= 50:
            quality_score += 10
        
        # NEW: Center distance score (0-20 points)
        if center_distance_km <= 10:
            quality_score += 20
        elif center_distance_km <= 25:
            quality_score += 15
        elif center_distance_km <= 50:
            quality_score += 10
        elif center_distance_km <= 75:
            quality_score += 5
        
        # Seasonal preference (0-10 points, reduced from 20)
        if acquisition_date.month in self.config.PREFERRED_MONTHS:
            quality_score += 10
        elif acquisition_date.month in [5, 10]:  # Shoulder months
            quality_score += 5
        
        # Solar angle bonus (0-10 points, reduced from 15)
        sun_zenith = properties.get('s2:mean_solar_zenith', 90)
        if sun_zenith < 30:  # High sun angle
            quality_score += 10
        elif sun_zenith < 40:
            quality_score += 7
        elif sun_zenith < 50:
            quality_score += 3
        
        return {
            'id': item.id,
            'stac_item': item,
            'acquisition_date': acquisition_date.strftime('%Y-%m-%d'),
            'cloud_cover': cloud_cover,
            'data_coverage': data_coverage,
            'sun_zenith': sun_zenith,
            'sun_azimuth': properties.get('s2:mean_solar_azimuth', 0),
            'quality_score': quality_score,
            'mgrs_tile': properties.get('s2:mgrs_tile', 'unknown'),
            'processing_level': properties.get('s2:processing_level', 'L2A'),
            'constellation': properties.get('constellation', 'sentinel-2'),
            # NEW spatial metrics
            'overlap_percentage': overlap_percentage,
            'center_distance_km': center_distance_km,
            'spatial_fit_score': overlap_percentage - center_distance_km  # Combined spatial metric
        }

    def process_scene(self, scene_data: Dict[str, Any], zone) -> Optional[SceneData]:
        """
        Process and download a Sentinel-2 scene with enhanced spatial metadata
        """
        item = scene_data['stac_item']
        scene_id = scene_data['id']
        
        try:
            # Create local directory for this scene
            zone_dir = SATELLITE_DIR / zone.name.lower().replace(' ', '_') / 'sentinel2'
            scene_dir = zone_dir / scene_id
            scene_dir.mkdir(parents=True, exist_ok=True)
            
            # Download priority bands for archaeological analysis
            file_paths = {}
            available_bands = []
            
            logger.info(f"Downloading bands for {scene_id}")
            
            # Map band names to actual asset names in STAC
            band_mapping = {
                'B02': 'blue',     # Blue
                'B03': 'green',    # Green
                'B04': 'red',      # Red
                'B05': 'rededge1', # Red edge 1
                'B07': 'rededge3', # Red edge 3
                'B08': 'nir',      # NIR
                'B11': 'swir16',   # SWIR 1610nm
                'B12': 'swir22'    # SWIR 2190nm
            }
            
            for band_code in self.config.PRIORITY_BANDS:
                try:
                    # Try the mapped asset name first
                    asset_name = band_mapping.get(band_code, band_code.lower())
                    if asset_name in item.assets:
                        asset = item.assets[asset_name]
                        band_file = self.download_band(asset, scene_dir, band_code)
                        if band_file and band_file.exists():
                            file_paths[band_code] = band_file
                            available_bands.append(band_code)
                            logger.debug(f"  ✓ Downloaded {band_code} ({asset_name})")
                        else:
                            logger.warning(f"  ❌ Failed to download {band_code}")
                    else:
                        # Try alternative naming
                        alt_names = [band_code, band_code.lower(), f"{band_code.lower()}-cog"]
                        found = False
                        for alt_name in alt_names:
                            if alt_name in item.assets:
                                asset = item.assets[alt_name]
                                band_file = self.download_band(asset, scene_dir, band_code)
                                if band_file and band_file.exists():
                                    file_paths[band_code] = band_file
                                    available_bands.append(band_code)
                                    logger.debug(f"  ✓ Downloaded {band_code} ({alt_name})")
                                    found = True
                                    break
                        if not found:
                            logger.warning(f"  ⚠️ Band {band_code} not available in scene")
                except Exception as e:
                    logger.warning(f"Error downloading band {band_code}: {e}")
                    continue
            
            if len(available_bands) < 4:  # Need minimum bands
                logger.warning(f"Insufficient bands downloaded for {scene_id}: {len(available_bands)}")
                return None
            
            # Create enhanced metadata with spatial fit information
            metadata = {
                'provider': 'sentinel-2',
                'acquisition_date': scene_data['acquisition_date'],
                'cloud_cover': scene_data['cloud_cover'],
                'data_coverage': scene_data['data_coverage'],
                'quality_score': scene_data['quality_score'],
                'mgrs_tile': scene_data['mgrs_tile'],
                'processing_level': scene_data['processing_level'],
                'constellation': scene_data['constellation'],
                'sun_zenith': scene_data['sun_zenith'],
                'sun_azimuth': scene_data['sun_azimuth'],
                'scene_directory': str(scene_dir),
                'stac_url': item.get_self_href(),
                'spatial_resolution': self.get_band_resolutions(),
                'spectral_bands': len(available_bands),
                # NEW: Spatial fit metadata
                'overlap_percentage': scene_data.get('overlap_percentage', 0),
                'center_distance_km': scene_data.get('center_distance_km', 0),
                'spatial_fit_score': scene_data.get('spatial_fit_score', 0),
                'archaeological_suitability': self.assess_archaeological_potential(scene_data, available_bands)
            }
            
            # Create SceneData object
            scene_obj = SceneData(
                zone_id=zone.name.lower().replace(' ', '_'),
                provider='sentinel-2',
                scene_id=scene_id,
                file_paths=file_paths,
                available_bands=available_bands,
                metadata=metadata
            )
            
            # Save scene metadata
            metadata_file = scene_dir / 'scene_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"✓ Scene processed: {len(available_bands)} bands, quality: {scene_data['quality_score']:.1f}, overlap: {scene_data.get('overlap_percentage', 0):.1f}%")
            
            return scene_obj
            
        except Exception as e:
            logger.error(f"Error processing scene {scene_id}: {e}")
            return None

    def download_band(self, asset: pystac.Asset, scene_dir: Path, band_name: str) -> Optional[Path]:
        """Download a single Sentinel-2 band from AWS S3"""
        try:
            # Get the HTTPS URL for the band
            band_url = asset.href
            
            # Create local filename
            band_file = scene_dir / f"{band_name}.tif"
            
            # Skip if already downloaded
            if band_file.exists():
                logger.debug(f"Band {band_name} already exists, skipping download")
                return band_file
            
            # Download with streaming
            logger.debug(f"Downloading {band_name} from {band_url}")
            
            response = self.session.get(band_url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Write to file
            with open(band_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Verify the file is a valid GeoTIFF
            try:
                with rasterio.open(band_file) as src:
                    if src.count == 0:
                        logger.warning(f"Downloaded file {band_file} appears to be empty")
                        band_file.unlink()
                        return None
            except Exception as e:
                logger.warning(f"Downloaded file {band_file} is not a valid GeoTIFF: {e}")
                if band_file.exists():
                    band_file.unlink()
                return None
            
            return band_file
            
        except Exception as e:
            logger.error(f"Error downloading band {band_name}: {e}")
            return None
    
    def get_band_resolutions(self) -> Dict[str, int]:
        """Return spatial resolutions for Sentinel-2 bands"""
        return {
            'B01': 60, 'B02': 10, 'B03': 10, 'B04': 10,
            'B05': 20, 'B06': 20, 'B07': 20, 'B08': 10,
            'B8A': 20, 'B09': 60, 'B10': 60, 'B11': 20, 'B12': 20
        }
    
    def assess_archaeological_potential(self, scene_data: Dict, available_bands: List[str]) -> Dict[str, Any]:
        """Assess archaeological analysis potential with spatial fit considerations"""
        assessment = {
            'overall_score': 0,
            'vegetation_analysis': False,
            'soil_analysis': False,
            'red_edge_available': False,
            'swir_available': False,
            'spatial_suitability': False,
            'limitations': []
        }
        
        # Check for vegetation analysis capability (red-edge bands)
        red_edge_bands = [b for b in ['B05', 'B06', 'B07'] if b in available_bands]
        if len(red_edge_bands) >= 2:
            assessment['red_edge_available'] = True
            assessment['vegetation_analysis'] = True
            assessment['overall_score'] += 30
        else:
            assessment['limitations'].append('Limited red-edge bands for vegetation stress detection')
        
        # Check for soil analysis capability (SWIR bands)
        swir_bands = [b for b in ['B11', 'B12'] if b in available_bands]
        if len(swir_bands) >= 1:
            assessment['swir_available'] = True
            assessment['soil_analysis'] = True
            assessment['overall_score'] += 25
        else:
            assessment['limitations'].append('Missing SWIR bands for soil composition analysis')
        
        # Check for basic multispectral analysis
        core_bands = [b for b in ['B02', 'B03', 'B04', 'B08'] if b in available_bands]
        if len(core_bands) >= 4:
            assessment['overall_score'] += 20
        else:
            assessment['limitations'].append('Missing core visible/NIR bands')
        
        # Quality bonuses
        if scene_data['cloud_cover'] < 10:
            assessment['overall_score'] += 15
        
        if scene_data['data_coverage'] > 90:
            assessment['overall_score'] += 10
        
        # NEW: Spatial fit evaluation
        overlap_pct = scene_data.get('overlap_percentage', 0)
        center_dist = scene_data.get('center_distance_km', float('inf'))
        
        if overlap_pct >= 85 and center_dist <= 25:
            assessment['spatial_suitability'] = True
            assessment['overall_score'] += 15
        elif overlap_pct >= 70 and center_dist <= 50:
            assessment['overall_score'] += 10
        else:
            assessment['limitations'].append(f'Suboptimal spatial fit: {overlap_pct:.1f}% overlap, {center_dist:.1f}km from center')
            
        return assessment

    def create_composite_for_analysis(self, scene_data: SceneData, aoi_polygon: Optional[Polygon] = None) -> Optional[xr.Dataset]:
        """
        Create analysis-ready composite from Sentinel-2 bands with optional AOI clipping
        """
        try:
            # Load available bands
            band_arrays = {}
            crs = None
            transform = None
            
            for band_code, file_path in scene_data.file_paths.items():
                try:
                    with rioxarray.open_rasterio(file_path) as da:
                        # Clip to AOI if provided
                        if aoi_polygon:
                            da = da.rio.clip([aoi_polygon], crs=da.rio.crs)
                        
                        band_arrays[band_code] = da.squeeze()
                        
                        # Store CRS and transform from first band
                        if crs is None:
                            crs = da.rio.crs
                            transform = da.rio.transform()
                            
                except Exception as e:
                    logger.warning(f"Error loading band {band_code}: {e}")
                    continue
            
            if not band_arrays:
                logger.error("No bands could be loaded for composite creation")
                return None
            
            # Create xarray Dataset
            composite = xr.Dataset(band_arrays)
            
            # Add metadata
            composite.attrs.update({
                'provider': 'sentinel-2-improved',
                'scene_id': scene_data.scene_id,
                'zone_id': scene_data.zone_id,
                'crs': str(crs),
                'spatial_resolution': scene_data.metadata.get('spatial_resolution', {}),
                'overlap_percentage': scene_data.metadata.get('overlap_percentage', 0),
                'center_distance_km': scene_data.metadata.get('center_distance_km', 0),
                'quality_score': scene_data.metadata.get('quality_score', 0),
                'archaeological_suitability': scene_data.metadata.get('archaeological_suitability', {})
            })
            
            logger.info(f"✓ Created composite for {scene_data.scene_id} with {len(band_arrays)} bands")
            
            return composite
            
        except Exception as e:
            logger.error(f"Error creating composite for {scene_data.scene_id}: {e}")
            return None

# Integration functions for your existing pipeline
def create_sentinel2_provider() -> Sentinel2Provider:
    """Create and configure IMPROVED Sentinel-2 provider"""
    return Sentinel2Provider()

def test_sentinel2_targeting(zone_id: str = 'negro_madeira') -> bool:
    """Test the IMPROVED Sentinel-2 targeting for a specific zone"""
    try:
        if zone_id not in TARGET_ZONES:
            logger.error(f"Zone {zone_id} not found in TARGET_ZONES")
            return False
            
        provider = Sentinel2Provider()
        scenes = provider.search_scenes_improved(TARGET_ZONES[zone_id], max_results=3)
        
        if scenes:
            logger.info(f"✓ IMPROVED Sentinel-2 targeting test successful for {zone_id}")
            for i, scene in enumerate(scenes[:3]):
                logger.info(f"  Scene {i+1}: {scene['id']}")
                logger.info(f"    Overlap: {scene['overlap_percentage']:.1f}%")
                logger.info(f"    Distance: {scene['center_distance_km']:.1f}km")
                logger.info(f"    Quality: {scene['quality_score']:.1f}")
                logger.info(f"    Cloud: {scene['cloud_cover']:.1f}%")
            return True
        else:
            logger.warning(f"No spatially-suitable Sentinel-2 scenes found for {zone_id}")
            return False
            
    except Exception as e:
        logger.error(f"IMPROVED Sentinel-2 targeting test failed: {e}")
        return False

# For command-line testing
if __name__ == "__main__":
    print("🛰️ Testing IMPROVED Sentinel-2 Provider with Precise Targeting...")
    
    # Test the improved targeting
    success = test_sentinel2_targeting('negro_madeira')
    
    if success:
        print("✅ IMPROVED Sentinel-2 provider test successful!")
        print("\nKey improvements:")
        print("- Spatial overlap filtering (minimum 70% overlap with AOI)")
        print("- Center distance filtering (maximum 50km from AOI center)")
        print("- Enhanced quality scoring with spatial fit weighting")
        print("- Expanded search area to find better candidates")
        print("- Precise location targeting eliminates distant/offset scenes")
    else:
        print("❌ IMPROVED Sentinel-2 provider test failed")
        print("Check internet connection and API access")
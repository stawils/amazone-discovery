"""
Sentinel-2 AWS Open Data Provider for Archaeological Discovery
Optimized for Amazon basin archaeological analysis using 13-band multispectral data
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
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Core dependencies for Sentinel-2 STAC
from pystac_client import Client
import pystac
import geopandas as gpd
from shapely.geometry import box, Point
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
    ARCHAEOLOGICAL_BANDS: Dict[str, str] = {
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
    }
    
    # Priority bands for archaeological analysis (highest resolution + key bands)
    PRIORITY_BANDS: List[str] = ['B02', 'B03', 'B04', 'B05', 'B07', 'B08', 'B11', 'B12']
    
    # Quality filters
    MAX_CLOUD_COVER: float = 20.0
    MIN_DATA_COVERAGE: float = 80.0
    
    # Preferred dates (dry season for Amazon)
    PREFERRED_MONTHS: List[int] = [6, 7, 8, 9]  # June-September
    
    # Temporal range
    DEFAULT_START_DATE: str = "2023-01-01"
    DEFAULT_END_DATE: str = "2024-12-31"

class Sentinel2Error(Exception):
    """Custom exception for Sentinel-2 related errors"""
    pass

class Sentinel2Provider(BaseProvider):
    """
    Advanced Sentinel-2 provider for archaeological discovery
    
    Features:
    - 13-band multispectral analysis optimized for archaeology
    - Red-edge bands for vegetation stress detection (crop marks)
    - SWIR bands for soil composition analysis (terra preta)
    - 5-day revisit cycle for temporal analysis
    - Cloud-optimized GeoTIFF access via AWS
    - STAC API integration for efficient search
    """
    
    def __init__(self, config: Optional[Sentinel2Config] = None):
        self.config = config or Sentinel2Config()
        self.stac_client = Client.open(self.config.STAC_API_URL)
        self.session = requests.Session()
        
        logger.info("🛰️ Sentinel-2 AWS Provider initialized")
        logger.info(f"STAC API: {self.config.STAC_API_URL}")
        logger.info(f"Collection: {self.config.L2A_COLLECTION}")
    
    def download_data(self, zones: List[str], max_scenes: int = 3) -> List[SceneData]:
        """
        Download Sentinel-2 data for archaeological analysis
        
        Args:
            zones: List of zone IDs from TARGET_ZONES
            max_scenes: Maximum scenes per zone
            
        Returns:
            List of SceneData objects with downloaded Sentinel-2 data
        """
        
        all_scene_data = []
        
        for zone_id in zones:
            if zone_id not in TARGET_ZONES:
                logger.warning(f"Unknown zone: {zone_id}")
                continue
                
            zone = TARGET_ZONES[zone_id]
            logger.info(f"\n🎯 Processing {zone.name} with Sentinel-2")
            
            try:
                # Search for optimal scenes
                scenes = self.search_scenes(zone, max_scenes * 2)  # Get more to filter best
                
                if not scenes:
                    logger.warning(f"No suitable Sentinel-2 scenes found for {zone.name}")
                    continue
                
                # Process best scenes
                zone_scenes = []
                for i, scene in enumerate(scenes[:max_scenes]):
                    logger.info(f"Processing scene {i+1}/{max_scenes}: {scene['id']}")
                    
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
        
        logger.info(f"🎯 Sentinel-2 download complete: {len(all_scene_data)} scenes total")
        return all_scene_data
    
    def search_scenes(self, zone, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search for optimal Sentinel-2 scenes for archaeological analysis
        
        Args:
            zone: TargetZone object
            max_results: Maximum number of results
            
        Returns:
            List of scene metadata dictionaries sorted by quality score
        """
        
        logger.info(f"🔍 Searching Sentinel-2 scenes for {zone.name}")
        
        try:
            # Create search parameters
            bbox = zone.bbox  # (south, west, north, east)
            
            # Search with STAC API
            search = self.stac_client.search(
                collections=[self.config.L2A_COLLECTION],
                bbox=bbox,
                datetime=f"{self.config.DEFAULT_START_DATE}/{self.config.DEFAULT_END_DATE}",
                limit=max_results * 3,  # Get more to filter
                query={
                    "eo:cloud_cover": {"lt": self.config.MAX_CLOUD_COVER}
                }
            )
            
            items = list(search.items())
            logger.info(f"Found {len(items)} candidate scenes")
            
            if not items:
                return []
            
            # Score and filter scenes for archaeological suitability
            scored_scenes = []
            
            for item in items:
                try:
                    score_data = self.score_archaeological_suitability(item, zone)
                    if score_data['quality_score'] > 40:  # Minimum threshold
                        scored_scenes.append(score_data)
                except Exception as e:
                    logger.warning(f"Error scoring scene {item.id}: {e}")
                    continue
            
            # Sort by quality score
            scored_scenes.sort(key=lambda x: x['quality_score'], reverse=True)
            
            logger.info(f"✓ Selected {len(scored_scenes)} high-quality scenes")
            return scored_scenes[:max_results]
            
        except Exception as e:
            raise Sentinel2Error(f"Scene search failed for {zone.name}: {e}")
    
    def score_archaeological_suitability(self, item: pystac.Item, zone) -> Dict[str, Any]:
        """
        Score Sentinel-2 scene for archaeological analysis suitability
        
        Args:
            item: STAC item
            zone: TargetZone object
            
        Returns:
            Dictionary with scene metadata and quality score
        """
        
        properties = item.properties
        
        # Extract key metadata
        cloud_cover = properties.get('eo:cloud_cover', 100)
        data_coverage = properties.get('s2:data_coverage', 0)
        acquisition_date = datetime.fromisoformat(properties['datetime'].replace('Z', '+00:00'))
        
        # Calculate quality score
        quality_score = 0
        
        # Cloud cover score (0-30 points)
        if cloud_cover <= 5:
            quality_score += 30
        elif cloud_cover <= 10:
            quality_score += 25
        elif cloud_cover <= 15:
            quality_score += 15
        elif cloud_cover <= 20:
            quality_score += 5
        
        # Data coverage score (0-20 points)
        if data_coverage >= 95:
            quality_score += 20
        elif data_coverage >= 90:
            quality_score += 15
        elif data_coverage >= 80:
            quality_score += 10
        
        # Seasonal preference (0-20 points)
        if acquisition_date.month in self.config.PREFERRED_MONTHS:
            quality_score += 20
        elif acquisition_date.month in [5, 10]:  # Shoulder months
            quality_score += 10
        
        # Recent data bonus (0-15 points)
        days_old = (datetime.now(acquisition_date.tzinfo) - acquisition_date).days
        if days_old < 365:
            quality_score += 15
        elif days_old < 730:
            quality_score += 10
        elif days_old < 1095:
            quality_score += 5
        
        # Solar angle bonus (0-15 points)
        sun_azimuth = properties.get('s2:mean_solar_azimuth', 0)
        sun_zenith = properties.get('s2:mean_solar_zenith', 90)
        if sun_zenith < 30:  # High sun angle
            quality_score += 15
        elif sun_zenith < 40:
            quality_score += 10
        elif sun_zenith < 50:
            quality_score += 5
        
        return {
            'id': item.id,
            'stac_item': item,
            'acquisition_date': acquisition_date.strftime('%Y-%m-%d'),
            'cloud_cover': cloud_cover,
            'data_coverage': data_coverage,
            'sun_zenith': sun_zenith,
            'sun_azimuth': sun_azimuth,
            'quality_score': quality_score,
            'mgrs_tile': properties.get('s2:mgrs_tile', 'unknown'),
            'processing_level': properties.get('s2:processing_level', 'L2A'),
            'constellation': properties.get('constellation', 'sentinel-2')
        }
    
    def process_scene(self, scene_data: Dict[str, Any], zone) -> Optional[SceneData]:
        """
        Process and download a Sentinel-2 scene for archaeological analysis
        
        Args:
            scene_data: Scene metadata from search
            zone: TargetZone object
            
        Returns:
            SceneData object or None if processing failed
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
            
            for band_name in self.config.PRIORITY_BANDS:
                try:
                    if band_name in item.assets:
                        asset = item.assets[band_name]
                        band_file = self.download_band(asset, scene_dir, band_name)
                        
                        if band_file and band_file.exists():
                            file_paths[band_name] = band_file
                            available_bands.append(band_name)
                            logger.debug(f"  ✓ Downloaded {band_name}")
                        else:
                            logger.warning(f"  ❌ Failed to download {band_name}")
                    else:
                        logger.warning(f"  ⚠️ Band {band_name} not available in scene")
                        
                except Exception as e:
                    logger.warning(f"Error downloading band {band_name}: {e}")
                    continue
            
            if len(available_bands) < 4:  # Need minimum bands
                logger.warning(f"Insufficient bands downloaded for {scene_id}: {len(available_bands)}")
                return None
            
            # Create enhanced metadata for archaeological analysis
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
            
            logger.info(f"✓ Scene processed: {len(available_bands)} bands, quality score: {scene_data['quality_score']}")
            
            return scene_obj
            
        except Exception as e:
            logger.error(f"Error processing scene {scene_id}: {e}")
            return None
    
    def download_band(self, asset: pystac.Asset, scene_dir: Path, band_name: str) -> Optional[Path]:
        """
        Download a single Sentinel-2 band from AWS S3
        
        Args:
            asset: STAC asset object
            scene_dir: Local directory for the scene
            band_name: Band identifier (e.g., 'B04')
            
        Returns:
            Path to downloaded file or None if failed
        """
        
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
        """
        Assess the archaeological analysis potential of a Sentinel-2 scene
        
        Args:
            scene_data: Scene metadata
            available_bands: List of successfully downloaded bands
            
        Returns:
            Dictionary with archaeological assessment
        """
        
        assessment = {
            'overall_score': 0,
            'vegetation_analysis': False,
            'soil_analysis': False,
            'red_edge_available': False,
            'swir_available': False,
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
        
        # Archaeological indices that can be calculated
        indices_possible = []
        if all(b in available_bands for b in ['B04', 'B08']):
            indices_possible.append('NDVI')
        if all(b in available_bands for b in ['B04', 'B05']):
            indices_possible.append('NDRE (Red Edge)')
        if all(b in available_bands for b in ['B08', 'B11']):
            indices_possible.append('Terra Preta Index')
        if all(b in available_bands for b in ['B11', 'B12']):
            indices_possible.append('Clay Mineral Index')
        
        assessment['indices_available'] = indices_possible
        assessment['spectral_richness'] = len(available_bands)
        
        return assessment
    
    def create_composite_for_analysis(self, scene_data: SceneData, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Create a multi-band composite optimized for archaeological analysis
        
        Args:
            scene_data: SceneData object with downloaded bands
            output_path: Optional output path for composite
            
        Returns:
            Path to created composite or None if failed
        """
        
        try:
            if not output_path:
                scene_dir = Path(scene_data.metadata['scene_directory'])
                output_path = scene_dir / 'archaeological_composite.tif'
            
            # Priority band order for archaeological analysis
            band_order = ['B02', 'B03', 'B04', 'B05', 'B07', 'B08', 'B11', 'B12']
            
            # Collect available bands in priority order
            bands_to_stack = []
            band_names = []
            
            for band in band_order:
                if band in scene_data.file_paths:
                    bands_to_stack.append(scene_data.file_paths[band])
                    band_names.append(band)
            
            if len(bands_to_stack) < 4:
                logger.warning("Insufficient bands for composite creation")
                return None
            
            # Stack bands using rioxarray
            band_arrays = []
            profile = None
            
            for i, band_path in enumerate(bands_to_stack):
                with rasterio.open(band_path) as src:
                    band_data = src.read(1)
                    band_arrays.append(band_data)
                    
                    if profile is None:
                        profile = src.profile
                        profile.update({
                            'count': len(bands_to_stack),
                            'compress': 'lzw',
                            'tiled': True,
                            'blockxsize': 512,
                            'blockysize': 512
                        })
            
            # Write composite
            with rasterio.open(output_path, 'w', **profile) as dst:
                for i, band_array in enumerate(band_arrays):
                    dst.write(band_array, i + 1)
                
                # Add band descriptions
                dst.descriptions = band_names
            
            logger.info(f"✓ Archaeological composite created: {output_path}")
            logger.info(f"  Bands included: {', '.join(band_names)}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating archaeological composite: {e}")
            return None

# Integration function for your existing pipeline
def create_sentinel2_provider() -> Sentinel2Provider:
    """Create and configure Sentinel-2 provider for archaeological pipeline"""
    return Sentinel2Provider()

# Convenience function for quick testing
def test_sentinel2_access(zone_id: str = 'negro_madeira') -> bool:
    """
    Test Sentinel-2 access for a specific zone
    
    Args:
        zone_id: Zone ID to test
        
    Returns:
        True if successful, False otherwise
    """
    
    try:
        provider = Sentinel2Provider()
        scenes = provider.search_scenes(TARGET_ZONES[zone_id], max_results=1)
        
        if scenes:
            logger.info(f"✓ Sentinel-2 access test successful for {zone_id}")
            logger.info(f"  Found scene: {scenes[0]['id']}")
            logger.info(f"  Quality score: {scenes[0]['quality_score']}")
            logger.info(f"  Cloud cover: {scenes[0]['cloud_cover']}%")
            return True
        else:
            logger.warning(f"No Sentinel-2 scenes found for {zone_id}")
            return False
            
    except Exception as e:
        logger.error(f"Sentinel-2 access test failed: {e}")
        return False

if __name__ == "__main__":
    # Test the Sentinel-2 provider
    print("🛰️ Testing Sentinel-2 AWS Provider...")
    
    # Test basic functionality
    success = test_sentinel2_access('negro_madeira')
    
    if success:
        print("✅ Sentinel-2 provider test successful!")
        print("\nKey features available:")
        print("- 13-band multispectral analysis")
        print("- Red-edge bands for vegetation stress (crop marks)")
        print("- SWIR bands for soil analysis (terra preta)")
        print("- 5-day revisit cycle")
        print("- Cloud-optimized GeoTIFF access via AWS")
        print("- STAC API integration")
    else:
        print("❌ Sentinel-2 provider test failed")
        print("Check internet connection and API access")

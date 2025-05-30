"""
Google Earth Engine Provider for Archaeological Discovery
Alternative satellite data source with cloud processing capabilities
"""

import ee
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
import time

from src.core.config import TARGET_ZONES, APIConfig, DetectionConfig, SATELLITE_DIR
from .data_objects import SceneData, BaseProvider

logger = logging.getLogger(__name__)

class GEEError(Exception):
    """Custom exception for Google Earth Engine errors"""
    pass

class GoogleEarthEngineProvider:
    """Google Earth Engine provider for archaeological satellite data"""
    
    def __init__(self, service_account_path: str = None):
        """
        Initialize Google Earth Engine
        
        Args:
            service_account_path: Path to service account JSON file (optional)
        """
        self.authenticated = False
        self.service_account_path = service_account_path or os.getenv('GEE_SERVICE_ACCOUNT_PATH')
        self.project = os.getenv('GEE_PROJECT_ID')
        
        try:
            self._authenticate()
            self.authenticated = True
            logger.info("✅ Google Earth Engine authenticated successfully")
        except Exception as e:
            raise GEEError(f"Failed to authenticate with Google Earth Engine: {e}")
    
    def _authenticate(self):
        """Authenticate with Google Earth Engine"""
        
        if self.service_account_path and Path(self.service_account_path).exists():
            # Service account authentication (for production)
            try:
                credentials = ee.ServiceAccountCredentials(
                    email=None,  # Will be read from the JSON file
                    key_file=self.service_account_path
                )
                ee.Initialize(credentials, project=self.project)
                logger.info("✅ Authenticated with service account")
            except Exception as e:
                logger.warning(f"Service account auth failed: {e}")
                # Fall back to user authentication
                self._user_authenticate()
        else:
            # User authentication (for development/testing)
            self._user_authenticate()
    
    def _user_authenticate(self):
        """User authentication for development"""
        try:
            ee.Authenticate()  # This will open browser for first-time auth
            ee.Initialize(project=self.project)
            logger.info("✅ Authenticated with user account")
        except Exception as e:
            # Try to initialize without explicit auth (if already authenticated)
            try:
                ee.Initialize(project=self.project)
                logger.info("✅ Using existing authentication")
            except Exception as init_error:
                raise GEEError(f"Authentication failed: {e}, Initialization failed: {init_error}")
    
    def search_landsat_scenes(self, zone_id: str, 
                            start_date: str = "2023-01-01",
                            end_date: str = "2024-12-31",
                            max_cloud_cover: float = 20.0) -> Dict[str, Any]:
        """
        Search for Landsat scenes using Google Earth Engine
        
        Args:
            zone_id: Target zone identifier
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format 
            max_cloud_cover: Maximum cloud cover percentage
            
        Returns:
            Dictionary with scene information and image collection
        """
        
        if zone_id not in TARGET_ZONES:
            raise GEEError(f"Unknown zone: {zone_id}")
        
        zone = TARGET_ZONES[zone_id]
        logger.info(f"🔍 Searching GEE for {zone.name}")
        
        try:
            # Define area of interest using config (center and radius or bbox)
            if hasattr(zone, 'bbox') and zone.bbox:
                # Use bounding box if available
                bbox = zone.bbox  # (south, west, north, east)
                aoi = ee.Geometry.Rectangle([bbox[1], bbox[0], bbox[3], bbox[2]])
            else:
                # Use center and search_radius_km to create a buffer
                lon, lat = zone.center
                radius_m = getattr(zone, 'search_radius_km', 20) * 1000
                aoi = ee.Geometry.Point([lon, lat]).buffer(radius_m).bounds()
            
            # Get Landsat 8/9 Collection 2 Surface Reflectance
            landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            landsat9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            
            # Merge collections
            landsat = landsat8.merge(landsat9)
            
            # Filter by location, date, and cloud cover
            filtered = (landsat
                       .filterBounds(aoi)
                       .filterDate(start_date, end_date)
                       .filter(ee.Filter.lt('CLOUD_COVER', max_cloud_cover))
                       .sort('CLOUD_COVER')  # Sort by cloud cover (best first)
                       )
            
            # Get collection info
            collection_size = filtered.size().getInfo()
            logger.info(f"✅ Found {collection_size} Landsat scenes for {zone.name}")
            
            if collection_size == 0:
                logger.warning(f"No scenes found for {zone.name} with <{max_cloud_cover}% clouds")
                return {
                    'zone_id': zone_id,
                    'zone_name': zone.name,
                    'collection_size': 0,
                    'scenes': [],
                    'image_collection': None
                }
            
            # Get scene metadata (limit to first 10 for performance)
            scene_list = filtered.limit(10)
            scene_info = []
            
            # Extract scene metadata
            def extract_scene_info(image):
                props = image.toDictionary().getInfo()
                return {
                    'scene_id': props.get('LANDSAT_PRODUCT_ID', 'unknown'),
                    'date': props.get('DATE_ACQUIRED', 'unknown'),
                    'cloud_cover': props.get('CLOUD_COVER', 100),
                    'sun_elevation': props.get('SUN_ELEVATION', 0),
                    'spacecraft': props.get('SPACECRAFT_ID', 'unknown'),
                    'processing_level': 'L2SR',
                    'bounds': aoi.bounds().getInfo()
                }
            
            # Get info for top scenes
            scenes = scene_list.toList(10)
            for i in range(min(5, collection_size)):  # Top 5 scenes
                try:
                    image = ee.Image(scenes.get(i))
                    scene_info.append(extract_scene_info(image))
                except Exception as e:
                    logger.warning(f"Error getting scene {i} info: {e}")
            
            result = {
                'zone_id': zone_id,
                'zone_name': zone.name,
                'aoi': aoi,
                'collection_size': collection_size,
                'scenes': scene_info,
                'image_collection': filtered,
                'best_image': filtered.first() if collection_size > 0 else None
            }
            
            logger.info(f"✅ GEE search complete for {zone.name}")
            return result
            
        except Exception as e:
            raise GEEError(f"Scene search failed for {zone.name}: {e}")
    
    def process_archaeological_analysis(self, zone_id: str, 
                                      max_scenes: int = 3) -> Dict[str, Any]:
        """
        Run archaeological analysis using Google Earth Engine cloud processing
        
        Args:
            zone_id: Target zone identifier
            max_scenes: Maximum number of scenes to process
            
        Returns:
            Dictionary with analysis results
        """
        
        logger.info(f"🧠 Running GEE archaeological analysis for {zone_id}")
        
        try:
            # Search for scenes
            search_result = self.search_landsat_scenes(zone_id)
            
            if search_result['collection_size'] == 0:
                return {
                    'zone_id': zone_id,
                    'success': False,
                    'error': 'No suitable scenes found',
                    'results': {}
                }
            
            zone = TARGET_ZONES[zone_id]
            collection = search_result['image_collection']
            aoi = search_result['aoi']
            
            # Take the best scenes (lowest cloud cover)
            best_images = collection.limit(max_scenes)
            
            # Ensure best_images is an ImageCollection
            if isinstance(best_images, ee.List):
                best_images = ee.ImageCollection.fromImages(best_images)
            
            # Calculate median composite (reduces clouds and noise)
            composite = best_images.median().clip(aoi)
            
            # Extract spectral bands (Landsat Collection 2 SR band names)
            bands = {
                'blue': composite.select('SR_B2'),    # Blue
                'green': composite.select('SR_B3'),   # Green  
                'red': composite.select('SR_B4'),     # Red
                'nir': composite.select('SR_B5'),     # NIR
                'swir1': composite.select('SR_B6'),   # SWIR1
                'swir2': composite.select('SR_B7')    # SWIR2
            }
            
            # Apply scaling factors (Landsat Collection 2 SR)
            def apply_scale_factors(image):
                optical_bands = image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
                scaled = optical_bands.multiply(0.0000275).add(-0.2)
                return scaled.copyProperties(image, ['system:time_start'])
            
            scaled_composite = apply_scale_factors(composite)
            
            # Calculate spectral indices using GEE cloud processing
            indices = self._calculate_spectral_indices_gee(scaled_composite)
            
            # Detect terra preta signatures
            terra_preta_results = self._detect_terra_preta_gee(indices, aoi)
            
            # Export results for local processing (if needed)
            export_data = self._prepare_export_data(scaled_composite, indices, aoi, zone_name=zone.name)
            
            analysis_results = {
                'zone_id': zone_id,
                'zone_name': zone.name,
                'success': True,
                'scenes_used': min(max_scenes, search_result['collection_size']),
                'processing_date': datetime.now().isoformat(),
                'terra_preta': terra_preta_results,
                'spectral_indices': {
                    'available': list(indices.keys()),
                    'computed_on_cloud': True
                },
                'export_data': export_data,
                'gee_analysis': True
            }
            
            logger.info(f"✅ GEE analysis complete for {zone.name}")
            return analysis_results
            
        except Exception as e:
            logger.error(f"GEE analysis failed for {zone_id}: {e}")
            return {
                'zone_id': zone_id,
                'success': False,
                'error': str(e),
                'results': {}
            }
    
    def _calculate_spectral_indices_gee(self, image: ee.Image) -> Dict[str, ee.Image]:
        """Calculate spectral indices using Google Earth Engine"""
        # Ensure input is an ee.Image
        if not isinstance(image, ee.Image):
            image = ee.Image(image)
        # Extract bands
        blue = image.select('SR_B2')
        green = image.select('SR_B3')
        red = image.select('SR_B4')
        nir = image.select('SR_B5')
        swir1 = image.select('SR_B6')
        swir2 = image.select('SR_B7')
        
        # Calculate indices using GEE operations
        indices = {
            'ndvi': nir.subtract(red).divide(nir.add(red)).rename('NDVI'),
            'ndwi': green.subtract(nir).divide(green.add(nir)).rename('NDWI'),
            'terra_preta': nir.subtract(swir1).divide(nir.add(swir1)).rename('TERRA_PRETA'),
            'clay_minerals': swir1.divide(swir2).rename('CLAY_MINERALS'),
            'brightness': blue.pow(2).add(green.pow(2)).add(red.pow(2)).divide(3).sqrt().rename('BRIGHTNESS')
        }
        
        return indices
    
    def _detect_terra_preta_gee(self, indices: Dict[str, ee.Image], aoi: ee.Geometry) -> Dict[str, Any]:
        """Detect terra preta signatures using Google Earth Engine"""
        # Terra preta detection criteria
        ndvi = indices['ndvi']
        tp_index = indices['terra_preta']
        # Create mask for terra preta signatures and name the band
        tp_mask = (tp_index.gt(DetectionConfig.TERRA_PRETA_INDEX_MIN)
                  .And(ndvi.gt(DetectionConfig.TERRA_PRETA_NDVI_MIN))
                  .And(ndvi.lt(0.8)))
        tp_mask = tp_mask.rename('terra_preta')  # Ensure band is named
        # Calculate statistics
        tp_area = tp_mask.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=30,
            maxPixels=1e9
        )
        total_area = ee.Image.pixelArea().reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=30,
            maxPixels=1e9
        )
        # Get results
        try:
            tp_area_dict = tp_area.getInfo() or {}
            total_area_dict = total_area.getInfo() or {}
            tp_area_value = tp_area_dict.get('terra_preta', 0)
            total_area_value = total_area_dict.get('area', 1)
            coverage_percent = (tp_area_value / total_area_value) * 100 if total_area_value else 0
            results = {
                'detection_method': 'google_earth_engine_cloud',
                'total_tp_area_m2': float(tp_area_value),
                'coverage_percent': float(coverage_percent),
                'detection_criteria': {
                    'tp_index_min': DetectionConfig.TERRA_PRETA_INDEX_MIN,
                    'ndvi_min': DetectionConfig.TERRA_PRETA_NDVI_MIN,
                    'ndvi_max': 0.8
                },
                'cloud_processed': True
            }
            logger.info(f"Terra preta coverage: {coverage_percent:.2f}% of AOI")
            return results
        except Exception as e:
            logger.warning(f"Error getting terra preta statistics: {e}")
            return {
                'detection_method': 'google_earth_engine_cloud',
                'error': str(e),
                'cloud_processed': True
            }
    
    def _prepare_export_data(self, composite: ee.Image, indices: Dict[str, ee.Image], 
                           aoi: ee.Geometry, zone_name: str = None) -> Dict[str, str]:
        """Prepare data export URLs for local processing if needed"""
        try:
            # Ensure all bands are ee.Image
            export_image = composite
            if not isinstance(export_image, ee.Image):
                export_image = ee.Image(export_image)
            # Select only the 6 reflectance bands for export
            export_image = export_image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
            # Try direct download (scale=30)
            try:
                url = export_image.getDownloadURL({
                    'region': aoi,
                    'scale': 30,
                    'format': 'GEO_TIFF'
                })
                if zone_name:
                    local_dir = Path('data/satellite') / zone_name.lower().replace(' ', '_')
                    local_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Download the GeoTIFF for {zone_name} from the following URL:")
                    logger.info(url)
                    logger.info(f"Save it to: {local_dir}/exported_data.tif")
                return {
                    'download_url': url,
                    'format': 'GeoTIFF',
                    'scale': 30,
                    'bands': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
                    'note': 'URL valid for limited time - download promptly'
                }
            except Exception as e:
                # If error is due to size, fall back to Google Drive export
                if 'must be less than or equal to' in str(e):
                    logger.warning(f"Direct download too large, exporting to Google Drive instead...")
                    folder = 'GEE_Exports'
                    file_prefix = f"{zone_name}_exported_data"
                    task = ee.batch.Export.image.toDrive(
                        image=export_image,
                        description=f"{zone_name}_export",
                        folder=folder,
                        fileNamePrefix=file_prefix,
                        region=aoi,
                        scale=30,
                        fileFormat='GeoTIFF'
                    )
                    task.start()
                    logger.info(f"Started export to Google Drive for {zone_name}. Folder: {folder}, File: {file_prefix}.tif")
                    logger.info("Monitor export progress in the Earth Engine Code Editor Tasks tab or with ee.batch.Task.list().")
                    return {
                        'drive_export': True,
                        'drive_folder': folder,
                        'drive_file': f"{file_prefix}.tif",
                        'note': f"Export started to Google Drive folder '{folder}'. Download from Drive after completion. Monitor in GEE Code Editor Tasks tab."
                    }
                else:
                    logger.warning(f"Error preparing export data: {e}")
                    return {
                        'error': str(e),
                        'note': 'Export data preparation failed'
                    }
        except Exception as e:
            logger.warning(f"Error preparing export data: {e}")
            return {
                'error': str(e),
                'note': 'Export data preparation failed'
            }
    
    def batch_analyze_zones(self, zones: List[str] = None, 
                           max_scenes_per_zone: int = 3) -> Dict[str, Dict[str, Any]]:
        """Analyze multiple zones using Google Earth Engine, returning both old results and SceneData objects"""
        if zones is None:
            # Default to Priority 1 zones
            zones = [k for k, v in TARGET_ZONES.items() if v.priority == 1]
        logger.info(f"🚀 Starting GEE batch analysis for {len(zones)} zones")
        results = {}
        for zone_id in zones:
            if zone_id not in TARGET_ZONES:
                logger.warning(f"Unknown zone: {zone_id}")
                continue
            logger.info(f"\n🎯 Processing {TARGET_ZONES[zone_id].name}")
            try:
                zone_result = self.process_archaeological_analysis(zone_id, max_scenes_per_zone)
                scene_data_list = []
                if zone_result.get('success'):
                    # Try to extract file path or download url
                    export_data = zone_result.get('export_data', {})
                    file_paths = {}
                    available_bands = zone_result.get('spectral_indices', {}).get('available', [])
                    # Prefer local file if available, else use download_url
                    if 'download_url' in export_data:
                        file_paths['exported_data'] = export_data['download_url']
                    elif 'drive_file' in export_data:
                        file_paths['exported_data'] = export_data['drive_file']
                    # Compose metadata
                    metadata = {
                        'zone_name': zone_result.get('zone_name'),
                        'processing_date': zone_result.get('processing_date'),
                        'scenes_used': zone_result.get('scenes_used'),
                        'gee_analysis': True,
                        'terra_preta_coverage': zone_result.get('terra_preta', {}).get('coverage_percent'),
                        'export_note': export_data.get('note'),
                    }
                    # Use processing_date as a unique scene_id if nothing else
                    scene_id = f"gee_{zone_id}_{zone_result.get('processing_date', '')}"
                    scenedata = SceneData(
                        zone_id=zone_id,
                        provider='gee',
                        scene_id=scene_id,
                        file_paths=file_paths,
                        available_bands=available_bands,
                        metadata=metadata,
                        features=zone_result.get('spectral_indices', {})
                    )
                    scene_data_list.append(scenedata)
                results[zone_id] = {
                    'result': zone_result,
                    'scene_data': scene_data_list
                }
                # Brief pause between zones to be respectful to GEE
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error processing {zone_id}: {e}")
                results[zone_id] = {
                    'result': {'success': False, 'error': str(e)},
                    'scene_data': []
                }
        logger.info(f"✅ GEE batch analysis complete: {len(results)} zones processed")
        return results

# Convenience function
def analyze_with_gee(zones: List[str] = None, max_scenes: int = 3) -> Dict[str, Dict]:
    """Quick function to analyze zones using Google Earth Engine"""
    
    try:
        gee_provider = GoogleEarthEngineProvider()
        return gee_provider.batch_analyze_zones(zones, max_scenes)
    except GEEError as e:
        logger.error(f"GEE analysis failed: {e}")
        return {}

class GEEProvider(BaseProvider):
    """
    Google Earth Engine provider implementing the BaseProvider interface.
    """
    def __init__(self, provider=None):
        self.provider = provider or GoogleEarthEngineProvider()

    def download_data(self, zones: list, max_scenes: int = 3) -> list:
        """
        Download data for the given zones using the GEE provider.
        Returns a list of SceneData objects.
        """
        results = self.provider.batch_analyze_zones(zones, max_scenes)
        scene_data = []
        for zone, result in results.items():
            scene_data.extend(result.get('scene_data', []))
        return scene_data

if __name__ == "__main__":
    # Test Google Earth Engine provider
    print("🌍 Testing Google Earth Engine provider...")
    
    try:
        # Initialize provider
        gee_provider = GoogleEarthEngineProvider()
        print("✅ GEE provider initialized")
        
        # Test search for Negro-Madeira zone
        result = gee_provider.search_landsat_scenes('negro_madeira')
        print(f"✅ Found {result['collection_size']} scenes for {result['zone_name']}")
        
        if result['collection_size'] > 0:
            print(f"Best scene: {result['scenes'][0]['scene_id']}")
            print(f"Cloud cover: {result['scenes'][0]['cloud_cover']}%")
        
        # Test archaeological analysis
        print("\n🧠 Testing archaeological analysis...")
        analysis = gee_provider.process_archaeological_analysis('negro_madeira', max_scenes=1)
        
        if analysis['success']:
            print("✅ Archaeological analysis completed")
            print(f"Terra preta coverage: {analysis['terra_preta'].get('coverage_percent', 0):.2f}%")
        else:
            print(f"❌ Analysis failed: {analysis.get('error', 'Unknown error')}")
        
    except GEEError as e:
        print(f"❌ GEE Error: {e}")
        print("Make sure you're authenticated with Google Earth Engine")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

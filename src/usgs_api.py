"""
USGS M2M API Handler for Archaeological Discovery
Automated satellite data acquisition and management
"""
import requests
import json
import os
import time
import tarfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import logging
import socket

from .config import APIConfig, DetectionConfig, SATELLITE_DIR, TargetZone
from .data_objects import SceneData, BaseProvider

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class USGSAPIError(Exception):
    """Custom exception for USGS API errors"""
    pass

class USGSArchaeologyAPI:
    """USGS M2M API handler optimized for archaeological discovery"""
    
    def __init__(self, username=None, password=None, token=None, enabled=True, base_url=None, timeout=None):
        self.enabled = enabled
        if not self.enabled:
            self.usgs_api = None
            return
        self.username = username or APIConfig.USGS_USERNAME
        self.password = password or getattr(APIConfig, "USGS_PASSWORD", None)
        self.token = token or APIConfig.USGS_TOKEN
        self.base_url = base_url or APIConfig.USGS_API_URL or "https://m2m.cr.usgs.gov/api/api/json/stable/"
        self.timeout = timeout or int(os.getenv("USGS_API_TIMEOUT", "60"))
        self.session = requests.Session()
        self.authenticated = False
        self.api_key = None
        
        if not self.username or not self.token:
            raise USGSAPIError("USGS credentials not provided. Set USGS_USERNAME and USGS_TOKEN environment variables.")
        
        self.authenticate()
    
    def _dispatch_request(self, endpoint, params=None, enforce_login=False, timeout=None):
        # Authenticate if needed
        if endpoint != 'login-token' and not getattr(self, 'api_key', None):
            self.authenticate()
        url = self.base_url + endpoint
        headers = {}
        if getattr(self, 'api_key', None):
            headers['X-Auth-Token'] = self.api_key
        payload = json.dumps(params) if params else '{}'
        try:
            response = self.session.post(url, data=payload, headers=headers, timeout=timeout or self.timeout)
            result = response.json()
        except Exception as e:
            raise USGSAPIError(f"Network or JSON error during request to {endpoint}: {str(e)}")
        # Robust error handling
        for key in ['errorCode', 'errorMessage', 'data', 'requestId', 'version']:
            if key not in result:
                raise USGSAPIError(f"Missing '{key}' in API response for {endpoint}")
        if result['errorCode'] is not None:
            raise USGSAPIError(f"USGS API error: {result['errorCode']}: {result['errorMessage']}")
        return result['data']
    
    def authenticate(self) -> bool:
        """Authenticate with USGS M2M API using token and set API key for session headers"""
        params = {
            "username": self.username,
            "token": self.token,
            "authType": "EROS"
            # Optionally add 'userContext' if you want to track contactId/ip
        }
        data = self._dispatch_request('login-token', params)
        self.api_key = data
        self.session.headers.update({'X-Auth-Token': self.api_key})
        self.authenticated = True
        logger.info("✓ Successfully authenticated with USGS M2M API (token)")
        return True
    
    def search_landsat_scenes(self, zone: TargetZone, 
                            start_date: str = "2023-01-01", 
                            end_date: str = "2024-12-31",
                            max_results: int = 50) -> List[Dict]:
        """Search for optimal Landsat scenes covering target zone"""
        
        if not self.authenticated:
            self.authenticate()
        
        # Create spatial filter from bounding box
        bbox = zone.bbox
        spatial_filter = {
            "filterType": "mbr",
            "lowerLeft": {
                "latitude": bbox[0],  # south
                "longitude": bbox[1]  # west
            },
            "upperRight": {
                "latitude": bbox[2],  # north
                "longitude": bbox[3]  # east
            }
        }
        
        # Create temporal filter
        temporal_filter = {
            "startDate": start_date,
            "endDate": end_date
        }
        
        # Search parameters optimized for archaeological analysis
        search_params = {
            "datasetName": "landsat_ot_c2_l2",  # Landsat Collection 2 Level 2
            "spatialFilter": spatial_filter,
            "temporalFilter": temporal_filter,
            "maxResults": max_results,
            "sortOrder": "DESC",  # Most recent first
            "sortField": "acquisitionDate"
        }
        
        try:
            data = self._dispatch_request('scene-search', search_params)
            scenes = data['results']
            logger.info(f"✓ Found {len(scenes)} scenes for {zone.name}")
            
            return self._filter_optimal_scenes(scenes, zone)
            
        except USGSAPIError as e:
            raise USGSAPIError(f"Scene search failed: {e}")
    
    def _filter_optimal_scenes(self, scenes: List[Dict], zone: TargetZone) -> List[Dict]:
        """Filter scenes by quality metrics for archaeological analysis"""
        
        filtered_scenes = []
        
        for scene in scenes:
            try:
                # Extract metadata
                metadata = {item['fieldName']: item['value'] for item in scene.get('metadata', [])}
                
                # Get acquisition date
                acq_date = datetime.fromisoformat(scene['temporalCoverage']['startDate'].replace('Z', '+00:00'))
                
                # Quality filters
                cloud_cover = float(metadata.get('Cloud Cover Land', 100))
                sun_elevation = float(metadata.get('Sun Elevation L0RA', 0))
                
                # Prefer dry season (June-September) for better visibility
                is_dry_season = acq_date.month in DetectionConfig.PREFERRED_MONTHS
                
                # Quality score calculation
                quality_score = 0
                if cloud_cover <= DetectionConfig.MAX_CLOUD_COVER:
                    quality_score += (DetectionConfig.MAX_CLOUD_COVER - cloud_cover) / DetectionConfig.MAX_CLOUD_COVER * 40
                
                if sun_elevation > 30:  # Good solar angle
                    quality_score += 20
                
                if is_dry_season:
                    quality_score += 20
                
                # Recent data bonus
                days_old = (datetime.now(acq_date.tzinfo) - acq_date).days
                if days_old < 365:
                    quality_score += 20
                
                if quality_score >= 40:  # Minimum quality threshold
                    scene_info = {
                        'scene_id': scene['entityId'],
                        'display_id': scene['displayId'],
                        'acquisition_date': acq_date.strftime('%Y-%m-%d'),
                        'cloud_cover': cloud_cover,
                        'sun_elevation': sun_elevation,
                        'is_dry_season': is_dry_season,
                        'quality_score': quality_score,
                        'browse_url': scene.get('browse', [{}])[0].get('browsePath', ''),
                        'metadata': metadata
                    }
                    filtered_scenes.append(scene_info)
                    
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Error processing scene {scene.get('entityId', 'unknown')}: {str(e)}")
                continue
        
        # Sort by quality score
        filtered_scenes.sort(key=lambda x: x['quality_score'], reverse=True)
        
        logger.info(f"✓ Filtered to {len(filtered_scenes)} high-quality scenes")
        return filtered_scenes[:10]  # Top 10 scenes
    
    def get_download_options(self, scene_id: str) -> List[Dict]:
        """Get download options for a scene"""
        
        if not self.authenticated:
            self.authenticate()
        
        download_params = {
            "datasetName": "landsat_ot_c2_l2",
            "entityIds": [scene_id]
        }
        
        try:
            data = self._dispatch_request('download-options', download_params)
            return data
            
        except USGSAPIError as e:
            raise USGSAPIError(f"Download options failed: {e}")
    
    def download_scene(self, scene_id: str, zone_name: str, 
                      download_surface_reflectance: bool = True,
                      bands: Optional[List[str]] = None) -> Optional[List[Path]]:
        """Download only the required bands for a scene using USGS M2M API, with robust fallback logic."""
        # Default to 6 Amazon/Landsat SR bands if not specified
        if bands is None:
            bands = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
        level1_bands = ["B2", "B3", "B4", "B5", "B6", "B7"]
        zone_dir = SATELLITE_DIR / zone_name.lower().replace(' ', '_')
        zone_dir.mkdir(exist_ok=True, parents=True)
        try:
            # Get download options with secondary file groups
            download_params = {
                "datasetName": "landsat_ot_c2_l2",
                "entityIds": [scene_id],
                "includeSecondaryFileGroups": True
            }
            options = self._dispatch_request('download-options', download_params)
            if not options:
                logger.warning(f"No download options available for {scene_id}")
                return None
            downloads = []
            # 1. Try to match required SR bands (flexible match)
            for product in options:
                if product.get("secondaryDownloads"):
                    for secondary in product["secondaryDownloads"]:
                        for band in bands:
                            # Flexible match: ignore extension, substring match
                            if secondary.get("bulkAvailable") and band in secondary.get("displayId", ""):
                                downloads.append({
                                    "entityId": secondary["entityId"],
                                    "productId": secondary["id"]
                                })
            if downloads:
                logger.info(f"Requesting download for SR bands: {bands}")
            else:
                # 2. Fallback: Try Level-1 bands
                logger.warning(f"No required SR bands found for {scene_id}. Trying Level-1 bands as fallback.")
                for product in options:
                    if product.get("secondaryDownloads"):
                        for secondary in product["secondaryDownloads"]:
                            for band in level1_bands:
                                # Flexible match for Level-1 bands
                                if secondary.get("bulkAvailable") and band in secondary.get("displayId", ""):
                                    downloads.append({
                                        "entityId": secondary["entityId"],
                                        "productId": secondary["id"]
                                    })
                if downloads:
                    logger.info(f"Requesting download for Level-1 bands: {level1_bands}")
            # 3. Fallback: Download all available secondary band files
            if not downloads:
                logger.warning(f"No required bands found for {scene_id}. Downloading all available secondary band files as last resort.")
                for product in options:
                    if product.get("secondaryDownloads"):
                        for secondary in product["secondaryDownloads"]:
                            if secondary.get("bulkAvailable"):
                                downloads.append({
                                    "entityId": secondary["entityId"],
                                    "productId": secondary["id"]
                                })
                if downloads:
                    logger.info(f"Requesting download for ALL available secondary band files.")
            if not downloads:
                logger.warning(f"No suitable band files found for {scene_id}. Available secondary downloads:")
                for product in options:
                    if product.get("secondaryDownloads"):
                        for secondary in product["secondaryDownloads"]:
                            logger.warning(f"  - {secondary.get('displayId', 'Unknown')} (bulkAvailable: {secondary.get('bulkAvailable', False)})")
                return None
            # Submit download request
            label = datetime.now().strftime("%Y%m%d_%H%M%S")
            download_req_payload = {
                "downloads": downloads,
                "label": label
            }
            download_request_results = self._dispatch_request('download-request', download_req_payload)
            # Retrieve download URLs
            available = download_request_results.get('availableDownloads', [])
            preparing = download_request_results.get('preparingDownloads', [])
            filepaths = []
            # Download available files
            for result in available:
                url = result['url']
                filename = result['url'].split('/')[-1].split('?')[0]
                filepath = zone_dir / filename
                if filepath.exists():
                    logger.info(f"✓ File already exists: {filepath}")
                    filepaths.append(filepath)
                    continue
                logger.info(f"Downloading: {filename} ...")
                with requests.get(url, stream=True) as response:
                    response.raise_for_status()
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                logger.info(f"✓ Downloaded: {filepath}")
                filepaths.append(filepath)
            # If some downloads are still preparing, poll download-retrieve
            if preparing:
                preparing_ids = [d['downloadId'] for d in preparing]
                logger.info(f"Waiting for {len(preparing_ids)} files to be ready...")
                while preparing_ids:
                    time.sleep(30)
                    retrieve_payload = {"label": label}
                    retrieve_results = self._dispatch_request('download-retrieve', retrieve_payload)
                    for result in retrieve_results.get('available', []):
                        if result['downloadId'] in preparing_ids:
                            url = result['url']
                            filename = url.split('/')[-1].split('?')[0]
                            filepath = zone_dir / filename
                            if not filepath.exists():
                                logger.info(f"Downloading: {filename} ...")
                                with requests.get(url, stream=True) as response:
                                    response.raise_for_status()
                                    with open(filepath, 'wb') as f:
                                        for chunk in response.iter_content(chunk_size=8192):
                                            if chunk:
                                                f.write(chunk)
                                logger.info(f"✓ Downloaded: {filepath}")
                            filepaths.append(filepath)
                            preparing_ids.remove(result['downloadId'])
            return filepaths
        except requests.RequestException as e:
            raise USGSAPIError(f"Network error during download: {str(e)}")
        except Exception as e:
            raise USGSAPIError(f"Download error: {str(e)}")
    
    def batch_download_zones(self, zones: List[str] = None, 
                           max_scenes_per_zone: int = 3) -> Dict[str, Dict[str, Any]]:
        """Download optimal scenes for multiple target zones, returning both file paths and SceneData objects"""
        from .config import TARGET_ZONES
        if zones is None:
            # Download for priority 1 zones first
            zones = [k for k, v in TARGET_ZONES.items() if v.priority == 1]
        downloads = {}
        for zone_key in zones:
            if zone_key not in TARGET_ZONES:
                logger.warning(f"Unknown zone: {zone_key}")
                continue
            zone = TARGET_ZONES[zone_key]
            logger.info(f"\n🎯 Processing {zone.name}")
            try:
                # Search for scenes
                scenes = self.search_landsat_scenes(zone)
                if not scenes:
                    logger.warning(f"No suitable scenes found for {zone.name}")
                    downloads[zone_key] = {"file_paths": [], "scene_data": []}
                    continue
                # Download best scenes
                zone_filepaths = []
                zone_scenedata = []
                for i, scene in enumerate(scenes[:max_scenes_per_zone]):
                    logger.info(f"Downloading scene {i+1}/{max_scenes_per_zone}: {scene['display_id']}")
                    logger.info(f"Quality: {scene['quality_score']:.1f}, Cloud: {scene['cloud_cover']:.1f}%")
                    filepaths = self.download_scene(scene['scene_id'], zone.name)
                    if filepaths:
                        zone_filepaths.append(filepaths)
                        # Map band names to file paths (try to infer from filename)
                        band_map = {}
                        available_bands = []
                        for fp in filepaths:
                            name = fp.name.lower()
                            # Try to extract band name from filename
                            for band in ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "B2", "B3", "B4", "B5", "B6", "B7"]:
                                if band.lower() in name:
                                    band_map[band] = fp
                                    available_bands.append(band)
                        # Remove duplicates
                        available_bands = list(set(available_bands))
                        # Compose metadata
                        metadata = {
                            "acquisition_date": scene.get("acquisition_date"),
                            "cloud_cover": scene.get("cloud_cover"),
                            "sun_elevation": scene.get("sun_elevation"),
                            "is_dry_season": scene.get("is_dry_season"),
                            "quality_score": scene.get("quality_score"),
                            "browse_url": scene.get("browse_url"),
                            "display_id": scene.get("display_id"),
                        }
                        scenedata = SceneData(
                            zone_id=zone_key,
                            provider="usgs",
                            scene_id=scene["scene_id"],
                            file_paths=band_map,
                            available_bands=available_bands,
                            metadata=metadata,
                        )
                        zone_scenedata.append(scenedata)
                        # Brief pause between downloads
                        time.sleep(2)
                downloads[zone_key] = {"file_paths": zone_filepaths, "scene_data": zone_scenedata}
                logger.info(f"✓ Downloaded {len(zone_filepaths)} scenes for {zone.name}")
            except USGSAPIError as e:
                logger.error(f"Error processing {zone.name}: {str(e)}")
                downloads[zone_key] = {"file_paths": [], "scene_data": []}
            # Pause between zones to be respectful to the API
            time.sleep(5)
        return downloads
    
    def logout(self):
        """Clean logout from USGS API"""
        if self.authenticated:
            try:
                url = f"{self.base_url}logout"
                response = self.session.post(url)
                self.authenticated = False
                logger.info("✓ Logged out from USGS API")
            except:
                pass  # Logout errors are not critical
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logout()

# Convenience function for quick access
def download_archaeological_data(zones: List[str] = None, max_scenes: int = 3) -> Dict[str, List[Path]]:
    """Quick function to download archaeological satellite data"""
    
    with USGSArchaeologyAPI() as api:
        return api.batch_download_zones(zones, max_scenes)

class USGSProvider(BaseProvider):
    """
    USGS satellite data provider implementing the BaseProvider interface.
    """
    def __init__(self, api=None):
        self.api = api or USGSArchaeologyAPI()

    def download_data(self, zones: list, max_scenes: int = 3) -> list:
        """
        Download data for the given zones using the USGS API.
        Returns a list of SceneData objects.
        """
        results = self.api.batch_download_zones(zones, max_scenes)
        scene_data = []
        for zone, result in results.items():
            scene_data.extend(result.get('scene_data', []))
        return scene_data

if __name__ == "__main__":
    # Test the API
    print("Testing USGS API...")
    try:
        api = USGSArchaeologyAPI()
        print("✓ Authentication successful")
        
        # Test search for Negro-Madeira zone
        from .config import TARGET_ZONES
        zone = TARGET_ZONES['negro_madeira']
        scenes = api.search_landsat_scenes(zone)
        print(f"✓ Found {len(scenes)} scenes for {zone.name}")
        
        if scenes:
            print(f"Best scene: {scenes[0]['display_id']} (Quality: {scenes[0]['quality_score']:.1f})")
        
    except USGSAPIError as e:
        print(f"❌ API Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
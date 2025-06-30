"""
NASA GEDI Provider for Archaeological Discovery - FIXED VERSION
Space-based LiDAR for Amazon forest structure analysis

This file should replace src/providers/gedi_provider.py

The main fixes:
1. Proper zone name normalization (fixes upper-naporegion -> upper_napo)
2. Better GEDI L2A HDF5 data extraction
3. Improved metric file creation for analysis step
4. Fixed scene directory path handling
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import requests

from src.core.data_objects import SceneData, BaseProvider

logger = logging.getLogger(__name__)


class GEDIProvider(BaseProvider):
    """NASA GEDI Provider for Archaeological Discovery - FIXED."""

    def __init__(
        self,
        earthdata_username: str | None = None,
        earthdata_password: str | None = None,
    ) -> None:
        self.username = earthdata_username or os.getenv("EARTHDATA_USERNAME")
        self.password = earthdata_password or os.getenv("EARTHDATA_PASSWORD")

        # Import SATELLITE_DIR here, inside __init__
        from src.core.config import SATELLITE_DIR
        
        # Define instance-specific cache directories
        self.raw_hdf5_cache_dir = SATELLITE_DIR / "gedi" / "raw_hdf5_cache"
        self.processed_metrics_cache_dir = SATELLITE_DIR / "gedi" / "processed_metrics_cache"
        
        # Ensure these directories exist
        self.raw_hdf5_cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_metrics_cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        if self.username and self.password:
            # Use Bearer token authentication if password looks like a JWT token
            if self.password.count('.') == 2 and len(self.password) > 100:
                self.session.headers.update({"Authorization": f"Bearer {self.password}"})
                logger.info("🛰️ GEDI Provider: Bearer token detected. Using token-based authentication.")
            else:
                # Fall back to basic auth for backward compatibility
                self.session.auth = (self.username, self.password)
                logger.info("🛰️ GEDI Provider: Using HTTP Basic Auth with username/password.")
        else:
            logger.info("🛰️ GEDI Provider: No explicit credentials provided. Will rely on .netrc file for authentication if present.")

        self.base_urls = {
            "earthdata_search": "https://cmr.earthdata.nasa.gov/search",
            "harmony": "https://harmony.earthdata.nasa.gov",
            "lpdaac": (
                "https://lpdaac.usgs.gov/data/get-started-data/collection-overview/"
                "missions/gedi-mission/"
            ),
        }

        logger.info("🛰️ GEDI Provider initialized for Amazon archaeological discovery")

    def download_data(self, zones: List[str], max_scenes: int = 3) -> List[SceneData]:
        """Download GEDI data for the given zones with proper zone name handling."""

        # Import TARGET_ZONES here, inside the method
        from src.core.config import TARGET_ZONES

        all_scene_data: List[SceneData] = []

        if isinstance(zones, str):
            zones = [zones]

        for zone_id in zones:
            # FIX: Handle zone name mapping properly
            normalized_zone_id = self._normalize_zone_id(zone_id)
            
            if normalized_zone_id not in TARGET_ZONES:
                logger.warning("Unknown zone: %s (normalized: %s)", zone_id, normalized_zone_id)
                continue

            zone = TARGET_ZONES[normalized_zone_id]
            logger.info("🎯 Downloading GEDI data for %s", zone.name)

            try:
                granules = self.search_gedi_data(zone, max_scenes)
                if not granules:
                    logger.warning("No GEDI data found for %s", zone.name)
                    continue

                for i, granule in enumerate(granules[:max_scenes]):
                    logger.info(
                        "Processing GEDI granule %s/%s: %s",
                        i + 1,
                        len(granules),
                        granule["id"],
                    )

                    scene_data = self.process_gedi_granule(granule, zone, normalized_zone_id)
                    if scene_data:
                        all_scene_data.append(scene_data)
                        logger.info("✅ Successfully processed %s", granule["id"])
                    else:
                        logger.warning("❌ Failed to process %s", granule["id"])

            except Exception as exc:  # noqa: BLE001
                logger.error("Error processing zone %s: %s", zone.name, exc)
                continue

        logger.info(
            "🎯 GEDI download complete: %s granules processed", len(all_scene_data)
        )
        return all_scene_data

    def _normalize_zone_id(self, zone_id: str) -> str:
        """Normalize zone IDs to match TARGET_ZONES keys - CRITICAL FIX."""
        # Handle common zone ID variations found in logs
        zone_mapping = {
            "upper-naporegion": "upper_napo",
            "upper_naporegion": "upper_napo", 
            "uppernaporegion": "upper_napo",
            "upper-napo": "upper_napo",
            "upper_napo_region": "upper_napo",
            "upper-napo-region": "upper_napo",
            "negro-madeira": "negro_madeira",
            "negro_madeira_confluence": "negro_madeira",
            "trombetas-river": "trombetas",
            "trombetas_river": "trombetas",
            "trombetas_river_junction": "trombetas",
            "upper-xingu": "upper_xingu",
            "upper_xingu_region": "upper_xingu",
            "maranon-river": "maranon",
            "maranon_river": "maranon",
            "maranon_river_system": "maranon"
        }
        
        normalized = zone_id.lower().replace(" ", "_").replace("-", "_")
        return zone_mapping.get(normalized, normalized)

    def search_gedi_data(self, zone: Any, max_results: int = 10) -> List[Dict]:
        """Search for GEDI granules covering the zone."""

        try:
            bbox = zone.bbox  # (south, west, north, east)
            
            search_params = {
                "collection_concept_id": "C2142771958-LPCLOUD",  # GEDI L2A V002
                "bounding_box": f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}",  # west,south,east,north
                "page_size": max_results,
                "page_num": 1,
            }

            search_url = f"{self.base_urls['earthdata_search']}/granules"
            
            logger.info(f"🔍 Searching GEDI data with parameters: {search_params}")
            logger.info(f"🌍 Bounding box: {search_params['bounding_box']} (west,south,east,north)")
            
            headers = {"Accept": "application/json"}
            
            response = self.session.get(search_url, params=search_params, headers=headers, timeout=60)
            
            logger.info(f"🔗 Full request URL: {response.url}")
            
            if response.status_code != 200:
                logger.error(f"❌ CMR API error: {response.status_code}")
                # Try to get more detailed error from JSON response if available
                try:
                    error_details = response.json()
                    logger.error(f"Error details: {error_details}")
                except json.JSONDecodeError:
                    logger.error(f"Response text (first 500 chars): {response.text[:500]}")
                return []
            
            response.raise_for_status()

            search_results = response.json()
            granules = search_results.get("feed", {}).get("entry", [])
            total_results = len(granules)
            
            logger.info(f"📊 Found {total_results} GEDI granules for {zone.name}")
            
            if not granules:
                logger.warning("No GEDI granules found for %s.", zone.name)
                return []

            processed: List[Dict] = []
            for granule in granules:
                try:
                    g_info = self.extract_granule_metadata(granule, zone)
                    if g_info:
                        processed.append(g_info)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Error processing granule metadata: %s", exc)

            processed.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

            logger.info(
                "Found %s relevant GEDI granules for %s", len(processed), zone.name
            )
            return processed

        except Exception as exc:  # noqa: BLE001
            logger.error("GEDI search failed for %s: %s", zone.name, exc, exc_info=True)
            return []

    def extract_granule_metadata(self, granule: Dict, zone: Any) -> Optional[Dict]:
        """Extract relevant metadata from a GEDI granule."""

        try:
            granule_id = granule.get(
                "producer_granule_id", granule.get("title", "unknown")
            )

            time_start = granule.get("time_start", "")

            relevance_score = 0
            if time_start:
                try:
                    acq_date = datetime.strptime(time_start[:19], "%Y-%m-%dT%H:%M:%S")
                    days_old = (datetime.now() - acq_date).days
                    if days_old < 365:
                        relevance_score += 20
                    elif days_old < 730:
                        relevance_score += 15
                    elif days_old < 1095:
                        relevance_score += 10
                except Exception:  # noqa: BLE001
                    pass

            if "boxes" in granule:
                try:
                    for box_str in granule["boxes"]:
                        coords = [float(x) for x in box_str.split()]
                        if len(coords) == 4:
                            overlap = self.calculate_bbox_overlap(coords, zone.bbox)
                            relevance_score += overlap * 30
                except Exception:  # noqa: BLE001
                    pass

            return {
                "id": granule_id,
                "granule_metadata": granule,
                "acquisition_date": time_start,
                "relevance_score": relevance_score,
                "download_urls": self.extract_download_urls(granule),
            }

        except Exception as exc:  # noqa: BLE001
            logger.error("Error extracting granule metadata: %s", exc)
            return None

    def calculate_bbox_overlap(
        self, granule_bbox: List[float], zone_bbox: Tuple[float, float, float, float]
    ) -> float:
        """Calculate overlap ratio between granule and zone."""

        try:
            if len(granule_bbox) != 4:
                return 0

            g_west, g_south, g_east, g_north = granule_bbox
            z_south, z_west, z_north, z_east = zone_bbox

            overlap_w = max(g_west, z_west)
            overlap_s = max(g_south, z_south)
            overlap_e = min(g_east, z_east)
            overlap_n = min(g_north, z_north)

            if overlap_w < overlap_e and overlap_s < overlap_n:
                overlap_area = (overlap_e - overlap_w) * (overlap_n - overlap_s)
                zone_area = (z_east - z_west) * (z_north - z_south)
                return overlap_area / zone_area if zone_area > 0 else 0

            return 0

        except Exception as exc:  # noqa: BLE001
            logger.warning("Error calculating bbox overlap: %s", exc)
            return 0

    def extract_download_urls(self, granule: Dict) -> List[str]:
        """Extract HDF5 download URLs from granule links."""
        urls = []
        if "links" in granule:
            for link in granule["links"]:
                rel = link.get("rel", "")
                href = link.get("href", "")
                # Look for data links containing HDF5 files
                # The rel for data links is "http://esipfed.org/ns/fedsearch/1.1/data#"
                if ("data#" in rel and href.endswith(".h5")) or \
                   (rel.endswith("data#") and href.endswith(".h5")):
                    # Prefer HTTPS URLs over S3 URLs for authentication compatibility
                    if href.startswith("https://"):
                        urls.insert(0, href)  # Insert at beginning to prioritize HTTPS
                    else:
                        urls.append(href)
        if not urls:
            logger.warning("No HDF5 download URLs found in granule: %s", granule.get("id"))
        return urls

    def download_gedi_hdf5(self, url: str, granule_specific_cache_dir: Path) -> Optional[Path]:
        """
        Download a GEDI HDF5 file to a granule-specific cache directory.
        Skips download if file already exists and is valid.
        """
        try:
            granule_specific_cache_dir.mkdir(parents=True, exist_ok=True)
            
            filename = url.split("/")[-1]
            # Basic sanitization for filename, though usually not an issue with URLs
            filename = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in filename)
            if not filename.endswith(".h5"): # Ensure it's an h5 file
                logger.error(f"Invalid filename generated or URL does not point to .h5: {filename} from {url}")
                return None

            output_path = granule_specific_cache_dir / filename

            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info("CACHE HIT: Using existing GEDI HDF5 file: %s", output_path)
                return output_path

            logger.info("CACHE MISS: Downloading GEDI HDF5 from %s to %s", url, output_path)
            
            # Use stream=True for large files
            response = self.session.get(url, stream=True, timeout=300) # 5 min timeout for download
            response.raise_for_status()

            with open(output_path, "wb") as f_out:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    f_out.write(chunk)
            
            if output_path.stat().st_size > 0:
                logger.info("✅ Successfully downloaded GEDI HDF5: %s", output_path)
                return output_path
            else:
                logger.warning("❌ Downloaded GEDI HDF5 file is empty: %s", output_path)
                if output_path.exists(): # Clean up empty file
                    output_path.unlink()
                return None

        except requests.exceptions.RequestException as req_exc:
            logger.error("Request error downloading GEDI HDF5 %s: %s", url, req_exc)
        except IOError as io_exc:
            logger.error("I/O error with GEDI HDF5 file %s: %s", output_path if 'output_path' in locals() else filename, io_exc)
        except Exception as e:
            logger.error("Unexpected error downloading GEDI HDF5 %s: %s", url, e, exc_info=True)
        return None

    def check_cache_status(self, zone_id: str) -> Dict[str, Any]:
        """Check cache status for a given zone (placeholder)."""
        # This method might be useful for more detailed cache reporting later
        normalized_zone_id = self._normalize_zone_id(zone_id)
        
        raw_cache_zone_dir = self.raw_hdf5_cache_dir 
        # Granule specific dirs are one level down, so listing them might be too much here.
        # Instead, one could check for the existence of the zone's processed metrics.
        processed_cache_zone_dir = self.processed_metrics_cache_dir / normalized_zone_id
        
        return {
            "zone_id": normalized_zone_id,
            "raw_hdf5_cache_base": str(self.raw_hdf5_cache_dir),
            "processed_metrics_cache_dir_for_zone": str(processed_cache_zone_dir),
            "processed_metrics_exist_for_zone": processed_cache_zone_dir.exists() and any(processed_cache_zone_dir.iterdir())
        }

    def process_gedi_granule(self, granule: Dict, zone: Any, zone_id: str) -> Optional[SceneData]:
        """
        Process a single GEDI granule: download, extract metrics, and cache.
        Uses defined cache directories.
        """
        granule_id = granule.get("id")
        if not granule_id:
            logger.warning("Granule missing ID, cannot process.")
            return None

        # Sanitize granule_id for path compatibility
        granule_id_for_path = "".join(c if c.isalnum() else "_" for c in granule_id)

        download_urls = granule.get("download_urls", [])
        if not download_urls:
            logger.warning("No download URLs for granule %s", granule_id)
            return None
        
        # For GEDI, typically one HDF5 per granule
        download_url = download_urls[0] 
        
        # Define cache paths
        # Raw HDF5 is cached per zone_name and then granule_id_for_path
        raw_hdf5_granule_cache_dir = self.raw_hdf5_cache_dir / zone.name / granule_id_for_path
        
        # Processed metrics are cached per zone.name and granule
        processed_metrics_zone_dir = self.processed_metrics_cache_dir / zone.name # Use zone.name
        processed_metrics_zone_dir.mkdir(parents=True, exist_ok=True) # Ensure zone directory exists
        metrics_file_path = processed_metrics_zone_dir / f"{granule_id_for_path}_metrics.json"

        hdf5_file: Optional[Path] = None
        metrics_data: Optional[Dict[str, Any]] = None # Using Any for loaded metrics structure

        # 1. Download HDF5 (uses internal caching)
        hdf5_file = self.download_gedi_hdf5(download_url, raw_hdf5_granule_cache_dir)
        if not hdf5_file:
            logger.error("Failed to download HDF5 for granule %s", granule_id)
            return None

        # 2. Process metrics (check cache first)
        if metrics_file_path.exists() and metrics_file_path.stat().st_size > 0:
            logger.info("CACHE HIT: Using cached metrics for %s from %s", granule_id, metrics_file_path)
            try:
                with open(metrics_file_path, "r") as f:
                    metrics_data = json.load(f)
                # Basic validation of loaded metrics (e.g. check for expected keys)
                if not isinstance(metrics_data, dict) or not metrics_data: # Add more checks if needed
                    logger.warning("Cached metrics file %s is invalid or empty. Re-extracting.", metrics_file_path)
                    metrics_data = None 
            except json.JSONDecodeError:
                logger.warning("Error decoding cached metrics JSON from %s. Re-extracting.", metrics_file_path)
                metrics_data = None
            except Exception as e:
                logger.error("Error loading cached metrics from %s: %s. Re-extracting.", metrics_file_path, e)
                metrics_data = None
        
        if metrics_data is None: # Cache miss or invalid cache for metrics
            logger.info("CACHE MISS: Extracting GEDI metrics for %s from %s", granule_id, hdf5_file)
            # Ensure hdf5_file is not None and exists before passing to extraction
            if hdf5_file and hdf5_file.exists():
                extracted_metrics_arrays = self.extract_archaeological_metrics(hdf5_file, zone) # Returns dict of np.arrays
                
                if extracted_metrics_arrays:
                    # Convert numpy arrays to lists for JSON serialization
                    metrics_data_serializable = {}
                    for key, arr in extracted_metrics_arrays.items():
                        if isinstance(arr, np.ndarray):
                            metrics_data_serializable[key] = arr.tolist()
                        else: # Should not happen based on extract_archaeological_metrics typing
                            metrics_data_serializable[key] = arr 
                    
                    try:
                        with open(metrics_file_path, "w") as f:
                            json.dump(metrics_data_serializable, f, indent=2)
                        logger.info("✅ Saved GEDI metrics to %s", metrics_file_path)
                        metrics_data = metrics_data_serializable # Use the saved (serializable) data
                    except IOError as e:
                        logger.error("Error saving GEDI metrics to %s: %s", metrics_file_path, e)
                        # Proceed without cached metrics if saving fails
                    except TypeError as e:
                        logger.error("Error serializing metrics for %s: %s. Check for non-serializable types.", granule_id, e)

                else:
                    logger.warning("Failed to extract metrics for %s", granule_id)
            else:
                 logger.error("HDF5 file path is invalid or file does not exist. Cannot extract metrics for %s", granule_id)


        # Construct SceneData
        file_paths: Dict[str, Path] = {}
        if hdf5_file and hdf5_file.exists(): # Ensure hdf5_file is valid
            file_paths["hdf5_raw_file"] = hdf5_file
        if metrics_file_path.exists() and metrics_data: # Ensure metrics were successfully loaded/extracted
            file_paths["processed_metrics_file"] = metrics_file_path
        
        if not file_paths: # If no essential files are available
            logger.warning("No valid file paths (HDF5 or metrics) for granule %s. Skipping SceneData creation.", granule_id)
            return None

        # Consolidate metadata for SceneData
        scene_metadata = {
            "granule_id": granule_id,
            "acquisition_date": granule.get("acquisition_date"),
            "original_granule_metadata": granule.get("granule_metadata"), # Full original metadata if needed
            "relevance_score": granule.get("relevance_score"),
            "download_url_used": download_url,
            # Add metrics directly to metadata if they are small enough and useful
            # For large metrics, keep them in the separate file and SceneData.features can point to it
            # Or, if metrics_data is already loaded, it can be passed to SceneData constructor's 'features' arg
        }

        # If metrics_data was loaded/extracted, it can be passed as 'features'
        # The SceneData object can then decide how to handle it (e.g. store directly or provide access)
        scene_features = metrics_data if metrics_data else {}

        return SceneData(
            zone_id=zone_id,
            provider="gedi",
            scene_id=granule_id_for_path, # Use the path-safe ID
            file_paths=file_paths,
            available_bands=[],  # GEDI doesn't have bands in the traditional sense, metrics are features
            metadata=scene_metadata,
            features=scene_features # Pass loaded/extracted metrics here
        )

    def _verify_hdf5_file(self, hdf5_file: Path) -> bool:
        """
        Verify HDF5 file integrity by checking basic structure and file size.
        Returns True if file appears valid, False if corrupted.
        """
        try:
            if not hdf5_file.exists():
                return False
                
            # Check if file size is reasonable (> 100MB for GEDI files)
            file_size = hdf5_file.stat().st_size
            if file_size < 100 * 1024 * 1024:  # 100MB minimum
                logger.warning("HDF5 file too small (%d bytes): %s", file_size, hdf5_file)
                return False
            
            # Try to open and read basic structure
            with h5py.File(hdf5_file, "r") as hf:
                # Check for expected BEAM groups
                beam_names = [name for name in hf.keys() if name.startswith("BEAM")]
                if not beam_names:
                    logger.warning("No BEAM groups found in HDF5 file: %s", hdf5_file)
                    return False
                    
                # Try to access one beam's basic structure
                test_beam = beam_names[0]
                beam_group = hf[test_beam]
                
                # Check for essential datasets
                if "lat_lowestmode" not in beam_group or "lon_lowestmode" not in beam_group:
                    logger.warning("Missing essential datasets in HDF5 file: %s", hdf5_file)
                    return False
                    
            return True
            
        except (OSError, KeyError, Exception) as e:
            logger.warning("HDF5 file verification failed for %s: %s", hdf5_file, str(e))
            return False

    def extract_archaeological_metrics(
        self, hdf5_file: Path, zone: Any
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract key archaeological metrics from GEDI L2A HDF5 for a specific zone.
        Focuses on metrics indicative of clearings, mounds, and earthworks.
        Returns a dictionary of numpy arrays.
        
        Includes adaptive zone expansion for small zones with no GEDI coverage.
        """
        # Check if hdf5_file is None or does not exist
        if not hdf5_file or not hdf5_file.exists():
            logger.error(f"HDF5 file path is invalid or file does not exist: {hdf5_file}")
            return None

        metrics: Dict[str, List[Any]] = {
            "latitude": [], "longitude": [], "elevation_ground": [], "elevation_canopy_top": [],
            "canopy_height": [], "energy_total": [], "sensitivity": [], 
            "quality_flag": [], "beam_type": [], "shot_number": []
        }
        
        required_datasets = [
            "lat_lowestmode", "lon_lowestmode", "elev_lowestmode", "elev_highestreturn",
            "rh", "energy_total", "sensitivity", "quality_flag", "shot_number"
        ]
        # beam_type is not directly in L2A shots, it's implicit by BEAM group.

        try:
            # Check if file is corrupted by verifying file size and basic structure
            if not self._verify_hdf5_file(hdf5_file):
                logger.error("🚨 HDF5 file appears corrupted: %s", hdf5_file)
                logger.info("📁 Removing corrupted file for re-download...")
                if os.path.exists(hdf5_file):
                    os.remove(hdf5_file)
                    logger.info("✅ Corrupted file removed successfully")
                return None
                
            with h5py.File(hdf5_file, "r") as hf:
                # Iterate over the four GEDI beams (BEAM0000 to BEAM0011 or BEAM0101 to BEAM1011)
                # Common beams are BEAM0000, BEAM0001, BEAM0010, BEAM0011 (power beams)
                # and BEAM0101, BEAM0110, BEAM1000, BEAM1011 (coverage beams)
                beam_names = [name for name in hf.keys() if name.startswith("BEAM")]

                if not beam_names:
                    logger.warning("No BEAM groups found in HDF5 file: %s", hdf5_file)
                    return None

                # First pass: collect all coordinates to test adaptive expansion
                all_lats = []
                all_lons = []
                
                for beam_name in beam_names:
                    beam_group = hf[beam_name]
                    if ("lat_lowestmode" in beam_group and "lon_lowestmode" in beam_group and 
                        len(beam_group["lat_lowestmode"][:]) > 0):
                        all_lats.extend(beam_group["lat_lowestmode"][:])
                        all_lons.extend(beam_group["lon_lowestmode"][:])
                
                if not all_lats:
                    logger.warning("No coordinate data found in any beam: %s", hdf5_file)
                    return None
                
                all_lats = np.array(all_lats)
                all_lons = np.array(all_lons)
                
                # Apply adaptive zone expansion if needed
                expansion_applied, active_bbox, shot_count = self.adaptive_zone_expansion(
                    zone, all_lats, all_lons, min_shots=50, max_expansion=0.2
                )
                
                if expansion_applied:
                    logger.info(f"Using expanded zone boundaries for {zone.name}")
                    # Create a temporary zone object with expanded boundaries
                    class ExpandedZone:
                        def __init__(self, original_zone, new_bbox):
                            self.name = f"{original_zone.name} (expanded)"
                            self.bbox = new_bbox
                    
                    active_zone = ExpandedZone(zone, active_bbox)
                else:
                    active_zone = zone
                
                # Second pass: extract data using active zone boundaries
                for beam_name in beam_names:
                    beam_group = hf[beam_name]
                    
                    # Check if all required datasets exist in this beam
                    if not all(ds_name in beam_group for ds_name in required_datasets):
                        logger.debug("Beam %s in %s is missing some required datasets. Skipping.", beam_name, hdf5_file)
                        continue
                    
                    # Basic check for data presence (e.g. non-empty lowestmode lat)
                    if len(beam_group["lat_lowestmode"][:]) == 0:
                        logger.debug("Beam %s in %s has no data points. Skipping.", beam_name, hdf5_file)
                        continue

                    lats = beam_group["lat_lowestmode"][:]
                    lons = beam_group["lon_lowestmode"][:]
                    
                    # Filter points within the active zone's bounding box (may be expanded)
                    zone_mask = self.filter_points_in_zone(lats, lons, active_zone)
                    
                    if not np.any(zone_mask):
                        logger.debug("No GEDI shots from beam %s fall within active zone %s.", beam_name, active_zone.name)
                        continue
                    
                    logger.info("Found %s shots from beam %s in active zone %s", np.sum(zone_mask), beam_name, active_zone.name)

                    # Apply mask and append data
                    metrics["latitude"].extend(lats[zone_mask])
                    metrics["longitude"].extend(lons[zone_mask])
                    metrics["elevation_ground"].extend(beam_group["elev_lowestmode"][:][zone_mask])
                    metrics["elevation_canopy_top"].extend(beam_group["elev_highestreturn"][:][zone_mask])
                    
                    # Calculate canopy height from RH100 (98th percentile height)
                    rh_data = beam_group["rh"][:]  # Shape: (N, 101) - relative heights 0-100%
                    rh100_values = rh_data[:, 100][zone_mask]  # Get 100th percentile (canopy top)
                    metrics["canopy_height"].extend(rh100_values)
                    
                    metrics["energy_total"].extend(beam_group["energy_total"][:][zone_mask])
                    metrics["sensitivity"].extend(beam_group["sensitivity"][:][zone_mask])
                    metrics["quality_flag"].extend(beam_group["quality_flag"][:][zone_mask])
                    metrics["shot_number"].extend(beam_group["shot_number"][:][zone_mask])
                    
                    # Determine beam type (power or coverage) based on typical naming or metadata if available
                    # For simplicity, let's assume BEAM00xx are power, others coverage.
                    # A more robust way would check metadata if present.
                    beam_type_val = 0 # 0 for coverage, 1 for power
                    if beam_name in ["BEAM0000", "BEAM0001", "BEAM0010", "BEAM0011"]:
                        beam_type_val = 1 
                    metrics["beam_type"].extend([beam_type_val] * np.sum(zone_mask))


            # Convert lists to numpy arrays
            final_metrics: Dict[str, np.ndarray] = {}
            for key, val_list in metrics.items():
                if val_list:                 # Only add if there's data
                    final_metrics[key] = np.array(val_list)
                else: # If a key has no data after all beams, ensure it's an empty array of appropriate type
                    # This helps detector to expect consistent types
                    if key in ["latitude", "longitude", "elevation_ground", "elevation_canopy_top", 
                               "canopy_height", "energy_total", "sensitivity"]:
                        final_metrics[key] = np.array([], dtype=np.float64)
                    elif key in ["quality_flag", "beam_type"]:
                         final_metrics[key] = np.array([], dtype=np.int8) # Or appropriate int type
                    elif key == "shot_number":
                        final_metrics[key] = np.array([], dtype=np.uint64) # Or appropriate int type
            
            if not final_metrics or not final_metrics.get("latitude").size > 0 : # Check if any lats were actually added
                 logger.warning("No GEDI data points extracted within active zone %s from file %s.", active_zone.name, hdf5_file)
                 return None # Return None if no data points for any key

            shots_extracted = final_metrics["latitude"].size if "latitude" in final_metrics else 0
            zone_desc = f"{zone.name} (expanded)" if expansion_applied else zone.name
            logger.info("Extracted %s GEDI shots for %s from %s", shots_extracted, zone_desc, hdf5_file)
            
            if expansion_applied:
                logger.info("Zone expansion successfully provided %s additional shots", shots_extracted)
            return final_metrics

        except FileNotFoundError:
            logger.error("HDF5 file not found: %s", hdf5_file)
            return None
        except Exception as e:
            logger.error("Error extracting metrics from HDF5 %s: %s", hdf5_file, e, exc_info=True)
            return None

    def filter_points_in_zone(
        self, lats: np.ndarray, lons: np.ndarray, zone: Any
    ) -> np.ndarray:
        """Filter GEDI points to those within the zone's bounding box."""
        # zone.bbox is (south, west, north, east)
        south, west, north, east = zone.bbox
        
        mask = (
            (lats >= south) & (lats <= north) & (lons >= west) & (lons <= east)
        )
        return mask
    
    def adaptive_zone_expansion(
        self, zone: Any, lats: np.ndarray, lons: np.ndarray, 
        min_shots: int = 50, max_expansion: float = 0.2
    ) -> Tuple[bool, Tuple[float, float, float, float], int]:
        """
        Adaptively expand zone boundaries to ensure minimum GEDI coverage.
        
        Args:
            zone: Original zone object
            lats, lons: All available GEDI coordinates
            min_shots: Minimum number of shots required
            max_expansion: Maximum expansion in degrees
            
        Returns:
            (expanded, new_bbox, shot_count)
        """
        original_bbox = zone.bbox
        south, west, north, east = original_bbox
        
        # Test original zone first
        original_mask = self.filter_points_in_zone(lats, lons, zone)
        original_count = np.sum(original_mask)
        
        if original_count >= min_shots:
            logger.info(f"Zone {zone.name} has sufficient GEDI coverage: {original_count} shots")
            return False, original_bbox, original_count
        
        # Calculate zone size
        zone_size_deg = max(north - south, east - west)
        logger.info(f"Zone {zone.name} too small for GEDI: {original_count} shots (need {min_shots})")
        logger.info(f"Original zone size: {zone_size_deg:.3f}° ({zone_size_deg * 111:.1f} km)")
        
        # Try expanding in steps
        expansion_steps = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]  # Progressive expansion
        
        for expansion in expansion_steps:
            if expansion > max_expansion:
                break
                
            # Expand zone symmetrically around center
            center_lat = (north + south) / 2
            center_lon = (east + west) / 2
            
            expanded_bbox = (
                center_lat - (zone_size_deg / 2 + expansion),  # south
                center_lon - (zone_size_deg / 2 + expansion),  # west
                center_lat + (zone_size_deg / 2 + expansion),  # north
                center_lon + (zone_size_deg / 2 + expansion),  # east
            )
            
            # Test expanded zone
            expanded_mask = (
                (lats >= expanded_bbox[0]) & (lats <= expanded_bbox[2]) & 
                (lons >= expanded_bbox[1]) & (lons <= expanded_bbox[3])
            )
            expanded_count = np.sum(expanded_mask)
            
            logger.info(f"Testing expansion by {expansion:.3f}° ({expansion * 111:.1f} km): {expanded_count} shots")
            
            if expanded_count >= min_shots:
                expansion_km = expansion * 111
                logger.info(f"✅ Expanded zone {zone.name} by {expansion:.3f}° ({expansion_km:.1f} km)")
                logger.info(f"   Original: {original_count} shots, Expanded: {expanded_count} shots")
                return True, expanded_bbox, expanded_count
        
        # If no expansion worked, return original
        logger.warning(f"❌ Could not find sufficient GEDI coverage for {zone.name} even with maximum expansion")
        logger.warning(f"Consider using a larger zone or multiple GEDI granules")
        return False, original_bbox, original_count
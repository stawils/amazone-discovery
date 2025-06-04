"""
NASA GEDI Provider for Archaeological Discovery - FIXED VERSION
Space-based LiDAR for Amazon forest structure analysis
Compatible with existing pipeline architecture
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
from src.core.config import TARGET_ZONES, SATELLITE_DIR

logger = logging.getLogger(__name__)


class GEDIProvider(BaseProvider):
    """NASA GEDI Provider for Archaeological Discovery."""

    def __init__(
        self,
        earthdata_username: str | None = None,
        earthdata_password: str | None = None,
    ) -> None:
        self.username = earthdata_username or os.getenv("EARTHDATA_USERNAME")
        self.password = earthdata_password or os.getenv("EARTHDATA_PASSWORD")

        self.session = requests.Session()
        if self.username and self.password:
            self.session.auth = (self.username, self.password)
            logger.info("🛰️ GEDI Provider: Credentials explicitly provided (e.g., via environment variables). Attempting HTTP Basic Auth.")
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
        """Download GEDI data for the given zones."""

        all_scene_data: List[SceneData] = []

        if isinstance(zones, str):
            zones = [zones]

        for zone_id in zones:
            if zone_id not in TARGET_ZONES:
                logger.warning("Unknown zone: %s", zone_id)
                continue

            zone = TARGET_ZONES[zone_id]
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

                    scene_data = self.process_gedi_granule(granule, zone)
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

    def search_gedi_data(self, zone: Any, max_results: int = 10) -> List[Dict]:
        """Search for GEDI granules covering the zone - FIXED VERSION."""

        try:
            bbox = zone.bbox  # (south, west, north, east)
            
            # Let's try a broader search first to see if GEDI data exists
            search_params = {
                "collection_concept_id": "C2142771958-LPCLOUD",  # GEDI L2A V002
                "bounding_box": f"{bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}",  # west,south,east,north
                "page_size": max_results,
                "page_num": 1,
                # CMR API expects Accept header for response format, not a format parameter
            }

            search_url = f"{self.base_urls['earthdata_search']}/granules"
            
            logger.info(f"🔍 Searching GEDI data with parameters: {search_params}")
            logger.info(f"🌍 Bounding box: {search_params['bounding_box']} (west,south,east,north)")
            
            # Try a much broader search to test if the API works at all
            # GEDI covers latitudes between 51.6°N and 51.6°S
            broader_search_params = {
                "collection_concept_id": "C2142771958-LPCLOUD",
                "bounding_box": "-180,-51,180,51",  # Global GEDI coverage area
                "page_size": 1,  # Just get one result to test
            }
            
            # Set headers for JSON response format
            headers = {"Accept": "application/json"}
            
            logger.info(f"🌍 Testing broader GEDI search to verify API access...")
            broad_response = self.session.get(search_url, params=broader_search_params, headers=headers, timeout=60)
            
            if broad_response.status_code == 200:
                broad_results = broad_response.json()
                broad_total = broad_results.get("feed", {}).get("totalResults", 0)
                logger.info(f"✅ GEDI API Test: Broader (global) search found {broad_total} granules.")
                
                if broad_total == 0:
                    logger.info("ℹ️ GEDI API Test: Broader (global) search found 0 granules. This is an API availability check, specific search for the zone will follow.")
            else:
                logger.error(f"❌ Broader search also failed: {broad_response.status_code}")
                logger.error(f"Response: {broad_response.text[:300]}")
            
            # Now try the original search with proper headers
            response = self.session.get(search_url, params=search_params, headers=headers, timeout=60)
            
            # Log the full URL for debugging
            logger.info(f"🔗 Full request URL: {response.url}")
            
            if response.status_code != 200:
                logger.error(f"❌ CMR API error: {response.status_code}")
                logger.error(f"Response text: {response.text[:500]}")
                # No longer creating synthetic data here.
                return []
            
            response.raise_for_status()

            search_results = response.json()
            total_results = search_results.get("feed", {}).get("totalResults", 0)
            granules = search_results.get("feed", {}).get("entry", [])
            
            logger.info(f"📊 CMR search results: {total_results} total, {len(granules)} returned")
            
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

            if "cloud_cover" in granule:
                try:
                    cloud = float(granule.get("cloud_cover", 100))
                    if cloud < 20:
                        relevance_score += 15
                    elif cloud < 50:
                        relevance_score += 5
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
        """Extract HDF5 download URLs from granule metadata."""

        urls: List[str] = []
        if "links" in granule:
            for link in granule["links"]:
                if link.get("rel") == "http://esipfed.org/ns/fedsearch/1.1/data#":
                    href = link.get("href", "")
                    if href.endswith(".h5"):
                        urls.append(href)
        return urls

    def process_gedi_granule(self, granule: Dict, zone: Any) -> Optional[SceneData]:
        """Process a GEDI granule and return a SceneData object."""

        try:
            granule_id = granule["id"]

            zone_dir = SATELLITE_DIR / zone.name.lower().replace(" ", "_") / "gedi"
            granule_dir = zone_dir / granule_id
            granule_dir.mkdir(parents=True, exist_ok=True)

            file_paths: Dict[str, Path] = {}
            available_bands: List[str] = []

            urls = granule.get("download_urls", [])
            if not urls:
                logger.warning(f"No download URLs for granule {granule_id}. Will result in no data for this granule.")
                # available_bands will remain empty, and the granule processing will yield no SceneData.
            else: # This block executes if actual download URLs ARE present.
                logger.info(f"Granule {granule_id} has {len(urls)} download URLs. Processing them now.")
                processed_metrics_for_granule: Dict[str, List[np.ndarray]] = {} # Store lists of arrays if merging strategies are complex

                for url_idx, url in enumerate(urls):
                    logger.debug(f"Processing URL {url_idx+1}/{len(urls)}: {url} for granule {granule_id}")
                    hdf5_file = self.download_gedi_hdf5(url, granule_dir)
                    logger.debug(f"Result from download_gedi_hdf5 for url {url}: {hdf5_file}") # LOG 1
                    if hdf5_file:
                        logger.debug(f"HDF5 file path seems valid ({hdf5_file}), attempting to extract metrics.") # LOG 2
                        metrics = self.extract_archaeological_metrics(hdf5_file, zone)
                        if metrics:
                            logger.debug(f"Successfully extracted metric keys {list(metrics.keys())} from {hdf5_file.name}")
                            # Storing metrics from potentially multiple HDF5 files per granule
                            for key, value_array in metrics.items():
                                if key not in processed_metrics_for_granule:
                                    processed_metrics_for_granule[key] = []
                                processed_metrics_for_granule[key].append(value_array)
                        else:
                            logger.warning(f"Could not extract metrics from {hdf5_file.name} for url {url}")
                    else:
                        logger.warning(f"Failed to download or validate HDF5 file from URL: {url}")
                
                # After processing all URLs, consolidate and save the metrics
                if processed_metrics_for_granule:
                    final_consolidated_metrics: Dict[str, np.ndarray] = {}
                    for key, list_of_arrays in processed_metrics_for_granule.items():
                        if list_of_arrays:
                            # Example consolidation: concatenate. This might need adjustment based on metric structure.
                            # Ensure all arrays in list_of_arrays are not None and are numpy arrays.
                            valid_arrays = [arr for arr in list_of_arrays if isinstance(arr, np.ndarray)]
                            if valid_arrays:
                                try:
                                    final_consolidated_metrics[key] = np.concatenate(valid_arrays)
                                    logger.debug(f"Consolidated metric '{key}' from {len(valid_arrays)} array(s). Resulting shape: {final_consolidated_metrics[key].shape}")
                                except ValueError as ve:
                                    logger.error(f"ValueError during concatenation for metric '{key}': {ve}. Arrays: {[arr.shape for arr in valid_arrays]}", exc_info=True)
                                    # Fallback: use the first array if concatenation fails
                                    if len(valid_arrays) > 0:
                                        final_consolidated_metrics[key] = valid_arrays[0]
                                        logger.warning(f"Used first array for metric '{key}' due to concatenation error.")
                            else:
                                logger.warning(f"No valid numpy arrays found for metric key '{key}' during consolidation.")
                        else:
                             logger.warning(f"No arrays found for metric key '{key}' during consolidation attempt.")


                    for metric_name, data_array in final_consolidated_metrics.items():
                        metric_file = granule_dir / f"{metric_name}.npy"
                        np.save(metric_file, data_array)
                        file_paths[metric_name] = metric_file
                        if metric_name not in available_bands: # Ensure no duplicates
                           available_bands.append(metric_name)
                    logger.info(f"Saved processed metrics for granule {granule_id} from HDF5 files. Final bands: {available_bands}")
                else:
                    logger.warning(f"No metrics were successfully processed from any HDF5 URL for granule {granule_id}")

            if not available_bands:
                logger.warning("No usable data extracted for %s (all download/processing attempts failed or no URLs)", granule_id)
                return None

            metadata = {
                "provider": "gedi",
                "product_type": "L2A",
                "acquisition_date": granule.get("acquisition_date", ""),
                "relevance_score": granule.get("relevance_score", 0),
                "spatial_resolution": "25m_footprints",
                "coverage_type": "point_cloud",
                "granule_directory": str(granule_dir),
                "archaeological_metrics": list(available_bands),
                "original_format": "synthetic" if not urls else "HDF5",
                "beam_count": 8,
                "mission": "GEDI",
                "platform": "International_Space_Station",
            }

            scene_data = SceneData(
                zone_id=zone.name.lower().replace(" ", "_"),
                provider="gedi",
                scene_id=granule_id,
                file_paths=file_paths,
                available_bands=available_bands,
                metadata=metadata,
            )

            metadata_file = granule_dir / "gedi_metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            logger.info(
                "✅ GEDI granule processed: %s metrics extracted", len(available_bands)
            )
            return scene_data

        except Exception as exc:  # noqa: BLE001
            logger.error("Error processing GEDI granule %s: %s", granule["id"], exc)
            return None

    def download_gedi_hdf5(self, url: str, output_dir: Path) -> Optional[Path]:
        logger.debug(f"Entering download_gedi_hdf5 for URL: {url}, Output Dir: {output_dir}")
        try:
            filename = url.split("/")[-1]
            output_file = output_dir / filename
            logger.debug(f"Output file path constructed: {output_file}")

            if output_file.exists():
                logger.info(f"GEDI file already exists, attempting to validate: {output_file}")
                # Validate existing file
                try:
                    with h5py.File(output_file, "r") as f_val:
                        keys = list(f_val.keys())
                        logger.debug(f"Existing HDF5 {filename} opened successfully. Keys: {keys}")
                        if len(keys) == 0:
                            logger.warning(f"Existing HDF5 file {filename} is empty (no keys). Will attempt re-download.")
                            # Continue to download logic by not returning here.
                        else:
                            logger.info(f"Existing HDF5 file {filename} seems valid. Size: {output_file.stat().st_size} bytes. Skipping download.")
                            logger.debug(f"Returning existing file: {output_file}")
                            return output_file
                except Exception as exc_val:
                    logger.warning(f"Existing HDF5 file {filename} is not valid ({exc_val}). Will attempt re-download.", exc_info=True)
                    # Continue to download logic

            logger.info(f"Attempting to download GEDI HDF5: {filename} from {url}")
            logger.debug(f"Initiating self.session.get() for {url}")
            response = self.session.get(url, stream=True, timeout=600)
            logger.debug(f"Response status code: {response.status_code}, Content-Length: {response.headers.get('Content-Length')}")
            response.raise_for_status() # Will raise an HTTPError if status is 4xx/5xx

            logger.debug(f"Starting file write to {output_file}")
            with open(output_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.debug(f"File writing complete for {output_file}")

            if output_file.exists():
                file_size = output_file.stat().st_size
                logger.debug(f"File {output_file} exists after download. Size: {file_size} bytes.")
                if file_size == 0:
                    logger.warning(f"Downloaded file {filename} is empty (0 bytes). Deleting and returning None.")
                    output_file.unlink()
                    logger.debug(f"Returning None for empty file {filename}")
                    return None
            else:
                logger.warning(f"File {output_file} does not exist after download attempt. Returning None.")
                return None

            logger.debug(f"Starting HDF5 validation for {output_file}")
            try:
                with h5py.File(output_file, "r") as f_val:
                    keys = list(f_val.keys())
                    logger.debug(f"HDF5 {filename} opened successfully after download. Keys: {keys}")
                    if len(keys) == 0:
                        logger.warning(
                            f"Downloaded HDF5 file {filename} is empty (no keys) after validation. Deleting."
                        )
                        output_file.unlink()
                        logger.debug(f"Returning None for HDF5 file with no keys: {filename}")
                        return None
            except Exception as exc_h5_val:
                logger.error(f"Downloaded file {filename} is not a valid HDF5: {exc_h5_val}", exc_info=True)
                if output_file.exists():
                    output_file.unlink()
                logger.debug(f"Returning None for invalid HDF5: {filename}")
                return None

            logger.info(f"✅ Successfully downloaded and validated GEDI HDF5: {filename}")
            logger.debug(f"Returning successfully downloaded file: {output_file}")
            return output_file

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTPError during GEDI HDF5 download from {url}: {http_err}", exc_info=True)
            if http_err.response.status_code == 401:
                logger.error("HTTP 401 Unauthorized: This indicates an authentication failure. "
                             "Please ensure your NASA Earthdata login credentials are correctly "
                             "configured, either via EARTHDATA_USERNAME and EARTHDATA_PASSWORD "
                             "environment variables or a .netrc file in your home directory. "
                             "A .netrc file is often required for script-based access.")
            elif http_err.response.status_code == 403:
                logger.error("HTTP 403 Forbidden: You are authenticated, but do not have permission "
                             "to access the requested resource. Please check your data subscriptions "
                             "or permissions on NASA Earthdata.")
            logger.debug(f"Returning None due to HTTPError for {url}")
            return None
        except Exception as exc:
            logger.error(f"Generic error downloading GEDI HDF5 from {url}: {exc}", exc_info=True)
            logger.debug(f"Returning None due to generic error for {url}")
            return None

    def extract_archaeological_metrics(
        self, hdf5_file: Path, zone: Any
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract archaeological metrics from a GEDI HDF5 file."""

        try:
            logger.info(f"ENTERING extract_archaeological_metrics for {hdf5_file.name}") # New prominent log
            extracted: Dict[str, np.ndarray] = {}
            with h5py.File(hdf5_file, "r") as f:
                logger.debug(f"Inspecting HDF5 file: {hdf5_file.name}")
                beam_groups = [key for key in f.keys() if key.startswith("BEAM")]
                logger.debug(f"Found {len(beam_groups)} beam groups: {beam_groups}")
                if not beam_groups:
                    logger.warning("No beam groups found in %s", hdf5_file)
                    return None

                all_lats: List[float] = []
                all_lons: List[float] = []
                all_rh95: List[float] = []
                all_rh100: List[float] = []
                all_ground: List[float] = []
                all_quality: List[float] = []

                for beam in beam_groups:
                    try:
                        logger.debug(f"Processing beam: {beam}")
                        beam_group = f[beam]
                        essential_datasets = ["lat_lowestmode", "lon_lowestmode", "rh", "elev_lowestmode", "quality_flag"]
                        present_datasets = [ds for ds in essential_datasets if ds in beam_group]
                        missing_datasets = [ds for ds in essential_datasets if ds not in beam_group]
                        if "rh" in present_datasets and not isinstance(beam_group["rh"], h5py.Group):
                            logger.warning(f"Dataset 'rh' in beam {beam} is not a group as expected.")
                            present_datasets.remove("rh") # Treat as missing if not a group
                            missing_datasets.append("rh (not a group)")
                        elif "rh" in present_datasets:
                            rh_sub_datasets = ["rh_95", "rh_100"] # Example sub-datasets under 'rh' group
                            rh_present_sub = [sd for sd in rh_sub_datasets if sd in beam_group["rh"]]
                            rh_missing_sub = [sd for sd in rh_sub_datasets if sd not in beam_group["rh"]]
                            logger.debug(f"Beam {beam} 'rh' group check: Present sub-datasets: {rh_present_sub}, Missing sub-datasets: {rh_missing_sub}")
                        logger.debug(f"Beam {beam}: Present essential datasets: {present_datasets}. Missing: {missing_datasets}.")

                        if "lat_lowestmode" not in beam_group or "lon_lowestmode" not in beam_group:
                            logger.warning(f"Beam {beam} is missing lat_lowestmode or lon_lowestmode. Skipping beam.")
                            continue

                        if (
                            "lat_lowestmode" in beam_group
                            and "lon_lowestmode" in beam_group
                        ):
                            lats = beam_group["lat_lowestmode"][:]
                            lons = beam_group["lon_lowestmode"][:]
                            logger.debug(f"Beam {beam}: Found {len(lats)} points before zone filtering.")

                            rh95 = beam_group.get("rh", {}).get("rh_95", [])
                            rh100 = beam_group.get("rh", {}).get("rh_100", [])
                            ground = beam_group.get("elev_lowestmode", [])
                            quality = beam_group.get("quality_flag", [])

                            mask = self.filter_points_in_zone(lats, lons, zone)
                            logger.debug(f"Beam {beam}: Found {np.sum(mask)} points after zone filtering.")

                            if np.any(mask):
                                current_lats = lats[mask]
                                current_lons = lons[mask]
                                num_points_in_beam_after_filter = len(current_lats)

                                all_lats.extend(current_lats)
                                all_lons.extend(current_lons)

                                # For optional datasets, ensure they have the same length as lats before masking
                                rh_group = beam_group.get("rh")
                                if rh_group and isinstance(rh_group, h5py.Group):
                                    rh95_data = rh_group.get("rh_95")
                                    if rh95_data is not None and len(rh95_data) == len(lats):
                                        all_rh95.extend(rh95_data[:][mask])
                                    else: # Pad with NaNs if missing or mismatched length for this beam
                                        all_rh95.extend([np.nan] * num_points_in_beam_after_filter)
                                        if rh95_data is None: logger.debug(f"Beam {beam}: rh/rh_95 not found or None.")
                                        elif len(rh95_data) != len(lats): logger.debug(f"Beam {beam}: rh/rh_95 length mismatch (expected {len(lats)}, got {len(rh95_data)}).")

                                    rh100_data = rh_group.get("rh_100")
                                    if rh100_data is not None and len(rh100_data) == len(lats):
                                        all_rh100.extend(rh100_data[:][mask])
                                    else: # Pad with NaNs
                                        all_rh100.extend([np.nan] * num_points_in_beam_after_filter)
                                        if rh100_data is None: logger.debug(f"Beam {beam}: rh/rh_100 not found or None.")
                                        elif len(rh100_data) != len(lats): logger.debug(f"Beam {beam}: rh/rh_100 length mismatch.")

                                else: # rh group itself is missing or not a group
                                    all_rh95.extend([np.nan] * num_points_in_beam_after_filter)
                                    all_rh100.extend([np.nan] * num_points_in_beam_after_filter)
                                    if not rh_group: logger.debug(f"Beam {beam}: 'rh' group not found.")
                                    elif not isinstance(rh_group, h5py.Group): logger.debug(f"Beam {beam}: 'rh' dataset is not a group.")

                                ground_data = beam_group.get("elev_lowestmode")
                                if ground_data is not None and len(ground_data) == len(lats):
                                    all_ground.extend(ground_data[:][mask])
                                else: # Pad with NaNs
                                    all_ground.extend([np.nan] * num_points_in_beam_after_filter)
                                    if ground_data is None: logger.debug(f"Beam {beam}: elev_lowestmode not found or None.")
                                    elif len(ground_data) != len(lats): logger.debug(f"Beam {beam}: elev_lowestmode length mismatch.")

                                quality_data = beam_group.get("quality_flag")
                                if quality_data is not None and len(quality_data) == len(lats):
                                    all_quality.extend(quality_data[:][mask])
                                else: # Pad with a default/NaN quality flag, e.g., -1 or np.nan
                                    all_quality.extend([np.nan] * num_points_in_beam_after_filter) # Or use 0 or -1 if NaN is problematic for int arrays later
                                    if quality_data is None: logger.debug(f"Beam {beam}: quality_flag not found or None.")
                                    elif len(quality_data) != len(lats): logger.debug(f"Beam {beam}: quality_flag length mismatch.")
                    except Exception as exc:  # noqa: BLE001
                        logger.error(f"Error processing beam {beam} in {hdf5_file.name}: {exc}", exc_info=True)

                if len(all_lats) > 0:
                    extracted["coordinates"] = np.column_stack((all_lats, all_lons))

                    # Only add other arrays if they have meaningful data (not all NaNs)
                    # and convert lists to numpy arrays here.
                    if any(not np.isnan(x) for x in all_rh95): # Check if there's at least one non-NaN value
                        extracted["canopy_height_95"] = np.array(all_rh95)
                    else:
                        logger.debug("No valid rh95 data collected across all beams.")

                    if any(not np.isnan(x) for x in all_rh100):
                        extracted["canopy_height_100"] = np.array(all_rh100)
                    else:
                        logger.debug("No valid rh100 data collected across all beams.")

                    if any(not np.isnan(x) for x in all_ground):
                        extracted["ground_elevation"] = np.array(all_ground)
                    else:
                        logger.debug("No valid ground_elevation data collected across all beams.")

                    if any(not np.isnan(x) for x in all_quality): # Assuming quality_flags can be float due to NaNs
                        extracted["quality_flags"] = np.array(all_quality)
                    else:
                        logger.debug("No valid quality_flags data collected across all beams.")

                    # Keep the logic for canopy_gaps, elevation_anomalies, and detector calls,
                    # but ensure they can handle potential NaNs in their inputs if these arrays are added.
                    # For example, detect_canopy_gaps might need np.nanmean or np.nanstd.
                    # The placeholder detectors don't do much yet, so this is fine for now.
                    if "canopy_height_95" in extracted and "canopy_height_100" in extracted: # Check if keys exist after NaN filtering
                        extracted["canopy_gaps"] = self.detect_canopy_gaps(
                            extracted["canopy_height_95"], extracted["canopy_height_100"] # Pass the numpy arrays
                        )
                    if "ground_elevation" in extracted: # Check if key exists
                        extracted["elevation_anomalies"] = (
                            self.detect_elevation_anomalies(extracted["ground_elevation"]) # Pass the numpy array
                        )

                    # Advanced detection using dedicated algorithms
                    if "coordinates" in extracted and extracted["coordinates"].size > 0:
                        try:
                            from src.detectors.gedi_detector import (
                                detect_archaeological_clearings,
                                detect_archaeological_earthworks,
                            )

                            # Prepare inputs for detectors, using .get for safety, providing empty arrays if not present
                            # The detectors themselves also handle empty/NaN inputs.
                            rh95_input = extracted.get("canopy_height_95", np.array([]))
                            rh100_input = extracted.get("canopy_height_100", np.array([]))
                            ground_elev_input = extracted.get("ground_elevation", np.array([]))
                            coords_input = extracted["coordinates"] # Already checked for presence

                            logger.debug(f"Calling clearing detector. RH95 size: {rh95_input.size}, RH100 size: {rh100_input.size}, Coords size: {coords_input.size}")
                            clearing_results = detect_archaeological_clearings(
                                rh95_input,
                                rh100_input,
                                coords_input
                            )
                            # The detector returns a dict like {"potential_clearings": [...], "feature_type": "clearing"}
                            extracted["potential_clearings"] = clearing_results.get("potential_clearings", [])
                            logger.info(f"Found {len(extracted['potential_clearings'])} potential clearing points.")

                            logger.debug(f"Calling earthwork detector. Ground elev size: {ground_elev_input.size}, Coords size: {coords_input.size}")
                            earthwork_results = detect_archaeological_earthworks(
                                ground_elev_input,
                                coords_input
                            )
                            # The detector returns a dict like {"potential_earthworks": [...], "feature_type": "earthwork"}
                            extracted["potential_earthworks"] = earthwork_results.get("potential_earthworks", [])
                            logger.info(f"Found {len(extracted['potential_earthworks'])} potential earthwork points.")

                        except ImportError as imp_err:
                            logger.error(f"Could not import GEDI detectors: {imp_err}")
                        except Exception as exc:
                            logger.error(f"Error during GEDI advanced detection: {exc}", exc_info=True)
                    else:
                        logger.debug("Skipping advanced GEDI detection as no coordinate data was extracted.")

                if extracted:
                    logger.info(f"✅ Extracted {len(extracted)} types of metrics from {hdf5_file.name} (e.g., {list(extracted.keys())[:3]}) with {len(extracted.get('coordinates', []))} data points.")
                else:
                    logger.warning(f"No metrics extracted from {hdf5_file.name}.")
                return extracted if extracted else None

        except Exception as exc:  # noqa: BLE001
            logger.error("Error extracting GEDI metrics: %s", exc)
            return None

    def filter_points_in_zone(
        self, lats: np.ndarray, lons: np.ndarray, zone: Any
    ) -> np.ndarray:
        """Filter GEDI points that fall within the target zone."""

        try:
            bbox = zone.bbox
            lat_mask = (lats >= bbox[0]) & (lats <= bbox[2])
            lon_mask = (lons >= bbox[1]) & (lons <= bbox[3])
            return lat_mask & lon_mask
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error filtering points for zone: %s", exc)
            return np.zeros(len(lats), dtype=bool)

    def detect_canopy_gaps(self, rh95: List[float], rh100: List[float]) -> np.ndarray:
        """Detect potential canopy gaps from RH metrics."""

        try:
            rh95_array = np.array(rh95)
            rh100_array = np.array(rh100)
            gap_threshold = 15.0
            gaps = (rh95_array < gap_threshold) | (rh100_array < gap_threshold)
            return gaps.astype(float)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error detecting canopy gaps: %s", exc)
            return np.array([])

    def detect_elevation_anomalies(self, ground_elevations: List[float]) -> np.ndarray:
        """Detect elevation anomalies from ground elevation data."""

        try:
            elev_array = np.array(ground_elevations)
            mean_elev = np.mean(elev_array)
            std_elev = np.std(elev_array)
            anomaly_threshold = 1.5 * std_elev
            anomalies = np.abs(elev_array - mean_elev) > anomaly_threshold
            return anomalies.astype(float)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error detecting elevation anomalies: %s", exc)
            return np.array([])

    def save_as_geotiff(self, data_array: np.ndarray, output_file: Path) -> None:
        """Save GEDI data as a simple numpy-based GeoTIFF placeholder."""

        try:
            np.save(output_file.with_suffix(".npy"), data_array)
            metadata = {
                "data_type": "gedi_point_data",
                "array_shape": data_array.shape,
                "data_format": "numpy_array",
                "description": "GEDI archaeological metrics",
                "coordinate_system": "WGS84",
            }
            with open(output_file.with_suffix(".json"), "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error saving GEDI data as GeoTIFF: %s", exc)


def test_gedi_provider() -> bool:
    """Test GEDI provider integration with existing pipeline."""

    try:
        provider = GEDIProvider()
        test_zones = ["negro_madeira"]
        scenes = provider.download_data(test_zones, max_scenes=1)
        if scenes:
            scene = scenes[0]
            logger.info("✅ GEDI provider test successful!")
            logger.info("  Scene ID: %s", scene.scene_id)
            logger.info("  Provider: %s", scene.provider)
            logger.info("  Available bands: %s", scene.available_bands)
            logger.info("  Metadata keys: %s", list(scene.metadata.keys()))
            return True
        logger.warning("No GEDI scenes returned")
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("GEDI provider test failed: %s", exc)
        return False


if __name__ == "__main__":
    print("🛰️ Testing GEDI Provider...")
    SUCCESS = test_gedi_provider()
    if SUCCESS:
        print("✅ GEDI provider ready for archaeological discovery!")
    else:
        print("❌ GEDI provider test failed")
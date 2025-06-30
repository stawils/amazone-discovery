from typing import List, Dict, Optional, Any
from pathlib import Path
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

# Removed: from .config import SATELLITE_DIR, LIDAR_DIR

logger = logging.getLogger(__name__)

class SceneData:
    """
    Standardized data object representing a single downloaded scene for archaeological analysis.
    Encapsulates all relevant metadata, file paths, and feature/band availability.
    Now includes a dedicated path for a processed composite file.
    """
    def __init__(
        self,
        zone_id: str,
        provider: str,
        scene_id: str,
        file_paths: Dict[str, Path],  # Paths to individual raw/source bands
        available_bands: List[str], # List of keys in file_paths that are available
        metadata: Optional[Dict[str, Any]] = None,
        features: Optional[Dict[str, Any]] = None,
        composite_file_path: Optional[Path] = None, # Path to the main processed composite file
        provider_name: Optional[str] = None # Added to link back to the provider
    ):
        """
        Args:
            zone_id: Identifier for the target zone
            provider: Data provider (e.g., 'gee', 'sentinel2')
            scene_id: Unique scene identifier (provider-specific)
            file_paths: Mapping of band/feature names to file paths of raw/source bands
            available_bands: List of available bands/features (e.g., ['blue', 'green', ...])
            metadata: Additional metadata (acquisition date, cloud cover, etc.)
            features: Optional, provider-specific features (e.g., GEE indices)
            composite_file_path: Optional path to a processed multi-band composite file.
            provider_name: Optional name of the provider
        """
        self.zone_id = zone_id
        self.provider = provider
        self.scene_id = scene_id
        self.file_paths = file_paths # Should refer to raw/source bands
        self.available_bands = available_bands # Corresponds to keys in file_paths
        self.metadata = metadata or {}
        self.features = features or {}
        self.composite_file_path = composite_file_path
        self.provider_name = provider_name

    def has_band(self, band: str) -> bool:
        """Check if a given raw band/feature is available for this scene."""
        return band in self.available_bands and band in self.file_paths

    def get_band_path(self, band: str) -> Optional[Path]:
        """Get the file path for a given raw band/feature, if available."""
        return self.file_paths.get(band)

    def __repr__(self):
        parts = [
            f"SceneData(zone_id='{self.zone_id}'",
            f"provider='{self.provider}'",
            f"scene_id='{self.scene_id}'",
            f"available_bands={self.available_bands}",
            f"raw_band_files_count={len(self.file_paths)}"
        ]
        if self.composite_file_path:
            parts.append(f"composite_file_path='{self.composite_file_path.name}'")
        else:
            parts.append("composite_file_path=None")
        if self.metadata.get('acquisition_date'):
             date_str = self.metadata['acquisition_date']
             if isinstance(date_str, datetime): # Ensure it's str for repr
                 date_str = date_str.isoformat()
             parts.append(f"date='{date_str.split('T')[0]}'") # Just date part
        parts.append(f"metadata_keys={list(self.metadata.keys())}")
        return ", ".join(parts) + ")"

    def get_default_path(self, provider_name: str) -> Path:
        """Returns a default path for storing/finding this scene's data."""
        # Import here to avoid circular dependency with config.py
        from .config import SATELLITE_DIR, LIDAR_DIR
        
        # Determine base directory based on provider type
        if "gedi" in provider_name.lower():
            base_dir = LIDAR_DIR / "gedi"
        elif "sentinel" in provider_name.lower():
            base_dir = SATELLITE_DIR / "sentinel2"
        elif "landsat" in provider_name.lower():
            base_dir = SATELLITE_DIR / "landsat"
        else:
            # Default to satellite directory for unknown providers
            base_dir = SATELLITE_DIR / provider_name.lower()
        
        # Create zone-specific subdirectory
        zone_dir = base_dir / self.zone_id
        
        # Return path with scene ID as filename
        if self.composite_file_path:
            return self.composite_file_path
        else:
            # Default composite filename
            return zone_dir / f"{self.scene_id}_composite.tif"

class BaseProvider(ABC):
    """
    Abstract base class for satellite data providers.
    All providers must implement the download_data method, returning a list of SceneData objects.
    """
    @abstractmethod
    def download_data(self, zones: List[str], max_scenes: int = 3) -> List[SceneData]:
        """
        Download or process data for the given zones.
        Args:
            zones: List of zone IDs to process.
            max_scenes: Maximum number of scenes per zone.
        Returns:
            List of SceneData objects for all processed scenes.
        """
        pass 
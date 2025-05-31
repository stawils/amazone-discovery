from typing import List, Dict, Optional, Any
from pathlib import Path
from abc import ABC, abstractmethod

class SceneData:
    """
    Standardized data object representing a single downloaded scene for archaeological analysis.
    Encapsulates all relevant metadata, file paths, and feature/band availability.
    """
    def __init__(
        self,
        zone_id: str,
        provider: str,
        scene_id: str,
        file_paths: Dict[str, Path],  # e.g. {'blue': Path(...), 'nir': Path(...)}
        available_bands: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        features: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            zone_id: Identifier for the target zone
            provider: Data provider (e.g., 'gee')
            scene_id: Unique scene identifier (provider-specific)
            file_paths: Mapping of band/feature names to file paths
            available_bands: List of available bands/features (e.g., ['blue', 'green', ...])
            metadata: Additional metadata (acquisition date, cloud cover, etc.)
            features: Optional, provider-specific features (e.g., GEE indices)
        """
        self.zone_id = zone_id
        self.provider = provider
        self.scene_id = scene_id
        self.file_paths = file_paths
        self.available_bands = available_bands
        self.metadata = metadata or {}
        self.features = features or {}

    def has_band(self, band: str) -> bool:
        """Check if a given band/feature is available for this scene."""
        return band in self.available_bands

    def get_band_path(self, band: str) -> Optional[Path]:
        """Get the file path for a given band/feature, if available."""
        return self.file_paths.get(band)

    def __repr__(self):
        return (
            f"SceneData(zone_id={self.zone_id}, provider={self.provider}, scene_id={self.scene_id}, "
            f"bands={self.available_bands}, files={list(self.file_paths.keys())})"
        )

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
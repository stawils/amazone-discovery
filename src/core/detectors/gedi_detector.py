"""GEDI LiDAR detection algorithms for archaeological analysis."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path
import logging

from ..config import GEDIConfig, TargetZone

logger = logging.getLogger(__name__)


def cluster_nearby_points(
    points: np.ndarray, min_cluster_size: int = 5, eps: float = 0.001
) -> List[Dict[str, Any]]:
    """Cluster nearby points using DBSCAN."""
    if len(points) == 0:
        return []
    clustering = DBSCAN(eps=eps, min_samples=min_cluster_size).fit(points)
    labels = clustering.labels_
    clusters = []
    for label in set(labels):
        if label == -1:
            continue
        cluster_points = points[labels == label]
        center = cluster_points.mean(axis=0)
        clusters.append({"center": tuple(center), "count": len(cluster_points)})
    return clusters


def cluster_elevation_features(
    points: np.ndarray, min_cluster_size: int = 3, eps: float = 0.001
) -> List[Dict[str, Any]]:
    """Cluster elevation anomaly points."""
    return cluster_nearby_points(points, min_cluster_size, eps)


def detect_linear_patterns(
    points: np.ndarray, min_points: int = 5
) -> List[Dict[str, Any]]:
    """Detect simple linear patterns using linear regression."""
    if len(points) < min_points:
        return []
    from sklearn.linear_model import LinearRegression

    X = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    if r2 > 0.8:
        return [{"coordinates": points.tolist(), "r2": float(r2)}]
    return []


def detect_archaeological_clearings(
    rh95_data: np.ndarray, rh100_data: np.ndarray, coordinates: np.ndarray | None = None
) -> Dict[str, Any]:
    """Detect potential archaeological clearings from canopy height metrics."""
    gap_threshold = GEDIConfig.CANOPY_GAP_THRESHOLD
    gaps95 = rh95_data < gap_threshold
    gaps100 = rh100_data < gap_threshold
    significant = gaps95 | gaps100
    clusters = []
    if coordinates is not None and len(coordinates) == len(significant):
        clusters = cluster_nearby_points(coordinates[significant])
    return {
        "gap_points": significant.astype(float),
        "gap_clusters": clusters,
        "archaeological_potential": len(clusters) * 2,
    }


def detect_archaeological_earthworks(
    elevation_data: np.ndarray, coordinates: np.ndarray
) -> Dict[str, Any]:
    """Detect elevation anomalies that may represent earthworks."""
    mean_elev = float(np.mean(elevation_data))
    std_elev = float(np.std(elevation_data))
    anomaly_threshold = GEDIConfig.ELEVATION_ANOMALY_THRESHOLD * std_elev
    high_anom = elevation_data > (mean_elev + anomaly_threshold)
    low_anom = elevation_data < (mean_elev - anomaly_threshold)

    mound_clusters = cluster_elevation_features(coordinates[high_anom])
    ditch_patterns = detect_linear_patterns(coordinates[low_anom])

    return {
        "mound_candidates": high_anom.astype(float),
        "ditch_candidates": low_anom.astype(float),
        "mound_clusters": mound_clusters,
        "linear_features": ditch_patterns,
        "archaeological_potential": len(mound_clusters) * 3 + len(ditch_patterns) * 2,
    }


class GEDIArchaeologicalDetector:
    """Detector that analyzes GEDI metric files."""

    def __init__(self, zone: TargetZone) -> None:
        self.zone = zone
        self.detection_results: Dict[str, Any] = {}

    def analyze_scene(self, scene_dir: Path) -> Dict[str, Any]:
        try:
            coords_file = scene_dir / "coordinates.npy"
            rh95_file = scene_dir / "canopy_height_95.npy"
            rh100_file = scene_dir / "canopy_height_100.npy"
            elev_file = scene_dir / "ground_elevation.npy"
            if (
                not coords_file.exists()
                or not rh95_file.exists()
                or not elev_file.exists()
            ):
                return {"success": False, "error": "Missing metric files"}

            coordinates = np.load(coords_file)
            rh95 = np.load(rh95_file)
            rh100 = np.load(rh100_file) if rh100_file.exists() else rh95
            ground = np.load(elev_file)

            clearing_results = detect_archaeological_clearings(rh95, rh100, coordinates)
            earthwork_results = detect_archaeological_earthworks(ground, coordinates)

            self.detection_results = {
                "success": True,
                "zone": self.zone.name,
                "clearing_results": clearing_results,
                "earthwork_results": earthwork_results,
                "total_features": len(clearing_results.get("gap_clusters", []))
                + len(earthwork_results.get("mound_clusters", [])),
            }

            return self.detection_results
        except Exception as exc:  # noqa: BLE001
            logger.error("GEDI detector error: %s", exc)
            return {"success": False, "error": str(exc)}

    def export_detections_to_geojson(self, output_path: Path) -> bool:
        """Export detection results to GeoJSON."""
        if not self.detection_results or not self.detection_results.get("success"):
            logger.warning("No successful detection results to export")
            return False

        import json

        features: List[Dict[str, Any]] = []
        for cluster in self.detection_results.get("clearing_results", {}).get(
            "gap_clusters", []
        ):
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [cluster["center"][0], cluster["center"][1]],
                    },
                    "properties": {
                        "feature_type": "canopy_gap_cluster",
                        "count": cluster["count"],
                    },
                }
            )

        for cluster in self.detection_results.get("earthwork_results", {}).get(
            "mound_clusters", []
        ):
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [cluster["center"][0], cluster["center"][1]],
                    },
                    "properties": {
                        "feature_type": "elevation_mound",
                        "count": cluster["count"],
                    },
                }
            )

        fc = {"type": "FeatureCollection", "features": features}
        with open(output_path, "w") as f:
            json.dump(fc, f)
        return True

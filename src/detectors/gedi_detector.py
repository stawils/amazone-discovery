import numpy as np
import logging

logger = logging.getLogger(__name__)

def detect_archaeological_clearings(
    canopy_height_rh95: np.ndarray,
    canopy_height_rh100: np.ndarray,
    coordinates: np.ndarray,
    ch_threshold: float = 5.0,  # Canopy height below 5m considered a potential clearing
    min_points_for_clearing: int = 3 # Minimum number of GEDI shots to define a clearing event
) -> dict:
    """
    Detects potential archaeological clearings based on GEDI canopy height metrics.
    Clearings are identified as points with canopy height below a specified threshold.
    """
    logger.info(f"Running detect_archaeological_clearings with threshold {ch_threshold}m")

    if coordinates.size == 0 or (canopy_height_rh95.size == 0 and canopy_height_rh100.size == 0) :
        logger.debug("Not enough data for clearing detection (coordinates or canopy height missing).")
        return {"potential_clearings": [], "feature_type": "clearing"}

    # Prioritize rh95 if available, else use rh100
    if canopy_height_rh95.size > 0:
        ch_metric = canopy_height_rh95
        logger.debug(f"Using RH95 for clearing detection. Points: {len(ch_metric)}")
    elif canopy_height_rh100.size > 0:
        ch_metric = canopy_height_rh100
        logger.debug(f"Using RH100 for clearing detection. Points: {len(ch_metric)}")
    else:
        logger.debug("No canopy height data available for clearing detection.")
        return {"potential_clearings": [], "feature_type": "clearing"}

    # Filter out NaN values first to avoid runtime warnings with comparisons
    valid_indices = ~np.isnan(ch_metric)
    ch_metric_filtered = ch_metric[valid_indices]
    coordinates_filtered = coordinates[valid_indices]

    if ch_metric_filtered.size == 0:
        logger.debug("No valid (non-NaN) canopy height data after filtering.")
        return {"potential_clearings": [], "feature_type": "clearing"}

    potential_clearing_indices = np.where(ch_metric_filtered < ch_threshold)[0]

    detected_clearings = []
    if len(potential_clearing_indices) >= min_points_for_clearing:
        # For simplicity, returning individual points. Clustering could be added here.
        for idx in potential_clearing_indices:
            detected_clearings.append({
                "coordinates": coordinates_filtered[idx].tolist(),
                "canopy_height": float(ch_metric_filtered[idx]),
                "description": "Potential clearing (low canopy height)"
            })
        logger.info(f"Detected {len(detected_clearings)} GEDI points indicative of clearings.")
    else:
        logger.info(f"Not enough GEDI points ({len(potential_clearing_indices)}) met clearing threshold for a significant event.")

    return {"potential_clearings": detected_clearings, "feature_type": "clearing"}

def detect_archaeological_earthworks(
    ground_elevation: np.ndarray,
    coordinates: np.ndarray,
    z_score_threshold: float = 2.0, # Points with elevation Z-score > 2 or < -2
    min_points_for_earthwork: int = 2 # Minimum number of GEDI shots to define an earthwork event
) -> dict:
    """
    Detects potential archaeological earthworks (e.g., mounds, depressions)
    based on GEDI ground elevation anomalies.
    Anomalies are identified as points where elevation significantly deviates
    from the mean (using Z-score).
    """
    logger.info(f"Running detect_archaeological_earthworks with Z-score threshold {z_score_threshold}")

    if ground_elevation.size == 0 or coordinates.size == 0:
        logger.debug("Not enough data for earthwork detection (ground elevation or coordinates missing).")
        return {"potential_earthworks": [], "feature_type": "earthwork"}

    # Filter out NaN values
    valid_indices = ~np.isnan(ground_elevation)
    ground_elevation_filtered = ground_elevation[valid_indices]
    coordinates_filtered = coordinates[valid_indices]

    if ground_elevation_filtered.size < min_points_for_earthwork: # Need enough points to calculate mean/std
        logger.debug(f"Not enough valid (non-NaN) ground elevation points ({len(ground_elevation_filtered)}) for meaningful Z-score calculation.")
        return {"potential_earthworks": [], "feature_type": "earthwork"}

    mean_elev = np.mean(ground_elevation_filtered)
    std_elev = np.std(ground_elevation_filtered)

    logger.debug(f"Earthwork detection: Mean elevation={mean_elev}, Std Dev={std_elev}")

    if std_elev == 0: # Avoid division by zero if all elevations are the same
        logger.debug("Standard deviation of elevation is zero. Cannot calculate Z-scores.")
        return {"potential_earthworks": [], "feature_type": "earthwork"}

    z_scores = (ground_elevation_filtered - mean_elev) / std_elev

    potential_earthwork_indices = np.where(np.abs(z_scores) > z_score_threshold)[0]

    detected_earthworks = []
    if len(potential_earthwork_indices) >= min_points_for_earthwork:
        # For simplicity, returning individual points. Spatial analysis could be added.
        for idx in potential_earthwork_indices:
            detected_earthworks.append({
                "coordinates": coordinates_filtered[idx].tolist(),
                "elevation": float(ground_elevation_filtered[idx]),
                "z_score": float(z_scores[idx]),
                "description": "Potential earthwork (anomalous elevation)"
            })
        logger.info(f"Detected {len(detected_earthworks)} GEDI points indicative of earthworks.")
    else:
        logger.info(f"Not enough GEDI points ({len(potential_earthwork_indices)}) met earthwork Z-score threshold for a significant event.")

    return {"potential_earthworks": detected_earthworks, "feature_type": "earthwork"}

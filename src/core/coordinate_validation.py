"""
Unified Coordinate Validation System
Ensures all maps use consistent geographic coordinates (WGS84) across the entire pipeline
"""

import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CoordinateBounds:
    """Geographic bounds for different regions"""
    min_lon: float
    max_lon: float
    min_lat: float
    max_lat: float
    name: str

# Define regional coordinate bounds
AMAZON_BOUNDS = CoordinateBounds(-80.0, -44.0, -20.0, 10.0, "Amazon Basin")

ZONE_BOUNDS = {
    'upper_napo': CoordinateBounds(-74.0, -71.0, -1.5, 0.5, "Upper Napo"),
    'upper_napo_micro': CoordinateBounds(-73.0, -72.0, -1.0, 0.0, "Upper Napo Micro"),
    'upper_napo_micro_small': CoordinateBounds(-72.8, -72.2, -0.8, -0.2, "Upper Napo Micro Small"),
    'negro_madeira': CoordinateBounds(-62.0, -58.0, -4.0, -2.0, "Negro-Madeira"),
    'trombetas': CoordinateBounds(-58.0, -54.0, -2.5, -0.5, "Trombetas"),
    'upper_xingu': CoordinateBounds(-55.0, -54.0, -12.5, -11.0, "Upper Xingu"),
    'maranon': CoordinateBounds(-76.0, -74.0, -5.0, -3.0, "Marañón")
}

def validate_coordinates(
    lon: float, 
    lat: float, 
    context: str = "",
    zone_id: Optional[str] = None,
    strict: bool = False
) -> bool:
    """
    Validate geographic coordinates against Amazon/zone bounds
    
    Args:
        lon: Longitude (WGS84)
        lat: Latitude (WGS84)  
        context: Description for logging
        zone_id: Optional zone for stricter validation
        strict: If True, use zone bounds; if False, use Amazon bounds
    
    Returns:
        True if coordinates are valid
    """
    # Check for NaN or None
    if lon is None or lat is None or np.isnan(lon) or np.isnan(lat):
        logger.warning(f"Invalid coordinates (None/NaN): lon={lon}, lat={lat} for {context}")
        return False
    
    # Check for obviously wrong coordinates (UTM values are much larger)
    if abs(lon) > 180 or abs(lat) > 90:
        logger.error(f"Coordinates appear to be in UTM format: lon={lon}, lat={lat} for {context}")
        return False
    
    # Choose bounds based on strictness and zone availability
    if strict and zone_id and zone_id in ZONE_BOUNDS:
        bounds = ZONE_BOUNDS[zone_id]
        logger.debug(f"Using strict {bounds.name} bounds for validation")
    else:
        bounds = AMAZON_BOUNDS
        
    # Validate against bounds
    if not (bounds.min_lon <= lon <= bounds.max_lon):
        logger.warning(f"Longitude {lon:.6f} outside {bounds.name} bounds [{bounds.min_lon}, {bounds.max_lon}] for {context}")
        return False
        
    if not (bounds.min_lat <= lat <= bounds.max_lat):
        logger.warning(f"Latitude {lat:.6f} outside {bounds.name} bounds [{bounds.min_lat}, {bounds.max_lat}] for {context}")
        return False
        
    return True

def validate_coordinate_array(
    coordinates: np.ndarray, 
    context: str = "",
    zone_id: Optional[str] = None,
    remove_invalid: bool = True
) -> np.ndarray:
    """
    Validate array of coordinates and optionally filter invalid ones
    
    Args:
        coordinates: Array of shape (N, 2) with [lon, lat] pairs
        context: Description for logging
        zone_id: Optional zone for validation
        remove_invalid: If True, filter out invalid coordinates
    
    Returns:
        Validated coordinate array (possibly filtered)
    """
    if coordinates.size == 0:
        return coordinates
        
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        logger.error(f"Invalid coordinate array shape {coordinates.shape} for {context} - expected (N, 2)")
        return np.array([]).reshape(0, 2)
    
    # Choose bounds
    bounds = ZONE_BOUNDS.get(zone_id, AMAZON_BOUNDS)
    
    # Vectorized validation
    valid_lon = (coordinates[:, 0] >= bounds.min_lon) & (coordinates[:, 0] <= bounds.max_lon)
    valid_lat = (coordinates[:, 1] >= bounds.min_lat) & (coordinates[:, 1] <= bounds.max_lat)
    valid_mask = valid_lon & valid_lat
    
    # Check for UTM coordinates (typically > 180 in absolute value)
    utm_mask = (np.abs(coordinates[:, 0]) > 180) | (np.abs(coordinates[:, 1]) > 90)
    if np.any(utm_mask):
        logger.error(f"Found {np.sum(utm_mask)} UTM coordinates in {context} - these need conversion to geographic")
        valid_mask = valid_mask & ~utm_mask
    
    invalid_count = np.sum(~valid_mask)
    if invalid_count > 0:
        logger.warning(f"Found {invalid_count}/{len(coordinates)} invalid coordinates in {context}")
        
        if remove_invalid:
            coordinates = coordinates[valid_mask]
            logger.info(f"Filtered to {len(coordinates)} valid coordinates for {context}")
        
    return coordinates

def fix_invalid_coordinates(
    coordinates: Union[List[float], Tuple[float, float]], 
    zone_id: str,
    context: str = ""
) -> List[float]:
    """
    Provide fallback coordinates for invalid ones based on zone center
    
    Args:
        coordinates: Invalid coordinate pair
        zone_id: Zone identifier for fallback selection  
        context: Description for logging
        
    Returns:
        Valid [lon, lat] coordinates
    """
    logger.warning(f"Fixing invalid coordinates {coordinates} for {context} in zone {zone_id}")
    
    # Zone center fallbacks
    zone_centers = {
        'upper_napo': [-72.5, -0.5],
        'upper_napo_micro': [-72.5, -0.5], 
        'upper_napo_micro_small': [-72.5, -0.5],
        'negro_madeira': [-60.0, -3.0],
        'trombetas': [-56.0, -1.5],
        'upper_xingu': [-54.5, -11.7],
        'maranon': [-75.0, -4.0]
    }
    
    fallback = zone_centers.get(zone_id, [-60.0, -3.0])  # Default to Amazon center
    logger.info(f"Using fallback coordinates {fallback} for zone {zone_id}")
    return fallback

def ensure_geojson_format(coordinates: Union[List, Tuple, np.ndarray]) -> List[float]:
    """
    Ensure coordinates are in proper GeoJSON format [lon, lat]
    
    Args:
        coordinates: Coordinate pair in various formats
        
    Returns:
        [longitude, latitude] as required by GeoJSON spec
    """
    if isinstance(coordinates, np.ndarray):
        coordinates = coordinates.tolist()
    elif isinstance(coordinates, tuple):
        coordinates = list(coordinates)
        
    if not isinstance(coordinates, list) or len(coordinates) != 2:
        raise ValueError(f"Invalid coordinate format: {coordinates}")
        
    # Ensure numeric types
    lon, lat = float(coordinates[0]), float(coordinates[1])
    
    return [lon, lat]

def detect_coordinate_system(coordinates: np.ndarray) -> str:
    """
    Detect if coordinates are in geographic (WGS84) or projected (UTM) system
    
    Args:
        coordinates: Array of coordinate pairs
        
    Returns:
        'geographic', 'utm', or 'unknown'
    """
    if coordinates.size == 0:
        return 'unknown'
        
    # Check coordinate ranges
    lon_range = np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])
    lat_range = np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])
    
    # UTM coordinates typically have much larger values and ranges
    if np.any(np.abs(coordinates) > 1000):
        return 'utm'
    elif np.all(np.abs(coordinates[:, 0]) <= 180) and np.all(np.abs(coordinates[:, 1]) <= 90):
        return 'geographic'
    else:
        return 'unknown'

def standardize_coordinates_for_export(
    features: List[Dict[str, Any]], 
    zone_id: str,
    context: str = ""
) -> List[Dict[str, Any]]:
    """
    Standardize all coordinates in a feature list for export
    
    Args:
        features: List of feature dictionaries
        zone_id: Zone identifier
        context: Description for logging
        
    Returns:
        Features with validated and standardized coordinates
    """
    standardized_features = []
    
    for i, feature in enumerate(features):
        try:
            # Handle different coordinate keys
            coords = None
            coord_key = None
            
            for key in ['coordinates', 'center', 'location', 'position']:
                if key in feature and feature[key] is not None:
                    coords = feature[key]
                    coord_key = key
                    break
                    
            if coords is None:
                logger.warning(f"No coordinates found in feature {i} for {context}")
                continue
                
            # Validate and fix if needed
            if isinstance(coords, (list, tuple, np.ndarray)) and len(coords) >= 2:
                lon, lat = float(coords[0]), float(coords[1])
                
                if validate_coordinates(lon, lat, f"{context}_feature_{i}", zone_id):
                    # Coordinates are valid, ensure GeoJSON format
                    feature[coord_key] = ensure_geojson_format([lon, lat])
                    standardized_features.append(feature)
                else:
                    # Fix invalid coordinates
                    fixed_coords = fix_invalid_coordinates([lon, lat], zone_id, f"{context}_feature_{i}")
                    feature[coord_key] = fixed_coords
                    feature['coordinate_fixed'] = True
                    standardized_features.append(feature)
            else:
                logger.warning(f"Invalid coordinate format in feature {i}: {coords} for {context}")
                
        except Exception as e:
            logger.error(f"Error standardizing coordinates for feature {i} in {context}: {e}")
            continue
            
    logger.info(f"Standardized {len(standardized_features)}/{len(features)} features for {context}")
    return standardized_features
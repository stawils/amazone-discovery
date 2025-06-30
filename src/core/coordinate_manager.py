"""
Unified Coordinate Management System
Ensures single source of truth for all coordinate operations
No fallbacks, strict validation, complete control
"""

from __future__ import annotations
import logging
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from shapely.geometry import Point, LineString, Polygon

logger = logging.getLogger(__name__)


class CoordinateManager:
    """
    Single source of truth for all coordinate operations.
    Ensures all features have valid geographic coordinates from creation.
    """
    
    def __init__(self, transform: Any = None, crs: Any = None):
        """Initialize with raster transform and CRS"""
        self.transform = transform
        self.crs = crs
        self._validate_initialization()
    
    def _validate_initialization(self) -> None:
        """Validate that coordinate manager is properly initialized"""
        if not self.transform:
            raise ValueError("CoordinateManager requires valid raster transform")
        if not self.crs:
            raise ValueError("CoordinateManager requires valid CRS")
        
        # Test coordinate conversion to ensure it works
        try:
            test_lon, test_lat = self._pixel_to_geographic(100, 100)
            if not self._is_valid_geographic(test_lon, test_lat):
                raise ValueError(f"Invalid test conversion result: {test_lon}, {test_lat}")
        except Exception as e:
            raise ValueError(f"Coordinate conversion test failed: {e}")
    
    def _pixel_to_geographic(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates directly to geographic coordinates
        Returns: (longitude, latitude)
        """
        import rasterio.transform
        import pyproj
        
        try:
            # Step 1: Pixel to UTM
            utm_x, utm_y = rasterio.transform.xy(self.transform, pixel_y, pixel_x)
            
            # Step 2: UTM to Geographic
            transformer = pyproj.Transformer.from_crs(
                self.crs, 'EPSG:4326', always_xy=True
            )
            lon, lat = transformer.transform(utm_x, utm_y)
            
            return float(lon), float(lat)
            
        except Exception as e:
            raise ValueError(f"Failed to convert pixel ({pixel_x}, {pixel_y}) to geographic: {e}")
    
    def _is_valid_geographic(self, lon: float, lat: float) -> bool:
        """Validate that coordinates are valid geographic coordinates"""
        return (-180 <= lon <= 180) and (-90 <= lat <= 90)
    
    def create_point_feature(
        self, 
        pixel_x: float, 
        pixel_y: float, 
        properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a point feature with guaranteed geographic coordinates
        """
        # Convert to geographic coordinates
        lon, lat = self._pixel_to_geographic(pixel_x, pixel_y)
        
        # Validate coordinates
        if not self._is_valid_geographic(lon, lat):
            raise ValueError(f"Invalid geographic coordinates: {lon}, {lat}")
        
        # Create geometry and feature
        geometry = Point(lon, lat)
        
        feature = {
            'geometry': geometry,
            'coordinates': [lon, lat],  # GeoJSON format: [longitude, latitude]
            'pixel_coordinates': [pixel_x, pixel_y],
            **properties
        }
        
        logger.debug(f"Created point feature: pixel({pixel_x:.2f}, {pixel_y:.2f}) -> geo({lon:.6f}, {lat:.6f})")
        return feature
    
    def create_line_feature(
        self,
        pixel_coords: List[Tuple[float, float]],
        properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a line feature with guaranteed geographic coordinates
        """
        if len(pixel_coords) < 2:
            raise ValueError("Line feature requires at least 2 points")
        
        # Convert all points to geographic
        geo_coords = []
        for px_x, px_y in pixel_coords:
            lon, lat = self._pixel_to_geographic(px_x, px_y)
            if not self._is_valid_geographic(lon, lat):
                raise ValueError(f"Invalid geographic coordinates: {lon}, {lat}")
            geo_coords.append((lon, lat))
        
        # Create geometry
        geometry = LineString(geo_coords)
        
        # Calculate center point for coordinates field
        center_lon = sum(coord[0] for coord in geo_coords) / len(geo_coords)
        center_lat = sum(coord[1] for coord in geo_coords) / len(geo_coords)
        
        feature = {
            'geometry': geometry,
            'coordinates': [center_lon, center_lat],  # Line center
            'pixel_coordinates': pixel_coords,
            'geographic_line_coords': geo_coords,
            **properties
        }
        
        logger.debug(f"Created line feature: {len(pixel_coords)} points -> center({center_lon:.6f}, {center_lat:.6f})")
        return feature
    
    def create_polygon_feature(
        self,
        pixel_coords: List[Tuple[float, float]],
        properties: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a polygon feature with guaranteed geographic coordinates
        """
        if len(pixel_coords) < 3:
            raise ValueError("Polygon feature requires at least 3 points")
        
        # Convert all points to geographic
        geo_coords = []
        for px_x, px_y in pixel_coords:
            lon, lat = self._pixel_to_geographic(px_x, px_y)
            if not self._is_valid_geographic(lon, lat):
                raise ValueError(f"Invalid geographic coordinates: {lon}, {lat}")
            geo_coords.append((lon, lat))
        
        # Ensure polygon is closed
        if geo_coords[0] != geo_coords[-1]:
            geo_coords.append(geo_coords[0])
        
        # Create geometry
        geometry = Polygon(geo_coords)
        
        # Calculate centroid for coordinates field
        centroid = geometry.centroid
        
        feature = {
            'geometry': geometry,
            'coordinates': [centroid.x, centroid.y],  # Polygon centroid
            'pixel_coordinates': pixel_coords,
            'geographic_polygon_coords': geo_coords,
            **properties
        }
        
        logger.debug(f"Created polygon feature: {len(pixel_coords)} points -> centroid({centroid.x:.6f}, {centroid.y:.6f})")
        return feature
    
    def create_region_feature(
        self,
        region_mask: np.ndarray,
        labeled_mask: np.ndarray,
        region_id: int,
        properties: Dict[str, Any],
        existing_features: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a feature from a labeled region with guaranteed geographic coordinates
        Includes duplicate detection and area validation
        """
        # Find region pixels
        region_pixels = np.where(labeled_mask == region_id)
        if len(region_pixels[0]) == 0:
            raise ValueError(f"No pixels found for region {region_id}")
        
        # Calculate centroid in pixel coordinates
        centroid_y = float(np.mean(region_pixels[0]))
        centroid_x = float(np.mean(region_pixels[1]))
        
        # Convert to geographic coordinates
        lon, lat = self._pixel_to_geographic(centroid_x, centroid_y)
        if not self._is_valid_geographic(lon, lat):
            raise ValueError(f"Invalid geographic coordinates: {lon}, {lat}")
        
        # Additional coordinate validation for Amazon region
        if not self._is_valid_amazon_coordinates(lon, lat):
            raise ValueError(f"Coordinates {lon:.6f}, {lat:.6f} outside Amazon region")
        
        # Check for duplicates if existing features provided
        if existing_features:
            if self._is_duplicate_location(lon, lat, existing_features):
                logger.warning(f"Duplicate feature detected at ({lon:.6f}, {lat:.6f}) - skipping")
                raise ValueError(f"Duplicate feature at coordinates {lon:.6f}, {lat:.6f}")
        
        # Create geometry
        geometry = Point(lon, lat)
        
        # Calculate additional properties with validation
        area_pixels = len(region_pixels[0])
        area_m2 = properties.get('area_m2', area_pixels * 100)  # Fallback area calculation
        
        # Validate area is reasonable
        if not self._is_valid_area(area_m2):
            logger.warning(f"Feature {region_id} has suspicious area: {area_m2} m²")
        
        feature = {
            'geometry': geometry,
            'coordinates': [lon, lat],  # Geographic centroid
            'pixel_centroid': [centroid_x, centroid_y],
            'area_pixels': area_pixels,
            'region_id': region_id,
            'coordinate_validation': {
                'amazon_region': True,
                'geographic_valid': True,
                'area_validated': self._is_valid_area(area_m2)
            },
            **properties
        }
        
        logger.debug(f"Created region feature {region_id}: pixel({centroid_x:.2f}, {centroid_y:.2f}) -> geo({lon:.6f}, {lat:.6f})")
        return feature
    
    def validate_feature(self, feature: Dict[str, Any]) -> bool:
        """
        Validate that a feature has proper coordinates
        """
        # Check required fields
        if 'coordinates' not in feature:
            logger.error("Feature missing 'coordinates' field")
            return False
        
        if 'geometry' not in feature:
            logger.error("Feature missing 'geometry' field")
            return False
        
        # Validate coordinates
        coords = feature['coordinates']
        if not isinstance(coords, list) or len(coords) != 2:
            logger.error(f"Invalid coordinates format: {coords}")
            return False
        
        lon, lat = coords
        if not self._is_valid_geographic(lon, lat):
            logger.error(f"Invalid geographic coordinates: {lon}, {lat}")
            return False
        
        # Validate geometry consistency
        if hasattr(feature['geometry'], 'x') and hasattr(feature['geometry'], 'y'):
            geom_lon, geom_lat = feature['geometry'].x, feature['geometry'].y
            if not self._is_valid_geographic(geom_lon, geom_lat):
                logger.error(f"Invalid geometry coordinates: {geom_lon}, {geom_lat}")
                return False
            
            # Check consistency between coordinates and geometry
            if abs(lon - geom_lon) > 0.001 or abs(lat - geom_lat) > 0.001:
                logger.warning(f"Coordinates mismatch: coords={coords}, geometry=({geom_lon:.6f}, {geom_lat:.6f})")
        
        return True
    
    def _is_valid_amazon_coordinates(self, lon: float, lat: float) -> bool:
        """Validate coordinates are within Amazon basin bounds"""
        return (-80 <= lon <= -44) and (-20 <= lat <= 10)
    
    def _is_duplicate_location(self, lon: float, lat: float, existing_features: List[Dict[str, Any]], threshold_m: float = 100.0) -> bool:
        """Check if coordinates are too close to existing features (potential duplicate)"""
        threshold_deg = threshold_m / 111000  # Convert meters to approximate degrees
        
        for feature in existing_features:
            existing_coords = feature.get('coordinates', [])
            if len(existing_coords) == 2:
                existing_lon, existing_lat = existing_coords
                distance = ((lon - existing_lon)**2 + (lat - existing_lat)**2)**0.5
                if distance < threshold_deg:
                    return True
        return False
    
    def _is_valid_area(self, area_m2: float) -> bool:
        """Validate that area is within reasonable archaeological bounds"""
        return 1000 <= area_m2 <= 1000000  # 0.1 hectare to 100 hectares
    
    def detect_suspicious_patterns(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect systematic errors and suspicious patterns in feature collection"""
        if not features:
            return {'status': 'no_features'}
        
        # Extract areas
        areas = [f.get('area_m2', 0) for f in features if 'area_m2' in f]
        if not areas:
            return {'status': 'no_areas'}
        
        # Check for identical areas (processing artifacts)
        from collections import Counter
        area_counts = Counter(areas)
        identical_areas = {area: count for area, count in area_counts.items() if count > 5}
        
        # Check for impossible spatial arrangements
        coords = [f.get('coordinates', []) for f in features if 'coordinates' in f]
        valid_coords = [c for c in coords if len(c) == 2]
        
        min_distances = []
        if len(valid_coords) > 1:
            for i, coord1 in enumerate(valid_coords):
                distances_to_others = []
                for j, coord2 in enumerate(valid_coords):
                    if i != j:
                        distance_m = ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)**0.5 * 111000
                        distances_to_others.append(distance_m)
                if distances_to_others:
                    min_distances.append(min(distances_to_others))
        
        # Check for spatial impossibilities
        spatial_impossibilities = []
        for i, feature in enumerate(features):
            if 'area_m2' in feature and 'coordinates' in feature:
                area_m2 = feature['area_m2']
                feature_size_m = (area_m2)**0.5  # Approximate diameter
                
                if min_distances and i < len(min_distances):
                    min_distance = min_distances[i]
                    if feature_size_m > min_distance * 0.8:  # Feature larger than 80% of distance to nearest neighbor
                        spatial_impossibilities.append({
                            'feature_index': i,
                            'area_m2': area_m2,
                            'approximate_size_m': feature_size_m,
                            'min_distance_to_neighbor_m': min_distance,
                            'issue': 'Feature too large for spatial separation'
                        })
        
        return {
            'status': 'analyzed',
            'total_features': len(features),
            'identical_areas': identical_areas,
            'spatial_impossibilities': spatial_impossibilities,
            'suspicious_patterns_detected': len(identical_areas) > 0 or len(spatial_impossibilities) > 0
        }
    
    def get_coordinate_info(self) -> Dict[str, Any]:
        """Get information about the coordinate system"""
        return {
            'transform': str(self.transform) if self.transform else None,
            'crs': str(self.crs) if self.crs else None,
            'epsg_code': self.crs.to_epsg() if self.crs else None,
            'status': 'initialized'
        }
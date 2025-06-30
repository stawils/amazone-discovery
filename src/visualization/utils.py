"""
Utility Functions for Archaeological Visualization
Data processing and coordinate validation utilities
"""

import logging
import numpy as np
import geopandas as gpd
from typing import Dict, List, Tuple, Optional, Any
from shapely.geometry import Point, Polygon, box
import pyproj
from pyproj import Transformer

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes and analyzes archaeological data for visualization"""
    
    def __init__(self):
        """Initialize data processor"""
        self.amazon_utm_zones = {
            'north': 'EPSG:32618',  # UTM Zone 18N
            'central': 'EPSG:32619',  # UTM Zone 19N  
            'south': 'EPSG:32620'   # UTM Zone 20N
        }
    
    def calculate_optimal_bounds(self, map_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate optimal map bounds from all data sources"""
        
        all_geometries = []
        
        # Collect all geometries
        for data_type, data in map_data.items():
            if hasattr(data, 'geometry') and not data.empty:
                all_geometries.extend(data.geometry.tolist())
        
        if not all_geometries:
            logger.warning("No geometries found, using default Amazon bounds")
            return {
                'north': -2.0, 'south': -8.0,
                'east': -65.0, 'west': -75.0,
                'center_lat': -5.0, 'center_lon': -70.0,
                'optimal_zoom': 16  # High zoom even for default bounds
            }
        
        # Calculate bounds
        lats = [geom.y if hasattr(geom, 'y') else geom.centroid.y for geom in all_geometries]
        lons = [geom.x if hasattr(geom, 'x') else geom.centroid.x for geom in all_geometries]
        
        bounds = {
            'north': max(lats),
            'south': min(lats),
            'east': max(lons), 
            'west': min(lons),
            'center_lat': (max(lats) + min(lats)) / 2,
            'center_lon': (max(lons) + min(lons)) / 2
        }
        
        # Calculate optimal zoom level
        bounds['optimal_zoom'] = self._calculate_zoom_level(bounds)
        
        logger.info(f"📊 Calculated bounds: {bounds['south']:.3f}°S to {bounds['north']:.3f}°S, "
                   f"{bounds['west']:.3f}°W to {bounds['east']:.3f}°W")
        
        return bounds
    
    def _calculate_zoom_level(self, bounds: Dict[str, float]) -> int:
        """Calculate optimal zoom level based on bounds - Very high zoom for detailed satellite imagery"""
        
        lat_range = bounds['north'] - bounds['south']
        lon_range = bounds['east'] - bounds['west']
        max_range = max(lat_range, lon_range)
        
        # Much higher zoom levels for detailed archaeological feature detection
        if max_range > 5.0:
            return 14  # Increased from 8
        elif max_range > 2.0:
            return 16  # Increased from 10
        elif max_range > 1.0:
            return 17  # Increased from 12
        elif max_range > 0.5:
            return 18  # Increased from 14
        else:
            return 19  # Increased from 16 - Maximum detail for small areas
    
    def calculate_feature_density(self, map_data: Dict[str, Any], grid_size_km: float = 5.0) -> Dict[str, Any]:
        """Calculate feature density across the area"""
        
        density_analysis = {
            'total_features': 0,
            'features_per_km2': 0.0,
            'hotspots': [],
            'coverage_area_km2': 0.0
        }
        
        try:
            # Count total features
            for data_type, data in map_data.items():
                if hasattr(data, '__len__'):
                    density_analysis['total_features'] += len(data)
            
            # Calculate coverage area (simplified)
            bounds = self.calculate_optimal_bounds(map_data)
            lat_range = bounds['north'] - bounds['south']
            lon_range = bounds['east'] - bounds['west']
            
            # Approximate area calculation (not precise but good enough for density)
            area_km2 = abs(lat_range * lon_range) * 111.32 * 111.32  # Rough conversion
            density_analysis['coverage_area_km2'] = area_km2
            
            if area_km2 > 0:
                density_analysis['features_per_km2'] = density_analysis['total_features'] / area_km2
            
            logger.info(f"📈 Density: {density_analysis['features_per_km2']:.2f} features/km²")
            
        except Exception as e:
            logger.warning(f"Density calculation failed: {e}")
        
        return density_analysis
    
    def validate_data_quality(self, map_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality and completeness"""
        
        quality_report = {
            'total_datasets': len(map_data),
            'valid_datasets': 0,
            'total_features': 0,
            'invalid_coordinates': 0,
            'missing_attributes': 0,
            'quality_score': 0.0
        }
        
        for data_type, data in map_data.items():
            if hasattr(data, '__len__') and len(data) > 0:
                quality_report['valid_datasets'] += 1
                quality_report['total_features'] += len(data)
                
                # Check for invalid coordinates
                if hasattr(data, 'geometry'):
                    invalid_geoms = data.geometry.isna().sum()
                    quality_report['invalid_coordinates'] += invalid_geoms
        
        # Calculate quality score
        if quality_report['total_features'] > 0:
            valid_ratio = 1 - (quality_report['invalid_coordinates'] / quality_report['total_features'])
            completeness_ratio = quality_report['valid_datasets'] / max(quality_report['total_datasets'], 1)
            quality_report['quality_score'] = (valid_ratio + completeness_ratio) / 2
        
        logger.info(f"✅ Data quality score: {quality_report['quality_score']:.2f}")
        return quality_report


class CoordinateValidator:
    """Validates and transforms coordinates for archaeological features"""
    
    def __init__(self):
        """Initialize coordinate validator"""
        self.amazon_bounds = {
            'north': 5.0,    # Northern Venezuela/Guyana
            'south': -18.0,  # Southern Bolivia  
            'east': -44.0,   # Eastern Brazil
            'west': -84.0    # Western Peru/Ecuador
        }
        
        # Set up coordinate transformers
        self.wgs84 = pyproj.CRS('EPSG:4326')
        self.utm_18n = pyproj.CRS('EPSG:32618')
        
    def validate_coordinates(self, lat: float, lon: float) -> Tuple[bool, str]:
        """Validate if coordinates are within Amazon region"""
        
        if not (-90 <= lat <= 90):
            return False, f"Invalid latitude: {lat} (must be -90 to 90)"
        
        if not (-180 <= lon <= 180):
            return False, f"Invalid longitude: {lon} (must be -180 to 180)"
        
        # Check if within Amazon bounds
        if not (self.amazon_bounds['south'] <= lat <= self.amazon_bounds['north']):
            return False, f"Latitude {lat} outside Amazon region"
        
        if not (self.amazon_bounds['west'] <= lon <= self.amazon_bounds['east']):
            return False, f"Longitude {lon} outside Amazon region"
        
        return True, "Valid coordinates"
    
    def clean_coordinate_data(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Clean and validate coordinate data in GeoDataFrame"""
        
        if gdf.empty:
            return gdf
        
        original_count = len(gdf)
        
        # Remove invalid geometries
        gdf = gdf[gdf.geometry.notnull()]
        gdf = gdf[gdf.is_valid]
        
        # Validate coordinates are in Amazon region
        valid_mask = gdf.geometry.apply(
            lambda geom: self.validate_coordinates(geom.y, geom.x)[0] if hasattr(geom, 'x') else False
        )
        
        gdf_clean = gdf[valid_mask].copy()
        
        removed_count = original_count - len(gdf_clean)
        if removed_count > 0:
            logger.warning(f"🧹 Removed {removed_count} invalid coordinates from {original_count} features")
        
        return gdf_clean
    
    def transform_to_utm(self, lat: float, lon: float) -> Tuple[float, float, str]:
        """Transform WGS84 coordinates to appropriate UTM zone"""
        
        # Determine UTM zone based on longitude
        utm_zone = int((lon + 180) / 6) + 1
        
        # For Amazon, typically zones 18-21N
        if utm_zone < 18:
            utm_zone = 18
        elif utm_zone > 21:
            utm_zone = 21
        
        # Create transformer for this UTM zone
        utm_crs = pyproj.CRS(f'EPSG:326{utm_zone:02d}')  # UTM North
        transformer = Transformer.from_crs(self.wgs84, utm_crs, always_xy=True)
        
        x, y = transformer.transform(lon, lat)
        
        return x, y, f"UTM {utm_zone}N"
    
    def calculate_distances(self, points: List[Tuple[float, float]]) -> np.ndarray:
        """Calculate distances between points in meters"""
        
        if len(points) < 2:
            return np.array([])
        
        # Convert to UTM for accurate distance calculation
        utm_points = []
        for lat, lon in points:
            x, y, _ = self.transform_to_utm(lat, lon)
            utm_points.append((x, y))
        
        utm_points = np.array(utm_points)
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(utm_points)):
            for j in range(i + 1, len(utm_points)):
                dist = np.sqrt((utm_points[i][0] - utm_points[j][0])**2 + 
                              (utm_points[i][1] - utm_points[j][1])**2)
                distances.append(dist)
        
        return np.array(distances)
    
    def create_buffer_zone(self, gdf: gpd.GeoDataFrame, buffer_km: float = 1.0) -> gpd.GeoDataFrame:
        """Create buffer zones around features for analysis"""
        
        if gdf.empty:
            return gdf
        
        # Convert to UTM for accurate buffering
        gdf_utm = gdf.to_crs('EPSG:32619')  # UTM 19N for central Amazon
        
        # Create buffer (convert km to meters)
        buffer_m = buffer_km * 1000
        gdf_utm['buffer_geometry'] = gdf_utm.geometry.buffer(buffer_m)
        
        # Convert back to WGS84
        gdf_buffered = gdf_utm.to_crs('EPSG:4326')
        
        logger.info(f"🔵 Created {buffer_km}km buffer zones for {len(gdf)} features")
        
        return gdf_buffered


class StatisticsCalculator:
    """Calculate comprehensive statistics for archaeological data"""
    
    def __init__(self):
        """Initialize statistics calculator"""
        pass
    
    def calculate_comprehensive_stats(self, map_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive statistics across all data types"""
        
        stats = {
            'overview': {
                'total_datasets': len(map_data),
                'total_features': 0,
                'data_types': list(map_data.keys())
            },
            'by_source': {},
            'spatial_analysis': {},
            'confidence_distribution': {},
            'recommendations': []
        }
        
        # Calculate per-source statistics
        for data_type, data in map_data.items():
            if hasattr(data, '__len__'):
                source_stats = {
                    'feature_count': len(data),
                    'data_type': data_type
                }
                
                # Add confidence statistics if available
                if hasattr(data, 'confidence'):
                    confidence_values = data['confidence'].dropna()
                    if not confidence_values.empty:
                        source_stats['confidence'] = {
                            'mean': float(confidence_values.mean()),
                            'median': float(confidence_values.median()),
                            'min': float(confidence_values.min()),
                            'max': float(confidence_values.max())
                        }
                
                stats['by_source'][data_type] = source_stats
                stats['overview']['total_features'] += source_stats['feature_count']
        
        # Generate recommendations
        stats['recommendations'] = self._generate_recommendations(stats)
        
        return stats
    
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate analysis recommendations based on statistics"""
        
        recommendations = []
        
        total_features = stats['overview']['total_features']
        
        if total_features == 0:
            recommendations.append("No features detected - consider expanding search area or adjusting detection parameters")
        elif total_features < 5:
            recommendations.append("Low feature count - verify data coverage and detection sensitivity")
        elif total_features > 100:
            recommendations.append("High feature density - consider prioritization filters")
        
        # Check data source balance
        source_counts = [stats['by_source'][src]['feature_count'] for src in stats['by_source']]
        if len(source_counts) > 1:
            max_count = max(source_counts)
            min_count = min(source_counts)
            if max_count > min_count * 10:
                recommendations.append("Unbalanced data sources - investigate coverage differences")
        
        return recommendations
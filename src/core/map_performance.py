"""
Map Performance Optimization Utilities
GPU acceleration, viewport culling, and intelligent clustering for archaeological maps
"""

import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("🚀 GPU acceleration available with CuPy")
except (ImportError, AttributeError, Exception) as e:
    GPU_AVAILABLE = False
    cp = None
    logger.info(f"💻 Using CPU for computations (CuPy issue: {type(e).__name__})")

@dataclass
class ViewportBounds:
    """Viewport boundaries for efficient rendering"""
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    buffer_factor: float = 1.2

    def contains_point(self, lat: float, lon: float) -> bool:
        """Check if point is within buffered viewport"""
        lat_buffer = (self.max_lat - self.min_lat) * (self.buffer_factor - 1) / 2
        lon_buffer = (self.max_lon - self.min_lon) * (self.buffer_factor - 1) / 2
        
        return (
            self.min_lat - lat_buffer <= lat <= self.max_lat + lat_buffer and
            self.min_lon - lon_buffer <= lon <= self.max_lon + lon_buffer
        )

class ArchaeologicalMapOptimizer:
    """High-performance optimization for archaeological feature rendering"""
    
    def __init__(self, enable_gpu: bool = True):
        self.enable_gpu = enable_gpu and GPU_AVAILABLE
        self.logger = logging.getLogger(__name__)
        
        # Performance settings
        self.settings = {
            'max_features_without_clustering': 500,
            'cluster_distance_meters': 100,
            'viewport_buffer_factor': 1.2,
            'confidence_threshold': 0.4,
            'min_cluster_size': 3,
            'max_cluster_radius_pixels': 50
        }
        
        if self.enable_gpu:
            self.logger.info("🚀 GPU-accelerated map optimization enabled")
        else:
            self.logger.info("💻 CPU-based map optimization enabled")
    
    def optimize_features(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize features for general display (no specific viewport)
        
        Args:
            features: List of archaeological features
            
        Returns:
            Optimized feature list for efficient rendering
        """
        try:
            # Step 1: Confidence filtering
            quality_features = self._filter_by_confidence(features)
            self.logger.debug(f"Confidence filtering: {len(features)} → {len(quality_features)} features")
            
            # Step 2: Deduplication
            deduplicated_features = self._deduplicate_features(quality_features)
            self.logger.debug(f"Deduplication: {len(quality_features)} → {len(deduplicated_features)} features")
            
            # Step 3: Sort by confidence and area for prioritization
            prioritized_features = self._prioritize_features(deduplicated_features)
            
            return prioritized_features
            
        except Exception as e:
            self.logger.error(f"Error optimizing features: {e}")
            return features[:self.settings['max_features_without_clustering']]  # Fallback
    
    def optimize_features_for_viewport(self, features: List[Dict[str, Any]], 
                                     viewport: ViewportBounds) -> List[Dict[str, Any]]:
        """
        Optimize features for current viewport with spatial culling
        
        Args:
            features: List of archaeological features
            viewport: Current viewport bounds
            
        Returns:
            Optimized feature list for efficient rendering
        """
        try:
            # Step 1: Viewport culling
            visible_features = self._cull_features_by_viewport(features, viewport)
            self.logger.debug(f"Viewport culling: {len(features)} → {len(visible_features)} features")
            
            # Step 2: Confidence filtering
            quality_features = self._filter_by_confidence(visible_features)
            self.logger.debug(f"Confidence filtering: {len(visible_features)} → {len(quality_features)} features")
            
            # Step 3: Intelligent clustering if needed
            if len(quality_features) > self.settings['max_features_without_clustering']:
                clustered_features = self._cluster_features(quality_features, viewport)
                self.logger.debug(f"Clustering: {len(quality_features)} → {len(clustered_features)} features")
                return clustered_features
            
            return quality_features
            
        except Exception as e:
            self.logger.error(f"Error optimizing features: {e}")
            return features[:self.settings['max_features_without_clustering']]  # Fallback
    
    def _cull_features_by_viewport(self, features: List[Dict[str, Any]], 
                                  viewport: ViewportBounds) -> List[Dict[str, Any]]:
        """Remove features outside viewport bounds"""
        visible_features = []
        
        for feature in features:
            coords = feature.get('coordinates', [])
            if len(coords) >= 2:
                lon, lat = coords[0], coords[1]
                if viewport.contains_point(lat, lon):
                    visible_features.append(feature)
        
        return visible_features
    
    def _filter_by_confidence(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter features by confidence threshold"""
        threshold = self.settings['confidence_threshold']
        return [f for f in features if f.get('confidence', 0.0) >= threshold]
    
    def _cluster_features(self, features: List[Dict[str, Any]], 
                         viewport: ViewportBounds) -> List[Dict[str, Any]]:
        """
        Intelligently cluster nearby features for performance
        Uses GPU acceleration if available
        """
        if len(features) < self.settings['min_cluster_size']:
            return features
        
        try:
            # Extract coordinates
            coordinates = np.array([[f['coordinates'][1], f['coordinates'][0]] 
                                  for f in features if len(f.get('coordinates', [])) >= 2])
            
            if len(coordinates) < self.settings['min_cluster_size']:
                return features
            
            # Use GPU acceleration if available
            if self.enable_gpu:
                clusters = self._gpu_cluster_features(coordinates, features)
            else:
                clusters = self._cpu_cluster_features(coordinates, features)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error clustering features: {e}")
            return features
    
    def _gpu_cluster_features(self, coordinates: np.ndarray, 
                             features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """GPU-accelerated feature clustering using CuPy"""
        try:
            # Transfer data to GPU
            gpu_coords = cp.asarray(coordinates)
            
            # Simple distance-based clustering on GPU
            cluster_distance = self._meters_to_degrees(self.settings['cluster_distance_meters'])
            clusters = []
            used_indices = cp.zeros(len(gpu_coords), dtype=bool)
            
            for i in range(len(gpu_coords)):
                if used_indices[i]:
                    continue
                
                # Find nearby points using GPU
                center = gpu_coords[i]
                distances = cp.linalg.norm(gpu_coords - center, axis=1)
                nearby_mask = distances < cluster_distance
                nearby_indices = cp.where(nearby_mask)[0]
                
                if len(nearby_indices) >= self.settings['min_cluster_size']:
                    # Create cluster
                    cluster_coords = gpu_coords[nearby_indices]
                    cluster_center = cp.mean(cluster_coords, axis=0)
                    
                    # Get highest confidence feature as representative
                    nearby_features = [features[int(idx)] for idx in cp.asnumpy(nearby_indices)]
                    best_feature = max(nearby_features, key=lambda f: f.get('confidence', 0.0))
                    
                    # Create cluster feature
                    cluster_feature = best_feature.copy()
                    cluster_feature.update({
                        'coordinates': [float(cluster_center[1]), float(cluster_center[0])],
                        'clustered_count': len(nearby_indices),
                        'clustered_features': nearby_features,
                        'cluster_confidence': float(cp.mean([f.get('confidence', 0.0) 
                                                           for f in nearby_features]))
                    })
                    
                    clusters.append(cluster_feature)
                    used_indices[nearby_indices] = True
                else:
                    # Add individual feature
                    clusters.append(features[i])
                    used_indices[i] = True
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"GPU clustering failed: {e}")
            return self._cpu_cluster_features(coordinates, features)
    
    def _cpu_cluster_features(self, coordinates: np.ndarray, 
                             features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """CPU-based feature clustering fallback"""
        try:
            from sklearn.cluster import DBSCAN
            
            # Convert distance to degrees
            cluster_distance = self._meters_to_degrees(self.settings['cluster_distance_meters'])
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(
                eps=cluster_distance, 
                min_samples=self.settings['min_cluster_size']
            ).fit(coordinates)
            
            labels = clustering.labels_
            clusters = []
            
            # Process clusters
            for label in set(labels):
                if label == -1:  # Noise points (not clustered)
                    noise_indices = np.where(labels == label)[0]
                    for idx in noise_indices:
                        clusters.append(features[idx])
                else:
                    # Create cluster representative
                    cluster_indices = np.where(labels == label)[0]
                    cluster_features = [features[idx] for idx in cluster_indices]
                    cluster_coords = coordinates[cluster_indices]
                    
                    # Use centroid and best feature
                    center = np.mean(cluster_coords, axis=0)
                    best_feature = max(cluster_features, key=lambda f: f.get('confidence', 0.0))
                    
                    cluster_feature = best_feature.copy()
                    cluster_feature.update({
                        'coordinates': [float(center[1]), float(center[0])],
                        'clustered_count': len(cluster_indices),
                        'clustered_features': cluster_features,
                        'cluster_confidence': np.mean([f.get('confidence', 0.0) 
                                                     for f in cluster_features])
                    })
                    
                    clusters.append(cluster_feature)
            
            return clusters
            
        except ImportError:
            self.logger.warning("scikit-learn not available, using simple clustering")
            return self._simple_cluster_features(coordinates, features)
        except Exception as e:
            self.logger.error(f"CPU clustering failed: {e}")
            return features
    
    def _simple_cluster_features(self, coordinates: np.ndarray, 
                                features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple distance-based clustering without external dependencies"""
        cluster_distance = self._meters_to_degrees(self.settings['cluster_distance_meters'])
        clusters = []
        used = np.zeros(len(coordinates), dtype=bool)
        
        for i in range(len(coordinates)):
            if used[i]:
                continue
            
            # Find nearby points
            center = coordinates[i]
            distances = np.linalg.norm(coordinates - center, axis=1)
            nearby_mask = distances < cluster_distance
            nearby_indices = np.where(nearby_mask)[0]
            
            if len(nearby_indices) >= self.settings['min_cluster_size']:
                # Create cluster
                cluster_coords = coordinates[nearby_indices]
                cluster_center = np.mean(cluster_coords, axis=0)
                nearby_features = [features[idx] for idx in nearby_indices]
                
                best_feature = max(nearby_features, key=lambda f: f.get('confidence', 0.0))
                cluster_feature = best_feature.copy()
                cluster_feature.update({
                    'coordinates': [float(cluster_center[1]), float(cluster_center[0])],
                    'clustered_count': len(nearby_indices),
                    'clustered_features': nearby_features,
                    'cluster_confidence': np.mean([f.get('confidence', 0.0) 
                                                 for f in nearby_features])
                })
                
                clusters.append(cluster_feature)
                used[nearby_indices] = True
            else:
                clusters.append(features[i])
                used[i] = True
        
        return clusters
    
    def _meters_to_degrees(self, meters: float, lat: float = 0.0) -> float:
        """Convert meters to approximate degrees at given latitude"""
        # Rough conversion: 1 degree ≈ 111,320 meters at equator
        meters_per_degree = 111320 * math.cos(math.radians(lat))
        return meters / meters_per_degree
    
    def calculate_optimal_zoom(self, features: List[Dict[str, Any]]) -> int:
        """Calculate optimal zoom level for feature distribution"""
        if not features:
            return 10
        
        # Extract coordinates
        lats = [f['coordinates'][1] for f in features if len(f.get('coordinates', [])) >= 2]
        lons = [f['coordinates'][0] for f in features if len(f.get('coordinates', [])) >= 2]
        
        if not lats or not lons:
            return 10
        
        # Calculate bounding box
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        max_range = max(lat_range, lon_range)
        
        # Zoom level based on coordinate range
        if max_range > 10:
            return 4
        elif max_range > 5:
            return 6
        elif max_range > 2:
            return 8
        elif max_range > 1:
            return 10
        elif max_range > 0.5:
            return 12
        else:
            return 14
    
    def _deduplicate_features(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate features within a small radius"""
        if len(features) <= 1:
            return features
            
        dedup_distance = self._meters_to_degrees(50)  # 50m deduplication radius
        deduplicated = []
        used_indices = set()
        
        for i, feature in enumerate(features):
            if i in used_indices:
                continue
                
            coords = feature.get('coordinates', [])
            if len(coords) < 2:
                deduplicated.append(feature)
                continue
                
            lat, lon = coords[1], coords[0]
            
            # Find nearby features
            duplicates = [i]
            for j, other_feature in enumerate(features[i+1:], start=i+1):
                if j in used_indices:
                    continue
                    
                other_coords = other_feature.get('coordinates', [])
                if len(other_coords) < 2:
                    continue
                    
                other_lat, other_lon = other_coords[1], other_coords[0]
                distance = math.sqrt((lat - other_lat)**2 + (lon - other_lon)**2)
                
                if distance < dedup_distance:
                    duplicates.append(j)
            
            # Keep the highest confidence feature among duplicates
            best_feature = max([features[idx] for idx in duplicates], 
                             key=lambda f: f.get('confidence', 0.0))
            deduplicated.append(best_feature)
            used_indices.update(duplicates)
            
        return deduplicated
    
    def _prioritize_features(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort features by priority score (confidence * area)"""
        def priority_score(feature):
            confidence = feature.get('confidence', 0.5)
            area = feature.get('area_m2', 1000)  # Default 1000 m²
            return confidence * math.log10(max(area, 1))
            
        return sorted(features, key=priority_score, reverse=True)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance optimization settings"""
        return {
            'gpu_enabled': self.enable_gpu,
            'gpu_available': GPU_AVAILABLE,
            'settings': self.settings.copy(),
            'optimization_features': [
                'viewport_culling',
                'confidence_filtering', 
                'intelligent_clustering',
                'deduplication',
                'prioritization',
                'gpu_acceleration' if self.enable_gpu else 'cpu_fallback'
            ]
        }
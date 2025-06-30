"""
Visualization Components
Modular components for building archaeological maps
"""

import logging
from typing import Dict, List, Any, Optional
import json
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureRenderer:
    """Handles rendering of archaeological features with enhanced styling"""
    
    def __init__(self):
        self.archaeological_icons = {
            # GEDI LiDAR Only features
            'gedi_clearing': {'icon': '🏘️', 'color': '#228B22', 'category': 'gedi_only'},
            'gedi_mound': {'icon': '⛰️', 'color': '#8B4513', 'category': 'gedi_only'},
            'gedi_linear': {'icon': '⛰️', 'color': '#4169E1', 'category': 'gedi_only'},
            
            # Sentinel-2 Only features
            'terra_preta': {'icon': '🌱', 'color': '#8A2BE2', 'category': 'sentinel2_only'},
            'crop_mark': {'icon': '🌾', 'color': '#4B0082', 'category': 'sentinel2_only'},
            'geometric_circle': {'icon': '⭕', 'color': '#9370DB', 'category': 'sentinel2_only'},
            'geometric_rectangle': {'icon': '⬜', 'color': '#9370DB', 'category': 'sentinel2_only'},
            'geometric_line': {'icon': '📏', 'color': '#9370DB', 'category': 'sentinel2_only'},
            
            # Multi-Sensor Validated
            'convergent_high': {'icon': '🎯', 'color': '#DC143C', 'category': 'cross_validated'},
            'convergent_medium': {'icon': '🎯', 'color': '#FF8C00', 'category': 'cross_validated'},
            
            # Priority Investigation
            'priority_1': {'icon': '🚩', 'color': '#FF1493', 'category': 'priority'},
            'priority_2': {'icon': '🚩', 'color': '#FF6B35', 'category': 'priority'},
            'priority_3': {'icon': '🚩', 'color': '#FFD700', 'category': 'priority'}
        }
    
    def _extract_coordinates(self, geometry):
        """Extract coordinates in GeoJSON format from Shapely geometry"""
        if geometry is None:
            logger.warning("Received None geometry, skipping coordinate extraction")
            return None
            
        try:
            if geometry.geom_type == 'Point':
                return [geometry.x, geometry.y]
            elif geometry.geom_type == 'LineString':
                return [[coord[0], coord[1]] for coord in geometry.coords]
            elif geometry.geom_type == 'Polygon':
                # Return exterior ring coordinates
                return [[[coord[0], coord[1]] for coord in geometry.exterior.coords]]
            else:
                # Fallback for complex geometries
                centroid = geometry.centroid
                if centroid is not None:
                    return [centroid.x, centroid.y]
                else:
                    logger.warning(f"Could not extract centroid from {geometry.geom_type}")
                    return None
        except Exception as e:
            logger.warning(f"Error extracting coordinates from geometry: {e}")
            return None
    
    def _process_feature_coordinate(self, all_coordinates: dict, coord_key: tuple, raw_data: dict, geometry, data_key: str):
        """Helper method to process a feature coordinate and merge with existing data"""
        
        # Determine provider from data_key if not already set
        if 'provider' not in raw_data:
            if 'gedi' in data_key:
                raw_data['provider'] = 'gedi'
            elif 'sentinel2' in data_key:
                raw_data['provider'] = 'sentinel2'
            else:
                raw_data['provider'] = raw_data.get('provider', 'unknown')
        
        if coord_key not in all_coordinates:
            all_coordinates[coord_key] = {
                'geometry': geometry,
                'data': raw_data,
                'sources': [data_key]
            }
        else:
            # Merge data, keeping priority information
            existing = all_coordinates[coord_key]['data']
            
            # Priority rank takes precedence
            if 'rank' in raw_data and raw_data['rank'] is not None:
                existing['rank'] = raw_data['rank']
            
            # Cross-validation data takes precedence  
            if raw_data.get('gedi_support', False):
                existing['gedi_support'] = True
            if raw_data.get('convergent_score', 0) > existing.get('convergent_score', 0):
                existing['convergent_score'] = raw_data['convergent_score']
                existing['convergence_distance_m'] = raw_data.get('convergence_distance_m')
            
            # Update provider if from a more specific source
            if 'gedi' in data_key and existing.get('provider') != 'gedi':
                existing['provider'] = 'gedi'
            elif 'sentinel2' in data_key and existing.get('provider') not in ['gedi']:
                existing['provider'] = 'sentinel2'
            
            all_coordinates[coord_key]['sources'].append(data_key)

    def create_feature_layers(self, map_data: Dict, theme: str) -> Dict[str, List]:
        """Create single truth icons - one icon per coordinate with all state information"""
        
        feature_layers = {
            'unified_features': []  # All features in one layer with state properties
        }
        
        # Process ALL data sources to get complete coordinate set
        all_coordinates = {}  # coordinate -> feature data
        
        logger.info(f"🔍 Processing ALL data sources to create single truth per coordinate")
        
        # Process each data source and merge by coordinates
        for data_key in ['gedi_only', 'sentinel2_only', 'combined', 'top_candidates']:
            if data_key in map_data and map_data[data_key] is not None:
                data = map_data[data_key]
                logger.info(f"📊 Processing {len(data)} features from {data_key}")
                
                # Handle both DataFrame and list of dictionaries formats
                if hasattr(data, 'iterrows'):
                    # DataFrame format
                    for idx, row in data.iterrows():
                        try:
                            # Check for None geometry
                            if row.geometry is None:
                                logger.debug(f"Skipping row {idx} from {data_key}: None geometry")
                                continue
                                
                            # Extract coordinates
                            if hasattr(row.geometry, 'coords'):
                                coords = list(row.geometry.coords)[0]
                            elif hasattr(row.geometry, 'x'):
                                coords = (row.geometry.x, row.geometry.y)
                            else:
                                logger.debug(f"Skipping row {idx} from {data_key}: unsupported geometry type")
                                continue
                            
                            # Round coordinates to avoid floating point duplicates
                            coord_key = (round(coords[0], 6), round(coords[1], 6))
                            
                            raw_data = row.to_dict() if hasattr(row, 'to_dict') else row
                            geometry = row.geometry
                            
                            self._process_feature_coordinate(all_coordinates, coord_key, raw_data, geometry, data_key)
                            
                        except Exception as e:
                            logger.warning(f"❌ Error processing DataFrame row {idx} from {data_key}: {e}")
                            continue
                            
                else:
                    # List of dictionaries format (from feature extraction)
                    for idx, feature_dict in enumerate(data):
                        try:
                            # Extract coordinates from dictionary
                            coords = feature_dict.get('coordinates')
                            if not coords or len(coords) != 2:
                                continue
                                
                            # Round coordinates to avoid floating point duplicates  
                            coord_key = (round(coords[0], 6), round(coords[1], 6))
                            
                            # Create geometry object from coordinates
                            from shapely.geometry import Point
                            geometry = Point(coords[0], coords[1])
                            
                            self._process_feature_coordinate(all_coordinates, coord_key, feature_dict, geometry, data_key)
                            
                        except Exception as e:
                            logger.warning(f"❌ Error processing dict feature {idx} from {data_key}: {e}")
                            continue
        
        logger.info(f"🎯 Found {len(all_coordinates)} unique coordinates after deduplication")
        
        # Create single feature per coordinate with proper state hierarchy
        for coord_key, coord_data in all_coordinates.items():
            try:
                raw_data = coord_data['data']
                geometry = coord_data['geometry']
                
                # Add coordinates to raw_data for tooltip methods
                if geometry and hasattr(geometry, 'x') and hasattr(geometry, 'y'):
                    raw_data['coordinates'] = [geometry.x, geometry.y]
                    raw_data['lat'] = geometry.y
                    raw_data['lon'] = geometry.x
                
                # Determine states
                gedi_support = raw_data.get('gedi_support', False)
                sentinel2_support = raw_data.get('sentinel2_support', False)
                convergent_score = raw_data.get('convergent_score', 0.0)
                is_cross_validated = gedi_support and sentinel2_support and convergent_score > 0.0 and raw_data.get('convergence_distance_m') is not None
                priority_rank = raw_data.get('rank', None)
                is_priority = priority_rank is not None and priority_rank <= 5
                provider = raw_data.get('provider', 'unknown')
                
                # Determine base feature type and icon (always keep consistent icon)
                if provider == 'gedi':
                    feature_type = self._classify_gedi_feature_from_data(raw_data)
                    base_icon_config = self.archaeological_icons.get(feature_type, self.archaeological_icons['gedi_clearing'])
                else:
                    feature_type = self._classify_sentinel2_feature_from_data(raw_data)
                    base_icon_config = self.archaeological_icons.get(feature_type, self.archaeological_icons['terra_preta'])
                
                # Use base icon config (JavaScript will handle red borders for cross-validated)
                icon_config = base_icon_config
                
                # Determine tooltip and category based on state hierarchy
                if is_priority:
                    tooltip_content = self._create_priority_tooltip_from_data(raw_data, feature_type, priority_rank)
                    primary_category = 'priority'
                elif is_cross_validated:
                    tooltip_content = self._create_convergent_tooltip_from_data(raw_data, feature_type)
                    primary_category = 'cross_validated'
                else:
                    if provider == 'gedi':
                        tooltip_content = self._create_gedi_only_tooltip_from_data(raw_data, feature_type)
                        primary_category = 'gedi'
                    else:
                        tooltip_content = self._create_sentinel2_only_tooltip_from_data(raw_data, feature_type)
                        primary_category = 'sentinel2'
                
                # Create single feature with ALL state information
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': geometry.geom_type,
                        'coordinates': self._extract_coordinates(geometry)
                    },
                    'properties': {
                        'feature_type': feature_type,
                        'icon': icon_config['icon'],
                        'color': icon_config['color'],
                        'tooltip': tooltip_content,
                        'confidence': raw_data.get('confidence', 0.0),
                        'provider': provider,
                        'original_provider': provider,  # Keep track of original source
                        'primary_category': primary_category,  # What icon/category it displays as
                        'is_cross_validated': is_cross_validated,
                        'is_priority': is_priority,
                        'priority_rank': priority_rank if is_priority else None,
                        'convergent_score': convergent_score if is_cross_validated else None,
                        'area_m2': raw_data.get('area_m2', 0),
                        'sources': coord_data['sources'],  # Track which datasets contributed
                        'raw_data': raw_data
                    }
                }
                
                feature_layers['unified_features'].append(feature)
                    
            except Exception as e:
                logger.warning(f"Error processing coordinate {coord_key}: {e}")
                continue
        
        logger.info(f"✅ Created {len(feature_layers['unified_features'])} single-truth features")
        
        # Show distribution by category
        categories = {}
        for feature in feature_layers['unified_features']:
            cat = feature['properties']['primary_category']
            categories[cat] = categories.get(cat, 0) + 1
        logger.info(f"📊 Feature distribution: {categories}")
        
        
        
        # Process convergence pairs as connecting lines
        if 'convergence_pairs' in map_data:
            feature_layers['convergence_lines'] = self._process_convergence_pairs(
                map_data['convergence_pairs'], theme
            )
            logger.info(f"🔗 Loaded {len(map_data['convergence_pairs'])} convergence pairs")
            
            # DIAGNOSTIC: Check if top candidates are diverse (they should be!)
            if 'top_candidates' in map_data and len(map_data['top_candidates']) > 0:
                types = map_data['top_candidates']['type'].value_counts()
                providers = map_data['top_candidates']['provider'].value_counts() if 'provider' in map_data['top_candidates'].columns else {}
                if len(types) == 1:
                    logger.warning(f"🚨 PIPELINE ISSUE: All top candidates are {types.index[0]} - should be diverse!")
                    logger.warning(f"💡 Expected: Mix of terra_preta, gedi_clearing, convergent features, etc.")
                if len(providers) == 1 and len(providers) > 0:
                    logger.warning(f"🚨 PIPELINE ISSUE: All top candidates from {providers.index[0]} - should include cross-validated!")
                    logger.warning(f"💡 Expected: Features with gedi_support=True, sentinel2_support=True")
        
        # Log feature counts for debugging
        for layer_name, features in feature_layers.items():
            logger.info(f"📊 {layer_name}: {len(features)} features")
        
        return feature_layers
    
    def _get_combined_coordinates(self, map_data: Dict) -> set:
        """Get coordinates from combined data to avoid duplicates"""
        combined_coords = set()
        if 'combined' in map_data:
            for idx, row in map_data['combined'].iterrows():
                try:
                    if hasattr(row.geometry, 'coords'):
                        coords = list(row.geometry.coords)[0]
                    elif hasattr(row.geometry, 'x'):
                        coords = (row.geometry.x, row.geometry.y)
                    else:
                        continue
                    combined_coords.add((round(coords[0], 6), round(coords[1], 6)))
                except Exception:
                    continue
        return combined_coords
    
    def _process_gedi_only_features(self, gedi_data, combined_coords: set, theme: str) -> List[Dict]:
        """Process GEDI features that are NOT in combined multi-sensor data"""
        features = []
        
        for idx, row in gedi_data.iterrows():
            try:
                # Extract coordinates
                if hasattr(row.geometry, 'coords'):
                    coords = list(row.geometry.coords)[0]
                elif hasattr(row.geometry, 'x'):
                    coords = (row.geometry.x, row.geometry.y)
                else:
                    continue
                
                # Check if this point has multi-sensor convergence
                coord_key = (round(coords[0], 6), round(coords[1], 6))
                is_convergent = coord_key in combined_coords
                
                # Show all GEDI features (both convergent and non-convergent)
                # The "only" designation now means "features from this sensor"
                feature_type = self._classify_gedi_feature(row)
                icon_config = self.archaeological_icons.get(feature_type, self.archaeological_icons['gedi_clearing'])
                
                # Create GEDI-specific tooltip
                tooltip_content = self._create_gedi_only_tooltip(row, feature_type)
                
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': row.geometry.geom_type,
                        'coordinates': self._extract_coordinates(row.geometry)
                    },
                    'properties': {
                        'feature_type': feature_type,
                        'icon': icon_config['icon'],
                        'color': icon_config['color'],
                        'category': icon_config['category'],
                        'tooltip': tooltip_content,
                        'confidence': getattr(row, 'confidence', 0.0),
                        'provider': 'gedi_only',
                        'data_source': 'NASA GEDI LiDAR',
                        'area_m2': getattr(row, 'area_m2', 0),  # Include area for radius display
                        'raw_data': row.to_dict()
                    }
                }
                
                features.append(feature)
                
            except Exception as e:
                logger.warning(f"Error processing GEDI-only feature {idx}: {e}")
                continue
        
        logger.info(f"🛰️ Processed {len(features)} GEDI-only features")
        return features
    
    def _process_sentinel2_only_features(self, sentinel2_data, combined_coords: set, theme: str) -> List[Dict]:
        """Process Sentinel-2 features that are NOT in combined multi-sensor data"""
        features = []
        
        for idx, row in sentinel2_data.iterrows():
            try:
                # Extract coordinates
                if hasattr(row.geometry, 'coords'):
                    coords = list(row.geometry.coords)[0]
                elif hasattr(row.geometry, 'x'):
                    coords = (row.geometry.x, row.geometry.y)
                else:
                    continue
                
                # Check if this point has multi-sensor convergence
                coord_key = (round(coords[0], 6), round(coords[1], 6))
                is_convergent = coord_key in combined_coords
                
                # Show all Sentinel-2 features (both convergent and non-convergent)
                # The "only" designation now means "features from this sensor"
                feature_type = self._classify_sentinel2_feature(row)
                icon_config = self.archaeological_icons.get(feature_type, self.archaeological_icons['terra_preta'])
                
                logger.debug(f"Sentinel-2 feature {idx}: type={feature_type}, icon={icon_config['icon']}")
                
                # Create Sentinel-2-specific tooltip
                tooltip_content = self._create_sentinel2_only_tooltip(row, feature_type)
                
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': row.geometry.geom_type,
                        'coordinates': self._extract_coordinates(row.geometry)
                    },
                    'properties': {
                        'feature_type': feature_type,
                        'icon': icon_config['icon'],
                        'color': icon_config['color'],
                        'category': icon_config['category'],
                        'tooltip': tooltip_content,
                        'confidence': getattr(row, 'confidence', 0.0),
                        'provider': 'sentinel2_only',
                        'data_source': 'ESA Sentinel-2 MSI',
                        'area_m2': getattr(row, 'area_m2', 0),  # Include area for radius display
                        'raw_data': row.to_dict()
                    }
                }
                
                features.append(feature)
                
            except Exception as e:
                logger.warning(f"Error processing Sentinel-2-only feature {idx}: {e}")
                continue
        
        logger.info(f"🛰️ Processed {len(features)} Sentinel-2-only features")
        return features
    
    def _process_gedi_features(self, gedi_data, theme: str) -> List[Dict]:
        """Process GEDI LiDAR features with enhanced tooltips"""
        
        features = []
        
        for idx, row in gedi_data.iterrows():
            try:
                # Extract coordinates
                if hasattr(row.geometry, 'coords'):
                    coords = list(row.geometry.coords)[0]
                elif hasattr(row.geometry, 'x'):
                    coords = (row.geometry.x, row.geometry.y)
                else:
                    continue
                
                # Determine feature type and styling
                feature_type = self._classify_gedi_feature(row)
                icon_config = self.archaeological_icons.get(feature_type, self.archaeological_icons['gedi_clearing'])
                
                # Create enhanced tooltip
                tooltip_content = self._create_gedi_tooltip(row, feature_type)
                
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': row.geometry.geom_type,
                        'coordinates': self._extract_coordinates(row.geometry)
                    },
                    'properties': {
                        'feature_type': feature_type,
                        'icon': icon_config['icon'],
                        'color': icon_config['color'],
                        'category': icon_config['category'],
                        'tooltip': tooltip_content,
                        'confidence': getattr(row, 'confidence', 0.0),
                        'provider': 'gedi',
                        'raw_data': row.to_dict()
                    }
                }
                
                features.append(feature)
                
            except Exception as e:
                logger.warning(f"Error processing GEDI feature {idx}: {e}")
                continue
        
        logger.info(f"🔍 Processed {len(features)} GEDI features")
        return features
    
    def _process_sentinel2_features(self, sentinel2_data, theme: str) -> List[Dict]:
        """Process Sentinel-2 multispectral features"""
        
        features = []
        
        for idx, row in sentinel2_data.iterrows():
            try:
                # Extract coordinates
                if hasattr(row.geometry, 'coords'):
                    coords = list(row.geometry.coords)[0]
                elif hasattr(row.geometry, 'x'):
                    coords = (row.geometry.x, row.geometry.y)
                else:
                    continue
                
                # Determine feature type
                feature_type = self._classify_sentinel2_feature(row)
                icon_config = self.archaeological_icons.get(feature_type, self.archaeological_icons['terra_preta'])
                
                # Create enhanced tooltip
                tooltip_content = self._create_sentinel2_tooltip(row, feature_type)
                
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': row.geometry.geom_type,
                        'coordinates': self._extract_coordinates(row.geometry)
                    },
                    'properties': {
                        'feature_type': feature_type,
                        'icon': icon_config['icon'],
                        'color': icon_config['color'],
                        'category': icon_config['category'],
                        'tooltip': tooltip_content,
                        'confidence': getattr(row, 'confidence', 0.0),
                        'provider': 'sentinel2',
                        'raw_data': row.to_dict()
                    }
                }
                
                features.append(feature)
                
            except Exception as e:
                logger.warning(f"Error processing Sentinel-2 feature {idx}: {e}")
                continue
        
        logger.info(f"🛰️ Processed {len(features)} Sentinel-2 features")
        return features
    
    def _process_convergent_features(self, combined_data, theme: str) -> List[Dict]:
        """Process combined features - both single-sensor and multi-sensor"""
        
        features = []
        
        for idx, row in combined_data.iterrows():
            try:
                # Get feature data
                raw_data = row.to_dict() if hasattr(row, 'to_dict') else row
                gedi_support = raw_data.get('gedi_support', False)
                convergent_score = raw_data.get('convergent_score', 0.0)
                provider = raw_data.get('provider', 'unknown')
                feature_source_type = raw_data.get('type', '').lower()
                
                logger.debug(f"Processing combined feature: type={feature_source_type}, provider={provider}, gedi_support={gedi_support}, score={convergent_score}")
                
                # Extract coordinates
                if hasattr(row.geometry, 'coords'):
                    coords = list(row.geometry.coords)[0]
                elif hasattr(row.geometry, 'x'):
                    coords = (row.geometry.x, row.geometry.y)
                else:
                    continue
                
                # Check if this is a priority site (top candidate) - SKIP if it is, let priority_features handle it
                priority_rank = raw_data.get('rank', None)
                is_priority_site = priority_rank is not None and priority_rank <= 5  # Top 5 candidates
                
                if is_priority_site:
                    # Store cross-validation info for priority sites to enhance them later
                    # BUT also process them here to ensure they get cross-validation flags
                    if not hasattr(self, '_priority_cross_validation_data'):
                        self._priority_cross_validation_data = {}
                    
                    coord_key = (round(coords[0], 6), round(coords[1], 6))
                    self._priority_cross_validation_data[coord_key] = {
                        'gedi_support': gedi_support,
                        'convergent_score': convergent_score,
                        'is_cross_validated': gedi_support and convergent_score > 0.0 and raw_data.get('convergence_distance_m') is not None,
                        'raw_data': raw_data,
                        'area_m2': raw_data.get('area_m2', 0)  # Preserve area data
                    }
                    # Don't skip - process priority sites here too so they get cross-validation styling
                    # Priority layer will override the icon but keep the cross-validation properties
                
                # Determine feature type and category based on provider
                if provider == 'sentinel2':
                    # Sentinel-2 features in combined dataset
                    feature_type = self._classify_sentinel2_feature(row)
                    category = 'sentinel2_only'
                    icon_config = self.archaeological_icons.get(feature_type, self.archaeological_icons['terra_preta'])
                    # Use convergent tooltip if cross-validated, otherwise use regular tooltip
                    if gedi_support and convergent_score > 0.0:
                        tooltip_content = self._create_convergent_tooltip(row, feature_type)
                    else:
                        tooltip_content = self._create_sentinel2_only_tooltip(row, feature_type)
                    
                elif provider == 'gedi':
                    # GEDI features in combined dataset
                    feature_type = self._classify_gedi_feature(row)
                    category = 'gedi_only'
                    icon_config = self.archaeological_icons.get(feature_type, self.archaeological_icons['gedi_clearing'])
                    # Use convergent tooltip if cross-validated, otherwise use regular tooltip
                    if gedi_support and convergent_score > 0.0:
                        tooltip_content = self._create_convergent_tooltip(row, feature_type)
                    else:
                        tooltip_content = self._create_gedi_only_tooltip(row, feature_type)
                    
                else:
                    # Fallback for unknown types
                    feature_type = 'convergent_medium'
                    category = 'cross_validated'
                    icon_config = self.archaeological_icons[feature_type]
                    tooltip_content = self._create_convergent_tooltip(row, feature_type)
                
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': row.geometry.geom_type,
                        'coordinates': self._extract_coordinates(row.geometry)
                    },
                    'properties': {
                        'feature_type': feature_type,
                        'icon': icon_config['icon'],
                        'color': icon_config['color'],
                        'category': category,
                        'tooltip': tooltip_content,
                        'confidence': getattr(row, 'confidence', 0.0),
                        'provider': category,
                        'is_cross_validated': gedi_support and convergent_score > 0.0 and raw_data.get('convergence_distance_m') is not None,  # Flag for red outline
                        'area_m2': getattr(row, 'area_m2', 0),  # Include area for radius display
                        'raw_data': row.to_dict()
                    }
                }
                
                features.append(feature)
                
            except Exception as e:
                logger.warning(f"Error processing combined feature {idx}: {e}")
                continue
        
        logger.info(f"🎯 Processed {len(features)} combined features (includes single-sensor and multi-sensor)")
        return features
    
    def _classify_original_feature_type(self, row) -> str:
        """Classify the original feature type for convergent features"""
        raw_data = row.to_dict() if hasattr(row, 'to_dict') else row
        feature_type = raw_data.get('type', '').lower()
        provider = raw_data.get('provider', '')
        
        # Return the classified type based on provider and feature characteristics
        if provider == 'gedi' or 'gedi' in feature_type:
            if 'clearing' in feature_type:
                return 'gedi_clearing'
            else:
                return 'gedi_mound'
        elif provider == 'sentinel2' or 'terra_preta' in feature_type:
            if 'terra_preta' in feature_type:
                return 'terra_preta'
            elif 'geometric' in feature_type:
                if 'circle' in feature_type:
                    return 'geometric_circle'
                elif 'rectangle' in feature_type:
                    return 'geometric_rectangle'
                else:
                    return 'geometric_line'
            else:
                return 'crop_mark'
        else:
            return 'convergent_high'  # Default fallback
    
    def _get_original_icon_for_feature(self, original_feature_type: str, feature_source_type: str) -> str:
        """Get the original icon for a feature type"""
        # Try to get the icon from our predefined icons first
        if original_feature_type in self.archaeological_icons:
            return self.archaeological_icons[original_feature_type]['icon']
        
        # Fallback logic based on feature characteristics
        if 'gedi' in original_feature_type or 'gedi' in feature_source_type:
            if 'clearing' in feature_source_type:
                return '🏘️'
            else:
                return '⛰️'
        elif 'terra_preta' in original_feature_type or 'terra_preta' in feature_source_type:
            return '🌱'
        elif 'geometric' in original_feature_type or 'geometric' in feature_source_type:
            if 'circle' in feature_source_type:
                return '⭕'
            elif 'rectangle' in feature_source_type:
                return '⬜'
            else:
                return '📏'
        else:
            return '🎯'  # Default convergent icon
    
    def _process_priority_features(self, priority_data, theme: str) -> List[Dict]:
        """Process high-priority investigation candidates"""
        
        features = []
        
        for idx, row in priority_data.iterrows():
            try:
                # Extract coordinates
                if hasattr(row.geometry, 'coords'):
                    coords = list(row.geometry.coords)[0]
                elif hasattr(row.geometry, 'x'):
                    coords = (row.geometry.x, row.geometry.y)
                else:
                    continue
                
                # Determine priority level and cross-validation status
                # Handle both attribute access and dictionary-style access
                try:
                    if hasattr(row, 'rank') and not callable(getattr(row, 'rank')):
                        priority_rank = row.rank
                    elif hasattr(row, 'to_dict'):
                        priority_rank = row.to_dict().get('rank', idx + 1)
                    else:
                        priority_rank = idx + 1
                except:
                    priority_rank = idx + 1
                
                # Check if this priority site is cross-validated using stored data or direct data
                coord_key = (round(coords[0], 6), round(coords[1], 6))
                cross_val_data = getattr(self, '_priority_cross_validation_data', {}).get(coord_key, {})
                
                # Use stored cross-validation data if available, otherwise check direct data
                if cross_val_data:
                    gedi_support = cross_val_data['gedi_support']
                    convergent_score = cross_val_data['convergent_score'] 
                    is_cross_validated = cross_val_data['is_cross_validated']
                    convergent_raw_data = cross_val_data['raw_data']
                else:
                    raw_data = row.to_dict() if hasattr(row, 'to_dict') else row
                    gedi_support = raw_data.get('gedi_support', False)
                    convergent_score = raw_data.get('convergent_score', 0.0)
                    is_cross_validated = gedi_support and convergent_score > 0.0 and raw_data.get('convergence_distance_m') is not None
                    convergent_raw_data = raw_data
                
                if priority_rank <= 2:
                    feature_type = 'priority_1'
                elif priority_rank <= 4:
                    feature_type = 'priority_2'
                else:
                    feature_type = 'priority_3'
                
                icon_config = self.archaeological_icons[feature_type]
                
                # Create enhanced tooltip - use convergent tooltip if cross-validated
                if is_cross_validated and convergent_raw_data:
                    tooltip_content = self._create_convergent_tooltip(
                        type('MockRow', (), convergent_raw_data)(), feature_type
                    )
                else:
                    tooltip_content = self._create_priority_tooltip(row, feature_type, priority_rank)
                
                feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': row.geometry.geom_type,
                        'coordinates': self._extract_coordinates(row.geometry)
                    },
                    'properties': {
                        'feature_type': feature_type,
                        'icon': icon_config['icon'],
                        'color': icon_config['color'],
                        'category': icon_config['category'],
                        'tooltip': tooltip_content,
                        'confidence': getattr(row, 'confidence', 0.0),
                        'priority_rank': priority_rank,
                        'provider': 'priority',
                        'is_cross_validated': is_cross_validated,  # Flag for red outline on priority sites
                        'area_m2': cross_val_data.get('area_m2', getattr(row, 'area_m2', 0)),  # Use enhanced area if available
                        'raw_data': row.to_dict()
                    }
                }
                
                features.append(feature)
                
            except Exception as e:
                logger.warning(f"Error processing priority feature {idx}: {e}")
                continue
        
        logger.info(f"⭐ Processed {len(features)} priority features")
        return features
    
    def _classify_gedi_feature(self, row) -> str:
        """Classify GEDI feature based on properties"""
        
        # Get type from row data - handle both GeoDataFrame rows and GeoJSON properties
        feature_type = ''
        
        # First try to get from the row/feature properties directly (export format)
        if hasattr(row, 'to_dict'):
            row_data = row.to_dict()
            feature_type = row_data.get('type', '').lower()
        elif hasattr(row, 'type'):
            feature_type = getattr(row, 'type', '').lower()
        # Fallback for dict-like objects (GeoJSON properties)
        elif isinstance(row, dict):
            feature_type = row.get('type', '').lower()
        
        logger.debug(f"Classifying GEDI feature with type: {feature_type}")
        
        if not feature_type:
            logger.warning(f"No type found for GEDI feature, using default")
            return 'gedi_clearing'
        
        if 'clearing' in feature_type or 'settlement' in feature_type:
            return 'gedi_clearing'
        elif 'mound' in feature_type or 'earthwork' in feature_type:
            return 'gedi_mound'
        elif 'linear' in feature_type or 'causeway' in feature_type:
            return 'gedi_linear'
        else:
            return 'gedi_clearing'  # Default
    
    def _classify_sentinel2_feature(self, row) -> str:
        """Classify Sentinel-2 feature based on properties"""
        
        # Get type from row data - handle both GeoDataFrame rows and GeoJSON properties  
        feature_type = ''
        
        # First try to get from the row/feature properties directly (export format)
        if hasattr(row, 'to_dict'):
            row_data = row.to_dict()
            feature_type = row_data.get('type', '').lower()
        elif hasattr(row, 'type'):
            feature_type = getattr(row, 'type', '').lower()
        # Fallback for dict-like objects (GeoJSON properties)
        elif isinstance(row, dict):
            feature_type = row.get('type', '').lower()
        
        logger.debug(f"Classifying Sentinel-2 feature with type: {feature_type}")
        
        if not feature_type:
            logger.warning(f"No type found for Sentinel-2 feature, using default")
            return 'terra_preta'
        
        if 'terra_preta' in feature_type or 'soil' in feature_type:
            return 'terra_preta'
        elif 'crop_mark' in feature_type or 'crop' in feature_type:
            return 'crop_mark'
        elif 'circle' in feature_type:
            return 'geometric_circle'
        elif 'rectangle' in feature_type or 'square' in feature_type:
            return 'geometric_rectangle'
        elif 'line' in feature_type or 'linear' in feature_type:
            return 'geometric_line'
        else:
            return 'terra_preta'  # Default
    
    def _create_unified_features(self, map_data: Dict, theme: str) -> List[Dict]:
        """Create unified features - ONE icon per location with multiple states"""
        
        # Dictionary to store unique features by coordinate
        unified_features = {}
        
        logger.info("🔄 Starting unified feature processing...")
        
        # Step 1: Process all combined data (has everything including cross-validation info)
        if 'combined' in map_data:
            logger.info(f"📊 Processing {len(map_data['combined'])} combined features")
            for idx, row in map_data['combined'].iterrows():
                try:
                    # Extract coordinates as unique key
                    if hasattr(row.geometry, 'coords'):
                        coords = list(row.geometry.coords)[0]
                    elif hasattr(row.geometry, 'x'):
                        coords = (row.geometry.x, row.geometry.y)
                    else:
                        continue
                    
                    coord_key = (round(coords[0], 6), round(coords[1], 6))
                    raw_data = row.to_dict() if hasattr(row, 'to_dict') else row
                    
                    # Determine feature states
                    gedi_support = raw_data.get('gedi_support', False)
                    convergent_score = raw_data.get('convergent_score', 0.0)
                    is_cross_validated = gedi_support and convergent_score > 0.0 and raw_data.get('convergence_distance_m') is not None
                    priority_rank = raw_data.get('rank', None)
                    is_priority = priority_rank is not None and priority_rank <= 5
                    provider = raw_data.get('provider', 'unknown')
                    
                    # Determine primary category (priority order: Cross-validated > Priority > Provider)
                    if is_cross_validated:
                        primary_category = 'cross_validated'
                        feature_type = 'convergent_high' if convergent_score > 0.7 else 'convergent_medium'
                        icon_config = self.archaeological_icons[feature_type]
                    elif is_priority:
                        primary_category = 'priority'
                        if priority_rank <= 2:
                            feature_type = 'priority_1'
                        elif priority_rank <= 4:
                            feature_type = 'priority_2'
                        else:
                            feature_type = 'priority_3'
                        icon_config = self.archaeological_icons[feature_type]
                    elif provider == 'gedi':
                        primary_category = 'gedi_only'
                        feature_type = self._classify_gedi_feature(row)
                        icon_config = self.archaeological_icons.get(feature_type, self.archaeological_icons['gedi_clearing'])
                    elif provider == 'sentinel2':
                        primary_category = 'sentinel2_only'
                        feature_type = self._classify_sentinel2_feature(row)
                        icon_config = self.archaeological_icons.get(feature_type, self.archaeological_icons['terra_preta'])
                    else:
                        continue  # Skip unknown providers
                    
                    # Create tooltip
                    if is_cross_validated:
                        tooltip_content = self._create_convergent_tooltip(row, feature_type)
                    elif is_priority:
                        tooltip_content = self._create_priority_tooltip(row, feature_type, priority_rank)
                    elif provider == 'gedi':
                        tooltip_content = self._create_gedi_only_tooltip(row, feature_type)
                    else:
                        tooltip_content = self._create_sentinel2_only_tooltip(row, feature_type)
                    
                    # Store unified feature
                    unified_features[coord_key] = {
                        'type': 'Feature',
                        'geometry': {
                            'type': row.geometry.geom_type,
                            'coordinates': self._extract_coordinates(row.geometry)
                        },
                        'properties': {
                            'feature_type': feature_type,
                            'icon': icon_config['icon'],
                            'color': icon_config['color'],
                            'primary_category': primary_category,
                            'tooltip': tooltip_content,
                            'confidence': raw_data.get('confidence', 0.0),
                            'provider': provider,
                            'is_cross_validated': is_cross_validated,
                            'is_priority': is_priority,
                            'priority_rank': priority_rank if is_priority else None,
                            'has_gedi': gedi_support,
                            'has_sentinel2': provider == 'sentinel2' or gedi_support,  # If has GEDI support, likely also has Sentinel-2
                            'area_m2': raw_data.get('area_m2', 0),
                            'raw_data': raw_data
                        }
                    }
                    
                except Exception as e:
                    logger.warning(f"Error processing unified feature {idx}: {e}")
                    continue
        
        # Convert to list
        feature_list = list(unified_features.values())
        
        logger.info(f"✅ Created {len(feature_list)} unified features from {len(unified_features)} unique locations")
        
        return feature_list
    
    def _get_gedi_interpretation(self, feature_type: str) -> str:
        """Get human-readable interpretation for GEDI feature"""
        interpretations = {
            'gedi_clearing': 'Ancient Settlement',
            'gedi_mound': 'Ceremonial Mound',
            'gedi_linear': 'Ancient Causeway'
        }
        return interpretations.get(feature_type, 'Archaeological Feature')
    
    def _get_sentinel2_interpretation(self, feature_type: str) -> str:
        """Get human-readable interpretation for Sentinel-2 feature"""
        interpretations = {
            'terra_preta': 'Terra Preta (Dark Earth)',
            'crop_mark': 'Crop Mark Anomaly',
            'geometric_circle': 'Circular Structure',
            'geometric_rectangle': 'Rectangular Structure',
            'geometric_line': 'Linear Feature'
        }
        return interpretations.get(feature_type, 'Spectral Anomaly')
    
    def _create_gedi_only_tooltip(self, row, feature_type: str) -> str:
        """Create focused GEDI tooltip with technical data only"""
        
        # Extract key data safely from export format
        raw_data = row.to_dict() if hasattr(row, 'to_dict') else row
        confidence = raw_data.get('confidence', 0.0)
        area_m2 = raw_data.get('area_m2', 0)
        # Try multiple field names for shot count
        shot_count = (raw_data.get('count', 0) or 
                     raw_data.get('pulse_density', 0) or 
                     raw_data.get('shots', 0) or
                     raw_data.get('lidar_shots', 0) or
                     raw_data.get('beam_count', 0))
        
        # Fallback: check sensor_details for pulse_density
        if shot_count == 0 and 'sensor_details' in raw_data:
            sensor_details = raw_data['sensor_details']
            shot_count = (sensor_details.get('pulse_density', 0) or
                         sensor_details.get('count', 0))
        zone = raw_data.get('zone', 'unknown')
        
        # Extract coordinates with comprehensive fallbacks
        lat, lon = 0.0, 0.0
        
        # Try geometry object first
        try:
            if hasattr(row, 'geometry') and row.geometry:
                if hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
                    lat, lon = row.geometry.y, row.geometry.x
            if lat == 0.0 and lon == 0.0:
                # Fallback to coordinate arrays
                coords = getattr(row.geometry, 'coords', None)
                if coords:
                    lon, lat = coords[0][0], coords[0][1]
                elif hasattr(row, 'geometry') and hasattr(row.geometry, 'coordinates'):
                    lon, lat = row.geometry.coordinates[0], row.geometry.coordinates[1]
        except:
            pass
        
        # Fallback to raw_data coordinates (GeoJSON format)
        if lat == 0.0 and lon == 0.0:
            if 'geometry' in raw_data and isinstance(raw_data['geometry'], dict):
                geom_coords = raw_data['geometry'].get('coordinates')
                if geom_coords and len(geom_coords) >= 2:
                    lon, lat = geom_coords[0], geom_coords[1]  # GeoJSON is [lng, lat]
            
            # Final fallback to direct coordinates field
            if lat == 0.0 and lon == 0.0:
                coords = raw_data.get('coordinates', [])
                if coords and len(coords) >= 2:
                    lon, lat = coords[0], coords[1]  # Assume [lng, lat]
                else:
                    lat = raw_data.get('lat', 0.0)
                    lon = raw_data.get('lon', 0.0)
        
        # Calculate area in hectares
        area_hectares = area_m2 / 10000 if area_m2 > 0 else 0
        shot_density = shot_count / area_hectares if area_hectares > 0 else 0
        
        # LiDAR analysis fields we just added
        gap_points = raw_data.get('gap_points_detected', shot_count)
        mean_elevation = raw_data.get('mean_elevation', None)
        elevation_std = raw_data.get('elevation_std', None)
        elevation_threshold = raw_data.get('elevation_anomaly_threshold', None)
        local_variance = raw_data.get('local_variance', None)
        pulse_density = raw_data.get('pulse_density', None)
        
        # Format enhanced LiDAR analysis section
        enhanced_analysis = ""
        if mean_elevation is not None:
            enhanced_analysis += f"<p><strong>Mean Elevation:</strong> {mean_elevation:.1f}m"
            if elevation_std is not None:
                enhanced_analysis += f" (std: {elevation_std:.1f}m)"
            enhanced_analysis += "</p>"
        
        if gap_points and gap_points != shot_count:
            enhanced_analysis += f"<p><strong>Gap Points Detected:</strong> {gap_points}</p>"
        
        if pulse_density is not None:
            enhanced_analysis += f"<p><strong>Pulse Density:</strong> {pulse_density:.3f} pts/m²</p>"
        
        if local_variance is not None:
            enhanced_analysis += f"<p><strong>Elevation Variance:</strong> {local_variance:.2f}</p>"
        
        if elevation_threshold is not None:
            enhanced_analysis += f"<p><strong>Anomaly Threshold:</strong> {elevation_threshold:.1f}m</p>"

        return f"""
        <div class="popup-detailed">
            <div class="popup-header">
                <span class="popup-title">🏘️ GEDI Detection</span>
                <span class="confidence-badge gedi">{confidence:.1%}</span>
            </div>
            <div class="popup-details">
                <p><strong>Coordinates:</strong> {lat:.6f}, {lon:.6f}</p>
                <p><strong>Area:</strong> {area_m2:,.0f} m² ({area_hectares:.2f} hectares)</p>
                <p><strong>Zone:</strong> {zone}</p>
                <p><strong>LiDAR Shots:</strong> {shot_count}</p>
                {f'<p><strong>Shot Density:</strong> {shot_density:.1f}/ha</p>' if shot_density > 0 else ''}
                {enhanced_analysis}
            </div>
        </div>
        """
    
    def _parse_json_field(self, raw_data: dict, field_name: str) -> dict:
        """Parse JSON string fields from export data"""
        import json
        field_value = raw_data.get(field_name)
        if isinstance(field_value, str):
            try:
                return json.loads(field_value)
            except (json.JSONDecodeError, TypeError):
                return {}
        elif isinstance(field_value, dict):
            return field_value
        return {}
    
    def _create_sentinel2_only_tooltip(self, row, feature_type: str) -> str:
        """Create detailed popup for Sentinel-2 features with comprehensive export data"""
        
        # Extract key data from export format
        raw_data = row.to_dict() if hasattr(row, 'to_dict') else row
        confidence = raw_data.get('confidence', 0.0)
        area_m2 = raw_data.get('area_m2', 0)
        feature_type_clean = raw_data.get('type', feature_type)
        selection_reason = raw_data.get('selection_reason', 'Spectral anomaly detected')
        
        # Cross-provider validation from export
        gedi_support = raw_data.get('gedi_support', False)
        sentinel2_support = raw_data.get('sentinel2_support', True)
        convergent_score = raw_data.get('convergent_score', 0.0)
        
        # Quality indicators - handle both dict and JSON string formats
        quality_raw = raw_data.get('quality_indicators', {})
        if isinstance(quality_raw, str) and quality_raw.startswith('{'):
            import json
            try:
                quality = json.loads(quality_raw)
            except:
                quality = {}
        else:
            quality = quality_raw if isinstance(quality_raw, dict) else {}
        
        archaeological_grade = raw_data.get('archaeological_grade', 'standard')
        zone = raw_data.get('zone', 'Unknown Zone')
        provider = raw_data.get('provider', 'sentinel2')
        
        # Get detailed spectral information
        confidence_level = quality.get('confidence_level', 'moderate') if quality else 'moderate'
        area_significance = quality.get('area_significance', 'medium') if quality else 'medium'
        
        # Archaeological assessment details  
        assessment = raw_data.get('archaeological_assessment', '')
        if isinstance(assessment, str) and assessment.startswith('{'):
            import json
            try:
                assessment_data = json.loads(assessment)
                interpretation = assessment_data.get('interpretation', 'Spectral archaeological signature')
                evidence_type = assessment_data.get('evidence_type', 'Multispectral analysis')
                cultural_context = assessment_data.get('cultural_context', 'Pre-Columbian')
            except:
                interpretation = 'Spectral archaeological signature'
                evidence_type = 'Multispectral analysis'
                cultural_context = 'Pre-Columbian'
        else:
            interpretation = 'Spectral archaeological signature'
            evidence_type = 'Multispectral analysis'
            cultural_context = 'Pre-Columbian'
        
        # Get icon based on type
        icon_map = {
            'terra_preta': '🌱',
            'crop_mark': '🌾', 
            'geometric_circle': '⭕',
            'geometric_rectangle': '⬜',
            'geometric_line': '📏'
        }
        icon = icon_map.get(feature_type_clean, '🌱')
        
        # Extract coordinates with comprehensive fallbacks
        lat, lon = 0.0, 0.0
        
        # Try geometry object first
        try:
            if hasattr(row, 'geometry') and row.geometry:
                if hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
                    lat, lon = row.geometry.y, row.geometry.x
            if lat == 0.0 and lon == 0.0:
                # Fallback to coordinate arrays
                coords = getattr(row.geometry, 'coords', None)
                if coords:
                    lon, lat = coords[0][0], coords[0][1]
                elif hasattr(row, 'geometry') and hasattr(row.geometry, 'coordinates'):
                    lon, lat = row.geometry.coordinates[0], row.geometry.coordinates[1]
        except:
            pass
        
        # Fallback to raw_data coordinates (GeoJSON format)
        if lat == 0.0 and lon == 0.0:
            if 'geometry' in raw_data and isinstance(raw_data['geometry'], dict):
                geom_coords = raw_data['geometry'].get('coordinates')
                if geom_coords and len(geom_coords) >= 2:
                    lon, lat = geom_coords[0], geom_coords[1]  # GeoJSON is [lng, lat]
            
            # Final fallback to direct coordinates field
            if lat == 0.0 and lon == 0.0:
                coords = raw_data.get('coordinates', [])
                if coords and len(coords) >= 2:
                    lon, lat = coords[0], coords[1]  # Assume [lng, lat]
                else:
                    lat = raw_data.get('lat', 0.0)
                    lon = raw_data.get('lon', 0.0)
        
        # Parse sensor details for technical parameters
        sensor_details = self._parse_json_field(raw_data, 'sensor_details')
        detection_algorithm = self._parse_json_field(raw_data, 'detection_algorithm')
        
        # Extract technical parameters
        spatial_resolution = 10  # Default Sentinel-2 resolution
        cloud_coverage = 0
        ndvi_threshold = 0.0
        soil_brightness = 0.0
        red_edge_ratio = 0.0
        
        # Extract computed spectral indices
        ndvi_value = 0.0
        ndre1_value = 0.0
        ndwi_value = 0.0
        terra_preta_enhanced = 0.0
        crop_mark_index = 0.0
        brightness_value = 0.0
        p_value = 0.0
        effect_size = 0.0
        
        if sensor_details:
            spatial_resolution = sensor_details.get('spatial_resolution_m', 10)
            cloud_coverage = sensor_details.get('cloud_coverage_percent', 0)
            
        if detection_algorithm:
            if 'parameters' in detection_algorithm:
                params = detection_algorithm['parameters']
                ndvi_threshold = params.get('ndvi_threshold', 0.0)
                soil_brightness = params.get('soil_brightness_index', 0.0)
                red_edge_ratio = params.get('red_edge_ratio', 0.0)
            
            # Extract computed spectral indices
            if 'spectral_indices' in detection_algorithm:
                indices = detection_algorithm['spectral_indices']
                ndvi_value = indices.get('ndvi', 0.0)
                ndre1_value = indices.get('ndre1', 0.0)
                ndwi_value = indices.get('ndwi', 0.0)
                terra_preta_enhanced = indices.get('terra_preta_enhanced', 0.0)
                crop_mark_index = indices.get('crop_mark_index', 0.0)
                brightness_value = indices.get('brightness', 0.0)
            
            # Extract statistical validation
            if 'statistical_validation' in detection_algorithm:
                stats = detection_algorithm['statistical_validation']
                p_value = stats.get('p_value', 0.0)
                effect_size = stats.get('effect_size', 0.0)
        
        return f"""
        <div class="popup-detailed">
            <div class="popup-header">
                <span class="popup-title">🌱 Sentinel-2 Detection</span>
                <span class="confidence-badge sentinel2">{confidence:.1%}</span>
            </div>
            <div class="popup-details">
                <p><strong>Coordinates:</strong> {lat:.6f}, {lon:.6f}</p>
                <p><strong>Area:</strong> {area_m2:,.0f} m² ({area_m2/10000:.2f} hectares)</p>
                <p><strong>Zone:</strong> {zone}</p>
                <p><strong>Type:</strong> {feature_type_clean}</p>
                {f'<p><strong>NDVI:</strong> {ndvi_value:.3f}</p>' if ndvi_value != 0.0 else ''}
                {f'<p><strong>Cloud Cover:</strong> {cloud_coverage:.0f}%</p>' if cloud_coverage > 0 else ''}
            </div>
        </div>
        """

    def _create_gedi_tooltip(self, row, feature_type: str) -> str:
        """Create concise popup for GEDI features with real export data"""
        
        # Use the unified method
        return self._create_gedi_only_tooltip(row, feature_type)
    
    def _create_sentinel2_tooltip(self, row, feature_type: str) -> str:
        """Create concise popup for Sentinel-2 features with real export data"""
        
        # Use the unified method
        return self._create_sentinel2_only_tooltip(row, feature_type)
    
    def _create_convergent_tooltip(self, row, feature_type: str) -> str:
        """Create concise popup for cross-provider convergent features with real export data"""
        
        # Extract key data from export format
        raw_data = row.to_dict() if hasattr(row, 'to_dict') else row
        
        confidence = raw_data.get('confidence', 0.0)
        combined_confidence = raw_data.get('combined_confidence', confidence)
        convergent_score = raw_data.get('convergent_score', 0.0)
        gedi_support = raw_data.get('gedi_support', True)  # Cross-validated should have both
        sentinel2_support = raw_data.get('sentinel2_support', True)
        convergence_distance = raw_data.get('convergence_distance_m', None)
        area_m2 = raw_data.get('area_m2', 0)
        convergence_type = raw_data.get('convergence_type', 'cross-sensor')
        
        # Extract coordinates with comprehensive fallbacks
        lat, lon = 0.0, 0.0
        
        # Try geometry object first
        try:
            if hasattr(row, 'geometry') and row.geometry:
                if hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
                    lat, lon = row.geometry.y, row.geometry.x
        except:
            pass
        
        # Fallback to raw_data coordinates (GeoJSON format)
        if lat == 0.0 and lon == 0.0:
            if 'geometry' in raw_data and isinstance(raw_data['geometry'], dict):
                geom_coords = raw_data['geometry'].get('coordinates')
                if geom_coords and len(geom_coords) >= 2:
                    lon, lat = geom_coords[0], geom_coords[1]  # GeoJSON is [lng, lat]
            
            # Final fallback to direct coordinates field
            if lat == 0.0 and lon == 0.0:
                coords = raw_data.get('coordinates', [])
                if coords and len(coords) >= 2:
                    lon, lat = coords[0], coords[1]  # Assume [lng, lat]
                else:
                    lat = raw_data.get('lat', 0.0)
                    lon = raw_data.get('lon', 0.0)
        
        # Calculate correlation strength from convergent score
        correlation_strength = convergent_score / 15.0 if convergent_score > 0 else combined_confidence or confidence
        
        return f"""
        <div class="popup-compact">
            <div class="popup-header">
                <span class="popup-title">🎯 Multi-Sensor Detection</span>
                <span class="confidence-badge priority">{(combined_confidence or confidence):.1%}</span>
            </div>
            <div class="popup-details">
                <p><strong>Coordinates:</strong> {lat:.6f}, {lon:.6f}</p>
                <p><strong>Area:</strong> {area_m2:,.0f} m² ({area_m2/10000:.2f} ha)</p>
                <p><strong>Combined Confidence:</strong> {(combined_confidence or confidence):.3f}</p>
                <p><strong>Convergence Score:</strong> {convergent_score:.1f}/15</p>
                <p><strong>Sensor Agreement:</strong> {correlation_strength:.3f}</p>
                {f'<p><strong>Separation Distance:</strong> {convergence_distance:.1f}m</p>' if convergence_distance else ''}
                <p><strong>Sources:</strong> GEDI + Sentinel-2 MSI</p>
            </div>
        </div>
        """
    
    def _create_priority_tooltip(self, row, feature_type: str, rank: int) -> str:
        """Create detailed popup for priority investigation features with comprehensive export data"""
        
        # Extract key data from export format
        raw_data = row.to_dict() if hasattr(row, 'to_dict') else row
        
        confidence = raw_data.get('confidence', 0.0)
        area_m2 = raw_data.get('area_m2', 0)
        feature_type_real = raw_data.get('type', feature_type)
        selection_reason = raw_data.get('selection_reason', 'High priority detection')
        
        # Cross-provider validation from export
        gedi_support = raw_data.get('gedi_support', False)
        sentinel2_support = raw_data.get('sentinel2_support', False)
        convergent_score = raw_data.get('convergent_score', 0.0)
        
        # Quality indicators
        quality = raw_data.get('quality_indicators', {})
        archaeological_grade = raw_data.get('archaeological_grade', 'standard')
        zone = raw_data.get('zone', 'Unknown Zone')
        provider = raw_data.get('provider', 'Multi-sensor')
        
        # Remove archaeological assessment details - keep only technical data
        
        # Investigation priority and field details
        field_investigation = raw_data.get('field_investigation_priority', 'Medium')
        
        # Extract coordinates with comprehensive fallbacks
        lat, lon = 0.0, 0.0
        
        # Try geometry.coordinates first (GeoJSON format)
        if hasattr(row, 'geometry') and row.geometry:
            if hasattr(row.geometry, 'x') and hasattr(row.geometry, 'y'):
                lat, lon = row.geometry.y, row.geometry.x
        
        # Fallback to raw_data coordinates
        if lat == 0.0 and lon == 0.0:
            if 'geometry' in raw_data and isinstance(raw_data['geometry'], dict):
                geom_coords = raw_data['geometry'].get('coordinates')
                if geom_coords and len(geom_coords) >= 2:
                    lon, lat = geom_coords[0], geom_coords[1]  # GeoJSON is [lng, lat]
            
            # Fallback to direct coordinates field
            if lat == 0.0 and lon == 0.0:
                coords = raw_data.get('coordinates', [])
                if coords and len(coords) >= 2:
                    lon, lat = coords[0], coords[1]  # Assume [lng, lat]
        
        return f"""
        <div class="popup-detailed priority">
            <div class="popup-header">
                <span class="popup-title">🚩 Ranked Detection #{rank}</span>
                <span class="confidence-badge priority">{confidence:.1%}</span>
            </div>
            <div class="popup-details">
                <p><strong>Coordinates:</strong> {lat:.6f}, {lon:.6f}</p>
                <p><strong>Area:</strong> {area_m2:,.0f} m² ({area_m2/10000:.2f} ha)</p>
                <p><strong>Detection Confidence:</strong> {confidence:.3f}</p>
                <p><strong>Priority Score:</strong> {convergent_score:.1f}/15</p>
                <p><strong>Data Sources:</strong> {provider}</p>
                <p><strong>Quality Grade:</strong> {archaeological_grade}</p>
                <p><strong>Selection Metric:</strong> {selection_reason}</p>
            </div>
        </div>
        """
    
    def _process_convergence_pairs(self, convergence_data, theme: str) -> List[Dict]:
        """Process convergence pairs as connecting lines and endpoint markers"""
        features = []
        
        for idx, row in convergence_data.iterrows():
            try:
                # Extract line coordinates
                if hasattr(row.geometry, 'coords'):
                    coords = list(row.geometry.coords)
                else:
                    logger.warning(f"Convergence pair {idx} has unsupported geometry type: {row.geometry.geom_type}")
                    continue
                
                # Extract metadata
                raw_data = row.to_dict() if hasattr(row, 'to_dict') else row
                correlation_strength = raw_data.get('strength', raw_data.get('correlation_strength', 0.0))
                distance_m = raw_data.get('distance_m', 0)
                gedi_type = raw_data.get('feature1_type', raw_data.get('gedi_type', 'unknown'))
                sentinel2_type = raw_data.get('feature2_type', raw_data.get('sentinel2_type', 'unknown'))
                
                # Determine line color based on correlation strength
                if correlation_strength >= 0.7:
                    line_color = '#DC143C'  # Strong correlation - red
                    line_weight = 3
                    line_opacity = 0.8
                elif correlation_strength >= 0.4:
                    line_color = '#FF8C00'  # Medium correlation - orange
                    line_weight = 2
                    line_opacity = 0.6
                else:
                    line_color = '#FFD700'  # Weak correlation - gold
                    line_weight = 1
                    line_opacity = 0.4
                
                # Create convergence line feature
                line_feature = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': coords
                    },
                    'properties': {
                        'feature_type': 'convergence_line',
                        'color': line_color,
                        'weight': line_weight,
                        'opacity': line_opacity,
                        'dashArray': '5, 5',  # Dashed line
                        'category': 'convergence',
                        'tooltip': self._create_convergence_tooltip(row),
                        'correlation_strength': correlation_strength,
                        'distance_m': distance_m,
                        'raw_data': raw_data
                    }
                }
                features.append(line_feature)
                
                # Create endpoint markers for each end of the line
                start_point = coords[0]
                end_point = coords[-1]
                
                # Start point marker (GEDI)
                start_marker = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': start_point
                    },
                    'properties': {
                        'feature_type': 'convergence_endpoint_gedi',
                        'icon': '🛰️',
                        'color': '#228B22',
                        'category': 'convergence',
                        'tooltip': f"GEDI Detection: {gedi_type}<br/>Coordinates: {start_point[1]:.6f}, {start_point[0]:.6f}<br/>Correlated with Sentinel-2 at {distance_m:.0f}m",
                        'radius': 6,
                        'stroke': True,
                        'strokeColor': line_color,
                        'strokeWeight': 2,
                        'raw_data': raw_data
                    }
                }
                features.append(start_marker)
                
                # End point marker (Sentinel-2)
                end_marker = {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': end_point
                    },
                    'properties': {
                        'feature_type': 'convergence_endpoint_sentinel2',
                        'icon': '🌱',
                        'color': '#8A2BE2',
                        'category': 'convergence',
                        'tooltip': f"Sentinel-2 Detection: {sentinel2_type}<br/>Coordinates: {end_point[1]:.6f}, {end_point[0]:.6f}<br/>Correlated with GEDI at {distance_m:.0f}m",
                        'radius': 6,
                        'stroke': True,
                        'strokeColor': line_color,
                        'strokeWeight': 2,
                        'raw_data': raw_data
                    }
                }
                features.append(end_marker)
                
            except Exception as e:
                logger.warning(f"Error processing convergence pair {idx}: {e}")
                continue
        
        logger.info(f"🔗 Processed {len(convergence_data)} convergence pairs into {len(features)} map features")
        return features
    
    def _create_convergence_tooltip(self, row) -> str:
        """Create tooltip for convergence line"""
        raw_data = row.to_dict() if hasattr(row, 'to_dict') else row
        
        correlation_strength = raw_data.get('strength', raw_data.get('correlation_strength', 0.0))
        distance_m = raw_data.get('distance_m', 0)
        gedi_type = raw_data.get('feature1_type', raw_data.get('gedi_type', 'unknown'))
        sentinel2_type = raw_data.get('feature2_type', raw_data.get('sentinel2_type', 'unknown'))
        
        # Format correlation strength
        correlation_label = "Strong" if correlation_strength >= 0.7 else "Medium" if correlation_strength >= 0.4 else "Weak"
        
        # Get statistical confidence from combined data
        combined_confidence = raw_data.get('combined_confidence', correlation_strength)
        
        return f"""
        <div class="popup-compact">
            <div class="popup-header">
                <span class="popup-title">🔗 Sensor Correlation</span>
                <span class="confidence-badge convergence">{correlation_strength:.1%}</span>
            </div>
            <div class="popup-details">
                <p><strong>Correlation Coefficient:</strong> {correlation_strength:.4f}</p>
                <p><strong>Spatial Separation:</strong> {distance_m:.1f}m</p>
                <p><strong>Feature 1:</strong> {gedi_type} (GEDI)</p>
                <p><strong>Feature 2:</strong> {sentinel2_type} (Sentinel-2)</p>
                <p><strong>Statistical Confidence:</strong> {combined_confidence:.3f}</p>
            </div>
        </div>
        """
    
    # New "_from_data" methods that work directly with raw_data dictionaries
    
    def _create_priority_tooltip_from_data(self, raw_data: Dict, feature_type: str, rank: int) -> str:
        """Create priority tooltip from raw data dictionary with full scientific data"""
        confidence = raw_data.get('confidence', 0.0)
        area_m2 = raw_data.get('area_m2', 0)
        area_hectares = area_m2 / 10000 if area_m2 > 0 else 0
        
        # Extract coordinates for display with comprehensive fallbacks
        lat, lon = 0.0, 0.0
        
        # Try geometry.coordinates first (GeoJSON format)
        if 'geometry' in raw_data and isinstance(raw_data['geometry'], dict):
            geom_coords = raw_data['geometry'].get('coordinates')
            if geom_coords and len(geom_coords) >= 2:
                lon, lat = geom_coords[0], geom_coords[1]  # GeoJSON is [lng, lat]
        
        # Fallback to direct coordinates field
        if lat == 0.0 and lon == 0.0:
            coords = raw_data.get('coordinates', [])
            if coords and len(coords) >= 2:
                lon, lat = coords[0], coords[1]  # Assume [lng, lat]
        
        # Fallback to direct lat/lon fields
        if lat == 0.0 and lon == 0.0:
            lat = raw_data.get('lat', 0.0)
            lon = raw_data.get('lon', 0.0)
        
        # Cross-provider validation data
        convergent_score = raw_data.get('convergent_score', 0.0)
        convergence_distance_m = raw_data.get('convergence_distance_m')
        gedi_support = raw_data.get('gedi_support', False)
        sentinel2_support = raw_data.get('sentinel2_support', False)
        combined_confidence = raw_data.get('combined_confidence')
        field_priority = raw_data.get('field_investigation_priority', 'medium')
        # Clean up field priority display
        if field_priority == 'unknown' or not field_priority:
            field_priority = 'medium'
        provider = raw_data.get('provider', 'unknown')
        
        # Build validation status
        validation_status = []
        if gedi_support:
            validation_status.append("GEDI LiDAR")
        if sentinel2_support:
            validation_status.append("Sentinel-2 MSI")
        validation_text = " + ".join(validation_status) if validation_status else "Single sensor"
        
        return f"""
        <div class="feature-popup">
            <div class="popup-header priority-header">
                <h3>🚩 Priority Investigation Site #{rank}</h3>
                <div class="priority-badge">{field_priority.upper()} PRIORITY</div>
            </div>
            <div class="popup-content">
                <div class="scientific-metrics">
                    <h4>📊 Detection Metrics</h4>
                    <p><strong>Coordinates:</strong> {lat:.6f}, {lon:.6f}</p>
                    <p><strong>Confidence:</strong> {confidence:.3f} ({confidence*100:.1f}%)</p>
                    <p><strong>Area:</strong> {area_hectares:.2f} hectares ({area_m2:,.0f} m²)</p>
                    <p><strong>Type:</strong> {feature_type.replace('_', ' ').title()}</p>
                    <p><strong>Primary Sensor:</strong> {provider.title()}</p>
                </div>
                
                <div class="validation-metrics">
                    <h4>🎯 Cross-Provider Validation</h4>
                    <p><strong>Validation Status:</strong> {validation_text}</p>
                    <p><strong>Convergent Score:</strong> {convergent_score:.3f}</p>
                    {f'<p><strong>Convergence Distance:</strong> {convergence_distance_m:.1f}m</p>' if convergence_distance_m else ''}
                    {f'<p><strong>Combined Confidence:</strong> {combined_confidence:.3f}</p>' if combined_confidence else ''}
                </div>
                
                <div class="investigation-status">
                    <h4>🔬 Field Investigation</h4>
                    <p><strong>Priority Rank:</strong> #{rank} of top candidates</p>
                    <p><strong>Investigation Priority:</strong> {field_priority.title()}</p>
                    <p><strong>Recommended Action:</strong> {"Immediate ground survey" if rank <= 3 else "Secondary investigation"}</p>
                </div>
            </div>
        </div>
        """
    
    def _create_convergent_tooltip_from_data(self, raw_data: Dict, feature_type: str) -> str:
        """Create convergent tooltip from raw data dictionary with full scientific data"""
        confidence = raw_data.get('confidence', 0.0)
        convergent_score = raw_data.get('convergent_score', 0.0)
        area_m2 = raw_data.get('area_m2', 0)
        area_hectares = area_m2 / 10000 if area_m2 > 0 else 0
        
        # Extract coordinates for display with comprehensive fallbacks
        lat, lon = 0.0, 0.0
        
        # Try geometry.coordinates first (GeoJSON format)
        if 'geometry' in raw_data and isinstance(raw_data['geometry'], dict):
            geom_coords = raw_data['geometry'].get('coordinates')
            if geom_coords and len(geom_coords) >= 2:
                lon, lat = geom_coords[0], geom_coords[1]  # GeoJSON is [lng, lat]
        
        # Fallback to direct coordinates field
        if lat == 0.0 and lon == 0.0:
            coords = raw_data.get('coordinates', [])
            if coords and len(coords) >= 2:
                lon, lat = coords[0], coords[1]  # Assume [lng, lat]
        
        # Fallback to direct lat/lon fields
        if lat == 0.0 and lon == 0.0:
            lat = raw_data.get('lat', 0.0)
            lon = raw_data.get('lon', 0.0)
        
        # Cross-provider validation data
        convergence_distance_m = raw_data.get('convergence_distance_m')
        gedi_support = raw_data.get('gedi_support', False)
        sentinel2_support = raw_data.get('sentinel2_support', False)
        combined_confidence = raw_data.get('combined_confidence')
        provider = raw_data.get('provider', 'unknown')
        convergence_type = raw_data.get('convergence_type', 'spatial')
        
        # Build validation details
        sensors = []
        if gedi_support:
            sensors.append("GEDI LiDAR (canopy structure)")
        if sentinel2_support:
            sensors.append("Sentinel-2 MSI (spectral signatures)")
        
        confidence_level = "HIGH" if convergent_score > 0.7 else "MEDIUM" if convergent_score > 0.4 else "LOW"
        
        return f"""
        <div class="feature-popup">
            <div class="popup-header convergent-header">
                <h3>🎯 Cross-Validated Archaeological Feature</h3>
                <div class="confidence-badge confidence-{confidence_level.lower()}">{confidence_level} CONFIDENCE</div>
            </div>
            <div class="popup-content">
                <div class="detection-summary">
                    <h4>🔬 Multi-Sensor Detection</h4>
                    <p><strong>Coordinates:</strong> {lat:.6f}, {lon:.6f}</p>
                    <p><strong>Convergent Score:</strong> {convergent_score:.3f} ({convergent_score*100:.1f}%)</p>
                    <p><strong>Primary Confidence:</strong> {confidence:.3f} ({confidence*100:.1f}%)</p>
                    {f'<p><strong>Combined Confidence:</strong> {combined_confidence:.3f} ({combined_confidence*100:.1f}%)</p>' if combined_confidence else ''}
                    <p><strong>Feature Type:</strong> {feature_type.replace('_', ' ').title()}</p>
                </div>
                
                <div class="sensor-validation">
                    <h4>📡 Sensor Convergence</h4>
                    {f'<p><strong>Convergence Distance:</strong> {convergence_distance_m:.1f}m (spatial overlap)</p>' if convergence_distance_m else ''}
                    <p><strong>Convergence Type:</strong> {convergence_type.title()}</p>
                    <p><strong>Validating Sensors:</strong></p>
                    <ul>
                        {''.join(f'<li>✓ {sensor}</li>' for sensor in sensors)}
                    </ul>
                </div>
                
                <div class="archaeological-metrics">
                    <h4>🏛️ Archaeological Significance</h4>
                    <p><strong>Area:</strong> {area_hectares:.2f} hectares ({area_m2:,.0f} m²)</p>
                    <p><strong>Primary Sensor:</strong> {provider.title()}</p>
                    <p><strong>Validation Status:</strong> Cross-provider validated</p>
                    <p><strong>Scientific Reliability:</strong> {"High - multiple sensor confirmation" if len(sensors) > 1 else "Moderate - single sensor detection"}</p>
                </div>
            </div>
        </div>
        """
    
    def _classify_gedi_feature_from_data(self, raw_data: Dict) -> str:
        """Classify GEDI feature from raw data dictionary"""
        feature_type = raw_data.get('type', 'gedi_clearing')
        if feature_type.startswith('gedi_'):
            return feature_type
        return f'gedi_{feature_type}'
    
    def _create_gedi_only_tooltip_from_data(self, raw_data: Dict, feature_type: str) -> str:
        """Create GEDI tooltip from raw data dictionary with full scientific data"""
        confidence = raw_data.get('confidence', 0.0)
        area_m2 = raw_data.get('area_m2', 0)
        area_hectares = area_m2 / 10000 if area_m2 > 0 else 0
        count = raw_data.get('count', 0)
        density = (count / area_hectares) if area_hectares > 0 else 0
        
        # Extract coordinates for display with comprehensive fallbacks
        lat, lon = 0.0, 0.0
        
        # Try geometry.coordinates first (GeoJSON format)
        if 'geometry' in raw_data and isinstance(raw_data['geometry'], dict):
            geom_coords = raw_data['geometry'].get('coordinates')
            if geom_coords and len(geom_coords) >= 2:
                lon, lat = geom_coords[0], geom_coords[1]  # GeoJSON is [lng, lat]
        
        # Fallback to direct coordinates field
        if lat == 0.0 and lon == 0.0:
            coords = raw_data.get('coordinates', [])
            if coords and len(coords) >= 2:
                lon, lat = coords[0], coords[1]  # Assume [lng, lat]
        
        # Fallback to direct lat/lon fields
        if lat == 0.0 and lon == 0.0:
            lat = raw_data.get('lat', 0.0)
            lon = raw_data.get('lon', 0.0)
        
        # GEDI-specific technical data
        provider = raw_data.get('provider', 'gedi')
        archaeological_grade = raw_data.get('archaeological_grade', 'standard')
        zone = raw_data.get('zone', 'unknown')
        
        # Statistical significance data (from GEDI analysis)
        p_value = raw_data.get('p_value', 0.001)
        significance_level = "HIGH" if p_value < 0.01 else "MEDIUM" if p_value < 0.05 else "LOW"
        
        # LiDAR analysis fields we just added
        gap_points = raw_data.get('gap_points_detected', count)
        mean_elevation = raw_data.get('mean_elevation', None)
        elevation_std = raw_data.get('elevation_std', None)
        elevation_threshold = raw_data.get('elevation_anomaly_threshold', None)
        local_variance = raw_data.get('local_variance', None)
        pulse_density = raw_data.get('pulse_density', None)
        
        # Format LiDAR analysis section
        lidar_analysis = ""
        if mean_elevation is not None:
            lidar_analysis += f"<p><strong>Elevation:</strong> {mean_elevation:.1f}m"
            if elevation_std is not None:
                lidar_analysis += f" (±{elevation_std:.1f}m)"
            lidar_analysis += "</p>"
        
        if gap_points and gap_points != count:
            lidar_analysis += f"<p><strong>Gap Points:</strong> {gap_points}</p>"
        
        if pulse_density is not None:
            lidar_analysis += f"<p><strong>Pulse Density:</strong> {pulse_density:.3f} pts/m²</p>"
        
        if local_variance is not None:
            lidar_analysis += f"<p><strong>Elevation Variance:</strong> {local_variance:.2f}</p>"
        
        if elevation_threshold is not None:
            lidar_analysis += f"<p><strong>Anomaly Threshold:</strong> {elevation_threshold:.1f}m</p>"

        return f"""
        <div class="popup-compact">
            <div class="popup-header">
                <span class="popup-title">🛰️ GEDI LiDAR Detection</span>
                <span class="confidence-badge gedi">{confidence*100:.1f}%</span>
            </div>
            <div class="popup-details">
                <p><strong>Coordinates:</strong> {lat:.6f}, {lon:.6f}</p>
                <p><strong>Area:</strong> {area_m2:,.0f} m² ({area_hectares:.2f} ha)</p>
                <p><strong>Type:</strong> {feature_type.replace('_', ' ').title()}</p>
                <p><strong>LiDAR Shots:</strong> {count} ({density:.1f} shots/ha)</p>
                <p><strong>Quality:</strong> {archaeological_grade.title()}</p>
                {lidar_analysis}
            </div>
        </div>
        """
    
    def _create_sentinel2_only_tooltip_from_data(self, raw_data: Dict, feature_type: str) -> str:
        """Create Sentinel-2 tooltip from raw data dictionary with full scientific data"""
        
        # Use the existing unified method
        return self._create_sentinel2_only_tooltip(type('MockRow', (), raw_data)(), feature_type)
    
    def _classify_sentinel2_feature_from_data(self, raw_data: Dict) -> str:
        """Classify Sentinel-2 feature from raw data dictionary"""
        feature_type = raw_data.get('type', 'terra_preta')
        return feature_type
    
    def _create_sentinel2_only_tooltip_from_data(self, raw_data: Dict, feature_type: str) -> str:
        """Create Sentinel-2 tooltip from raw data dictionary"""
        confidence = raw_data.get('confidence', 0.0)
        area_m2 = raw_data.get('area_m2', 0)
        area_hectares = area_m2 / 10000 if area_m2 > 0 else 0
        
        # Extract coordinates with comprehensive fallbacks
        lat, lon = 0.0, 0.0
        
        # Try geometry.coordinates first (GeoJSON format)
        if 'geometry' in raw_data and isinstance(raw_data['geometry'], dict):
            geom_coords = raw_data['geometry'].get('coordinates')
            if geom_coords and len(geom_coords) >= 2:
                lon, lat = geom_coords[0], geom_coords[1]  # GeoJSON is [lng, lat]
        
        # Fallback to direct coordinates field
        if lat == 0.0 and lon == 0.0:
            coords = raw_data.get('coordinates', [])
            if coords and len(coords) >= 2:
                lon, lat = coords[0], coords[1]  # Assume [lng, lat]
        
        # Fallback to direct lat/lon fields
        if lat == 0.0 and lon == 0.0:
            lat = raw_data.get('lat', 0.0)
            lon = raw_data.get('lon', 0.0)
        
        return f"""
        <div class="popup-compact">
            <div class="popup-header">
                <span class="popup-title">🌱 Sentinel-2 Detection</span>
                <span class="confidence-badge sentinel2">{confidence:.1%}</span>
            </div>
            <div class="popup-details">
                <p><strong>Coordinates:</strong> {lat:.6f}, {lon:.6f}</p>
                <p><strong>Area:</strong> {area_m2:,.0f} m² ({area_hectares:.2f} ha)</p>
                <p><strong>Type:</strong> {feature_type.replace('_', ' ').title()}</p>
                <p><strong>Data Source:</strong> ESA Sentinel-2</p>
            </div>
        </div>
        """


class LayerManager:
    """Manages map layers and controls"""
    
    def create_layer_controls(self, map_data: Dict) -> Dict[str, Any]:
        """Create layer control configuration"""
        
        layer_controls = {
            'base_layers': {
                'Satellite': 'Esri.WorldImagery',
                'Terrain': 'Esri.WorldShadedRelief',
                'Topographic': 'Esri.WorldTopoMap'
            },
            'overlay_layers': {}
        }
        
        # Add data-driven overlay layers with clear naming
        if 'gedi' in map_data:
            layer_controls['overlay_layers']['GEDI Only'] = 'gedi_only_features'
        
        if 'sentinel2' in map_data:
            layer_controls['overlay_layers']['Sentinel-2 Only'] = 'sentinel2_only_features'
        
        # Cross-validated features are shown with red outlines in their original layers (GEDI/Sentinel-2)
        # No separate layer needed since they stay in gedi_only_features or sentinel2_only_features
        
        if 'top_candidates' in map_data:
            layer_controls['overlay_layers']['Priority Sites'] = 'priority_features'
        
        if 'convergence_pairs' in map_data:
            layer_controls['overlay_layers']['Convergence Lines'] = 'convergence_lines'
        
        return layer_controls


class ControlPanel:
    """Creates analysis and control panels"""
    
    def create_analysis_panels(self, map_data: Dict) -> Dict[str, str]:
        """Create analysis panels with statistics and controls"""
        
        # Calculate statistics
        stats = self._calculate_statistics(map_data)
        
        # Create panels
        panels = {
            'statistics_panel': self._create_statistics_panel(stats),
            'filter_panel': self._create_filter_panel(map_data),
            'legend_panel': self._create_legend_panel(),
            'lidar_analysis_panel': self._create_lidar_analysis_panel(map_data),
            'archaeological_maps_panel': self._create_archaeological_maps_panel()
        }
        
        return panels
    
    def _calculate_statistics(self, map_data: Dict) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        
        stats = {
            'total_features': 0,
            'gedi_count': 0,
            'sentinel2_count': 0,
            'convergent_count': 0,
            'priority_count': 0,
            'high_confidence_count': 0
        }
        
        for data_type, data in map_data.items():
            if hasattr(data, '__len__'):
                count = len(data)
                stats['total_features'] += count
                
                if data_type == 'gedi':
                    stats['gedi_count'] = count
                elif data_type == 'sentinel2':
                    stats['sentinel2_count'] = count
                elif data_type == 'combined':
                    stats['convergent_count'] = count
                elif data_type == 'top_candidates':
                    stats['priority_count'] = count
        
        return stats
    
    def _create_statistics_panel(self, stats: Dict) -> str:
        """Create statistics panel HTML"""
        
        return f"""
        <div class="statistics-panel">
            <h4>📊 Detection Summary</h4>
            <div class="stats-grid">
                <div class="stat-item">
                    <span class="stat-number">{stats['total_features']}</span>
                    <span class="stat-label">Total Features</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{stats['gedi_count']}</span>
                    <span class="stat-label">GEDI LiDAR</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{stats['sentinel2_count']}</span>
                    <span class="stat-label">Sentinel-2</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{stats['convergent_count']}</span>
                    <span class="stat-label">Multi-sensor</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{stats['priority_count']}</span>
                    <span class="stat-label">Priority Sites</span>
                </div>
            </div>
        </div>
        """
    
    def _create_filter_panel(self, map_data: Dict) -> str:
        """Create filter panel HTML"""
        
        return """
        <div class="filter-panel">
            <h4>🔍 Feature Filters</h4>
            <div class="filter-controls">
                <div class="filter-group">
                    <label>Confidence Threshold:</label>
                    <input type="range" id="confidence-slider" min="0" max="100" value="70">
                    <span id="confidence-value">70%</span>
                </div>
                <div class="filter-group">
                    <label>Data Sources:</label>
                    <div class="checkbox-group">
                        <label><input type="checkbox" id="show-gedi" checked> GEDI</label>
                        <label><input type="checkbox" id="show-sentinel2" checked> Sentinel-2</label>
                        <label><input type="checkbox" id="show-convergent" checked> Multi-sensor</label>
                        <label><input type="checkbox" id="show-priority" checked> Priority</label>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _create_legend_panel(self) -> str:
        """Create legend panel HTML"""
        
        return """
        <div class="legend-panel">
            <h4>🗺️ Map Legend</h4>
            <div class="legend-items">
                <div class="legend-item">
                    <span class="legend-icon">🏘️</span>
                    <span class="legend-text">GEDI Settlement</span>
                </div>
                <div class="legend-item">
                    <span class="legend-icon">⛰️</span>
                    <span class="legend-text">GEDI Earthwork</span>
                </div>
                <div class="legend-item">
                    <span class="legend-icon">🌱</span>
                    <span class="legend-text">Terra Preta</span>
                </div>
                <div class="legend-item">
                    <span class="legend-icon">⭕</span>
                    <span class="legend-text">Geometric Pattern</span>
                </div>
                <div class="legend-item">
                    <span class="legend-icon">🎯</span>
                    <span class="legend-text">Multi-sensor</span>
                </div>
                <div class="legend-item">
                    <span class="legend-icon">🚩</span>
                    <span class="legend-text">Priority Site</span>
                </div>
            </div>
        </div>
        """
    
    def _create_lidar_analysis_panel(self, map_data: Dict) -> str:
        """Create LiDAR analysis panel showing elevation and gap statistics"""
        
        # Extract LiDAR analysis data from GEDI features
        gedi_features = []
        if 'gedi_only' in map_data:
            gedi_features.extend(map_data['gedi_only'].to_dict('records') if hasattr(map_data['gedi_only'], 'to_dict') else map_data['gedi_only'])
        if 'combined' in map_data:
            # Filter for GEDI features in combined data
            combined_data = map_data['combined']
            if hasattr(combined_data, 'iterrows'):
                for _, row in combined_data.iterrows():
                    if row.get('provider') == 'gedi' or 'gedi' in str(row.get('type', '')).lower():
                        gedi_features.append(row.to_dict())
        
        # Calculate LiDAR statistics
        elevations = []
        gap_points_total = 0
        pulse_densities = []
        elevation_stds = []
        feature_count = 0
        
        for feature in gedi_features:
            feature_count += 1
            
            # Collect elevation data
            mean_elev = feature.get('mean_elevation')
            if mean_elev is not None:
                elevations.append(mean_elev)
            
            # Collect gap points
            gap_points = feature.get('gap_points_detected', 0)
            if gap_points:
                gap_points_total += gap_points
            
            # Collect pulse density
            pulse_density = feature.get('pulse_density')
            if pulse_density is not None:
                pulse_densities.append(pulse_density)
            
            # Collect elevation standard deviations
            elev_std = feature.get('elevation_std')
            if elev_std is not None:
                elevation_stds.append(elev_std)
        
        # Calculate summary statistics
        avg_elevation = sum(elevations) / len(elevations) if elevations else 0
        elevation_range = f"{min(elevations):.1f} - {max(elevations):.1f}m" if elevations else "N/A"
        avg_pulse_density = sum(pulse_densities) / len(pulse_densities) if pulse_densities else 0
        avg_elevation_std = sum(elevation_stds) / len(elevation_stds) if elevation_stds else 0
        
        return f"""
        <div class="lidar-analysis-panel">
            <h4>📡 LiDAR Analysis Summary</h4>
            <div class="lidar-stats">
                <div class="lidar-stat-item">
                    <span class="stat-number">{feature_count}</span>
                    <span class="stat-label">GEDI Features</span>
                </div>
                <div class="lidar-stat-item">
                    <span class="stat-number">{gap_points_total:,}</span>
                    <span class="stat-label">Total Gap Points</span>
                </div>
                <div class="lidar-stat-item">
                    <span class="stat-number">{avg_elevation:.1f}m</span>
                    <span class="stat-label">Avg Elevation</span>
                </div>
                <div class="lidar-stat-item">
                    <span class="stat-number">{elevation_range}</span>
                    <span class="stat-label">Elevation Range</span>
                </div>
                <div class="lidar-stat-item">
                    <span class="stat-number">{avg_pulse_density:.3f}</span>
                    <span class="stat-label">Avg Pulse Density</span>
                </div>
                <div class="lidar-stat-item">
                    <span class="stat-number">±{avg_elevation_std:.1f}m</span>
                    <span class="stat-label">Avg Elevation Std</span>
                </div>
            </div>
        </div>
        """
    
    def _create_archaeological_maps_panel(self) -> str:
        """Create panel with links to generated archaeological maps"""
        
        return f"""
        <div class="archaeological-maps-panel">
            <h4>🏛️ Archaeological Visualizations</h4>
            <div class="archaeological-links">
                <p><strong>Advanced LiDAR Analysis Maps:</strong></p>
                <div class="map-links">
                    <div class="map-link-item">
                        <span class="map-icon">🗺️</span>
                        <span class="map-description">Digital Elevation Model - Ground surface through canopy</span>
                    </div>
                    <div class="map-link-item">
                        <span class="map-icon">🌳</span>
                        <span class="map-description">Canopy Height Model - Vegetation analysis</span>
                    </div>
                    <div class="map-link-item">
                        <span class="map-icon">🏛️</span>
                        <span class="map-description">Archaeological Features - Temples, buildings, structures</span>
                    </div>
                    <div class="map-link-item">
                        <span class="map-icon">🛤️</span>
                        <span class="map-description">Ancient Infrastructure - Roads, causeways, terraces</span>
                    </div>
                </div>
                <div class="archaeological-info">
                    <p><em>Archaeological visualizations are automatically generated and saved to the archaeological_maps directory. These maps use advanced LiDAR processing techniques proven at Maya and Angkor sites to reveal hidden structures through forest canopy.</em></p>
                </div>
            </div>
        </div>
        """
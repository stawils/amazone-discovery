"""
Archaeological Discovery Visualization Suite
Interactive maps, plots, and visualizations for archaeological analysis results
"""

import folium
from folium import plugins
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
import json
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
from folium.plugins import MarkerCluster

from .config import TARGET_ZONES, VisualizationConfig, ScoringConfig

logger = logging.getLogger(__name__)

class ArchaeologicalVisualizer:
    """Advanced visualization suite for archaeological discoveries"""
    
    def __init__(self):
        self.config = VisualizationConfig()
        self.colors = self.config.CONFIDENCE_COLORS
        
    def create_discovery_map(self, analysis_results: Dict[str, List[Dict]], 
                           scoring_results: Dict[str, Dict] = None,
                           output_path: Path = None) -> Path:
        """Create advanced interactive map with archaeological discoveries"""
        
        logger.info("Creating interactive archaeological discovery map...")
        
        if not analysis_results:
            logger.warning("No analysis results provided")
            return None
        
        # Calculate map center (Amazon basin center)
        amazon_center = [-5.0, -60.0]  # Approximate center of Amazon basin
        
        # Create base map with enhanced features
        m = folium.Map(
            location=amazon_center,
            zoom_start=5,
            tiles=None,  # We'll add custom tiles
            scrollWheelZoom=True,  # Enable zoom on mouse scroll
            control_scale=True,
            prefer_canvas=True  # Better performance for many markers
        )
        
        # Add enhanced tile layers
        folium.TileLayer(
            'OpenStreetMap',
            name='OpenStreetMap',
            attr='OpenStreetMap contributors'
        ).add_to(m)
        
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri WorldImagery',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Add terrain layer
        folium.TileLayer(
            tiles='https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.png',
            attr='Stamen Terrain',
            name='Terrain',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Add target zones with scoring visualization
        self._add_target_zones(m, scoring_results)
        
        # Add discoveries with enhanced clustering
        self._add_discoveries_with_cluster(m, analysis_results, scoring_results)
        
        # Add heatmap layer for discovery density
        self._add_discovery_heatmap(m, analysis_results)
        
        # Add enhanced legend
        self._add_legend(m)
        
        # Add layer control
        folium.LayerControl(position='topright', collapsed=False).add_to(m)
        
        # Add enhanced controls
        plugins.Fullscreen(position='topright').add_to(m)
        plugins.MeasureControl(position='bottomleft').add_to(m)
        plugins.MiniMap(toggle_display=True, position='bottomright').add_to(m)
        plugins.ScrollZoomToggler().add_to(m)
        plugins.LocateControl(auto_start=False, position='topleft').add_to(m)
        # SearchControl removed due to compatibility issues
        # plugins.SearchControl(
        #     position='topleft',
        #     url='https://nominatim.openstreetmap.org/search?format=json&q={s}',
        #     zoom=12
        # ).add_to(m)
        
        # Add export controls
        self._add_export_controls(m)
        
        # Add 3D terrain visualization if LiDAR data exists
        if self._has_lidar_data():
            self._add_3d_terrain(m)
        
        # Save map with enhanced features
        if output_path is None:
            output_path = Path("archaeological_discovery_map.html")
        
        # Add custom CSS and JS for enhanced interactivity
        m.get_root().header.add_child(folium.Element("""
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
        <link rel="stylesheet" href="https://unpkg.com/leaflet-easybutton@2.4.0/src/easy-button.css">
        <script src="https://unpkg.com/leaflet-easybutton@2.4.0/src/easy-button.js"></script>
        """))
        
        m.save(str(output_path))
        logger.info(f"✓ Enhanced interactive map saved: {output_path}")
        
        return output_path

    def _add_discovery_heatmap(self, map_obj: folium.Map, analysis_results: Dict):
        """Add heatmap layer showing discovery density"""
        heat_data = []
        
        for zone_id, zone_results in analysis_results.items():
            for scene_result in zone_results:
                if scene_result.get('success'):
                    # Add terra preta points
                    tp_patches = scene_result.get('terra_preta', {}).get('patches', [])
                    for patch in tp_patches:
                        if patch.get('centroid'):
                            lat, lon = patch['centroid'][1], patch['centroid'][0]
                            heat_data.append([lat, lon, patch.get('confidence', 0.5)])
                    
                    # Add geometric feature points
                    geom_features = scene_result.get('geometric_features', [])
                    for feature in geom_features:
                        if feature.get('type') == 'circle' and feature.get('center'):
                            lat, lon = feature['center'][1], feature['center'][0]
                            heat_data.append([lat, lon, feature.get('confidence', 0.5)])
        
        if heat_data:
            plugins.HeatMap(
                heat_data,
                name='Discovery Heatmap',
                min_opacity=0.3,
                radius=25,
                blur=15,
                max_zoom=17,
                gradient={0.4: 'blue', 0.6: 'lime', 0.8: 'orange', 1.0: 'red'}
            ).add_to(map_obj)

    def _add_export_controls(self, map_obj: folium.Map):
        """Add export functionality to the map"""
        export_html = """
        <div style="position: fixed; top: 10px; right: 10px; z-index: 1000;">
            <button onclick="exportMap()" style="padding: 8px; background: white; border: 2px solid grey; border-radius: 5px;">
                Export Map Data
            </button>
        </div>
        <script>
        function exportMap() {
            // Implement export functionality
            alert('Export functionality would save current map view and data');
        }
        </script>
        """
        map_obj.get_root().html.add_child(folium.Element(export_html))

    def _has_lidar_data(self) -> bool:
        """Check if LiDAR data exists for 3D visualization"""
        lidar_dir = Path("data/lidar")
        return lidar_dir.exists() and any(lidar_dir.iterdir())

    def _add_3d_terrain(self, map_obj: folium.Map):
        """Add 3D terrain visualization if LiDAR data exists"""
        # This would be implemented with actual LiDAR data processing
        # Placeholder for the 3D visualization integration
        map_obj.get_root().html.add_child(folium.Element("""
        <div style="position: fixed; bottom: 20px; right: 20px; z-index: 1000;">
            <button onclick="toggle3D()" style="padding: 8px; background: white; border: 2px solid grey; border-radius: 5px;">
                Toggle 3D View
            </button>
        </div>
        <script>
        function toggle3D() {
            // Would integrate with actual 3D visualization library
            alert('3D visualization would show here with LiDAR data');
        }
        </script>
        """))
    
    def _add_target_zones(self, map_obj: folium.Map, scoring_results: Dict = None):
        """Add target zones to map"""
        
        zone_group = folium.FeatureGroup(name="Target Zones")
        
        for zone_id, zone in TARGET_ZONES.items():
            # Get score if available
            score = 0
            classification = "Not analyzed"
            if scoring_results and zone_id in scoring_results:
                score = scoring_results[zone_id]['total_score']
                classification = scoring_results[zone_id]['classification']
            
            # Determine color based on score
            if score >= 10:
                color = self.colors['high_confidence']
                fill_color = self.colors['high_confidence']
                alpha = 0.3
            elif score >= 7:
                color = self.colors['probable_feature']
                fill_color = self.colors['probable_feature'] 
                alpha = 0.25
            elif score >= 4:
                color = self.colors['possible_anomaly']
                fill_color = self.colors['possible_anomaly']
                alpha = 0.2
            else:
                color = '#2E86AB'  # Blue for unanalyzed
                fill_color = '#2E86AB'
                alpha = 0.15
            
            # Create bounding box
            bbox = zone.bbox  # (south, west, north, east)
            bounds = [[bbox[0], bbox[1]], [bbox[2], bbox[3]]]
            
            # Add zone rectangle
            folium.Rectangle(
                bounds=bounds,
                popup=folium.Popup(
                    html=self._create_zone_popup(zone, score, classification),
                    max_width=400
                ),
                tooltip=f"{zone.name} (Score: {score}/15)",
                color=color,
                weight=2,
                fill=True,
                fillColor=fill_color,
                fillOpacity=alpha
            ).add_to(zone_group)
            
            # Add center marker
            folium.CircleMarker(
                location=[zone.center[0], zone.center[1]],
                radius=8,
                popup=folium.Popup(
                    html=self._create_zone_popup(zone, score, classification),
                    max_width=400
                ),
                tooltip=f"{zone.name}",
                color='white',
                weight=2,
                fill=True,
                fillColor=color,
                fillOpacity=0.8
            ).add_to(zone_group)
        
        zone_group.add_to(map_obj)
    
    def _add_discoveries_with_cluster(self, map_obj: folium.Map, analysis_results: Dict, scoring_results: Dict = None):
        """Add discovered features to map with enhanced clustering"""
        
        # Collect marker data for clusters
        tp_data = []
        geom_data = []
        
        for zone_id, zone_results in analysis_results.items():
            if not zone_results:
                continue
                
            zone = TARGET_ZONES[zone_id]
            
            for scene_result in zone_results:
                if not scene_result.get('success'):
                    continue
                
                # Process terra preta patches
                tp_patches = scene_result.get('terra_preta', {}).get('patches', [])
                for patch in tp_patches:
                    if patch.get('centroid'):
                        confidence = patch.get('confidence', 0)
                        lat, lon = patch['centroid'][1], patch['centroid'][0]
                        tp_data.append({
                            'location': [lat, lon],
                            'popup': self._create_tp_popup(patch, zone.name),
                            'tooltip': f"Terra Preta ({confidence:.2f} confidence)",
                            'icon': folium.Icon(color='green', icon='leaf', prefix='fa')
                        })
                
                # Process geometric features
                geom_features = scene_result.get('geometric_features', [])
                for feature in geom_features:
                    feature_type = feature.get('type', 'unknown')
                    if feature_type == 'circle' and feature.get('center'):
                        center = feature.get('center', [0, 0])
                        lat, lon = (center[1], center[0]) if abs(center[0]) > 180 else (center[0], center[1])
                        geom_data.append({
                            'location': [lat, lon],
                            'popup': self._create_geom_popup(feature, zone.name),
                            'tooltip': f"Circular Feature ({feature.get('diameter_m', 0):.0f}m)",
                            'icon': folium.Icon(color='orange', icon='circle', prefix='fa')
                        })
                    elif feature_type == 'line':
                        start = feature.get('start', [0, 0])
                        end = feature.get('end', [0, 0])
                        if abs(start[0]) > 180:
                            start_lat, start_lon = start[1], start[0]
                            end_lat, end_lon = end[1], end[0]
                        else:
                            start_lat, start_lon = start[0], start[1]
                            end_lat, end_lon = end[0], end[1]
                        mid_lat = (start_lat + end_lat) / 2
                        mid_lon = (start_lon + end_lon) / 2
                        geom_data.append({
                            'location': [mid_lat, mid_lon],
                            'popup': self._create_geom_popup(feature, zone.name),
                            'tooltip': f"Linear Feature ({feature.get('length_m', 0):.0f}m)",
                            'icon': folium.Icon(color='blue', icon='minus', prefix='fa')
                        })
                    elif feature_type == 'rectangle':
                        center = feature.get('center', [0, 0])
                        lat, lon = (center[1], center[0]) if abs(center[0]) > 180 else (center[0], center[1])
                        geom_data.append({
                            'location': [lat, lon],
                            'popup': self._create_geom_popup(feature, zone.name),
                            'tooltip': f"Rectangular Feature ({feature.get('area_m2', 0):.0f} m²)",
                            'icon': folium.Icon(color='purple', icon='square', prefix='fa')
                        })
        
        # Create clusters with collected data
        if tp_data:
            tp_cluster = plugins.FastMarkerCluster(
                [],  # Empty data list to satisfy API requirement
                name="Terra Preta Signatures",
                overlay=True,
                control=True,
                disable_clustering_at_zoom=15,
                spiderfy_on_max_zoom=True
            ).add_to(map_obj)
            
            for marker_data in tp_data:
                folium.Marker(
                    location=marker_data['location'],
                    popup=marker_data['popup'],
                    tooltip=marker_data['tooltip'],
                    icon=marker_data['icon']
                ).add_to(tp_cluster)
        
        if geom_data:
            geom_cluster = plugins.FastMarkerCluster(
                [],  # Empty data list to satisfy API requirement
                name="Geometric Features",
                overlay=True,
                control=True,
                disable_clustering_at_zoom=15,
                spiderfy_on_max_zoom=True
            ).add_to(map_obj)
            
            for marker_data in geom_data:
                folium.Marker(
                    location=marker_data['location'],
                    popup=marker_data['popup'],
                    tooltip=marker_data['tooltip'],
                    icon=marker_data['icon']
                ).add_to(geom_cluster)

    def _add_geometric_feature_marker(self, group: plugins.FastMarkerCluster, feature: Dict, zone_name: str):
        """Add geometric feature with enhanced styling and interactivity"""
        feature_type = feature.get('type', 'unknown')
        confidence = feature.get('confidence', 0)
        if feature_type == 'circle':
            center = feature.get('center', [0, 0])
            diameter = feature.get('diameter_m', 100)
            if abs(center[0]) > 180:
                lat, lon = center[1], center[0]
            else:
                lat, lon = center[0], center[1]
            folium.Marker(
                location=[lat, lon],
                popup=self._create_geom_popup(feature, zone_name),
                tooltip=f"Circular Feature ({diameter:.0f}m)",
                icon=folium.Icon(color='orange', icon='circle', prefix='fa')
            ).add_to(group)
        elif feature_type == 'line':
            start = feature.get('start', [0, 0])
            end = feature.get('end', [0, 0])
            length = feature.get('length_m', 0)
            if abs(start[0]) > 180:
                start_lat, start_lon = start[1], start[0]
                end_lat, end_lon = end[1], end[0]
            else:
                start_lat, start_lon = start[0], start[1]
                end_lat, end_lon = end[0], end[1]
            folium.Marker(
                location=[(start_lat + end_lat) / 2, (start_lon + end_lon) / 2],
                popup=self._create_geom_popup(feature, zone_name),
                tooltip=f"Linear Feature ({length:.0f}m)",
                icon=folium.Icon(color='blue', icon='minus', prefix='fa')
            ).add_to(group)
        elif feature_type == 'rectangle':
            center = feature.get('center', [0, 0])
            area = feature.get('area_m2', 0)
            if abs(center[0]) > 180:
                lat, lon = center[1], center[0]
            else:
                lat, lon = center[0], center[1]
            folium.Marker(
                location=[lat, lon],
                popup=self._create_geom_popup(feature, zone_name),
                tooltip=f"Rectangular Feature ({area:.0f} m²)",
                icon=folium.Icon(color='purple', icon='square', prefix='fa')
            ).add_to(group)
    
    def _create_zone_popup(self, zone, score: float, classification: str) -> str:
        """Create HTML popup for target zone"""
        
        html = f"""
        <div style="width: 300px;">
            <h4 style="margin-bottom: 10px; color: #2E86AB;">{zone.name}</h4>
            <hr style="margin: 10px 0;">
            
            <p><strong>Coordinates:</strong> {zone.center[0]:.4f}°, {zone.center[1]:.4f}°</p>
            <p><strong>Priority:</strong> {zone.priority} {'⭐' * (4 - zone.priority)}</p>
            <p><strong>Anomaly Score:</strong> {score}/15</p>
            <p><strong>Classification:</strong> <span style="color: {'red' if score >= 10 else 'orange' if score >= 7 else 'blue'};">{classification}</span></p>
            
            <hr style="margin: 10px 0;">
            <p><strong>Expected Features:</strong><br>{zone.expected_features}</p>
            <p><strong>Historical Evidence:</strong><br>{zone.historical_evidence}</p>
            
            <hr style="margin: 10px 0;">
            <p><strong>Search Radius:</strong> {zone.search_radius_km} km</p>
            <p><strong>Feature Size Range:</strong> {zone.min_feature_size_m}-{zone.max_feature_size_m}m</p>
        </div>
        """
        return html
    
    def _create_tp_popup(self, patch: Dict, zone_name: str) -> str:
        """Create HTML popup for terra preta patch"""
        
        area_ha = patch.get('area_m2', 0) / 10000
        confidence = patch.get('confidence', 0)
        tp_index = patch.get('mean_tp_index', 0)
        ndvi = patch.get('mean_ndvi', 0)
        
        html = f"""
        <div style="width: 250px;">
            <h4 style="margin-bottom: 10px; color: #228B22;">Terra Preta Signature</h4>
            <hr style="margin: 10px 0;">
            
            <p><strong>Zone:</strong> {zone_name}</p>
            <p><strong>Area:</strong> {area_ha:.2f} hectares</p>
            <p><strong>Confidence:</strong> {confidence:.2f}</p>
            <p><strong>Terra Preta Index:</strong> {tp_index:.3f}</p>
            <p><strong>NDVI:</strong> {ndvi:.3f}</p>
            
            <hr style="margin: 10px 0;">
            <p style="font-size: 12px; color: #666;">
                Terra preta indicates anthropogenic soils from ancient human settlements.
                Higher confidence suggests stronger archaeological potential.
            </p>
        </div>
        """
        return html
    
    def _create_geom_popup(self, feature: Dict, zone_name: str) -> str:
        """Create HTML popup for geometric feature"""
        
        feature_type = feature.get('type', 'unknown')
        expected = feature.get('expected_feature', 'unknown')
        confidence = feature.get('confidence', 0)
        
        html = f"""
        <div style="width: 250px;">
            <h4 style="margin-bottom: 10px; color: #FF6B35;">{feature_type.title()} Feature</h4>
            <hr style="margin: 10px 0;">
            
            <p><strong>Zone:</strong> {zone_name}</p>
            <p><strong>Type:</strong> {expected}</p>
            <p><strong>Confidence:</strong> {confidence:.2f}</p>
        """
        
        if feature_type == 'circle':
            diameter = feature.get('diameter_m', 0)
            area = feature.get('area_m2', 0)
            html += f"""
            <p><strong>Diameter:</strong> {diameter:.0f} meters</p>
            <p><strong>Area:</strong> {area/10000:.2f} hectares</p>
            """
        elif feature_type == 'line':
            length = feature.get('length_m', 0)
            html += f"<p><strong>Length:</strong> {length:.0f} meters</p>"
        elif feature_type == 'rectangle':
            area = feature.get('area_m2', 0)
            width = feature.get('width_m', 0)
            height = feature.get('height_m', 0)
            html += f"""
            <p><strong>Dimensions:</strong> {width:.0f} × {height:.0f} meters</p>
            <p><strong>Area:</strong> {area/10000:.2f} hectares</p>
            """
        
        html += """
            <hr style="margin: 10px 0;">
            <p style="font-size: 12px; color: #666;">
                Geometric patterns may indicate ancient settlements, earthworks, or agricultural features.
            </p>
        </div>
        """
        return html
    
    def _add_legend(self, map_obj: folium.Map):
        """Add interactive legend to map with toggle functionality"""
        
        legend_html = """
        <div id="arch-legend" style="
            position: fixed; 
            bottom: 50px; 
            left: 50px; 
            width: 280px; /* Adjusted width for a more compact look */
            background-color: rgba(255, 255, 255, 0.95); /* Slightly less transparent */
            border: 1px solid #ccc; /* Lighter border */
            border-radius: 8px;
            z-index: 9999;
            font-family: 'Segoe UI', Arial, sans-serif; /* More modern font */
            padding: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15); /* Softer shadow */
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h4 style="margin: 0; color: #333; font-size: 1.1em;">Archaeological Discovery Legend</h4>
                <button id="legend-toggle" style="
                    background: #007bff; /* Bootstrap primary blue */
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 6px 12px; /* Slightly larger padding */
                    cursor: pointer;
                    font-size: 0.9em;
                    transition: background-color 0.2s ease; /* Smooth transition */
                ">Show</button>
            </div>
            
            <div id="legend-content" style="max-height: 300px; overflow-y: auto; display: none; /* Initially hidden */">
                <div style="margin-bottom: 15px;">
                    <h5 style="margin: 10px 0; color: #555; border-bottom: 1px solid #eee; padding-bottom: 5px; font-size: 1em;">Target Zones</h5>
                    <div style="display: grid; grid-template-columns: 1fr; gap: 8px; font-size: 0.9em;">
                        <div><i class="fa fa-square" style="color: #e74c3c; width: 15px; text-align: center;"></i> High Confidence</div>
                        <div><i class="fa fa-square" style="color: #f39c12; width: 15px; text-align: center;"></i> Probable Feature</div>
                        <div><i class="fa fa-square" style="color: #f1c40f; width: 15px; text-align: center;"></i> Possible Anomaly</div>
                        <div><i class="fa fa-square" style="color: #3498db; width: 15px; text-align: center;"></i> Not Analyzed</div>
                    </div>
                </div>
                
                <div style="margin-bottom: 15px;">
                    <h5 style="margin: 10px 0; color: #555; border-bottom: 1px solid #eee; padding-bottom: 5px; font-size: 1em;">Feature Types</h5>
                    <div style="display: grid; grid-template-columns: 1fr; gap: 8px; font-size: 0.9em;">
                        <div><i class="fa fa-leaf" style="color: #27ae60; width: 15px; text-align: center;"></i> Terra Preta</div>
                        <div><i class="fa fa-circle" style="color: #d35400; width: 15px; text-align: center;"></i> Circular</div>
                        <div><i class="fa fa-minus" style="color: #2980b9; width: 15px; text-align: center;"></i> Linear</div>
                        <div><i class="fa fa-square" style="color: #8e44ad; width: 15px; text-align: center;"></i> Rectangular</div>
                    </div>
                </div>
                
                <div style="background: #f0f8ff; padding: 10px; border-radius: 4px; border: 1px solid #e0e0e0;">
                    <h5 style="margin: 5px 0; color: #555; font-size: 1em;">Confidence Levels</h5>
                    <ul style="padding-left: 20px; margin: 5px 0; font-size: 0.85em; line-height: 1.4;">
                        <li><strong>High</strong>: Multiple strong indicators</li>
                        <li><strong>Probable</strong>: Some evidence of activity</li>
                        <li><strong>Possible</strong>: Requires investigation</li>
                    </ul>
                </div>
            </div>
        </div>
        
        <script>
            document.getElementById('legend-toggle').addEventListener('click', function() {
                const content = document.getElementById('legend-content');
                if (content.style.display === 'none') {
                    content.style.display = 'block';
                    this.textContent = 'Hide';
                    this.style.backgroundColor = '#dc3545'; /* Red for hide */
                } else {
                    content.style.display = 'none';
                    this.textContent = 'Show';
                    this.style.backgroundColor = '#007bff'; /* Blue for show */
                }
            });
        </script>
        """
        
        map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    def create_scoring_dashboard(self, scoring_results: Dict[str, Dict], 
                               output_path: Path = None) -> Path:
        """Create interactive dashboard for scoring results"""
        
        if not scoring_results:
            logger.warning("No scoring results provided")
            return None
        
        logger.info("Creating scoring dashboard...")
        
        # Prepare data
        zones = []
        scores = []
        classifications = []
        evidence_counts = []
        
        for zone_id, result in scoring_results.items():
            zones.append(TARGET_ZONES[zone_id].name)
            scores.append(result['total_score'])
            classifications.append(result['classification'])
            evidence_counts.append(result['evidence_count'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Anomaly Scores by Zone', 'Score Distribution', 
                          'Classification Summary', 'Evidence vs Score'),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # Bar chart of scores
        fig.add_trace(
            go.Bar(x=zones, y=scores, name="Anomaly Score",
                  marker_color=['red' if s >= 10 else 'orange' if s >= 7 else 'yellow' if s >= 4 else 'gray' 
                               for s in scores]),
            row=1, col=1
        )
        
        # Histogram of scores
        fig.add_trace(
            go.Histogram(x=scores, nbinsx=10, name="Score Distribution",
                        marker_color='lightblue'),
            row=1, col=2
        )
        
        # Pie chart of classifications
        class_counts = {}
        for c in classifications:
            class_counts[c] = class_counts.get(c, 0) + 1
        
        fig.add_trace(
            go.Pie(labels=list(class_counts.keys()), values=list(class_counts.values()),
                  name="Classifications"),
            row=2, col=1
        )
        
        # Scatter plot: Evidence count vs Score
        fig.add_trace(
            go.Scatter(x=evidence_counts, y=scores, mode='markers',
                      text=zones, name="Evidence vs Score",
                      marker=dict(size=10, color=scores, colorscale='RdYlBu_r')),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Amazon Archaeological Discovery - Scoring Dashboard",
            showlegend=False,
            height=800
        )
        
        # Save dashboard
        if output_path is None:
            output_path = Path("scoring_dashboard.html")
        
        fig.write_html(str(output_path))
        logger.info(f"✓ Scoring dashboard saved: {output_path}")
        
        return output_path
    
    def create_feature_analysis_plots(self, analysis_results: Dict, 
                                    output_dir: Path = None) -> List[Path]:
        """Create detailed feature analysis plots"""
        
        if output_dir is None:
            output_dir = Path("analysis_plots")
        output_dir.mkdir(exist_ok=True)
        
        logger.info("Creating feature analysis plots...")
        
        plot_paths = []
        
        # 1. Terra preta analysis
        tp_plot_path = self._create_terra_preta_plots(analysis_results, output_dir)
        if tp_plot_path:
            plot_paths.append(tp_plot_path)
        
        # 2. Geometric feature analysis
        geom_plot_path = self._create_geometric_plots(analysis_results, output_dir)
        if geom_plot_path:
            plot_paths.append(geom_plot_path)
        
        # 3. Zone comparison
        comp_plot_path = self._create_zone_comparison(analysis_results, output_dir)
        if comp_plot_path:
            plot_paths.append(comp_plot_path)
        
        logger.info(f"✓ Created {len(plot_paths)} analysis plots")
        return plot_paths
    
    def _create_terra_preta_plots(self, analysis_results: Dict, output_dir: Path) -> Path:
        """Create terra preta analysis plots"""
        
        # Collect terra preta data
        tp_data = []
        
        for zone_id, zone_results in analysis_results.items():
            zone_name = TARGET_ZONES[zone_id].name
            
            for scene_result in zone_results:
                if scene_result.get('success'):
                    patches = scene_result.get('terra_preta', {}).get('patches', [])
                    
                    for patch in patches:
                        tp_data.append({
                            'zone': zone_name,
                            'area_ha': patch.get('area_m2', 0) / 10000,
                            'confidence': patch.get('confidence', 0),
                            'tp_index': patch.get('mean_tp_index', 0),
                            'ndvi': patch.get('mean_ndvi', 0)
                        })
        
        if not tp_data:
            return None
        
        df = pd.DataFrame(tp_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Terra Preta Analysis', fontsize=16, fontweight='bold')
        
        # Area distribution by zone
        df.boxplot(column='area_ha', by='zone', ax=axes[0, 0])
        axes[0, 0].set_title('Terra Preta Area Distribution by Zone')
        axes[0, 0].set_ylabel('Area (hectares)')
        
        # Confidence vs TP Index
        for zone in df['zone'].unique():
            zone_data = df[df['zone'] == zone]
            axes[0, 1].scatter(zone_data['tp_index'], zone_data['confidence'], 
                             label=zone, alpha=0.7)
        axes[0, 1].set_xlabel('Terra Preta Index')
        axes[0, 1].set_ylabel('Confidence')
        axes[0, 1].set_title('Confidence vs Terra Preta Index')
        axes[0, 1].legend()
        
        # NDVI distribution
        axes[1, 0].hist(df['ndvi'], bins=20, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('NDVI')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('NDVI Distribution of Terra Preta Patches')
        
        # Size vs Confidence
        scatter = axes[1, 1].scatter(df['area_ha'], df['confidence'], 
                                   c=df['tp_index'], cmap='viridis', alpha=0.7)
        axes[1, 1].set_xlabel('Area (hectares)')
        axes[1, 1].set_ylabel('Confidence')
        axes[1, 1].set_title('Area vs Confidence (colored by TP Index)')
        plt.colorbar(scatter, ax=axes[1, 1], label='TP Index')
        
        plt.tight_layout()
        
        output_path = output_dir / "terra_preta_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_geometric_plots(self, analysis_results: Dict, output_dir: Path) -> Path:
        """Create geometric feature analysis plots"""
        
        # Collect geometric feature data
        geom_data = []
        
        for zone_id, zone_results in analysis_results.items():
            zone_name = TARGET_ZONES[zone_id].name
            
            for scene_result in zone_results:
                if scene_result.get('success'):
                    features = scene_result.get('geometric_features', [])
                    
                    for feature in features:
                        data_point = {
                            'zone': zone_name,
                            'type': feature.get('type', 'unknown'),
                            'confidence': feature.get('confidence', 0)
                        }
                        
                        if feature['type'] == 'circle':
                            data_point['size'] = feature.get('diameter_m', 0)
                            data_point['area'] = feature.get('area_m2', 0) / 10000
                        elif feature['type'] == 'line':
                            data_point['size'] = feature.get('length_m', 0)
                            data_point['area'] = 0
                        elif feature['type'] == 'rectangle':
                            data_point['size'] = np.sqrt(feature.get('area_m2', 0))
                            data_point['area'] = feature.get('area_m2', 0) / 10000
                        
                        geom_data.append(data_point)
        
        if not geom_data:
            return None
        
        df = pd.DataFrame(geom_data)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Geometric Feature Analysis', fontsize=16, fontweight='bold')
        
        # Feature type distribution
        type_counts = df['type'].value_counts()
        axes[0, 0].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Feature Type Distribution')
        
        # Size distribution by type
        for ftype in df['type'].unique():
            type_data = df[df['type'] == ftype]
            axes[0, 1].hist(type_data['size'], alpha=0.7, label=ftype, bins=15)
        axes[0, 1].set_xlabel('Size (m)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Feature Size Distribution')
        axes[0, 1].legend()
        
        # Confidence by zone and type
        df_pivot = df.pivot_table(values='confidence', index='zone', columns='type', aggfunc='mean')
        sns.heatmap(df_pivot, annot=True, cmap='YlOrRd', ax=axes[1, 0])
        axes[1, 0].set_title('Average Confidence by Zone and Type')
        
        # Size vs Confidence
        for ftype in df['type'].unique():
            type_data = df[df['type'] == ftype]
            axes[1, 1].scatter(type_data['size'], type_data['confidence'], 
                             label=ftype, alpha=0.7)
        axes[1, 1].set_xlabel('Size (m)')
        axes[1, 1].set_ylabel('Confidence')
        axes[1, 1].set_title('Size vs Confidence by Feature Type')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        output_path = output_dir / "geometric_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def _create_zone_comparison(self, analysis_results: Dict, output_dir: Path) -> Path:
        """Create zone comparison plots"""
        
        # Collect zone summary data
        zone_data = []
        
        for zone_id, zone_results in analysis_results.items():
            zone = TARGET_ZONES[zone_id]
            
            tp_count = 0
            geom_count = 0
            total_tp_area = 0
            
            for scene_result in zone_results:
                if scene_result.get('success'):
                    tp_patches = scene_result.get('terra_preta', {}).get('patches', [])
                    tp_count += len(tp_patches)
                    total_tp_area += sum(p.get('area_m2', 0) for p in tp_patches)
                    
                    geom_features = scene_result.get('geometric_features', [])
                    geom_count += len(geom_features)
            
            zone_data.append({
                'zone': zone.name,
                'priority': zone.priority,
                'tp_patches': tp_count,
                'geometric_features': geom_count,
                'total_features': tp_count + geom_count,
                'tp_area_ha': total_tp_area / 10000,
                'scenes_analyzed': len([s for s in zone_results if s.get('success')])
            })
        
        df = pd.DataFrame(zone_data)
        
        if df.empty:
            return None
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Zone Comparison Analysis', fontsize=16, fontweight='bold')
        
        # Total features by zone
        bars = axes[0, 0].bar(df['zone'], df['total_features'], 
                             color=['red' if p == 1 else 'orange' if p == 2 else 'blue' 
                                   for p in df['priority']])
        axes[0, 0].set_title('Total Features Detected by Zone')
        axes[0, 0].set_ylabel('Number of Features')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Feature type breakdown
        width = 0.35
        x = np.arange(len(df))
        axes[0, 1].bar(x - width/2, df['tp_patches'], width, label='Terra Preta', alpha=0.8)
        axes[0, 1].bar(x + width/2, df['geometric_features'], width, label='Geometric', alpha=0.8)
        axes[0, 1].set_title('Feature Type Breakdown by Zone')
        axes[0, 1].set_ylabel('Number of Features')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(df['zone'], rotation=45)
        axes[0, 1].legend()
        
        # Terra preta area by zone
        axes[1, 0].pie(df['tp_area_ha'], labels=df['zone'], autopct='%1.1f%%')
        axes[1, 0].set_title('Terra Preta Area Distribution')
        
        # Priority vs Discovery rate
        axes[1, 1].scatter(df['priority'], df['total_features'], 
                          s=df['scenes_analyzed']*50, alpha=0.7)
        axes[1, 1].set_xlabel('Zone Priority')
        axes[1, 1].set_ylabel('Total Features')
        axes[1, 1].set_title('Priority vs Discovery Rate\n(bubble size = scenes analyzed)')
        
        # Add zone labels to scatter plot
        for i, row in df.iterrows():
            axes[1, 1].annotate(row['zone'].split()[0], 
                               (row['priority'], row['total_features']),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        output_path = output_dir / "zone_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

if __name__ == "__main__":
    # Test visualization
    print("Testing archaeological visualization...")
    
    # Mock test data
    mock_analysis = {
        'negro_madeira': [{
            'success': True,
            'terra_preta': {
                'patches': [
                    {'centroid': (-60.0, -3.167), 'area_m2': 5000, 'confidence': 0.8}
                ]
            },
            'geometric_features': [
                {'type': 'circle', 'center': (-60.0, -3.167), 'diameter_m': 200, 'confidence': 0.7}
            ]
        }]
    }
    
    mock_scoring = {
        'negro_madeira': {
            'total_score': 12,
            'classification': 'HIGH CONFIDENCE ARCHAEOLOGICAL SITE',
            'evidence_count': 4
        }
    }
    
    visualizer = ArchaeologicalVisualizer()
    
    # Test map creation
    map_path = visualizer.create_discovery_map(mock_analysis, mock_scoring, Path("test_map.html"))
    print(f"✓ Test map created: {map_path}")
    
    # Test dashboard
    dashboard_path = visualizer.create_scoring_dashboard(mock_scoring, Path("test_dashboard.html"))
    print(f"✓ Test dashboard created: {dashboard_path}")
    
    print("Visualization tests completed")

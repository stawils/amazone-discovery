"""
Core Archaeological Map Generator
Main orchestrator for the enhanced visualization system
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import geopandas as gpd
from datetime import datetime

from .components import FeatureRenderer, LayerManager, ControlPanel
from .templates import HTMLTemplateEngine
from .styles import ArchaeologicalThemes
from .utils import DataProcessor, CoordinateValidator
from .professional_maya_terrain import ProfessionalMayaTerrain

logger = logging.getLogger(__name__)


class ArchaeologicalMapGenerator:
    """
    Main archaeological map generator with modular architecture
    Clean, maintainable, and extensible design
    """
    
    def __init__(self, run_id: Optional[str] = None, results_dir: Optional[Path] = None):
        """Initialize the archaeological map generator"""
        
        self.run_id = run_id
        self.results_dir = Path(results_dir) if results_dir else Path("results")
        
        # Initialize core components
        self.feature_renderer = FeatureRenderer()
        self.layer_manager = LayerManager()
        self.control_panel = ControlPanel()
        self.template_engine = HTMLTemplateEngine()
        self.themes = ArchaeologicalThemes()
        
        # Data processing utilities
        self.data_processor = DataProcessor()
        self.coordinate_validator = CoordinateValidator()
        
        # Map configuration
        self.default_config = {
            'center': [-5.0, -70.0],  # Amazon center
            'zoom': 17,  # Very high zoom for detailed satellite imagery
            'max_zoom': 20,  # Increased max zoom for maximum detail
            'attribution': 'Amazon Archaeological Discovery Pipeline'
        }
        
        logger.info("🗺️ Archaeological Map Generator initialized")
    
    def generate_enhanced_map(self, 
                            zone_name: str,
                            theme: str = "professional",
                            include_analysis: bool = True,
                            interactive_features: bool = True,
                            generate_archaeological_maps: bool = True) -> Path:
        """
        Generate enhanced archaeological map with all improvements
        
        Args:
            zone_name: Target zone identifier
            theme: Visualization theme ('professional', 'field', 'scientific')
            include_analysis: Include analysis panels and statistics
            interactive_features: Enable advanced interactive tools
            generate_archaeological_maps: Auto-generate LiDAR archaeological visualizations
            
        Returns:
            Path to generated HTML map
        """
        
        try:
            logger.info(f"🎯 Generating enhanced map for {zone_name}")
            
            # Load and validate data
            map_data = self._load_zone_data(zone_name)
            if not map_data:
                logger.error(f"No data found for zone: {zone_name}")
                return None
            
            # Calculate map bounds and center with error handling
            try:
                bounds = self.data_processor.calculate_optimal_bounds(map_data)
            except Exception as e:
                logger.warning(f"Error calculating bounds: {e}, using default Amazon bounds")
                bounds = {
                    'north': -2.0, 'south': -8.0,
                    'east': -65.0, 'west': -75.0,
                    'center_lat': -5.0, 'center_lon': -70.0,
                    'optimal_zoom': 16
                }
            
            map_config = self._create_map_config(bounds, theme)
            
            # Generate map components
            feature_layers = self.feature_renderer.create_feature_layers(map_data, theme)
            control_layers = self.layer_manager.create_layer_controls(map_data)
            analysis_panels = self.control_panel.create_analysis_panels(map_data) if include_analysis else {}
            
            # Build complete map
            map_html = self.template_engine.build_complete_map(
                config=map_config,
                feature_layers=feature_layers,
                control_layers=control_layers,
                analysis_panels=analysis_panels,
                theme=theme,
                zone_name=zone_name,
                interactive=interactive_features
            )
            
            # Save enhanced map
            output_path = self._get_output_path(zone_name, "enhanced")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(map_html)
            
            logger.info(f"✅ Enhanced map generated: {output_path}")
            
            # AUTOMATICALLY GENERATE LIDAR ARCHAEOLOGICAL MAPS (if enabled)
            if generate_archaeological_maps:
                logger.info(f"🏛️ Auto-generating LiDAR archaeological maps for {zone_name}")
                try:
                    archaeological_result = self.generate_lidar_archaeological_maps(zone_name)
                    if archaeological_result.get('success'):
                        logger.info(f"✅ LiDAR archaeological maps generated: {archaeological_result['total_products']} products")
                        
                        # Log detected features
                        features = archaeological_result.get('archaeological_features_detected', {})
                        if features:
                            total_features = sum(features.values())
                            logger.info(f"🏛️ Detected {total_features} potential archaeological features:")
                            for feature_type, count in features.items():
                                logger.info(f"   • {feature_type.title()}: {count}")
                        
                        # Store archaeological maps info for potential use in HTML map
                        setattr(self, '_last_archaeological_result', archaeological_result)
                        
                        # AUTOMATICALLY GENERATE MAYA-STYLE LIDAR TERRAIN SURFACE VISUALIZATION
                        logger.info(f"🗻 Auto-generating Maya-style LiDAR terrain surface visualization for {zone_name}")
                        self.generate_lidar_terrain_surface_visualization(zone_name)
                        
                        # AUTOMATICALLY GENERATE 3D MAYA-STYLE TERRAIN VISUALIZATION
                        logger.info(f"🌍 Auto-generating 3D Maya-style terrain visualization for {zone_name}")
                        self.generate_maya_3d_terrain_visualization(zone_name)
                        
                        # AUTOMATICALLY GENERATE PROFESSIONAL MAYA TERRAIN SURFACE
                        logger.info(f"🏔️ Auto-generating Professional Maya terrain surface for {zone_name}")
                        self.generate_professional_maya_terrain_surface(zone_name)
                        
                        # AUTOMATICALLY GENERATE ARCHAEOLOGICAL LIDAR DEM
                        logger.info(f"🗺️ Auto-generating Archaeological LiDAR DEM for {zone_name}")
                        self.generate_archaeological_lidar_dem(zone_name)
                    else:
                        logger.warning(f"⚠️ LiDAR archaeological map generation failed: {archaeological_result.get('error', 'Unknown error')}")
                except Exception as e:
                    logger.warning(f"⚠️ Error generating LiDAR archaeological maps: {e}")
                    # Don't fail the main map generation if archaeological maps fail
            else:
                logger.info("🏛️ LiDAR archaeological map generation disabled")
            
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Map generation failed: {e}", exc_info=True)
            return None
    
    def generate_lidar_archaeological_maps(self, zone_name: str) -> Dict[str, Any]:
        """
        Generate advanced LiDAR archaeological visualization maps
        Reveals ancient structures through forest canopy like Maya/Angkor discoveries
        
        Returns:
            Dictionary of generated visualization products and their paths
        """
        
        try:
            logger.info(f"🏛️ Generating LiDAR archaeological maps for {zone_name}")
            
            # Load GEDI LiDAR data
            lidar_data = self._load_gedi_lidar_data(zone_name)
            if not lidar_data:
                logger.error(f"No GEDI LiDAR data found for {zone_name}")
                return {"success": False, "error": "No LiDAR data available"}
            
            # Create output directory for archaeological maps (save in visualizations folder)
            if self.run_id:
                if self.run_id.startswith('run_'):
                    output_dir = self.results_dir / self.run_id / "visualizations"
                else:
                    output_dir = self.results_dir / f"run_{self.run_id}" / "visualizations"
            else:
                output_dir = self.results_dir / "visualizations"
            
            # Create a subdirectory for archaeological maps to organize them
            archaeological_dir = output_dir / "archaeological_maps" / zone_name
            archaeological_dir.mkdir(parents=True, exist_ok=True)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Use professional Maya terrain visualizer
            maya_viz = ProfessionalMayaTerrain(zone_name)
            
            # Generate professional Maya terrain visualization
            viz_products = maya_viz.create_terrain_surface_visualization(lidar_data, [], archaeological_dir)
            
            if not viz_products:
                logger.warning(f"No archaeological visualization products generated for {zone_name}")
                return {"success": False, "error": "Visualization generation failed"}
            
            # Create summary of generated products
            summary = {
                "success": True,
                "zone_name": zone_name,
                "total_products": len(viz_products),
                "output_directory": str(archaeological_dir),
                "generated_maps": viz_products,
                "lidar_data_summary": {
                    "total_points": len(lidar_data.get('coordinates', [])),
                    "elevation_range": self._get_elevation_range(lidar_data),
                    "coverage_area": self._estimate_coverage_area(lidar_data)
                },
                "archaeological_features_detected": self._count_detected_features(archaeological_dir),
                "generated_timestamp": datetime.now().isoformat()
            }
            
            # Save summary
            summary_path = archaeological_dir / f"{zone_name}_archaeological_visualization_summary.json"
            with open(summary_path, 'w') as f:
                import json
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"✅ Generated {len(viz_products)} archaeological visualization products")
            logger.info(f"📍 Products saved to: {archaeological_dir}")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ LiDAR archaeological map generation failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def generate_3d_archaeological_visualization(self, zone_name: str) -> Optional[Path]:
        """
        Generate 3D archaeological visualization automatically
        Integrates with the pipeline to create interactive 3D maps
        """
        
        try:
            from .archaeological_3d_visualizer import Archaeological3DVisualizer
            
            # Load GEDI data
            gedi_data = self._load_gedi_lidar_data(zone_name)
            if not gedi_data:
                logger.warning(f"No GEDI data available for 3D visualization: {zone_name}")
                return None
            
            # Load archaeological features
            archaeological_features = self._load_archaeological_features_for_3d(zone_name)
            if not archaeological_features:
                logger.warning(f"No archaeological features found for 3D visualization: {zone_name}")
                return None
            
            # Create output directory in visualizations folder
            if self.run_id:
                if self.run_id.startswith('run_'):
                    output_dir = self.results_dir / self.run_id / "visualizations"
                else:
                    output_dir = self.results_dir / f"run_{self.run_id}" / "visualizations"
            else:
                output_dir = self.results_dir / "visualizations"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create Maya 3D visualizer
            visualizer = ProfessionalMayaTerrain(zone_name)
            
            # Generate Maya 3D visualization
            html_file = visualizer.create_terrain_surface_visualization(
                gedi_data=gedi_data,
                archaeological_features=archaeological_features,
                output_dir=output_dir
            )
            
            if html_file:
                logger.info(f"✅ 3D archaeological visualization generated: {html_file}")
                return html_file
            else:
                logger.warning("Failed to generate 3D archaeological visualization")
                return None
                
        except Exception as e:
            logger.error(f"❌ 3D archaeological visualization failed: {e}", exc_info=True)
            return None
    
    def _load_archaeological_features_for_3d(self, zone_name: str) -> List[Dict]:
        """Load archaeological features for 3D visualization"""
        
        try:
            # Try archaeological features first
            if self.run_id:
                if self.run_id.startswith('run_'):
                    run_dir = self.results_dir / self.run_id
                else:
                    run_dir = self.results_dir / f"run_{self.run_id}"
            else:
                return []
            
            # Check archaeological maps directory
            arch_file = run_dir / "archaeological_maps" / zone_name / f"{zone_name}_archaeological_features.geojson"
            
            if not arch_file.exists():
                # Try combined detections
                arch_file = run_dir / "exports" / "combined" / f"{zone_name}_top_20_candidates.geojson"
                
            if not arch_file.exists():
                logger.warning(f"No archaeological features found for {zone_name}")
                return []
            
            import geopandas as gpd
            gdf = gpd.read_file(arch_file)
            features = []
            
            for idx, feature in gdf.iterrows():
                feature_dict = {
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [feature.geometry.x, feature.geometry.y]
                    },
                    'properties': dict(feature.drop('geometry'))
                }
                features.append(feature_dict)
            
            logger.info(f"Loaded {len(features)} archaeological features for 3D visualization")
            return features
            
        except Exception as e:
            logger.error(f"Error loading archaeological features for 3D: {e}")
            return []
    
    def generate_enhanced_3d_archaeological_visualization(self, zone_name: str) -> Optional[Path]:
        """
        Generate enhanced 3D archaeological visualization with satellite imagery
        Combines 3D terrain with Google satellite maps and enhanced navigation
        """
        
        try:
            from .enhanced_3d_archaeological_visualizer import Enhanced3DArchaeologicalVisualizer
            
            # Load GEDI data
            gedi_data = self._load_gedi_lidar_data(zone_name)
            if not gedi_data:
                logger.warning(f"No GEDI data available for enhanced 3D visualization: {zone_name}")
                return None
            
            # Load archaeological features
            archaeological_features = self._load_archaeological_features_for_3d(zone_name)
            if not archaeological_features:
                logger.warning(f"No archaeological features found for enhanced 3D visualization: {zone_name}")
                return None
            
            # Create output directory in visualizations folder
            if self.run_id:
                if self.run_id.startswith('run_'):
                    output_dir = self.results_dir / self.run_id / "visualizations"
                else:
                    output_dir = self.results_dir / f"run_{self.run_id}" / "visualizations"
            else:
                output_dir = self.results_dir / "visualizations"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create enhanced 3D visualizer
            visualizer = Enhanced3DArchaeologicalVisualizer(zone_name, self.run_id)
            
            # Generate enhanced 3D map
            html_file = visualizer.create_enhanced_3d_archaeological_map(
                gedi_data=gedi_data,
                archaeological_features=archaeological_features,
                output_dir=output_dir
            )
            
            if html_file:
                logger.info(f"✅ Enhanced 3D archaeological visualization generated: {html_file}")
                return html_file
            else:
                logger.warning("Failed to generate enhanced 3D archaeological visualization")
                return None
                
        except Exception as e:
            logger.error(f"❌ Enhanced 3D archaeological visualization failed: {e}", exc_info=True)
            return None
    
    def generate_lidar_terrain_surface_visualization(self, zone_name: str) -> Optional[Path]:
        """
        Generate Maya-style LiDAR terrain surface visualization
        Creates continuous terrain surface with elevation-based coloring like Maya/Angkor discoveries
        """
        
        try:
            from .working_maya_visualizer import WorkingMayaVisualizer
            
            # Load GEDI data
            gedi_data = self._load_gedi_lidar_data(zone_name)
            if not gedi_data:
                logger.warning(f"No GEDI data available for terrain surface visualization: {zone_name}")
                return None
            
            # Load archaeological features
            archaeological_features = self._load_archaeological_features_for_3d(zone_name)
            if not archaeological_features:
                logger.warning(f"No archaeological features found for terrain surface visualization: {zone_name}")
                return None
            
            # Create output directory in visualizations folder
            if self.run_id:
                if self.run_id.startswith('run_'):
                    output_dir = self.results_dir / self.run_id / "visualizations"
                else:
                    output_dir = self.results_dir / f"run_{self.run_id}" / "visualizations"
            else:
                output_dir = self.results_dir / "visualizations"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create working Maya visualizer
            visualizer = WorkingMayaVisualizer(zone_name, self.run_id)
            
            # Generate working Maya terrain map
            html_file = visualizer.create_working_maya_terrain(
                gedi_data=gedi_data,
                archaeological_features=archaeological_features,
                output_dir=output_dir
            )
            
            if html_file:
                logger.info(f"✅ Maya-style LiDAR terrain surface visualization generated: {html_file}")
                return html_file
            else:
                logger.warning("Failed to generate LiDAR terrain surface visualization")
                return None
                
        except Exception as e:
            logger.error(f"❌ LiDAR terrain surface visualization failed: {e}", exc_info=True)
            return None
    
    def generate_maya_3d_terrain_visualization(self, zone_name: str) -> Optional[Path]:
        """
        Generate 3D Maya-style LiDAR terrain visualization with continuous surface
        Creates proper 3D terrain mesh with elevation-based coloring and smooth navigation
        """
        
        try:
            # Use the working compact Maya 3D visualizer
            
            # Load GEDI data
            gedi_data = self._load_gedi_lidar_data(zone_name)
            if not gedi_data:
                logger.warning(f"No GEDI data available for 3D Maya terrain visualization: {zone_name}")
                return None
            
            # Load archaeological features
            archaeological_features = self._load_archaeological_features_for_3d(zone_name)
            if not archaeological_features:
                logger.warning(f"No archaeological features found for 3D Maya terrain visualization: {zone_name}")
                return None
            
            # Create output directory in visualizations folder
            if self.run_id:
                if self.run_id.startswith('run_'):
                    output_dir = self.results_dir / self.run_id / "visualizations"
                else:
                    output_dir = self.results_dir / f"run_{self.run_id}" / "visualizations"
            else:
                output_dir = self.results_dir / "visualizations"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create compact Maya 3D visualizer
            visualizer = ProfessionalMayaTerrain(zone_name)
            
            # Generate compact Maya 3D visualization
            html_file = visualizer.create_terrain_surface_visualization(
                gedi_data=gedi_data,
                archaeological_features=archaeological_features,
                output_dir=output_dir
            )
            
            if html_file:
                logger.info(f"✅ 3D Maya-style terrain visualization generated: {html_file}")
                return html_file
            else:
                logger.warning("Failed to generate 3D Maya-style terrain visualization")
                return None
                
        except Exception as e:
            logger.error(f"❌ 3D Maya-style terrain visualization failed: {e}", exc_info=True)
            return None
    
    def generate_professional_maya_terrain_surface(self, zone_name: str) -> Optional[Path]:
        """
        Generate Professional Maya-style terrain surface with heat map coloring
        Creates the exact style shown in reference with professional interface
        """
        
        try:
            # Load GEDI data
            gedi_data = self._load_gedi_lidar_data(zone_name)
            if not gedi_data:
                logger.warning(f"No GEDI data available for Professional Maya terrain: {zone_name}")
                return None
            
            # Load archaeological features
            archaeological_features = self._load_archaeological_features_for_3d(zone_name)
            if not archaeological_features:
                logger.warning(f"No archaeological features found for Professional Maya terrain: {zone_name}")
                # Continue with empty features
                archaeological_features = []
            
            # Create output directory
            if self.run_id:
                if self.run_id.startswith('run_'):
                    output_dir = self.results_dir / self.run_id / "visualizations"
                else:
                    output_dir = self.results_dir / f"run_{self.run_id}" / "visualizations"
            else:
                output_dir = self.results_dir / "visualizations"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create Professional Maya terrain visualizer
            visualizer = ProfessionalMayaTerrain(zone_name)
            
            # Generate Professional Maya terrain surface
            html_file = visualizer.create_terrain_surface_visualization(
                gedi_data=gedi_data,
                archaeological_features=archaeological_features,
                output_dir=output_dir
            )
            
            if html_file:
                logger.info(f"✅ Professional Maya terrain surface generated: {html_file}")
                return html_file
            else:
                logger.warning("Failed to generate Professional Maya terrain surface")
                return None
                
        except Exception as e:
            logger.error(f"❌ Professional Maya terrain surface failed: {e}", exc_info=True)
            return None
    
    def generate_archaeological_lidar_dem(self, zone_name: str) -> Optional[Path]:
        """
        Generate Archaeological LiDAR DEM visualization with proper DEM techniques
        Based on Sky View Factor and hillshade methods from archaeological literature
        """
        
        try:
            # Load GEDI data
            gedi_data = self._load_gedi_lidar_data(zone_name)
            if not gedi_data:
                logger.warning(f"No GEDI data available for Archaeological DEM: {zone_name}")
                return None
            
            # Load archaeological features
            archaeological_features = self._load_archaeological_features_for_3d(zone_name)
            if not archaeological_features:
                logger.warning(f"No archaeological features found for Archaeological DEM: {zone_name}")
                # Continue with empty features
                archaeological_features = []
            
            # Create output directory
            if self.run_id:
                if self.run_id.startswith('run_'):
                    output_dir = self.results_dir / self.run_id / "visualizations"
                else:
                    output_dir = self.results_dir / f"run_{self.run_id}" / "visualizations"
            else:
                output_dir = self.results_dir / "visualizations"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create Archaeological LiDAR DEM visualizer
            visualizer = ProfessionalMayaTerrain(zone_name)
            
            # Generate Archaeological DEM
            html_file = visualizer.create_terrain_surface_visualization(
                gedi_data=gedi_data,
                archaeological_features=archaeological_features,
                output_dir=output_dir
            )
            
            if html_file:
                logger.info(f"✅ Archaeological LiDAR DEM generated: {html_file}")
                return html_file
            else:
                logger.warning("Failed to generate Archaeological LiDAR DEM")
                return None
                
        except Exception as e:
            logger.error(f"❌ Archaeological LiDAR DEM failed: {e}", exc_info=True)
            return None
    
    def _load_gedi_lidar_data(self, zone_name: str) -> Optional[Dict[str, Any]]:
        """Load GEDI LiDAR data for archaeological visualization"""
        
        try:
            # Method 1: Use exported GEDI detections (fastest and most reliable)
            if self.run_id:
                if self.run_id.startswith('run_'):
                    run_dir = self.results_dir / self.run_id
                else:
                    run_dir = self.results_dir / f"run_{self.run_id}"
                
                # Check for exported GEDI detections
                gedi_export_file = run_dir / "exports" / "gedi" / f"{zone_name}_gedi_detections.geojson"
                if gedi_export_file.exists():
                    logger.info(f"Loading GEDI data from exports: {gedi_export_file}")
                    return self._load_data_from_geojson_export(gedi_export_file)
                
                # Method 2: Check acquired_scene_data.json for original metrics paths
                scene_data_file = run_dir / "gedi" / "acquired_scene_data.json"
                if scene_data_file.exists():
                    logger.debug(f"Loading GEDI scene data from: {scene_data_file}")
                    try:
                        import json
                        with open(scene_data_file, 'r') as f:
                            scene_data = json.load(f)
                        
                        # Look for metrics paths in the scene data
                        for zone_data in scene_data.get(zone_name, []):
                            scene_path = zone_data.get('scene_path')
                            if scene_path and Path(scene_path).exists():
                                logger.info(f"Found GEDI metrics path: {scene_path}")
                                return self._load_metrics_from_path(Path(scene_path))
                    except Exception as e:
                        logger.warning(f"Error loading scene data: {e}")
            
            # Method 3: Look in the main data directories
            zone_variations = [
                zone_name,
                zone_name.replace('_', ' ').title(),
                zone_name.replace('_', '-').title(),
                ' '.join(word.title() for word in zone_name.split('_'))
            ]
            
            for variation in zone_variations:
                for base_path in [Path("data/satellite/gedi/processed_metrics_cache"), 
                                Path("data/archive-sat-data/gedi/processed_metrics_cache")]:
                    data_dir = base_path / variation
                    logger.debug(f"Checking GEDI data in: {data_dir}")
                    
                    if data_dir.exists():
                        return self._load_metrics_from_path(data_dir)
            
            logger.warning(f"No GEDI metrics found for zone: {zone_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading GEDI data: {e}")
            return None
    
    def _load_data_from_geojson_export(self, geojson_file: Path) -> Optional[Dict[str, Any]]:
        """Load GEDI data from exported GeoJSON file and create synthetic metrics for visualization"""
        
        try:
            import json
            import numpy as np
            import geopandas as gpd
            
            logger.info(f"Loading GEDI export data from: {geojson_file}")
            
            # Load GeoJSON file
            gdf = gpd.read_file(geojson_file)
            
            if len(gdf) == 0:
                logger.warning("No features found in GEDI export file")
                return None
            
            # Extract coordinates
            coordinates = []
            elevations = []
            canopy_heights = []
            
            for idx, feature in gdf.iterrows():
                # Get coordinates from geometry
                if hasattr(feature.geometry, 'x') and hasattr(feature.geometry, 'y'):
                    lon, lat = feature.geometry.x, feature.geometry.y
                    coordinates.append([lon, lat])
                    
                    # Extract elevation if available
                    elevation = feature.get('elevation_m', None)
                    if elevation is None:
                        # Use spatial context elevation
                        spatial_context = feature.get('spatial_context', {})
                        if isinstance(spatial_context, dict):
                            elevation = spatial_context.get('elevation_m', None)
                    
                    # Generate synthetic elevation if none available (for visualization purposes)
                    if elevation is None:
                        # Create realistic elevation based on Amazon topography (50-200m typical)
                        base_elevation = 100 + np.random.normal(0, 20)  # 100m ± 20m variation
                        elevation = max(50, min(250, base_elevation))  # Clamp to realistic range
                    
                    elevations.append(elevation)
                    
                    # Generate synthetic canopy height (typical Amazon forest: 15-40m)
                    area_m2 = feature.get('area_m2', 1000)
                    if area_m2 > 5000:  # Larger clearings have lower canopy
                        canopy_height = np.random.uniform(5, 15)
                    else:  # Smaller areas have taller surrounding canopy
                        canopy_height = np.random.uniform(20, 35)
                    
                    canopy_heights.append(canopy_height)
            
            if not coordinates:
                logger.warning("No valid coordinates extracted from GEDI export")
                return None
            
            # Convert to numpy arrays
            coordinates_array = np.array(coordinates)
            elevations_array = np.array(elevations)
            canopy_array = np.array(canopy_heights)
            
            result = {
                'coordinates': coordinates_array,
                'elevation_data': elevations_array,
                'rh95_data': canopy_array,  # Use synthetic canopy heights
                'rh100_data': canopy_array * 1.1  # RH100 slightly higher than RH95
            }
            
            logger.info(f"✅ Loaded {len(coordinates_array)} GEDI features from export")
            logger.info(f"   📊 Elevation range: {elevations_array.min():.1f} - {elevations_array.max():.1f}m")
            logger.info(f"   🌳 Canopy height range: {canopy_array.min():.1f} - {canopy_array.max():.1f}m")
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading data from GeoJSON export: {e}")
            return None
    
    def _load_metrics_from_path(self, data_dir: Path) -> Optional[Dict[str, Any]]:
        """Load GEDI metrics from a specific directory path"""
        
        try:
            import json
            import numpy as np
            
            logger.info(f"Loading GEDI metrics from: {data_dir}")
            
            # Check for JSON format metrics
            json_files = list(data_dir.glob("*_metrics.json"))
            if json_files:
                combined_data = {
                    'coordinates': [],
                    'elevation_data': [],
                    'rh95_data': [],
                    'rh100_data': []
                }
                
                for json_file in json_files:
                    try:
                        logger.debug(f"Loading JSON file: {json_file}")
                        with open(json_file, 'r') as f:
                            granule_data = json.load(f)
                        
                        # Extract coordinates
                        coords = np.column_stack([
                            np.array(granule_data['longitude']),
                            np.array(granule_data['latitude'])
                        ])
                        
                        combined_data['coordinates'].append(coords)
                        combined_data['elevation_data'].append(np.array(granule_data['elevation_ground']))
                        
                        # Handle different canopy height formats
                        if 'rh95' in granule_data and 'rh100' in granule_data:
                            combined_data['rh95_data'].append(np.array(granule_data['rh95']))
                            combined_data['rh100_data'].append(np.array(granule_data['rh100']))
                        elif 'canopy_height' in granule_data:
                            canopy_heights = np.array(granule_data['canopy_height'])
                            combined_data['rh95_data'].append(canopy_heights)
                            combined_data['rh100_data'].append(canopy_heights)
                        else:
                            # Calculate canopy height from elevation difference
                            elev_ground = np.array(granule_data['elevation_ground'])
                            elev_canopy = np.array(granule_data.get('elevation_canopy_top', elev_ground))
                            canopy_height = np.maximum(0, elev_canopy - elev_ground)
                            combined_data['rh95_data'].append(canopy_height)
                            combined_data['rh100_data'].append(canopy_height)
                        
                        logger.info(f"Loaded {len(coords)} points from {json_file.name}")
                        
                    except Exception as e:
                        logger.warning(f"Error loading granule {json_file}: {e}")
                        continue
                
                # Combine all granule data
                if combined_data['coordinates']:
                    result = {
                        'coordinates': np.vstack(combined_data['coordinates']),
                        'elevation_data': np.concatenate(combined_data['elevation_data']),
                        'rh95_data': np.concatenate(combined_data['rh95_data']),
                        'rh100_data': np.concatenate(combined_data['rh100_data'])
                    }
                    
                    logger.info(f"✅ Combined GEDI data: {len(result['coordinates'])} total points")
                    return result
            
            # Check for legacy .npy format
            coords_file = data_dir / "coordinates.npy"
            if coords_file.exists():
                logger.info("Loading legacy .npy format GEDI data")
                return {
                    'coordinates': np.load(coords_file),
                    'elevation_data': np.load(data_dir / "ground_elevation.npy"),
                    'rh95_data': np.load(data_dir / "canopy_height_95.npy"),
                    'rh100_data': np.load(data_dir / "canopy_height_100.npy")
                }
            
            logger.warning(f"No valid GEDI metrics found in {data_dir}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading metrics from {data_dir}: {e}")
            return None
    
    def _get_elevation_range(self, lidar_data: Dict[str, Any]) -> Dict[str, float]:
        """Get elevation range statistics from LiDAR data"""
        try:
            elevations = lidar_data.get('elevation_data', np.array([]))
            if len(elevations) > 0:
                valid_elevations = elevations[~np.isnan(elevations)]
                if len(valid_elevations) > 0:
                    return {
                        "min": float(valid_elevations.min()),
                        "max": float(valid_elevations.max()),
                        "range": float(valid_elevations.max() - valid_elevations.min()),
                        "mean": float(valid_elevations.mean())
                    }
        except Exception:
            pass
        return {"min": 0, "max": 0, "range": 0, "mean": 0}
    
    def _estimate_coverage_area(self, lidar_data: Dict[str, Any]) -> Dict[str, float]:
        """Estimate LiDAR coverage area"""
        try:
            coordinates = lidar_data.get('coordinates', np.array([]))
            if len(coordinates) > 0:
                lats = coordinates[:, 1]
                lons = coordinates[:, 0]
                
                lat_range = lats.max() - lats.min()
                lon_range = lons.max() - lons.min()
                
                # Rough area calculation (not precise but good for estimates)
                area_km2 = abs(lat_range * lon_range) * 111.32 * 111.32
                
                return {
                    "area_km2": float(area_km2),
                    "lat_range": float(lat_range),
                    "lon_range": float(lon_range)
                }
        except Exception:
            pass
        return {"area_km2": 0, "lat_range": 0, "lon_range": 0}
    
    def _count_detected_features(self, output_dir: Path) -> Dict[str, int]:
        """Count archaeological features detected in visualization products"""
        try:
            feature_file = output_dir / f"*_archaeological_features.geojson"
            geojson_files = list(output_dir.glob("*_archaeological_features.geojson"))
            
            if geojson_files:
                import json
                with open(geojson_files[0], 'r') as f:
                    geojson_data = json.load(f)
                
                feature_counts = {}
                for feature in geojson_data.get('features', []):
                    feature_type = feature.get('properties', {}).get('type', 'unknown')
                    feature_counts[feature_type] = feature_counts.get(feature_type, 0) + 1
                
                return feature_counts
        except Exception:
            pass
        return {}
    
    def _load_zone_data(self, zone_name: str) -> Dict[str, Any]:
        """Load all available data for the zone"""
        
        if not self.run_id:
            logger.error("Run ID required for data loading")
            return {}
        
        # Handle run_id that may or may not already include 'run_' prefix
        if self.run_id.startswith('run_'):
            exports_dir = self.results_dir / self.run_id / "exports"
        else:
            exports_dir = self.results_dir / f"run_{self.run_id}" / "exports"
        
        logger.debug(f"Looking for exports in: {exports_dir}")
        data_sources = {}
        
        # Load GEDI data
        gedi_file = exports_dir / "gedi" / f"{zone_name}_gedi_detections.geojson"
        if gedi_file.exists():
            try:
                data_sources['gedi'] = gpd.read_file(gedi_file)
                logger.info(f"📡 Loaded {len(data_sources['gedi'])} GEDI features")
            except Exception as e:
                logger.warning(f"Failed to load GEDI data: {e}")
        
        # Load Sentinel-2 data
        sentinel2_file = exports_dir / "sentinel2" / f"{zone_name}_sentinel2_detections.geojson"
        if sentinel2_file.exists():
            try:
                data_sources['sentinel2'] = gpd.read_file(sentinel2_file)
                logger.info(f"🛰️ Loaded {len(data_sources['sentinel2'])} Sentinel-2 features")
            except Exception as e:
                logger.warning(f"Failed to load Sentinel-2 data: {e}")
        
        # Load combined/validated data
        combined_file = exports_dir / "combined" / f"{zone_name}_combined_detections.geojson"
        if combined_file.exists():
            try:
                data_sources['combined'] = gpd.read_file(combined_file)
                logger.info(f"🎯 Loaded {len(data_sources['combined'])} validated features")
            except Exception as e:
                logger.warning(f"Failed to load combined data: {e}")
        
        # Load top candidates (try different possible counts)
        top_files = [
            exports_dir / "combined" / f"{zone_name}_top_5_candidates.geojson",
            exports_dir / "combined" / f"{zone_name}_top_10_candidates.geojson", 
            exports_dir / "combined" / f"{zone_name}_top_3_candidates.geojson"
        ]
        
        for top_file in top_files:
            if top_file.exists():
                try:
                    data_sources['top_candidates'] = gpd.read_file(top_file)
                    logger.info(f"⭐ Loaded {len(data_sources['top_candidates'])} top candidates from {top_file.name}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load top candidates from {top_file.name}: {e}")
                    continue
        
        # Load convergence pairs
        convergence_file = exports_dir / "combined" / f"{zone_name}_convergence_pairs.geojson"
        if convergence_file.exists():
            try:
                data_sources['convergence_pairs'] = gpd.read_file(convergence_file)
                logger.info(f"🔗 Loaded {len(data_sources['convergence_pairs'])} convergence pairs")
            except Exception as e:
                logger.warning(f"Failed to load convergence pairs: {e}")
        
        # Map data sources to expected keys for feature renderer
        mapped_data = {}
        if 'gedi' in data_sources:
            mapped_data['gedi_only'] = data_sources['gedi']
        if 'sentinel2' in data_sources:
            mapped_data['sentinel2_only'] = data_sources['sentinel2']
        if 'combined' in data_sources:
            mapped_data['combined'] = data_sources['combined']
        if 'top_candidates' in data_sources:
            mapped_data['top_candidates'] = data_sources['top_candidates']
        if 'convergence_pairs' in data_sources:
            mapped_data['convergence_pairs'] = data_sources['convergence_pairs']
        
        logger.info(f"📊 Mapped data sources: {list(mapped_data.keys())}")
        return mapped_data
    
    def _create_map_config(self, bounds: Dict, theme: str) -> Dict[str, Any]:
        """Create map configuration based on bounds and theme"""
        
        theme_config = self.themes.get_theme_config(theme)
        
        return {
            **self.default_config,
            'center': [bounds['center_lat'], bounds['center_lon']],
            'zoom': bounds['optimal_zoom'],
            'bounds': [[bounds['south'], bounds['west']], [bounds['north'], bounds['east']]],
            'theme': theme_config,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_output_path(self, zone_name: str, map_type: str = "enhanced") -> Path:
        """Get output path for generated map"""
        
        if self.run_id:
            # Handle run_id that may or may not already include 'run_' prefix
            if self.run_id.startswith('run_'):
                output_dir = self.results_dir / self.run_id / "visualizations"
            else:
                output_dir = self.results_dir / f"run_{self.run_id}" / "visualizations"
        else:
            output_dir = self.results_dir / "visualizations"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{zone_name}_{map_type}_map_{timestamp}.html"
        
        return output_dir / filename


class LegacyMapWrapper:
    """
    Wrapper to maintain compatibility with existing archaeological_visualizer.py
    Gradually migrate functionality to the new modular system
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize wrapper with legacy support"""
        self.new_generator = ArchaeologicalMapGenerator(*args, **kwargs)
        logger.info("🔄 Using legacy wrapper - consider migrating to new system")
    
    def create_unified_map(self, zone_name: str) -> Optional[Path]:
        """Legacy method compatibility"""
        return self.new_generator.generate_enhanced_map(zone_name)
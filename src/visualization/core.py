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
                            interactive_features: bool = True) -> Path:
        """
        Generate enhanced archaeological map with all improvements
        
        Args:
            zone_name: Target zone identifier
            theme: Visualization theme ('professional', 'field', 'scientific')
            include_analysis: Include analysis panels and statistics
            interactive_features: Enable advanced interactive tools
            
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
            
            # Calculate map bounds and center
            bounds = self.data_processor.calculate_optimal_bounds(map_data)
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
            return output_path
            
        except Exception as e:
            logger.error(f"❌ Map generation failed: {e}", exc_info=True)
            return None
    
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
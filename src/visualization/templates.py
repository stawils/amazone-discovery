"""
HTML Template Engine
Generates complete HTML maps with modern UI components
"""

import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class HTMLTemplateEngine:
    """Generates complete HTML maps with modern archaeological UI"""
    
    def __init__(self):
        self.base_template = self._load_base_template()
        self.css_styles = self._load_css_styles()
        self.javascript_code = self._load_javascript_code()
    
    def build_complete_map(self, 
                          config: Dict[str, Any],
                          feature_layers: Dict[str, List],
                          control_layers: Dict[str, Any],
                          analysis_panels: Dict[str, str],
                          theme: str = "professional",
                          zone_name: str = "Archaeological Zone",
                          interactive: bool = True) -> str:
        """Build complete HTML map with all components"""
        
        try:
            # Prepare map data
            map_data = {
                'config': config,
                'feature_layers': feature_layers,
                'control_layers': control_layers,
                'theme': theme,
                'zone_name': zone_name,
                'timestamp': datetime.now().isoformat(),
                'interactive': interactive
            }
            
            # Generate HTML components
            html_head = self._generate_html_head(theme, zone_name)
            html_body = self._generate_html_body(map_data, analysis_panels)
            html_scripts = self._generate_html_scripts(map_data)
            
            # Combine into complete HTML document
            complete_html = f"""
            <!DOCTYPE html>
            <html lang="en">
            {html_head}
            <body>
                {html_body}
                {html_scripts}
            </body>
            </html>
            """
            
            logger.info(f"✅ Generated complete HTML map for {zone_name}")
            return complete_html
            
        except Exception as e:
            logger.error(f"❌ HTML generation failed: {e}", exc_info=True)
            return self._generate_error_page(str(e))
    
    def _generate_html_head(self, theme: str, zone_name: str) -> str:
        """Generate HTML head with CSS and meta tags"""
        
        return f"""
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Archaeological Discovery - {zone_name}</title>
            
            <!-- Leaflet CSS -->
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" 
                  integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
            
            <!-- Font Awesome for icons -->
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
            
            <!-- Google Fonts -->
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
            
            <!-- Custom Archaeological Styles -->
            <style>
                {self.css_styles}
            </style>
        </head>
        """
    
    def _generate_html_body(self, map_data: Dict, analysis_panels: Dict[str, str]) -> str:
        """Generate HTML body with map and panels"""
        
        zone_name = map_data['zone_name']
        timestamp = map_data['timestamp']
        
        # Calculate total features for display from unified features
        total_features = len(map_data['feature_layers'].get('unified_features', []))
        
        return f"""
        <div id="app" class="archaeological-app">
            <!-- Header -->
            <header class="app-header">
                <div class="header-content">
                    <h1 class="app-title">
                        <i class="fas fa-map-marked-alt"></i>
                        Amazon Archaeological Discovery
                    </h1>
                    <div class="zone-info">
                        <h2 class="zone-name">{zone_name}</h2>
                        <p class="generation-time">Generated: {datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M')}</p>
                    </div>
                </div>
            </header>
            
            <!-- Main Content -->
            <main class="app-main">
                <!-- Map Container -->
                <div id="map" class="map-container"></div>
                
                <!-- Collapse Button -->
                <button id="collapse-btn" class="collapse-button" title="Hide Panels">
                    <i class="fas fa-chevron-right"></i>
                </button>
                
                <!-- Control Panels -->
                <div class="control-panels">
                    <!-- Summary Panel -->
                    <div class="panel summary-panel">
                        <div class="panel-header">
                            <h3><i class="fas fa-chart-bar"></i> Detection Summary</h3>
                        </div>
                        <div class="panel-content">
                            <div class="summary-stats">
                                <div class="stat-card clickable-filter" data-filter="all">
                                    <span class="stat-number">{len(map_data['feature_layers'].get('unified_features', []))}</span>
                                    <span class="stat-label">Total Features</span>
                                </div>
                                <div class="stat-card gedi-only clickable-filter" data-filter="gedi-only">
                                    <span class="stat-number">{len([f for f in map_data['feature_layers'].get('unified_features', []) if f.get('properties', {}).get('original_provider') == 'gedi' and not f.get('properties', {}).get('is_priority')])}</span>
                                    <span class="stat-label">🏘️ GEDI Only</span>
                                </div>
                                <div class="stat-card sentinel2-only clickable-filter" data-filter="sentinel2-only">
                                    <span class="stat-number">{len([f for f in map_data['feature_layers'].get('unified_features', []) if f.get('properties', {}).get('original_provider') == 'sentinel2' and not f.get('properties', {}).get('is_priority')])}</span>
                                    <span class="stat-label">🌱 Sentinel-2 Only</span>
                                </div>
                                <div class="stat-card convergent clickable-filter" data-filter="convergent" style="{'display: none;' if len([f for f in map_data['feature_layers'].get('unified_features', []) if f.get('properties', {}).get('is_cross_validated')]) == 0 else ''}">
                                    <span class="stat-number">{len([f for f in map_data['feature_layers'].get('unified_features', []) if f.get('properties', {}).get('is_cross_validated')])}</span>
                                    <span class="stat-label">🎯 Cross-Validated</span>
                                </div>
                                <div class="stat-card priority clickable-filter" data-filter="priority">
                                    <span class="stat-number">{len([f for f in map_data['feature_layers'].get('unified_features', []) if f.get('properties', {}).get('is_priority')])}</span>
                                    <span class="stat-label">🚩 Priority Sites</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Filter Panel -->
                    <div class="panel filter-panel">
                        <div class="panel-header">
                            <h3><i class="fas fa-filter"></i> Feature Filters</h3>
                        </div>
                        <div class="panel-content">
                            <div class="filter-group">
                                <label for="confidence-slider">Confidence Threshold:</label>
                                <div class="slider-container">
                                    <input type="range" id="confidence-slider" min="0" max="100" value="70" class="slider">
                                    <span id="confidence-value" class="slider-value">70%</span>
                                </div>
                            </div>
                            
                            <div class="filter-group">
                                <label>Feature Categories:</label>
                                <div class="checkbox-group">
                                    <label class="checkbox-label gedi-only-filter">
                                        <input type="checkbox" id="show-gedi-only" checked>
                                        <span class="checkmark"></span>
                                        🏘️ GEDI LiDAR Only
                                    </label>
                                    <label class="checkbox-label sentinel2-only-filter">
                                        <input type="checkbox" id="show-sentinel2-only" checked>
                                        <span class="checkmark"></span>
                                        🌱 Sentinel-2 Only
                                    </label>
                                    <label class="checkbox-label cross-validated-filter">
                                        <input type="checkbox" id="toggle-convergent" checked>
                                        <span class="checkmark"></span>
                                        🎯 Cross-Validated
                                    </label>
                                    <label class="checkbox-label priority-filter">
                                        <input type="checkbox" id="show-priority" checked>
                                        <span class="checkmark"></span>
                                        🚩 Priority Investigation
                                    </label>
                                </div>
                            </div>
                            
                            <div class="filter-group">
                                <label>Geometry Types:</label>
                                <div class="checkbox-group">
                                    <label class="checkbox-label">
                                        <input type="checkbox" id="show-points" checked>
                                        <span class="checkmark"></span>
                                        📍 Points
                                    </label>
                                    <label class="checkbox-label">
                                        <input type="checkbox" id="show-polygons" checked>
                                        <span class="checkmark"></span>
                                        ⭕ Polygons
                                    </label>
                                    <label class="checkbox-label">
                                        <input type="checkbox" id="show-lines" checked>
                                        <span class="checkmark"></span>
                                        📏 Lines
                                    </label>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Legend Panel -->
                    <div class="panel legend-panel">
                        <div class="panel-header legend-header-clickable" onclick="toggleLegend()">
                            <h3><i class="fas fa-map-signs"></i> Map Legend</h3>
                            <i class="fas fa-chevron-down legend-toggle-icon"></i>
                        </div>
                        <div class="panel-content legend-content">
                            <div class="legend-items">
                                <div class="legend-section">
                                    <h6>Single-Sensor Detections:</h6>
                                    <div class="legend-item gedi-only">
                                        <span class="legend-icon">🏘️</span>
                                        <span class="legend-text">GEDI LiDAR Only</span>
                                    </div>
                                    <div class="legend-item gedi-only">
                                        <span class="legend-icon">⛰️</span>
                                        <span class="legend-text">GEDI Earthwork Only</span>
                                    </div>
                                    <div class="legend-item sentinel2-only">
                                        <span class="legend-icon">🌱</span>
                                        <span class="legend-text">Terra Preta</span>
                                    </div>
                                    <div class="legend-item sentinel2-only">
                                        <span class="legend-icon">🌾</span>
                                        <span class="legend-text">Crop Marks</span>
                                    </div>
                                    <div class="legend-item sentinel2-only">
                                        <span class="legend-icon">⭕</span>
                                        <span class="legend-text">Circular Shapes</span>
                                    </div>
                                    <div class="legend-item sentinel2-only">
                                        <span class="legend-icon">⬜</span>
                                        <span class="legend-text">Rectangular Shapes</span>
                                    </div>
                                    <div class="legend-item sentinel2-only">
                                        <span class="legend-icon">📏</span>
                                        <span class="legend-text">Linear Features</span>
                                    </div>
                                </div>
                                
                                <div class="legend-section">
                                    <h6>Investigation Priority:</h6>
                                    <div class="legend-item convergent" style="{'display: none;' if len([f for f in map_data['feature_layers'].get('unified_features', []) if f.get('properties', {}).get('is_cross_validated')]) == 0 else ''}">
                                        <span class="legend-icon">🎯</span>
                                        <span class="legend-text">Cross-Validated</span>
                                    </div>
                                    <div class="legend-item priority">
                                        <span class="legend-icon">🚩</span>
                                        <span class="legend-text">Priority Investigation</span>
                                    </div>
                                </div>
                                
                                <div class="legend-section">
                                    <h6>Feature Indicators:</h6>
                                    <div class="legend-item cross-validated-outline">
                                        <span class="legend-icon" style="border: 3px solid #DC143C; border-radius: 50%; padding: 2px; display: inline-block;">⭕</span>
                                        <span class="legend-text">Red Outline = Cross-Validated</span>
                                    </div>
                                    <div class="legend-item convergence-lines" style="{'display: none;' if len(map_data['feature_layers'].get('convergence_lines', [])) == 0 else ''}">
                                        <span class="legend-icon" style="color: #FFD700;">━━━</span>
                                        <span class="legend-text">Convergence Lines</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </main>
            
            <!-- Zone Boundary Toggle -->
            <div class="zone-boundary-control">
                <label class="toggle-switch">
                    <input type="checkbox" id="zone-boundary-toggle" checked>
                    <span class="toggle-slider"></span>
                    <span class="toggle-label">Zone Boundaries</span>
                </label>
            </div>
        </div>
        """
    
    def _generate_html_scripts(self, map_data: Dict) -> str:
        """Generate JavaScript code for map functionality"""
        
        # Convert feature layers to JSON for JavaScript
        feature_layers_json = json.dumps(map_data['feature_layers'], default=str)
        config_json = json.dumps(map_data['config'], default=str)
        
        return f"""
        <!-- Leaflet JavaScript -->
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" 
                integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
        
        <!-- Map Application JavaScript -->
        <script>
            // Map data from Python
            const mapData = {config_json};
            const featureLayers = {feature_layers_json};
            
            {self.javascript_code}
        </script>
        """
    
    def _load_base_template(self) -> str:
        """Load base HTML template structure"""
        # This could be loaded from an external file in the future
        return "<!-- Base template loaded -->"
    
    def _load_css_styles(self) -> str:
        """Load comprehensive CSS styles for archaeological maps"""
        
        return """
        :root {
            --primary-archaeological: #8B4513;
            --secondary-forest: #228B22;
            --accent-discovery: #FF6B35;
            --confidence-high: #DC143C;
            --confidence-medium: #FFD700;
            --confidence-low: #87CEEB;
            --convergence: #4169E1;
            --priority: #FF1493;
            --background-dark: #2c3e50;
            --background-light: #ecf0f1;
            --text-primary: #2c3e50;
            --text-secondary: #7f8c8d;
            --border-color: #bdc3c7;
            --shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', Arial, sans-serif;
            background: linear-gradient(135deg, var(--background-dark), #34495e);
            color: var(--text-primary);
            overflow: hidden;
        }
        
        .archaeological-app {
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        /* Header Styles */
        .app-header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid var(--border-color);
            padding: 1rem 2rem;
            z-index: 1000;
        }
        
        .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .app-title {
            color: var(--primary-archaeological);
            font-size: 1.5rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .zone-info {
            text-align: right;
        }
        
        .zone-name {
            font-size: 1.2rem;
            font-weight: 500;
            color: var(--text-primary);
        }
        
        .generation-time {
            font-size: 0.875rem;
            color: var(--text-secondary);
        }
        
        /* Main Content */
        .app-main {
            flex: 1;
            display: flex;
            position: relative;
        }
        
        .map-container {
            flex: 1;
            position: relative;
            z-index: 1;
        }
        
        /* Control Panels */
        .control-panels {
            position: absolute;
            top: 1rem;
            right: 1rem;
            width: 320px;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            transition: transform 0.3s ease, opacity 0.3s ease;
        }
        
        .control-panels.collapsed {
            transform: translateX(100%);
            opacity: 0;
            pointer-events: none;
        }
        
        /* Collapse Button */
        .collapse-button {
            position: absolute;
            top: 1rem;
            right: 340px;
            width: 40px;
            height: 40px;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border: 2px solid var(--primary-archaeological);
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1001;
            box-shadow: var(--shadow);
            transition: all 0.3s ease;
            font-size: 16px;
            color: var(--primary-archaeological);
        }
        
        .collapse-button:hover {
            background: var(--primary-archaeological);
            color: white;
            transform: scale(1.1);
        }
        
        .control-panels.collapsed + .collapse-button {
            right: 1rem;
        }
        
        .control-panels.collapsed + .collapse-button i {
            transform: rotate(180deg);
        }
        
        .panel {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 8px;
            box-shadow: var(--shadow);
            overflow: hidden;
        }
        
        .panel-header {
            background: var(--primary-archaeological);
            color: white;
            padding: 0.75rem 1rem;
            font-weight: 500;
        }
        
        .panel-header h3 {
            font-size: 0.875rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .panel-content {
            padding: 1rem;
        }
        
        /* Summary Stats */
        .summary-stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
        }
        
        .stat-card {
            text-align: center;
            padding: 0.75rem;
            background: var(--background-light);
            border-radius: 6px;
            border: 1px solid var(--border-color);
            transition: all 0.2s ease;
        }
        
        .stat-card.clickable-filter {
            cursor: pointer;
            user-select: none;
        }
        
        .stat-card.clickable-filter:hover {
            background: var(--primary-archaeological);
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        
        .stat-card.clickable-filter.active {
            background: var(--accent-discovery);
            color: white;
            border-color: var(--accent-discovery);
        }
        
        .stat-card.clickable-filter:hover .stat-number,
        .stat-card.clickable-filter:hover .stat-label,
        .stat-card.clickable-filter.active .stat-number,
        .stat-card.clickable-filter.active .stat-label {
            color: white;
        }
        
        .stat-number {
            display: block;
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--primary-archaeological);
        }
        
        .stat-label {
            display: block;
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }
        
        /* Filter Controls */
        .filter-group {
            margin-bottom: 1rem;
        }
        
        .filter-group label {
            display: block;
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }
        
        .slider-container {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .slider {
            flex: 1;
            height: 6px;
            border-radius: 3px;
            background: var(--border-color);
            outline: none;
            -webkit-appearance: none;
        }
        
        .slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--primary-archaeological);
            cursor: pointer;
        }
        
        .slider-value {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--primary-archaeological);
            min-width: 35px;
        }
        
        .checkbox-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .checkbox-label {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.875rem;
            cursor: pointer;
        }
        
        .checkbox-label input[type="checkbox"] {
            display: none;
        }
        
        .checkmark {
            width: 16px;
            height: 16px;
            border: 2px solid var(--border-color);
            border-radius: 3px;
            position: relative;
        }
        
        .checkbox-label input[type="checkbox"]:checked + .checkmark {
            background: var(--primary-archaeological);
            border-color: var(--primary-archaeological);
        }
        
        .checkbox-label input[type="checkbox"]:checked + .checkmark::after {
            content: '✓';
            position: absolute;
            top: -2px;
            left: 1px;
            color: white;
            font-size: 12px;
        }
        
        /* Legend */
        .legend-items {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }
        
        /* Collapsible Legend */
        .legend-header-clickable {
            cursor: pointer;
            user-select: none;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background-color 0.2s ease;
        }
        
        .legend-header-clickable:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        
        .legend-toggle-icon {
            transition: transform 0.3s ease;
            font-size: 12px;
        }
        
        .legend-content {
            transition: max-height 0.3s ease, opacity 0.3s ease;
            overflow: hidden;
        }
        
        .legend-content.collapsed {
            max-height: 0;
            opacity: 0;
            padding-top: 0;
            padding-bottom: 0;
        }
        
        .legend-header-clickable.collapsed .legend-toggle-icon {
            transform: rotate(-90deg);
        }
        
        .legend-section {
            margin-bottom: 0.5rem;
        }
        
        .legend-section h6 {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--primary-archaeological);
            margin-bottom: 0.5rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 0.25rem;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.8rem;
            margin-bottom: 0.25rem;
            padding: 0.25rem;
            border-radius: 4px;
        }
        
        .legend-item.gedi-only {
            background-color: rgba(34, 139, 34, 0.1);
        }
        
        .legend-item.sentinel2-only {
            background-color: rgba(50, 205, 50, 0.1);
        }
        
        .legend-item.convergent {
            background-color: rgba(65, 105, 225, 0.1);
        }
        
        .legend-item.priority {
            background-color: rgba(220, 20, 60, 0.1);
        }
        
        .legend-icon {
            font-size: 1rem;
            width: 20px;
            text-align: center;
        }
        
        /* Mouse Coordinate Control */
        .mouse-coordinate-control {
            position: absolute;
            bottom: 6rem;  /* Position above zone boundary control */
            left: 1rem;
            z-index: 1000;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            box-shadow: var(--shadow);
            min-width: 220px;
        }
        
        .coordinate-display {
            font-size: 0.875rem;
            font-family: 'Courier New', monospace;
            color: var(--primary-archaeological);
            font-weight: 600;
        }
        
        /* Zone Boundary Control */
        .zone-boundary-control {
            position: absolute;
            bottom: 1rem;
            left: 1rem;
            z-index: 1000;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            box-shadow: var(--shadow);
        }
        
        .toggle-switch {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            cursor: pointer;
            font-size: 0.875rem;
        }
        
        .toggle-switch input[type="checkbox"] {
            display: none;
        }
        
        .toggle-slider {
            width: 40px;
            height: 20px;
            background: var(--border-color);
            border-radius: 10px;
            position: relative;
            transition: background 0.3s;
        }
        
        .toggle-slider::before {
            content: '';
            position: absolute;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: white;
            top: 2px;
            left: 2px;
            transition: transform 0.3s;
        }
        
        .toggle-switch input[type="checkbox"]:checked + .toggle-slider {
            background: var(--primary-archaeological);
        }
        
        .toggle-switch input[type="checkbox"]:checked + .toggle-slider::before {
            transform: translateX(20px);
        }
        
        /* Compact Popups */
        .popup-compact {
            max-width: 300px;
            font-family: 'Inter', Arial, sans-serif;
            font-size: 13px;
            line-height: 1.3;
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        .popup-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            background: var(--primary-archaeological);
            color: white;
            border-radius: 8px 8px 0 0;
            font-weight: 600;
        }
        
        .popup-title {
            font-size: 14px;
        }
        
        .popup-details {
            padding: 12px;
        }
        
        .popup-details p {
            margin: 6px 0;
            font-size: 12px;
        }
        
        .popup-details strong {
            color: var(--primary-archaeological);
        }
        
        .confidence-badge {
            background: var(--confidence-medium);
            color: var(--text-primary);
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        
        .confidence-badge.enhanced {
            background: var(--confidence-high);
            color: white;
        }
        
        .confidence-badge.priority {
            background: var(--priority);
            color: white;
        }
        
        .archaeological-tooltip h5 {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--primary-archaeological);
            margin: 0.75rem 0 0.25rem 0;
        }
        
        .archaeological-tooltip ul {
            margin: 0.5rem 0;
            padding-left: 1.25rem;
        }
        
        .archaeological-tooltip li {
            font-size: 0.8rem;
            margin-bottom: 0.25rem;
        }
        
        .archaeological-tooltip p {
            font-size: 0.8rem;
            margin-bottom: 0.5rem;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .header-content {
                flex-direction: column;
                gap: 0.5rem;
                text-align: center;
            }
            
            .control-panels {
                width: 100%;
                max-width: 320px;
                left: 1rem;
                right: 1rem;
            }
            
            .summary-stats {
                grid-template-columns: repeat(3, 1fr);
            }
        }
        
        /* Map Info Indicator Styles */
        .map-info-indicator {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 8px 12px;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            font-family: 'Inter', monospace;
            font-size: 12px;
            line-height: 1.4;
            border: 1px solid var(--border-color);
            min-width: 200px;
        }
        
        .zoom-info {
            display: flex;
            justify-content: space-between;
            color: var(--text-primary);
            font-weight: 500;
            margin-bottom: 4px;
            padding-bottom: 4px;
            border-bottom: 1px solid var(--border-color);
        }
        
        """
    
    def _load_javascript_code(self) -> str:
        """Load comprehensive JavaScript for map functionality"""
        
        return """
        // Initialize the archaeological map
        class ArchaeologicalMap {
            constructor() {
                this.map = null;
                this.layers = {};
                this.zoneBoundary = null;
                this.init();
            }
            
            init() {
                this.createMap();
                this.addBaseLayers();
                this.addFeatureLayers();
                this.addZoneBoundary();
                this.setupControls();
                this.setupEventListeners();
                this.setupPanelCollapse();
                // Apply initial filters to ensure consistent state
                this.updateAllFilters();
            }
            
            createMap() {
                // Initialize map with enhanced zoom configuration
                this.map = L.map('map', {
                    center: mapData.center,
                    zoom: mapData.zoom,
                    minZoom: 1,
                    maxZoom: 22,  // Support ultra-high zoom levels
                    zoomControl: false,
                    attributionControl: true,
                    zoomSnap: 0.25,  // Smoother zoom increments
                    zoomDelta: 0.5,  // Finer zoom control
                    wheelPxPerZoomLevel: 80  // Smoother mouse wheel zoom
                });
                
                // Enhanced zoom control with custom styling
                const zoomControl = L.control.zoom({ 
                    position: 'topleft',
                    zoomInTitle: 'Zoom in for higher resolution',
                    zoomOutTitle: 'Zoom out for broader view'
                });
                zoomControl.addTo(this.map);
                
                // Add custom zoom indicator
                const zoomIndicator = L.control({ position: 'bottomleft' });
                zoomIndicator.onAdd = function() {
                    const div = L.DomUtil.create('div', 'zoom-indicator');
                    div.innerHTML = '<span id="current-zoom">Z: ' + this._map.getZoom() + '</span><br><span id="resolution-info">~3m/px</span>';
                    return div;
                };
                zoomIndicator.addTo(this.map);
                
                // Update zoom indicator on zoom change
                this.map.on('zoomend', function() {
                    const zoom = this.getZoom();
                    const zoomSpan = document.getElementById('current-zoom');
                    const resSpan = document.getElementById('resolution-info');
                    
                    if (zoomSpan) zoomSpan.textContent = 'Z: ' + zoom;
                    if (resSpan) {
                        // Approximate resolution at equator
                        const resolution = 156543.034 / Math.pow(2, zoom);
                        if (resolution < 1) {
                            resSpan.textContent = '~' + (resolution * 100).toFixed(0) + 'cm/px';
                        } else if (resolution < 1000) {
                            resSpan.textContent = '~' + resolution.toFixed(1) + 'm/px';
                        } else {
                            resSpan.textContent = '~' + (resolution / 1000).toFixed(1) + 'km/px';
                        }
                    }
                });
                
                // Fit bounds if available
                if (mapData.bounds) {
                    this.map.fitBounds(mapData.bounds, { padding: [20, 20] });
                }
            }
            
            addBaseLayers() {
                // High-resolution satellite imagery layers
                
                // Esri World Imagery (High Quality) - Primary
                const satelliteHQ = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community',
                    maxZoom: 21,  // Increased from 18 to 21
                    minZoom: 1
                });
                
                // Google Satellite (Ultra High Quality) - Alternative
                const googleSat = L.tileLayer('https://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', {
                    attribution: 'Tiles &copy; Google',
                    subdomains: ['0', '1', '2', '3'],
                    maxZoom: 22,  // Google's highest zoom
                    minZoom: 1
                });
                
                // CARTO Satellite (Alternative High Quality)
                const cartoSat = L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
                    attribution: 'Tiles &copy; CARTO &copy; OpenStreetMap contributors',
                    subdomains: 'abcd',
                    maxZoom: 20,
                    minZoom: 1
                });
                
                // Mapbox Satellite (Very High Quality) - Premium option
                const mapboxSat = L.tileLayer('https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{z}/{x}/{y}?access_token=pk.eyJ1IjoibWFwYm94IiwiYSI6ImNpejY4NXVycTA2emYycXBndHRqcmZ3N3gifQ.rJcFIG214AriISLbB6B5aw', {
                    attribution: 'Tiles &copy; Mapbox &copy; OpenStreetMap',
                    maxZoom: 22,
                    minZoom: 1
                });
                
                // Set Google Satellite as default (highest quality)
                googleSat.addTo(this.map);
                this.layers.baseSatellite = googleSat;
                
                // Terrain and other layers
                const terrainLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}', {
                    attribution: 'Tiles &copy; Esri &mdash; Source: Esri',
                    maxZoom: 16  // Increased from 13
                });
                
                // OpenStreetMap for reference
                const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '&copy; OpenStreetMap contributors',
                    maxZoom: 19
                });
                
                this.layers.baseTerrain = terrainLayer;
                this.layers.baseOSM = osmLayer;
                
                // Enhanced layer control with quality indicators
                const baseLayers = {
                    "🛰️ Google Satellite (Ultra HQ - 22x)": googleSat,
                    "🌍 Esri World Imagery (HQ - 21x)": satelliteHQ,
                    "🗺️ Mapbox Satellite (HQ - 22x)": mapboxSat,
                    "🌄 Terrain Relief": terrainLayer,
                    "📍 OpenStreetMap": osmLayer
                };
                
                // Set Google Satellite as default - no layer control needed
                // Users can access different base layers through browser context menu if needed
            }
            
            addFeatureLayers() {
                // Create single unified layer group for all features
                this.layers.unifiedFeaturesGroup = L.layerGroup().addTo(this.map);
                this.layers.convergenceLinesGroup = L.layerGroup().addTo(this.map);
                
                // Set z-index for proper display order
                this.layers.unifiedFeaturesGroup.setZIndex(1000);
                this.layers.convergenceLinesGroup.setZIndex(100);
                
                // Add unified features and convergence lines
                this.addUnifiedFeatures();
                this.addConvergenceLines();
            }
            
            addUnifiedFeatures() {
                const features = featureLayers.unified_features || [];
                features.forEach(feature => {
                    const props = feature.properties;
                    const primaryCategory = props.primary_category || 'sentinel2';
                    const layer = this.createFeatureLayer(feature, primaryCategory);
                    
                    // Store additional metadata for filtering
                    layer.originalProvider = props.original_provider;
                    layer.isCrossValidated = props.is_cross_validated === true;
                    layer.isPriority = props.is_priority === true;
                    layer.primaryCategory = primaryCategory;
                    
                    layer.addTo(this.layers.unifiedFeaturesGroup);
                });
                console.log(`Added ${features.length} unified features`);
            }
            
            addConvergenceLines() {
                const features = featureLayers.convergence_lines || [];
                features.forEach(feature => {
                    const layer = this.createFeatureLayer(feature, 'convergence');
                    layer.addTo(this.layers.convergenceLinesGroup);
                });
                console.log(`Added ${features.length} convergence line features`);
            }
            
            createFeatureLayer(feature, category) {
                const props = feature.properties;
                const geometry = feature.geometry;
                
                let layer;
                const style = this.getGeometryStyle(props, category);
                
                // Create appropriate layer based on geometry type
                switch (geometry.type) {
                    case 'Polygon':
                        layer = this.createPolygonFeature(geometry, style, props);
                        break;
                    case 'LineString':
                        layer = this.createLineFeature(geometry, style, props);
                        break;
                    case 'Point':
                    default:
                        layer = this.createPointFeature(geometry, style, props);
                        break;
                }
                
                // Store feature data for filtering
                layer.feature = feature;
                layer.confidence = props.confidence || 0;
                
                // Add popup with area visualization
                const popup = L.popup({
                    maxWidth: 350,
                    className: 'archaeological-popup',
                    closeButton: true,
                    autoClose: true,
                    closeOnEscapeKey: true,
                    closeOnClick: true
                }).setContent(props.tooltip);
                
                layer.bindPopup(popup);
                
                // Add area visualization for point features
                if (geometry.type === 'Point' && layer.archaeologicalData) {
                    layer.on('popupopen', () => {
                        this.showFeatureArea(layer);
                    });
                    
                    layer.on('popupclose', () => {
                        this.hideFeatureArea(layer);
                    });
                }
                
                // Add interaction events
                this.addFeatureEvents(layer, props);
                
                return layer;
            }
            
            createPolygonFeature(geometry, style, props) {
                // Convert coordinates for Leaflet (GeoJSON uses [lon, lat], Leaflet uses [lat, lon])
                const leafletCoords = geometry.coordinates[0].map(coord => [coord[1], coord[0]]);
                
                const polygon = L.polygon(leafletCoords, {
                    ...style.polygon,
                    className: `archaeological-polygon ${props.feature_type}`
                });
                
                // Add center marker for better UX
                const bounds = polygon.getBounds();
                const center = bounds.getCenter();
                const centerMarker = L.marker(center, {
                    icon: this.createGeometricIcon(props, 'polygon'),
                    zIndexOffset: 1000
                });
                
                // Group polygon and center marker
                return L.layerGroup([polygon, centerMarker]);
            }
            
            createLineFeature(geometry, style, props) {
                // Convert coordinates for Leaflet
                const leafletCoords = geometry.coordinates.map(coord => [coord[1], coord[0]]);
                
                // Handle convergence lines with custom styling
                let lineOptions;
                if (props.feature_type === 'convergence_line') {
                    lineOptions = {
                        color: props.color || '#FFD700',
                        weight: props.weight || 2,
                        opacity: props.opacity || 0.8,
                        dashArray: props.dashArray || '5, 5',
                        className: `archaeological-line convergence-line`
                    };
                } else {
                    lineOptions = {
                        ...style.line,
                        className: `archaeological-line ${props.feature_type}`
                    };
                }
                
                const polyline = L.polyline(leafletCoords, lineOptions);
                
                // For convergence lines, don't add center marker to avoid clutter
                if (props.feature_type === 'convergence_line') {
                    return polyline;
                } else {
                    // Add center marker for other line types
                    const bounds = polyline.getBounds();
                    const center = bounds.getCenter();
                    const centerMarker = L.marker(center, {
                        icon: this.createGeometricIcon(props, 'line'),
                        zIndexOffset: 1000
                    });
                    
                    return L.layerGroup([polyline, centerMarker]);
                }
            }
            
            createPointFeature(geometry, style, props) {
                const coords = geometry.coordinates;
                const marker = L.marker([coords[1], coords[0]], { 
                    icon: this.createGeometricIcon(props, 'point')
                });
                
                // Store area information for radius display
                marker.archaeologicalData = {
                    area_m2: props.area_m2 || 0,
                    area_hectares: (props.area_m2 || 0) / 10000,
                    coordinates: [coords[1], coords[0]],
                    feature_type: props.feature_type || props.type
                };
                
                // Add area circle that shows on popup open
                marker.areaCircle = null;
                
                return marker;
            }
            
            getGeometryStyle(props, category) {
                const baseStyles = this.getBaseStyles(category);
                const confidence = props.confidence || 0;
                const type = props.type || '';
                
                // Adjust opacity based on confidence
                const opacity = Math.max(0.3, confidence);
                const fillOpacity = Math.max(0.1, confidence * 0.4);
                
                return {
                    polygon: {
                        color: baseStyles.color,
                        weight: 2,
                        opacity: opacity,
                        fillColor: baseStyles.fillColor,
                        fillOpacity: fillOpacity,
                        dashArray: type.includes('geometric') ? '5, 5' : null
                    },
                    line: {
                        color: baseStyles.color,
                        weight: 3,
                        opacity: opacity,
                        dashArray: type.includes('linear') ? '10, 5' : null
                    },
                    point: {
                        color: baseStyles.color
                    }
                };
            }
            
            getBaseStyles(category) {
                const styles = {
                    gedi: {
                        color: '#228B22',
                        fillColor: '#32CD32'
                    },
                    sentinel2: {
                        color: '#9370DB', 
                        fillColor: '#DDA0DD'
                    },
                    convergent: {
                        color: '#DC143C',
                        fillColor: '#FFB6C1'
                    },
                    priority: {
                        color: '#FF1493',
                        fillColor: '#FFC0CB'
                    }
                };
                
                return styles[category] || styles.sentinel2;
            }
            
            createGeometricIcon(props, geometryType) {
                const iconSize = geometryType === 'point' ? 32 : 24;
                const iconConfig = this.getIconConfig(props, geometryType);
                
                const categoryClass = props.category === 'priority' ? 'priority-marker' : '';
                const crossValidatedClass = props.is_cross_validated ? 'cross-validated-outline' : '';
                const iconHtml = `<div class="geometric-marker ${geometryType}-marker ${categoryClass} ${crossValidatedClass}" data-category="${props.category}" style="
                    background: ${iconConfig.background};
                    color: ${iconConfig.color};
                    border: ${iconConfig.borderWidth || '2px'} solid ${iconConfig.border};
                    border-radius: ${iconConfig.borderRadius};
                    width: ${iconSize}px;
                    height: ${iconSize}px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: ${iconConfig.fontSize}px;
                    box-shadow: ${iconConfig.boxShadow || '0 2px 8px rgba(0,0,0,0.4)'};
                    cursor: pointer;
                    transition: all 0.2s ease;
                    font-weight: bold;
                ">${iconConfig.symbol}</div>`;
                
                const divIconClass = `archaeological-icon geometric-icon ${categoryClass} ${crossValidatedClass}`;
                return L.divIcon({
                    html: iconHtml,
                    className: divIconClass,
                    iconSize: [iconSize, iconSize],
                    iconAnchor: [iconSize/2, iconSize/2],
                    popupAnchor: [0, -iconSize/2]
                });
            }
            
            getIconConfig(props, geometryType) {
                const type = props.feature_type || props.type || '';
                const primaryCategory = props.primary_category || 'sentinel2';
                const originalProvider = props.original_provider;
                const isPriority = props.is_priority === true;
                const isCrossValidated = props.is_cross_validated === true;
                
                // Use the icon from properties if available (from archaeological_icons)
                const icon = props.icon;
                
                // Enhanced colors by primary category with better visual logic
                const categoryColors = {
                    priority: { bg: '#FF1493', border: '#fff', accent: '#FFD700' },
                    cross_validated: { bg: '#DC143C', border: '#FFD700', accent: '#FFD700' },
                    gedi: { bg: '#228B22', border: '#fff', accent: '#90EE90' },
                    sentinel2: { bg: '#9370DB', border: '#fff', accent: '#DDA0DD' }
                };
                
                const colors = categoryColors[primaryCategory] || categoryColors.sentinel2;
                
                // Use the icon from properties if available (from archaeological_icons)
                if (icon && icon !== 'undefined' && icon !== '') {
                    // Enhanced visual logic for better clarity
                    let background, iconColor, border, borderWidth, boxShadow;
                    
                    if (isPriority) {
                        // Priority sites: bright red background with white icon
                        background = '#FF1493';
                        iconColor = 'white';
                        border = isCrossValidated ? '#FFD700' : '#fff';
                        borderWidth = isCrossValidated ? '4px' : '3px';
                        boxShadow = '0 0 10px rgba(255, 20, 147, 0.6)';
                    } else if (isCrossValidated) {
                        // Cross-validated: white background with colored icon and gold border
                        background = 'white';
                        iconColor = colors.bg;
                        border = '#FFD700';
                        borderWidth = '3px';
                        boxShadow = '0 0 8px rgba(220, 20, 60, 0.5)';
                    } else {
                        // Regular features: category color background with white icon
                        background = colors.bg;
                        iconColor = 'white';
                        border = colors.border;
                        borderWidth = '2px';
                        boxShadow = '0 2px 6px rgba(0, 0, 0, 0.3)';
                    }
                    
                    return {
                        symbol: icon,
                        background: background,
                        border: border,
                        borderWidth: borderWidth,
                        color: iconColor,
                        borderRadius: '50%',
                        fontSize: geometryType === 'point' ? 16 : 12,
                        boxShadow: boxShadow
                    };
                }
                
                // Fallback icon logic based on primary category and original provider
                let symbol, borderRadius;
                
                // Priority investigation - ALWAYS takes precedence
                if (isPriority || primaryCategory === 'priority') {
                    symbol = '🚩';  // Priority Investigation
                    borderRadius = '50%';
                }
                // Cross-validated features
                else if (isCrossValidated || primaryCategory === 'cross_validated') {
                    symbol = '🎯';  // Cross-Sensor Confirmed
                    borderRadius = '50%';
                }
                // GEDI features
                else if (primaryCategory === 'gedi' || originalProvider === 'gedi') {
                    if (type.includes('clearing') || type.includes('settlement')) {
                        symbol = '🏘️';  // GEDI LiDAR Only
                    } else {
                        symbol = '⛰️';  // GEDI Earthwork Only
                    }
                    borderRadius = '50%';
                }
                // Sentinel-2 features
                else if (primaryCategory === 'sentinel2' || originalProvider === 'sentinel2') {
                    if (type.includes('terra_preta') || type.includes('crop')) {
                        symbol = '🌱';  // Sentinel-2 Terra Preta Only
                    } else {
                        symbol = '⭕';  // Sentinel-2 Geometric Only
                    }
                    borderRadius = geometryType === 'polygon' ? '6px' : '50%';
                }
                // Fallback
                else {
                    symbol = '📍';
                    borderRadius = '50%';
                }
                
                // Enhanced fallback with improved visual logic
                let background, iconColor, border, borderWidth, boxShadow;
                
                if (isPriority) {
                    background = '#FF1493';
                    iconColor = 'white';
                    border = '#FFD700';
                    borderWidth = '3px';
                    boxShadow = '0 0 10px rgba(255, 20, 147, 0.6)';
                } else if (isCrossValidated) {
                    background = 'white';
                    iconColor = colors.bg;
                    border = '#FFD700';
                    borderWidth = '3px';
                    boxShadow = '0 0 8px rgba(220, 20, 60, 0.5)';
                } else {
                    background = colors.bg;
                    iconColor = 'white';
                    border = colors.border;
                    borderWidth = '2px';
                    boxShadow = '0 2px 6px rgba(0, 0, 0, 0.3)';
                }
                
                return {
                    symbol: symbol,
                    background: background,
                    border: border,
                    borderWidth: borderWidth,
                    color: iconColor,
                    borderRadius: borderRadius,
                    fontSize: geometryType === 'point' ? 16 : 12,
                    boxShadow: boxShadow
                };
            }
            
            addFeatureEvents(layer, props) {
                // Remove hover popup - only click to open
                layer.on('mouseover', function() {
                    // Only highlight effect, no popup on hover
                    this.eachLayer && this.eachLayer(l => {
                        if (l.setStyle && typeof l.setStyle === 'function') {
                            l.setStyle({ weight: l.options.weight + 1 });
                        }
                    });
                });
                
                layer.on('mouseout', function() {
                    // Reset highlight
                    this.eachLayer && this.eachLayer(l => {
                        if (l.setStyle && typeof l.setStyle === 'function') {
                            l.setStyle({ weight: l.options.weight - 1 });
                        }
                    });
                });
                
                // Click logging for debugging
                layer.on('click', function() {
                    console.log('Feature clicked:', {
                        confidence: props.confidence,
                        feature_type: props.feature_type,
                        category: props.category,
                        icon: props.icon,
                        geometryType: layer.feature?.geometry?.type,
                        provider: props.provider
                    });
                });
            }
            
            addZoneBoundary() {
                if (mapData.bounds) {
                    const bounds = mapData.bounds;
                    const rectangle = L.rectangle(bounds, {
                        color: '#FF6B35',
                        weight: 2,
                        fillOpacity: 0.1,
                        dashArray: '5, 5'
                    });
                    
                    this.zoneBoundary = rectangle.addTo(this.map);
                }
            }
            
            setupControls() {
                // Map selection tab completely removed as requested
                // Only keep enhanced zoom control with custom styling - no layer selection panel
                
                // Add mouse coordinate display above zone boundary control
                const coordIndicator = L.control({ position: 'topleft' });
                coordIndicator.onAdd = function() {
                    const div = L.DomUtil.create('div', 'mouse-coordinate-control');
                    div.innerHTML = '<div class="coordinate-display"><span>📍 <span id="mouse-coordinates">Move mouse for coordinates</span></span></div>';
                    return div;
                };
                coordIndicator.addTo(this.map);
                
                // Add mouse coordinate tracking
                this.map.on('mousemove', (e) => {
                    const lat = e.latlng.lat.toFixed(6);
                    const lng = e.latlng.lng.toFixed(6);
                    const coordSpan = document.getElementById('mouse-coordinates');
                    if (coordSpan) {
                        coordSpan.textContent = `Lat: ${lat}, Lng: ${lng}`;
                    }
                });
                
                // Update zoom level display
                this.map.on('zoomend', () => {
                    const zoomSpan = document.getElementById('zoom-level');
                    if (zoomSpan) {
                        zoomSpan.textContent = this.map.getZoom();
                    }
                });
                
                // Initialize zoom level display
                const initialZoomSpan = document.getElementById('zoom-level');
                if (initialZoomSpan) {
                    initialZoomSpan.textContent = this.map.getZoom();
                }
            }
            
            setupPanelCollapse() {
                const collapseBtn = document.getElementById('collapse-btn');
                const controlPanels = document.querySelector('.control-panels');
                
                if (collapseBtn && controlPanels) {
                    collapseBtn.addEventListener('click', () => {
                        controlPanels.classList.toggle('collapsed');
                        
                        // Update button title
                        const isCollapsed = controlPanels.classList.contains('collapsed');
                        collapseBtn.title = isCollapsed ? 'Show Panels' : 'Hide Panels';
                        
                        // Store state in localStorage
                        localStorage.setItem('panelsCollapsed', isCollapsed);
                    });
                    
                    // Restore previous state
                    const wasCollapsed = localStorage.getItem('panelsCollapsed') === 'true';
                    if (wasCollapsed) {
                        controlPanels.classList.add('collapsed');
                        collapseBtn.title = 'Show Panels';
                    }
                }
            }
            
            setupEventListeners() {
                // Confidence slider
                const slider = document.getElementById('confidence-slider');
                const value = document.getElementById('confidence-value');
                
                if (slider && value) {
                    slider.addEventListener('input', (e) => {
                        value.textContent = e.target.value + '%';
                        this.filterByConfidence(e.target.value / 100);
                    });
                }
                
                // Layer toggles with new structure
                const toggles = ['gedi-only', 'sentinel2-only', 'priority'];
                toggles.forEach(type => {
                    const checkbox = document.getElementById(`show-${type}`);
                    if (checkbox) {
                        checkbox.addEventListener('change', (e) => {
                            this.toggleLayer(type, e.target.checked);
                        });
                    }
                });
                
                // Cross-validated features toggle (affects features in all groups)
                const convergentToggle = document.getElementById('toggle-convergent');
                if (convergentToggle) {
                    convergentToggle.addEventListener('change', (e) => {
                        this.toggleCrossValidatedFeatures(e.target.checked);
                    });
                }
                
                // Geometry type filters
                const geometryTypes = ['points', 'polygons', 'lines'];
                geometryTypes.forEach(geomType => {
                    const checkbox = document.getElementById(`show-${geomType}`);
                    if (checkbox) {
                        checkbox.addEventListener('change', (e) => {
                            this.toggleGeometryType(geomType, e.target.checked);
                        });
                    }
                });
                
                // Zone boundary toggle
                const boundaryToggle = document.getElementById('zone-boundary-toggle');
                if (boundaryToggle) {
                    boundaryToggle.addEventListener('change', (e) => {
                        this.toggleZoneBoundary(e.target.checked);
                    });
                }
                
                // Clickable stat cards as filters
                document.querySelectorAll('.clickable-filter').forEach(card => {
                    card.addEventListener('click', (e) => {
                        this.handleStatCardClick(e.currentTarget);
                    });
                });
                
                // Enhanced styles for geometric features
                document.addEventListener('DOMContentLoaded', () => {
                    const style = document.createElement('style');
                    style.textContent = `
                        /* Geometric marker hover effects */
                        .geometric-marker:hover {
                            transform: scale(1.15);
                            z-index: 1000;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.5) !important;
                        }
                        
                        /* Polygon styling */
                        .archaeological-polygon {
                            transition: all 0.2s ease;
                        }
                        
                        .archaeological-polygon:hover {
                            filter: brightness(1.1);
                        }
                        
                        /* Line styling */
                        .archaeological-line {
                            transition: all 0.2s ease;
                        }
                        
                        .archaeological-line:hover {
                            filter: brightness(1.1);
                        }
                        
                        /* Icon type specific styling */
                        .polygon-marker {
                            background: linear-gradient(45deg, var(--bg-color, #9370DB), var(--accent-color, #DDA0DD)) !important;
                        }
                        
                        .line-marker {
                            background: linear-gradient(90deg, var(--bg-color, #4169E1), var(--accent-color, #87CEEB)) !important;
                        }
                        
                        /* Popup styling */
                        .archaeological-popup .leaflet-popup-content-wrapper {
                            border-radius: 12px;
                            box-shadow: 0 8px 24px rgba(0,0,0,0.2);
                            border: 1px solid rgba(255,255,255,0.1);
                        }
                        
                        .archaeological-popup .leaflet-popup-content {
                            margin: 0;
                            line-height: 1.4;
                            font-size: 14px;
                        }
                        
                        .archaeological-popup .leaflet-popup-tip {
                            border-top-color: white;
                        }
                        
                        /* Area visualization tooltip */
                        .area-tooltip {
                            background: rgba(0, 0, 0, 0.8) !important;
                            border: none !important;
                            border-radius: 6px !important;
                            color: white !important;
                            font-size: 12px !important;
                            font-weight: 500 !important;
                            padding: 6px 10px !important;
                            text-align: center !important;
                            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
                            line-height: 1.3 !important;
                        }
                        
                        .area-tooltip::before {
                            border-top-color: rgba(0, 0, 0, 0.8) !important;
                        }
                        
                        /* Zoom indicator styling */
                        .zoom-indicator {
                            background: rgba(0, 0, 0, 0.7);
                            color: white;
                            padding: 6px 10px;
                            border-radius: 6px;
                            font-size: 11px;
                            font-weight: 500;
                            font-family: 'Inter', monospace;
                            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
                            border: 1px solid rgba(255, 255, 255, 0.2);
                            pointer-events: none;
                            line-height: 1.3;
                        }
                        
                        /* Enhanced zoom control styling - positioned away from panels */
                        .leaflet-control-zoom {
                            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
                            z-index: 2000 !important;
                            background: rgba(255, 255, 255, 0.95) !important;
                            backdrop-filter: blur(10px) !important;
                            border-radius: 8px !important;
                            border: 2px solid var(--primary-archaeological) !important;
                            margin-left: 80px !important;
                            margin-top: 50px !important;
                        }
                        
                        /* Ensure all Leaflet controls are above all panels */
                        .leaflet-control-layers,
                        .leaflet-control-attribution,
                        .leaflet-control-scale {
                            z-index: 2000;
                        }
                        
                        .leaflet-control-zoom a {
                            font-size: 16px !important;
                            line-height: 30px !important;
                            width: 32px !important;
                            height: 32px !important;
                        }
                        
                        /* Confidence-based opacity animation */
                        .leaflet-interactive {
                            transition: opacity 0.3s ease, fill-opacity 0.3s ease;
                        }
                        
                        /* Enhanced legend styling */
                        .legend-section h6 {
                            margin-bottom: 8px;
                            color: #2c3e50;
                            font-weight: 600;
                        }
                        
                        .legend-item {
                            display: flex;
                            align-items: center;
                            margin-bottom: 4px;
                            padding: 2px 0;
                        }
                        
                        .legend-icon {
                            margin-right: 8px;
                            font-size: 16px;
                        }
                        
                        /* Geometric shape indicators in legend */
                        .legend-item.geometric::after {
                            content: " (Shape)";
                            font-size: 12px;
                            color: #7f8c8d;
                            font-style: italic;
                        }
                        
                        /* Priority marker z-index override - ensure they always appear on top */
                        .priority-marker,
                        .priority-marker .leaflet-marker-icon,
                        [data-category="priority"],
                        [data-category="priority"] .leaflet-marker-icon,
                        .archaeological-icon.priority-marker {
                            z-index: 1500;
                            position: relative;
                        }
                        
                        /* Cross-validated red outline styling */
                        .cross-validated-outline {
                            border: 3px solid #DC143C !important;
                            border-radius: 50% !important;
                        }
                    `;
                    document.head.appendChild(style);
                });
            }
            
            filterByConfidence(threshold) {
                console.log(`Filtering by confidence threshold: ${threshold} (${threshold * 100}%)`);
                this.updateAllFilters();
            }
            
            setFeatureVisibility(featureLayer, opacity, visible) {
                // Handle different layer types (single marker, layerGroup with polygon+marker, etc.)
                if (featureLayer.eachLayer) {
                    // This is a layerGroup (polygon/line with center marker)
                    featureLayer.eachLayer(layer => {
                        if (layer.setOpacity) {
                            layer.setOpacity(opacity);
                        }
                        if (layer.setStyle) {
                            // For polygons and lines, adjust stroke and fill opacity
                            const currentStyle = layer.options;
                            layer.setStyle({
                                opacity: opacity,
                                fillOpacity: currentStyle.fillOpacity ? currentStyle.fillOpacity * opacity : opacity * 0.3
                            });
                        }
                        if (layer.getElement && layer.getElement()) {
                            layer.getElement().style.display = visible ? '' : 'none';
                        }
                    });
                } else {
                    // This is a single marker
                    if (featureLayer.setOpacity) {
                        featureLayer.setOpacity(opacity);
                    }
                    if (featureLayer.getElement && featureLayer.getElement()) {
                        featureLayer.getElement().style.display = visible ? '' : 'none';
                    }
                }
            }
            
            toggleLayer(type, visible) {
                console.log(`Toggled ${type} layer: ${visible ? 'visible' : 'hidden'}`);
                this.updateAllFilters();
            }
            
            toggleZoneBoundary(visible) {
                if (this.zoneBoundary) {
                    if (visible) {
                        this.zoneBoundary.addTo(this.map);
                    } else {
                        this.map.removeLayer(this.zoneBoundary);
                    }
                }
            }
            
            toggleGeometryType(geometryType, visible) {
                this.updateAllFilters();
            }
            
            toggleCrossValidatedFeatures(visible) {
                // Toggle red outline styling for cross-validated features in unified layer
                if (this.layers.unifiedFeaturesGroup && this.layers.unifiedFeaturesGroup.eachLayer) {
                    this.layers.unifiedFeaturesGroup.eachLayer(featureLayer => {
                        const props = featureLayer.feature?.properties;
                        const isCrossValidated = featureLayer.isCrossValidated || props?.is_cross_validated === true;
                        
                        if (isCrossValidated) {
                            // Update the icon element if it exists
                            if (featureLayer.getElement) {
                                const iconElement = featureLayer.getElement().querySelector('.geometric-marker');
                                if (iconElement) {
                                    if (visible) {
                                        // Show red outline
                                        iconElement.style.border = '3px solid #DC143C';
                                        iconElement.classList.add('cross-validated-outline');
                                    } else {
                                        // Hide red outline
                                        iconElement.style.border = '2px solid #fff';
                                        iconElement.classList.remove('cross-validated-outline');
                                    }
                                }
                            }
                        }
                    });
                }
                
                // Update all filters to show/hide cross-validated features
                this.updateAllFilters();
                
                console.log(`Cross-validated features ${visible ? 'shown' : 'hidden'}`);
            }
            
            showFeatureArea(layer) {
                if (!layer.archaeologicalData || layer.areaCircle) return;
                
                const { area_m2, coordinates, feature_type } = layer.archaeologicalData;
                
                if (area_m2 > 0) {
                    // Calculate radius from area (assuming roughly circular feature)
                    const radiusMeters = Math.sqrt(area_m2 / Math.PI);
                    
                    // Create circle overlay
                    layer.areaCircle = L.circle(coordinates, {
                        radius: radiusMeters,
                        color: this.getAreaCircleColor(feature_type),
                        fillColor: this.getAreaCircleColor(feature_type),
                        fillOpacity: 0.15,
                        weight: 2,
                        opacity: 0.6,
                        dashArray: '8, 4'
                    }).addTo(this.map);
                    
                    // Add radius label
                    const areaHectares = area_m2 / 10000;
                    const radiusText = radiusMeters > 1000 ? 
                        `${(radiusMeters/1000).toFixed(1)}km radius` : 
                        `${Math.round(radiusMeters)}m radius`;
                    const areaText = areaHectares > 1 ?
                        `${areaHectares.toFixed(1)} ha` :
                        `${Math.round(area_m2)} m²`;
                    
                    layer.areaLabel = L.tooltip({
                        permanent: true,
                        direction: 'top',
                        className: 'area-tooltip',
                        offset: [0, -10]
                    })
                    .setContent(`${areaText}<br>${radiusText}`)
                    .setLatLng(coordinates);
                    
                    layer.areaLabel.addTo(this.map);
                }
            }
            
            hideFeatureArea(layer) {
                if (layer.areaCircle) {
                    this.map.removeLayer(layer.areaCircle);
                    layer.areaCircle = null;
                }
                if (layer.areaLabel) {
                    this.map.removeLayer(layer.areaLabel);
                    layer.areaLabel = null;
                }
            }
            
            getAreaCircleColor(feature_type) {
                const colorMap = {
                    'gedi_clearing': '#228B22',
                    'gedi_mound': '#8B4513', 
                    'terra_preta': '#32CD32',
                    'geometric_circle': '#9370DB',
                    'geometric_rectangle': '#9370DB',
                    'geometric_line': '#9370DB',
                    'convergent_high': '#DC143C',
                    'convergent_medium': '#FF8C00',
                    'priority_1': '#FF1493'
                };
                return colorMap[feature_type] || '#666666';
            }
            
            updateAllFilters() {
                // Get current filter states
                const showPoints = document.getElementById('show-points')?.checked ?? true;
                const showPolygons = document.getElementById('show-polygons')?.checked ?? true;
                const showLines = document.getElementById('show-lines')?.checked ?? true;
                
                const showGediOnly = document.getElementById('show-gedi-only')?.checked ?? true;
                const showSentinel2Only = document.getElementById('show-sentinel2-only')?.checked ?? true;
                const showPriority = document.getElementById('show-priority')?.checked ?? true;
                const showCrossValidated = document.getElementById('toggle-convergent')?.checked ?? true;
                
                // Convergence lines should be visible when EITHER provider is visible AND cross-validated is on
                const showConvergenceLines = (showGediOnly || showSentinel2Only) && showCrossValidated;
                
                const confidenceThreshold = (document.getElementById('confidence-slider')?.value ?? 70) / 100;
                
                // Filter unified features layer
                if (this.layers.unifiedFeaturesGroup && this.layers.unifiedFeaturesGroup.eachLayer) {
                    this.layers.unifiedFeaturesGroup.eachLayer(featureLayer => {
                        const feature = featureLayer.feature;
                        if (feature && feature.geometry) {
                            const geomType = feature.geometry.type.toLowerCase();
                            const confidence = featureLayer.confidence || 0;
                            const props = feature.properties || {};
                            
                            // Get feature classification from stored metadata or properties
                            const primaryCategory = featureLayer.primaryCategory || props.primary_category || 'sentinel2';
                            const originalProvider = featureLayer.originalProvider || props.original_provider || 'sentinel2';
                            const isPriority = featureLayer.isPriority || props.is_priority === true;
                            const isCrossValidated = featureLayer.isCrossValidated || props.is_cross_validated === true;
                            
                            let shouldShow = true;
                            
                            // Category-based filtering
                            if (primaryCategory === 'priority' || isPriority) {
                                // Priority sites - only show if priority filter is enabled
                                shouldShow = showPriority;
                            } else if (primaryCategory === 'cross_validated' || isCrossValidated) {
                                // Cross-validated features - respect both cross-validated toggle AND original provider toggle
                                if (!showCrossValidated) {
                                    shouldShow = false;
                                } else if (originalProvider === 'gedi' && !showGediOnly) {
                                    shouldShow = false;
                                } else if (originalProvider === 'sentinel2' && !showSentinel2Only) {
                                    shouldShow = false;
                                }
                            } else if (primaryCategory === 'gedi' || originalProvider === 'gedi') {
                                // GEDI-only features
                                shouldShow = showGediOnly;
                            } else if (primaryCategory === 'sentinel2' || originalProvider === 'sentinel2') {
                                // Sentinel-2-only features
                                shouldShow = showSentinel2Only;
                            }
                            
                            // Confidence threshold filter (except for priority sites which are always shown)
                            if (shouldShow && !isPriority) {
                                shouldShow = confidence >= confidenceThreshold;
                            }
                            
                            // Geometry type filter
                            if (shouldShow) {
                                if (geomType === 'point' && !showPoints) shouldShow = false;
                                else if (geomType === 'polygon' && !showPolygons) shouldShow = false;
                                else if (geomType === 'linestring' && !showLines) shouldShow = false;
                            }
                            
                            // Set visibility
                            this.setFeatureVisibility(featureLayer, shouldShow ? 1.0 : 0.2, shouldShow);
                        }
                    });
                }
                
                // Filter convergence lines separately
                if (this.layers.convergenceLinesGroup && this.layers.convergenceLinesGroup.eachLayer) {
                    this.layers.convergenceLinesGroup.eachLayer(featureLayer => {
                        const feature = featureLayer.feature;
                        if (feature && feature.geometry) {
                            const geomType = feature.geometry.type.toLowerCase();
                            let shouldShow = showConvergenceLines;
                            
                            // Geometry type filter for convergence lines
                            if (shouldShow) {
                                if (geomType === 'point' && !showPoints) shouldShow = false;
                                else if (geomType === 'polygon' && !showPolygons) shouldShow = false;
                                else if (geomType === 'linestring' && !showLines) shouldShow = false;
                            }
                            
                            this.setFeatureVisibility(featureLayer, shouldShow ? 1.0 : 0.2, shouldShow);
                        }
                    });
                }
                
                console.log(`Applied unified filters - Points: ${showPoints}, Polygons: ${showPolygons}, Lines: ${showLines}`);
                console.log(`Provider filters - GEDI: ${showGediOnly}, Sentinel-2: ${showSentinel2Only}, Priority: ${showPriority}, Cross-validated: ${showCrossValidated}`);
            }
            
            handleStatCardClick(card) {
                const filterType = card.dataset.filter;
                console.log(`Stat card clicked: ${filterType}`);
                
                if (filterType === 'all') {
                    // Show all categories
                    document.getElementById('show-gedi-only').checked = true;
                    document.getElementById('show-sentinel2-only').checked = true;
                    document.getElementById('toggle-convergent').checked = true;
                    document.getElementById('show-priority').checked = true;
                    
                    // Remove active state from all cards
                    document.querySelectorAll('.clickable-filter').forEach(c => c.classList.remove('active'));
                    card.classList.add('active');
                } else {
                    // Toggle the specific filter - map stat card filter names to checkbox IDs
                    const filterMapping = {
                        'gedi-only': 'show-gedi-only',
                        'sentinel2-only': 'show-sentinel2-only', 
                        'convergent': 'toggle-convergent',
                        'priority': 'show-priority'
                    };
                    
                    const checkboxId = filterMapping[filterType];
                    const checkbox = document.getElementById(checkboxId);
                    
                    if (checkbox) {
                        checkbox.checked = !checkbox.checked;
                        
                        // Update card active state
                        if (checkbox.checked) {
                            card.classList.add('active');
                        } else {
                            card.classList.remove('active');
                        }
                        
                        // Special handling for cross-validated toggle
                        if (filterType === 'convergent') {
                            this.toggleCrossValidatedFeatures(checkbox.checked);
                            return; // Don't call updateAllFilters again
                        }
                        
                        // Remove "all" active state if any specific filter is unchecked
                        const allCard = document.querySelector('[data-filter="all"]');
                        if (allCard && !checkbox.checked) {
                            allCard.classList.remove('active');
                        }
                    }
                }
                
                this.updateAllFilters();
            }
        }
        
        // Initialize the map when the page loads
        document.addEventListener('DOMContentLoaded', () => {
            new ArchaeologicalMap();
        });
        
        // Global function for legend toggle
        function toggleLegend() {
            const legendContent = document.querySelector('.legend-content');
            const legendIcon = document.querySelector('.legend-toggle-icon');
            const legendHeader = document.querySelector('.legend-header-clickable');
            
            if (legendContent && legendIcon && legendHeader) {
                legendContent.classList.toggle('collapsed');
                legendHeader.classList.toggle('collapsed');
                
                // Store collapsed state in localStorage
                const isCollapsed = legendContent.classList.contains('collapsed');
                localStorage.setItem('legendCollapsed', isCollapsed);
            }
        }
        
        // Restore legend state on page load
        document.addEventListener('DOMContentLoaded', () => {
            const wasCollapsed = localStorage.getItem('legendCollapsed') === 'true';
            if (wasCollapsed) {
                const legendContent = document.querySelector('.legend-content');
                const legendHeader = document.querySelector('.legend-header-clickable');
                if (legendContent && legendHeader) {
                    legendContent.classList.add('collapsed');
                    legendHeader.classList.add('collapsed');
                }
            }
        });
        """
    
    def _generate_error_page(self, error_message: str) -> str:
        """Generate error page when map generation fails"""
        
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Archaeological Discovery - Error</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    padding: 2rem;
                    background: #f8f9fa;
                    color: #721c24;
                }}
                .error-container {{
                    max-width: 600px;
                    margin: 0 auto;
                    background: white;
                    padding: 2rem;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #dc3545;
                    margin-bottom: 1rem;
                }}
            </style>
        </head>
        <body>
            <div class="error-container">
                <h1>❌ Map Generation Error</h1>
                <p>An error occurred while generating the archaeological map:</p>
                <pre>{error_message}</pre>
                <p>Please check the logs and try again.</p>
            </div>
        </body>
        </html>
        """
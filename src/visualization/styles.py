"""
Archaeological Themes and Styling System
Professional themes for different use cases
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ArchaeologicalThemes:
    """Professional archaeological visualization themes"""
    
    def __init__(self):
        self.themes = {
            'professional': self._professional_theme(),
            'field': self._field_investigation_theme(),
            'scientific': self._scientific_analysis_theme(),
            'presentation': self._presentation_theme()
        }
    
    def get_theme_config(self, theme_name: str = "professional") -> Dict[str, Any]:
        """Get complete theme configuration"""
        return self.themes.get(theme_name, self.themes['professional'])
    
    def _professional_theme(self) -> Dict[str, Any]:
        """Professional archaeological research theme"""
        return {
            'name': 'Professional Archaeological',
            'description': 'Clean, professional theme for research and documentation',
            
            'colors': {
                'primary': '#8B4513',        # Terra preta brown
                'secondary': '#228B22',      # Amazon forest green
                'accent': '#FF6B35',         # Discovery orange
                'background': '#FFFFFF',     # Clean white
                'text': '#2C3E50',          # Dark slate
                'border': '#BDC3C7'         # Light gray
            },
            
            'feature_styles': {
                'gedi_clearing': {
                    'color': '#228B22',
                    'icon': '🏘️',
                    'size': 'medium',
                    'opacity': 0.9
                },
                'gedi_mound': {
                    'color': '#8B4513', 
                    'icon': '⛰️',
                    'size': 'medium',
                    'opacity': 0.9
                },
                'terra_preta': {
                    'color': '#32CD32',
                    'icon': '🌱',
                    'size': 'medium', 
                    'opacity': 0.9
                },
                'geometric': {
                    'color': '#9370DB',
                    'icon': '⭕',
                    'size': 'medium',
                    'opacity': 0.9
                },
                'convergent': {
                    'color': '#4169E1',
                    'icon': '🎯',
                    'size': 'large',
                    'opacity': 1.0
                },
                'priority': {
                    'color': '#DC143C',
                    'icon': '🚩',
                    'size': 'large',
                    'opacity': 1.0
                }
            },
            
            'ui_elements': {
                'panel_background': 'rgba(255, 255, 255, 0.95)',
                'panel_border': '#E1E5E9',
                'header_background': '#8B4513',
                'header_text': '#FFFFFF',
                'button_primary': '#FF6B35',
                'button_secondary': '#BDC3C7'
            },
            
            'map_settings': {
                'default_zoom': 12,
                'max_zoom': 18,
                'attribution': 'Professional Archaeological Analysis'
            }
        }
    
    def _field_investigation_theme(self) -> Dict[str, Any]:
        """Field investigation and planning theme"""
        return {
            'name': 'Field Investigation',
            'description': 'High contrast theme optimized for field planning and navigation',
            
            'colors': {
                'primary': '#DC143C',        # High visibility red
                'secondary': '#FF8C00',      # Safety orange
                'accent': '#FFD700',         # High contrast gold
                'background': '#2C3E50',     # Dark background for visibility
                'text': '#FFFFFF',          # White text
                'border': '#34495E'         # Dark border
            },
            
            'feature_styles': {
                'gedi_clearing': {
                    'color': '#32CD32',
                    'icon': '🏠',
                    'size': 'large',
                    'opacity': 1.0,
                    'pulse': True
                },
                'gedi_mound': {
                    'color': '#FF8C00',
                    'icon': '▲',
                    'size': 'large', 
                    'opacity': 1.0,
                    'pulse': True
                },
                'terra_preta': {
                    'color': '#90EE90',
                    'icon': '●',
                    'size': 'large',
                    'opacity': 1.0
                },
                'geometric': {
                    'color': '#9370DB',
                    'icon': '◆',
                    'size': 'large',
                    'opacity': 1.0
                },
                'convergent': {
                    'color': '#FFD700',
                    'icon': '⭐',
                    'size': 'extra_large',
                    'opacity': 1.0,
                    'pulse': True
                },
                'priority': {
                    'color': '#DC143C',
                    'icon': '🎯',
                    'size': 'extra_large',
                    'opacity': 1.0,
                    'pulse': True,
                    'glow': True
                }
            },
            
            'ui_elements': {
                'panel_background': 'rgba(44, 62, 80, 0.95)',
                'panel_border': '#5D6D7E',
                'header_background': '#DC143C',
                'header_text': '#FFFFFF',
                'button_primary': '#FF8C00',
                'button_secondary': '#5D6D7E'
            },
            
            'map_settings': {
                'default_zoom': 14,
                'max_zoom': 18,
                'attribution': 'Field Investigation Planning'
            }
        }
    
    def _scientific_analysis_theme(self) -> Dict[str, Any]:
        """Scientific analysis and publication theme"""
        return {
            'name': 'Scientific Analysis',
            'description': 'Precise, data-focused theme for scientific analysis and publication',
            
            'colors': {
                'primary': '#2E8B57',        # Sea green for science
                'secondary': '#4682B4',      # Steel blue
                'accent': '#FF6347',         # Tomato for highlights
                'background': '#F8F9FA',     # Light scientific background
                'text': '#212529',          # Dark text for readability
                'border': '#DEE2E6'         # Subtle borders
            },
            
            'feature_styles': {
                'gedi_clearing': {
                    'color': '#2E8B57',
                    'icon': '■',
                    'size': 'small',
                    'opacity': 0.8,
                    'precision': True
                },
                'gedi_mound': {
                    'color': '#8B4513',
                    'icon': '▲',
                    'size': 'small',
                    'opacity': 0.8,
                    'precision': True
                },
                'terra_preta': {
                    'color': '#228B22',
                    'icon': '●',
                    'size': 'small',
                    'opacity': 0.8,
                    'precision': True
                },
                'geometric': {
                    'color': '#4682B4',
                    'icon': '◇',
                    'size': 'small',
                    'opacity': 0.8,
                    'precision': True
                },
                'convergent': {
                    'color': '#FF6347',
                    'icon': '⊕',
                    'size': 'medium',
                    'opacity': 0.9,
                    'precision': True
                },
                'priority': {
                    'color': '#DC143C',
                    'icon': '★',
                    'size': 'medium',
                    'opacity': 0.9,
                    'precision': True
                }
            },
            
            'ui_elements': {
                'panel_background': 'rgba(248, 249, 250, 0.98)',
                'panel_border': '#DEE2E6',
                'header_background': '#2E8B57',
                'header_text': '#FFFFFF',
                'button_primary': '#4682B4',
                'button_secondary': '#6C757D'
            },
            
            'map_settings': {
                'default_zoom': 11,
                'max_zoom': 18,
                'attribution': 'Scientific Archaeological Analysis',
                'grid_overlay': True,
                'coordinate_display': True
            }
        }
    
    def _presentation_theme(self) -> Dict[str, Any]:
        """Presentation and public outreach theme"""
        return {
            'name': 'Presentation',
            'description': 'Visually striking theme for presentations and public engagement',
            
            'colors': {
                'primary': '#8B008B',        # Dark magenta
                'secondary': '#FF1493',      # Deep pink
                'accent': '#00CED1',         # Dark turquoise
                'background': '#000000',     # Black for drama
                'text': '#FFFFFF',          # White text
                'border': '#8B008B'         # Magenta borders
            },
            
            'feature_styles': {
                'gedi_clearing': {
                    'color': '#00FF7F',
                    'icon': '🏛️',
                    'size': 'large',
                    'opacity': 1.0,
                    'glow': True,
                    'animation': 'pulse'
                },
                'gedi_mound': {
                    'color': '#FFD700',
                    'icon': '🗻',
                    'size': 'large',
                    'opacity': 1.0,
                    'glow': True,
                    'animation': 'pulse'
                },
                'terra_preta': {
                    'color': '#32CD32',
                    'icon': '🌿',
                    'size': 'large',
                    'opacity': 1.0,
                    'glow': True
                },
                'geometric': {
                    'color': '#9370DB',
                    'icon': '🔹',
                    'size': 'large',
                    'opacity': 1.0,
                    'glow': True
                },
                'convergent': {
                    'color': '#00CED1',
                    'icon': '✨',
                    'size': 'extra_large',
                    'opacity': 1.0,
                    'glow': True,
                    'animation': 'sparkle'
                },
                'priority': {
                    'color': '#FF1493',
                    'icon': '💎',
                    'size': 'extra_large',
                    'opacity': 1.0,
                    'glow': True,
                    'animation': 'sparkle'
                }
            },
            
            'ui_elements': {
                'panel_background': 'rgba(0, 0, 0, 0.9)',
                'panel_border': '#8B008B',
                'header_background': 'linear-gradient(45deg, #8B008B, #FF1493)',
                'header_text': '#FFFFFF',
                'button_primary': '#00CED1',
                'button_secondary': '#8B008B'
            },
            
            'map_settings': {
                'default_zoom': 13,
                'max_zoom': 18,
                'attribution': 'Archaeological Discovery Showcase',
                'dark_mode': True,
                'enhanced_effects': True
            }
        }
    
    def get_css_variables(self, theme_name: str = "professional") -> str:
        """Generate CSS custom properties for a theme"""
        
        theme = self.get_theme_config(theme_name)
        colors = theme.get('colors', {})
        ui = theme.get('ui_elements', {})
        
        css_vars = []
        
        # Color variables
        for name, value in colors.items():
            css_vars.append(f"  --color-{name}: {value};")
        
        # UI element variables
        for name, value in ui.items():
            css_vars.append(f"  --ui-{name}: {value};")
        
        return f":root {{\n" + "\n".join(css_vars) + "\n}"
    
    def get_feature_style(self, theme_name: str, feature_type: str) -> Dict[str, Any]:
        """Get specific feature styling for a theme"""
        
        theme = self.get_theme_config(theme_name)
        feature_styles = theme.get('feature_styles', {})
        
        return feature_styles.get(feature_type, {
            'color': '#666666',
            'icon': '●',
            'size': 'medium',
            'opacity': 0.8
        })
    
    def get_size_pixels(self, size_name: str) -> int:
        """Convert size names to pixel values"""
        
        size_map = {
            'small': 20,
            'medium': 30,
            'large': 40,
            'extra_large': 50
        }
        
        return size_map.get(size_name, 30)
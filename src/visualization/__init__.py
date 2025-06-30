"""
Modern Archaeological Visualization System
Modular, maintainable, and professional visualization components
"""

from .core import ArchaeologicalMapGenerator
from .components import (
    FeatureRenderer,
    LayerManager,
    ControlPanel
)
from .templates import HTMLTemplateEngine
from .styles import ArchaeologicalThemes

__version__ = "2.0.0"
__author__ = "Amazon Archaeological Discovery Pipeline"

# Main public interface
__all__ = [
    'ArchaeologicalMapGenerator',
    'FeatureRenderer',
    'LayerManager',
    'ControlPanel',
    'HTMLTemplateEngine',
    'ArchaeologicalThemes'
]
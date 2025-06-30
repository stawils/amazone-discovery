# Archaeological Analysis Framework - Methodology & Standards

## Overview

This document outlines the framework for archaeological analysis using the Amazon Archaeological Discovery Pipeline. It covers analysis methodology, validation procedures, and reporting standards for systematic archaeological research.

---

## Analysis Methodology Framework

### Multi-Evidence Convergence Analysis

The pipeline implements a multi-evidence convergence approach that combines different detection methods to increase archaeological confidence:

#### GEDI Space-Based LiDAR Analysis
```json
{
  "detection_parameters": {
    "gap_threshold": 15.0,
    "min_cluster_size": 3,
    "clustering_eps": 0.002
  },
  "output_format": {
    "clearings_detected": "integer",
    "gap_clusters": [
      {
        "center": "[longitude, latitude]",
        "count": "integer",
        "area_km2": "float",
        "confidence": "float (0-1)",
        "cluster_type": "string"
      }
    ]
  }
}
```

#### Sentinel-2 Multispectral Analysis
```json
{
  "detection_parameters": {
    "terra_preta_threshold": 0.12,
    "ndvi_threshold": 0.3,
    "min_area_m2": 5000
  },
  "output_format": {
    "terra_preta_patches": [
      {
        "coordinates": "[longitude, latitude]",
        "area_m2": "float",
        "mean_tp_index": "float",
        "confidence": "float (0-1)"
      }
    ]
  }
}
```

### Convergent Scoring Framework

```json
{
  "scoring_methodology": {
    "total_score": "0-14 point scale",
    "classification_thresholds": {
      "high_confidence": "10+ points",
      "probable": "7-9 points", 
      "possible": "4-6 points",
      "natural": "0-3 points"
    },
    "score_breakdown": {
      "geometric_patterns": "0-6 points",
      "terra_preta_signature": "0-2 points",
      "environmental_suitability": "0-1 points",
      "convergence_bonus": "0-3 points",
      "priority_zone_bonus": "0-2 points"
    }
  }
}
```

### Academic Validation Standards

```json
{
  "validation_framework": {
    "statistical_measures": {
      "cohens_d": "effect size metric",
      "p_value": "statistical significance",
      "confidence_interval": "[lower, upper] bounds"
    },
    "classification_criteria": {
      "exceptional": "cohens_d >= 0.8, p < 0.001",
      "high": "cohens_d >= 0.5, p < 0.01", 
      "medium": "cohens_d >= 0.3, p < 0.05",
      "low": "below thresholds"
    }
  }
}
```

---

## Performance Metrics Framework

### Detection Accuracy Assessment

| Metric Category | Measurement Method | Target Range |
|-----------------|-------------------|--------------|
| **Detection Density** | Features per km² | 0.1-5.0 (terra firme) |
| **Confidence Distribution** | Score histogram | Majority 60-85% range |
| **Multi-sensor Convergence** | Overlapping detections | >20% convergence rate |
| **False Positive Rate** | Ground truth comparison | <15% target |

### Quality Indicators

```json
{
  "primary_metrics": {
    "detection_density": "features per square kilometer",
    "flag_rate": "percentage of detections flagged",
    "confidence_distribution": "histogram of confidence scores",
    "convergence_rate": "multi-sensor agreement percentage"
  },
  "secondary_metrics": {
    "size_distribution": "feature size histogram",
    "geographic_clustering": "spatial distribution analysis",
    "spectral_coherence": "spectral signature consistency"
  }
}
```

---

## Analysis Workflow Standards

### Systematic Processing Pipeline

1. **Data Acquisition Phase**
   - Multi-sensor data collection
   - Quality assessment and filtering
   - Metadata preservation

2. **Detection Phase**
   - Provider-specific analysis
   - Feature extraction and classification
   - Confidence scoring

3. **Convergence Analysis Phase**
   - Multi-evidence aggregation
   - Spatial proximity analysis
   - Combined confidence calculation

4. **Validation Phase**
   - Academic standard validation
   - Quality assurance checks
   - Statistical significance testing

5. **Output Generation Phase**
   - Standardized report generation
   - GeoJSON export creation
   - Visualization preparation

### Processing Configuration

```python
# Standard analysis configuration
analysis_config = {
    "max_scenes": 3,
    "confidence_threshold": 0.5,
    "enable_gpu": True,
    "academic_validation": True,
    "output_formats": ["geojson", "report", "visualization"]
}
```

---

## Reporting Standards

### Analysis Report Structure

```json
{
  "report_sections": {
    "executive_summary": "key findings and significance",
    "methodology": "analysis methods and parameters",
    "results": "detection results and statistics",
    "quality_assessment": "validation and confidence metrics",
    "recommendations": "next steps and priorities"
  }
}
```

### Documentation Requirements

- **Methodology transparency**: All parameters and thresholds documented
- **Reproducibility**: Complete configuration preservation
- **Statistical rigor**: Academic validation included
- **Quality assurance**: Systematic validation checks

---

## Technical Implementation

### System Architecture Integration

The analysis framework integrates with the core pipeline architecture:

```python
from src.pipeline.modular_pipeline import ModularPipeline
from src.core.scoring import ConvergentAnomalyScorer

# Initialize analysis pipeline
pipeline = ModularPipeline(provider, run_id)
scorer = ConvergentAnomalyScorer(enable_academic_validation=True)

# Execute systematic analysis
results = pipeline.run(zones=target_zones, max_scenes=3)
```

### Quality Control Framework

```python
# Validation configuration
validation_config = {
    "density_thresholds": {"terra_firme": 5.0, "varzea": 50.0},
    "confidence_thresholds": {"minimum": 0.5, "high": 0.8},
    "statistical_requirements": {"min_p_value": 0.05, "min_effect_size": 0.3}
}
```

---

## Future Development

### Methodology Enhancements

- **Multi-temporal analysis**: Time-series change detection
- **Regional adaptation**: Zone-specific parameter tuning
- **Machine learning integration**: Advanced pattern recognition
- **Ground truth integration**: Field validation feedback loops

### Technical Improvements

- **Processing optimization**: Enhanced computational efficiency
- **Data integration**: Additional sensor modalities
- **Automation**: Streamlined analysis workflows
- **Scalability**: Large-scale survey capabilities

This framework provides the foundation for systematic archaeological analysis using satellite remote sensing technologies, ensuring scientific rigor and reproducible results.
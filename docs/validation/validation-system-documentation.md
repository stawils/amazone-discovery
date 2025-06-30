# Validation System Documentation

## Overview

The Amazon Archaeological Discovery Pipeline includes a comprehensive validation framework that cross-validates detected archaeological features against known control areas, calculates detection metrics, and flags potential issues for quality assurance.

## Validation Framework Architecture

### Core Component: `ArchaeologicalValidator`

**Location:** `/src/core/validation.py`

The validation system operates through several key components:

1. **Known Site Database** - Reference archaeological sites for true positive validation
2. **Control Area Database** - Known non-archaeological areas for false positive detection  
3. **Metrics Calculation Engine** - Detection density, confidence analysis, and quality assessment
4. **Flagging System** - Automated detection of suspicious patterns or low-quality results

## Configuration Parameters

### Confidence Thresholds

```python
# Archaeological research-aligned confidence thresholds
MINIMUM_CONFIDENCE = 0.85  # 85% minimum for archaeological features
MAXIMUM_CONFIDENCE = 0.98  # Flag unrealistically high confidence (>98%)
```

**Rationale:** Based on remote sensing archaeological literature where 85-90% confidence is considered reliable for feature detection.

### Detection Density Standards

```python
# Expected archaeological site density (per 100km²)
EXPECTED_MIN_DENSITY = 1    # Minimum sites per 100km²
EXPECTED_MAX_DENSITY = 200  # Maximum for high-density micro zones
OVER_DETECTION_THRESHOLD = 10  # Flag areas with >10 sites/km²
```

**Basis:** Derived from Amazon archaeological survey literature and adjusted for micro-zone analysis.

## Control Area Database

### Oil Infrastructure Locations

Real-world petroleum infrastructure used as negative controls:

```python
{
    "name": "Lago Agrio Oil Field",
    "lat": 0.085,      # Nueva Loja/Lago Agrio 
    "lon": -76.894,    # Major production center
    "radius_km": 5.0,  # Operational area
    "type": "oil_infrastructure"
},
{
    "name": "Shushufindi Oil Field", 
    "lat": -0.160,     # Shushufindi location
    "lon": -76.895,    # Second largest field
    "radius_km": 3.0,  # Field coverage
    "type": "oil_infrastructure"
}
```

### Protected Wetland Areas

UNESCO and government-protected wetland systems:

```python
{
    "name": "Cuyabeno Wetlands",
    "lat": -0.5,       # Wildlife Reserve center
    "lon": -76.0,      # Protected area
    "radius_km": 10.0, # Reserve boundary
    "type": "wetland"
},
{
    "name": "Yasuní Wetlands",
    "lat": -1.0,       # National Park center  
    "lon": -76.0,      # World Heritage site
    "radius_km": 15.0, # Park coverage
    "type": "wetland"
}
```

### Environmental Exclusion Zones

Areas with problematic environmental signatures for archaeological detection:

- **Pristine Forest**: High NDVI areas that mask archaeological signatures
- **White-sand Forest**: Unique soil signatures that interfere with terra preta detection
- **Modern Infrastructure**: Contemporary settlements and development

## Area Coverage Calculation

### Zone-Aware Area Estimation

The validation system uses target zone configuration for accurate density calculations:

```python
def _estimate_area_coverage(self, detection_coords):
    """Calculate detection area using zone configuration"""
    
    # 1. Identify target zone from detection coordinates
    center_lat, center_lon = np.mean(lats), np.mean(lons)
    closest_zone = self._find_closest_zone(center_lat, center_lon)
    
    # 2. Use zone search area for accurate density
    if closest_zone and distance < 50km:
        area_km2 = π × (zone.search_radius_km)²
        return area_km2
    
    # 3. Fallback to bounding box if zone not identified
    return self._calculate_bounding_box_area(coords)
```

**Example:** For `upper_napo_micro_small` with 5km search radius:
- **Area**: π × 5² = 78.54 km²
- **Density**: 139 detections / 78.54 km² = 1.77 features/km²

## Validation Metrics

### Core Metrics Calculated

1. **True Positive Rate**: Detections near known archaeological sites
2. **False Positive Rate**: Detections in control areas  
3. **Detection Density**: Features per km² for the survey area
4. **Confidence Distribution**: Statistical analysis of detection confidence
5. **Temporal Persistence**: Consistency across multiple time periods

### Quality Assessment Flags

#### High Severity Issues
- Detections below 85% confidence threshold
- Features in known oil infrastructure areas
- Unrealistically large features (>10 hectares)
- Detection density exceeding 10 features/km²

#### Medium Severity Issues  
- Detection density below expected range (1-5 per 100km²)
- Features not detected in previous years
- Confidence above 98% (potentially over-fitted)

#### Environmental Context Flags
- Detections in protected wetland areas
- Features in pristine forest zones
- Overlaps with modern infrastructure

## Integration with Pipeline

### Validation Workflow

1. **Data Collection**: Gather all detections from pipeline execution
2. **Coordinate Validation**: Verify geographic positioning accuracy
3. **Control Area Analysis**: Check for overlaps with negative control zones
4. **Density Analysis**: Calculate detection density using zone configuration
5. **Quality Assessment**: Apply confidence and environmental filters
6. **Report Generation**: Create detailed validation reports with recommendations

### Output Files Generated

```bash
/results/run_YYYYMMDD_HHMMSS/validation/
├── validation_results_[zone]_[timestamp].json     # Detailed metrics
├── flagged_detections_[zone]_[timestamp].csv      # Issues for review
└── validation_summary_[zone]_[timestamp].md       # Human-readable summary
```

## API Reference

### Primary Validation Method

```python
def validate_detections(self, detections: List[Dict], zone_name: str = None) -> Dict:
    """
    Validate archaeological detections against control areas and quality metrics
    
    Args:
        detections: List of detected features with coordinates and metadata
        zone_name: Target zone identifier for context-specific validation
        
    Returns:
        Dict containing:
        - total_detections: Number of features validated
        - validation_metrics: Quality assessment scores
        - flagged_detections: List of problematic features
        - recommendations: Suggested parameter adjustments
    """
```

### Utility Methods

```python
def _calculate_distance(self, lat1, lon1, lat2, lon2) -> float:
    """Calculate great circle distance between two points in kilometers"""

def _estimate_area_coverage(self, detection_coords) -> float:
    """Estimate survey area using zone configuration or bounding box"""

def _save_validation_results(self, results: Dict, zone_name: str):
    """Export validation results to JSON, CSV, and Markdown formats"""
```

## Configuration Management

### Threshold Adjustment

To modify validation parameters, edit `/src/core/validation.py`:

```python
# Confidence thresholds
elif confidence < 0.85:  # Adjust this value as needed
    flagged.append({
        "detection": detection,
        "flag_reason": "Below required confidence threshold (85%)",
        "severity": "high"
    })

# Density thresholds  
expected_max_density = 200  # Adjust based on survey expectations
expected_min_density = 1    # Modify for different survey intensities
```

### Control Area Updates

To add new control areas or modify existing ones:

```python
def _load_control_areas(self) -> List[Dict[str, Any]]:
    """Load known non-archaeological areas for validation"""
    return [
        {
            "name": "New Control Area",
            "lat": -0.123,     # Decimal degrees
            "lon": -72.456,    # Decimal degrees  
            "radius_km": 2.0,  # Coverage radius
            "type": "area_type",  # Category
            "confidence": 0.9  # Certainty level
        }
        # ... additional control areas
    ]
```

## Best Practices

### Survey Planning
- Review validation results before field surveys
- Use flagged detections to identify systematic issues
- Adjust detection parameters based on false positive rates

### Quality Assurance
- Validate control area coordinates against current maps
- Update thresholds based on field validation results  
- Monitor density metrics for survey area appropriateness

### Result Interpretation
- Focus on high-confidence, unflagged detections for priority investigation
- Use validation metrics to communicate detection quality to stakeholders
- Document validation parameters for reproducibility

## Maintenance and Updates

### Regular Updates Required
- **Control Area Verification**: Annual check of oil infrastructure and protected area boundaries
- **Threshold Calibration**: Adjust based on field validation results
- **Database Updates**: Add new known archaeological sites as discovered

### Performance Monitoring
- Track false positive rates across different survey areas
- Monitor validation processing time for large detection sets
- Assess accuracy of zone-based area calculations

## Integration Points

### Checkpoint System Integration
The validation framework integrates with the OpenAI checkpoint system to provide quality assessment for each pipeline execution.

### Export System Compatibility  
Validation results are compatible with the unified export manager and can be included in final survey reports.

### Visualization Integration
Flagged detections and validation metrics can be displayed on archaeological maps for visual quality assessment.

---

*This documentation covers validation system version 2.1 as implemented in the Amazon Archaeological Discovery Pipeline.*
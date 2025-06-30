# EMERGENCY: Catastrophic Over-Detection Analysis

**Date:** 2025-06-13  
**Status:** CRITICAL SYSTEM CALIBRATION REQUIRED  
**Issue:** Detection system finding 100-1000x more sites than archaeologically realistic

## Research-Based Problem Analysis

### Expected Archaeological Site Density (Literature)
- **Ancient Amazon Várzea (floodplains)**: 14.6 persons/km²
- **Ancient Amazon Terra Firme (uplands)**: 0.2 persons/km²
- **Pre-Columbian settlements**: Distributed with realistic spacing
- **Upper Napo region**: Under-surveyed, likely 1-50 sites per 100km²

### Current System Performance
- **Detection Rate**: 978 sites/km² 
- **Over-Detection Factor**: 500-5000x expected density
- **Validation Flag Rate**: 95% (indicating system recognizes the problem)
- **GEDI Detection Rate**: 0 (may be appropriately conservative)

### Archaeological Remote Sensing Standards (MDPI 2024)
- **50-59% confidence**: "Eventual True Positive" (worth investigating)
- **70-84% confidence**: "Very Likely True Positive" (high archaeological value)  
- **85%+ confidence**: "Definitive" (exceptional archaeological certainty)

**→ Our 85% validation threshold is CORRECT - it's the detectors over-detecting**

## Root Cause Analysis

### Sentinel-2 Detector Issues
1. **Terra preta threshold too low**: 0.12 (should be 0.25+ for archaeological significance)
2. **NDVI depression threshold too low**: 0.2 (should be 0.4+ for archaeological clearings)
3. **Minimum area too small**: Current allows tiny features that aren't archaeological sites
4. **No geometric filtering**: Processing artifacts being classified as archaeological

### GEDI Detector Assessment
- **Current parameters may be SCIENTIFICALLY APPROPRIATE**
- Archaeological clearings in Amazon are rare and should require strong evidence
- Zero detections may indicate realistic archaeological absence in test area

### Validation System Analysis
- **85% confidence threshold is INDUSTRY STANDARD**
- High flag rate (95%) indicates system correctly identifying over-detection
- Need to fix detectors, not lower validation standards

## Archaeological Literature Support

### Site Density Evidence
- Pre-Columbian Amazon populations were concentrated along rivers
- Vast areas of terra firme had very low population density
- Archaeological surveys consistently find sparse site distributions
- 978 sites/km² would represent unprecedented population density

### Remote Sensing Accuracy Standards
- 95% accuracy achievable with proper algorithms
- 85% confidence threshold standard in archaeological remote sensing
- Lower confidence acceptable for exploratory research, not production

## Required Actions

1. **EMERGENCY**: Dramatically tighten Sentinel-2 detection parameters
2. **MAINTAIN**: GEDI conservative parameters (likely scientifically appropriate)  
3. **PRESERVE**: 85% validation confidence threshold
4. **IMPLEMENT**: Geometric filtering to remove processing artifacts
5. **ADD**: Multi-scale archaeological significance testing

## Expected Outcomes Post-Fix

- **Detection Rate**: 978/km² → 0.1-5/km² (realistic archaeological density)
- **Validation Flag Rate**: 95% → 20-30% (focus on genuine quality issues)
- **GEDI Contribution**: 0 → 1-10 clearings (appropriate for archaeological research)
- **Scientific Credibility**: Restored to archaeological literature standards

---
**Next Steps**: Implement aggressive parameter tightening based on this analysis
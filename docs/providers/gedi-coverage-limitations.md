# GEDI Coverage Limitations and Mitigation Strategies

## Current GEDI Implementation

### What We Have
- **Data Source**: GEDI L2A footprint data (raw orbital tracks)
- **Coverage Pattern**: Linear strips with ~600m spacing between tracks
- **Temporal Range**: 2019-2023 (mission duration)
- **Spatial Resolution**: 25m diameter footprints, 60m along-track spacing

### Expected Coverage Patterns
When you see GEDI data in maps, the linear "strip" pattern is **normal and expected**:
- Each strip represents one orbital pass of the International Space Station
- Adjacent strips are typically 600m apart
- This is NOT a bug - it's how space-based LiDAR works

## Limitations and Archaeological Impact

### Geographic Coverage Gaps
- **Between-track gaps**: 600m spacing means ~70-80% area coverage
- **Temporal variation**: Different areas covered on different dates
- **Cloud interference**: Some tracks may have data gaps

### Archaeological Implications
- **Sites near tracks**: High-confidence detection with precise measurements
- **Sites between tracks**: May be missed by GEDI alone
- **Convergence analysis**: Requires Sentinel-2 data to fill spatial gaps

## Current Mitigation Strategies

### 1. Multi-Sensor Fusion
```
GEDI (high-precision strips) + Sentinel-2 (complete coverage) = Comprehensive analysis
```

### 2. Multi-Temporal Collection
- Pipeline can aggregate multiple GEDI passes over time
- Different seasons may provide different track coverage
- 4+ years of mission data maximizes spatial coverage

### 3. Statistical Confidence
- Focus on high-confidence detections within GEDI coverage area
- Use convergence scoring when multiple sensors agree
- Document coverage limitations in reports

## Future Enhancement Options

### Option 1: Enhanced Multi-Temporal (Medium Complexity)
- Aggregate 2019-2023 GEDI data for maximum coverage
- Query multiple orbital passes per zone
- **Effort**: 1-2 weeks development
- **Benefit**: +10-20% spatial coverage

### Option 2: GEDI L4B Gridded Products (High Complexity) 
- Use NASA's interpolated 1km grid products
- Implement statistical gap-filling algorithms
- **Effort**: 2-3 weeks development
- **Benefit**: Near-complete coverage but lower precision

### Option 3: ICESat-2 Integration (High Complexity)
- Add second space LiDAR with different orbital pattern
- Implement data fusion algorithms
- **Effort**: 3-4 weeks development
- **Benefit**: Complementary track coverage

## Recommendation

**For archaeological analysis, the current implementation is sufficient because:**

1. **Most archaeological sites are large enough** (>100m) to intersect GEDI tracks
2. **Convergence analysis with Sentinel-2** provides comprehensive coverage
3. **High-precision measurements** are more valuable than statistical interpolation
4. **Working pipeline** is better than complex system under development

**Document coverage gaps clearly in reports** rather than implementing complex workarounds that may reduce system reliability.

## Implementation Notes

If enhanced coverage is needed in future:
1. Start with multi-temporal aggregation (Option 1)
2. Evaluate coverage improvement before more complex options
3. Maintain backward compatibility with current pipeline
4. Add coverage quality metrics to reports
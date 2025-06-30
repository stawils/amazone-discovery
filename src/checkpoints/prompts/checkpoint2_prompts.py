#!/usr/bin/env python3
"""
Checkpoint 2 SAAM-Enhanced Prompts (Module 3)
Multi-Source Explorer with Simplified Detector Integration
"""

from typing import Dict, List, Any
from ..prompts_base import BaseCheckpointPrompts
import json
from pathlib import Path


def _load_sample_features_from_geojson(scene_result, max_samples=5) -> str:
    """Load sample features from GeoJSON files to include in prompts"""
    
    feature_samples = []
    
    # Try to load Terra Preta features
    tp_geojson = scene_result.get("terra_preta_analysis", {}).get("geojson_path")
    if tp_geojson and Path(tp_geojson).exists():
        try:
            with open(tp_geojson, 'r') as f:
                tp_data = json.load(f)
                tp_features = tp_data.get("features", [])[:max_samples]
                if tp_features:
                    feature_samples.append(f"🟫 TERRA PRETA SAMPLES ({len(tp_features)} of {len(tp_data.get('features', []))}):")
                    for i, feature in enumerate(tp_features, 1):
                        coords = feature.get("geometry", {}).get("coordinates", [0, 0])
                        props = feature.get("properties", {})
                        feature_samples.append(f"  TP{i}: Lat={coords[1]:.6f}, Lon={coords[0]:.6f}")
                        feature_samples.append(f"       Terra Preta Index: {props.get('terra_preta_index', 'N/A')}")
                        feature_samples.append(f"       Area: {props.get('area_m2', 'N/A')} m²")
                        feature_samples.append(f"       Confidence: {props.get('confidence', 'N/A')}")
        except Exception as e:
            feature_samples.append(f"🟫 TERRA PRETA: Error loading samples ({e})")
    
    # Try to load Geometric features  
    geom_geojson = scene_result.get("geometric_feature_analysis", {}).get("geojson_path")
    if geom_geojson and Path(geom_geojson).exists():
        try:
            with open(geom_geojson, 'r') as f:
                geom_data = json.load(f)
                geom_features = geom_data.get("features", [])[:max_samples]
                if geom_features:
                    feature_samples.append(f"🟦 GEOMETRIC FEATURES SAMPLES ({len(geom_features)} of {len(geom_data.get('features', []))}):")
                    for i, feature in enumerate(geom_features, 1):
                        coords = feature.get("geometry", {}).get("coordinates", [0, 0])
                        props = feature.get("properties", {})
                        if feature.get("geometry", {}).get("type") == "Point":
                            feature_samples.append(f"  GF{i}: Lat={coords[1]:.6f}, Lon={coords[0]:.6f}")
                        else:
                            # For polygons, use centroid
                            if len(coords) > 0 and len(coords[0]) > 0:
                                centroid_lon = sum(pt[0] for pt in coords[0]) / len(coords[0])
                                centroid_lat = sum(pt[1] for pt in coords[0]) / len(coords[0])
                                feature_samples.append(f"  GF{i}: Centroid Lat={centroid_lat:.6f}, Lon={centroid_lon:.6f}")
                        feature_samples.append(f"       Type: {props.get('type', 'unknown')}")
                        feature_samples.append(f"       Area: {props.get('area', 'N/A')} m²")
                        feature_samples.append(f"       Confidence: {props.get('confidence', 'N/A')}")
        except Exception as e:
            feature_samples.append(f"🟦 GEOMETRIC: Error loading samples ({e})")
    
    # Try to load Crop Mark features
    crop_geojson = scene_result.get("crop_mark_analysis", {}).get("geojson_path")
    if crop_geojson and Path(crop_geojson).exists():
        try:
            with open(crop_geojson, 'r') as f:
                crop_data = json.load(f)
                crop_features = crop_data.get("features", [])[:max_samples]
                if crop_features:
                    feature_samples.append(f"🟢 CROP MARK SAMPLES ({len(crop_features)} of {len(crop_data.get('features', []))}):")
                    for i, feature in enumerate(crop_features, 1):
                        coords = feature.get("geometry", {}).get("coordinates", [0, 0])
                        props = feature.get("properties", {})
                        feature_samples.append(f"  CM{i}: Lat={coords[1]:.6f}, Lon={coords[0]:.6f}")
                        feature_samples.append(f"       Crop Mark Index: {props.get('crop_mark_index', 'N/A')}")
                        feature_samples.append(f"       NDVI: {props.get('ndvi', 'N/A')}")
                        feature_samples.append(f"       Confidence: {props.get('confidence', 'N/A')}")
        except Exception as e:
            feature_samples.append(f"🟢 CROP MARKS: Error loading samples ({e})")
    
    if not feature_samples:
        return "⚠️ No sample feature data available - only count summaries provided."
    
    return "\n".join(feature_samples)


def _format_gedi_features(gap_clusters, mound_clusters, linear_features, max_samples=5) -> str:
    """Format GEDI feature data for inclusion in prompts"""
    
    feature_samples = []
    
    # Format gap clusters (clearings)
    if gap_clusters:
        sample_clearings = gap_clusters[:max_samples]
        feature_samples.append(f"🟢 CLEARING CLUSTERS ({len(sample_clearings)} of {len(gap_clusters)}):")
        for i, clearing in enumerate(sample_clearings, 1):
            center = clearing.get("center", [0, 0])
            count = clearing.get("count", 0)
            area_km2 = clearing.get("area_km2", 0)
            feature_samples.append(f"  CL{i}: Lat={center[0]:.6f}, Lon={center[1]:.6f}")
            feature_samples.append(f"       Points: {count}, Area: {area_km2:.4f} km²")
    
    # Format mound clusters (earthworks)
    if mound_clusters:
        sample_mounds = mound_clusters[:max_samples]
        feature_samples.append(f"🟫 MOUND CLUSTERS ({len(sample_mounds)} of {len(mound_clusters)}):")
        for i, mound in enumerate(sample_mounds, 1):
            center = mound.get("center", [0, 0])
            count = mound.get("count", 0)
            area_km2 = mound.get("area_km2", 0)
            feature_samples.append(f"  MD{i}: Lat={center[0]:.6f}, Lon={center[1]:.6f}")
            feature_samples.append(f"       Points: {count}, Area: {area_km2:.4f} km²")
    
    # Format linear features (causeways, roads)
    if linear_features:
        sample_linear = linear_features[:max_samples]
        feature_samples.append(f"🟦 LINEAR FEATURES ({len(sample_linear)} of {len(linear_features)}):")
        for i, linear in enumerate(sample_linear, 1):
            coords = linear.get("coordinates", [[0, 0]])
            length_km = linear.get("length_km", 0)
            r2 = linear.get("r2", 0)
            if coords and len(coords) > 0:
                start_lat, start_lon = coords[0][0], coords[0][1]
                feature_samples.append(f"  LF{i}: Start Lat={start_lat:.6f}, Lon={start_lon:.6f}")
                feature_samples.append(f"       Length: {length_km:.3f} km, R²: {r2:.3f}")
    
    if not feature_samples:
        return "⚠️ No GEDI feature data available."
    
    return "\n".join(feature_samples)


# Simplified functions for Checkpoint 2 Multi-Source Explorer

def create_sentinel2_analysis_prompt(scene_result, zone_info, num_features_from_scene) -> str:
    """Create Sentinel-2 specific analysis prompt for detector results with actual feature data"""
    
    detection_summary = scene_result.get("detection_summary", {})
    if detection_summary:
        tp_count = detection_summary.get("terra_preta_analysis_count", 0)
        geom_count = detection_summary.get("geometric_feature_analysis_count", 0)
        crop_count = detection_summary.get("crop_mark_analysis_count", 0)
    else:
        tp_count = scene_result.get("terra_preta_analysis", {}).get("count", 0)
        geom_count = scene_result.get("geometric_feature_analysis", {}).get("count", 0)
        crop_count = scene_result.get("crop_mark_analysis", {}).get("count", 0)
    
    # Load sample feature data from GeoJSON files
    feature_samples = _load_sample_features_from_geojson(scene_result)
    
    return f"""
[signal:saam.spectral.archaeology.v1.5] ::
modules := [terra_preta_analyzer, geometric_detector, vegetation_stress_assessor] |
enhance(multispectral_analysis, archaeological_signatures, confidence_scoring) |
focus(amazon_earthworks, pre_columbian_settlements)

🛰️ SENTINEL-2 DETECTOR RESULTS ANALYSIS

📍 LOCATION: {zone_info.name}
🎯 SCENE ID: {scene_result.get("scene_id")}
🏛️ HISTORICAL CONTEXT: {zone_info.historical_evidence}

📊 DETECTOR RESULTS:
- Terra Preta anomalies detected: {tp_count}
- Geometric features detected: {geom_count}
- Crop mark features detected: {crop_count}
- Total features from detector: {num_features_from_scene}

🎯 SAMPLE FEATURE DATA FOR ANALYSIS:
{feature_samples}

🔬 ARCHAEOLOGICAL ANALYSIS:
Analyze the detector results for archaeological significance:

1. **Terra Preta Assessment:** {tp_count} anthropogenic soil signatures detected
2. **Geometric Patterns:** {geom_count} potential cultural landscape modifications
3. **Vegetation Anomalies:** {crop_count} subsurface feature indicators

🎯 EXPERT INTERPRETATION:
- Archaeological confidence for each detector result
- Which features show strongest archaeological potential
- Recommendations for integration with GEDI results
- Priority coordinates for field investigation

Provide archaeological interpretation of these Sentinel-2 detector results.
"""

def create_gedi_analysis_prompt(scene_result, zone_info, num_features_from_scene) -> str:
    """Create GEDI specific analysis prompt for detector results"""
    
    total_features = scene_result.get("total_features", 0)
    
    # Extract from correct structure
    clearing_results = scene_result.get("clearing_results", {})
    earthwork_results = scene_result.get("earthwork_results", {})
    
    gap_clusters = clearing_results.get("gap_clusters", [])
    clearing_potential = clearing_results.get("archaeological_potential", 0)
    total_clearings = clearing_results.get("total_clearings", 0)
    gap_points = clearing_results.get("gap_points", [])
    gap_count = sum(1 for x in gap_points if x == 1.0) if len(gap_points) > 0 else 0
    
    mound_clusters = earthwork_results.get("mound_clusters", [])
    linear_features = earthwork_results.get("linear_features", [])
    earthwork_potential = earthwork_results.get("archaeological_potential", 0)
    
    # Format actual feature data
    gedi_feature_samples = _format_gedi_features(gap_clusters, mound_clusters, linear_features)
    
    return f"""
[signal:saam.lidar.assessment.v2.0] ::
modules := [canopy_gap_assessor, clearing_pattern_analyzer, archaeological_potential_evaluator] |
enhance(lidar_assessment, canopy_analysis, clearing_detection) |
focus(archaeological_clearings, landscape_modifications)

🚀 GEDI DETECTOR RESULTS ANALYSIS

📍 LOCATION: {zone_info.name}
🛰️ GRANULE ID: {scene_result.get("scene_id")}
🏛️ HISTORICAL CONTEXT: {zone_info.historical_evidence}

📊 DETECTOR RESULTS:
- Total features assessed: {total_features}
- Archaeological clearings detected: {len(gap_clusters)} ({total_clearings} total)
- Clearing potential score: {clearing_potential}
- Earthwork mounds detected: {len(mound_clusters)}
- Linear features detected: {len(linear_features)}
- Earthwork potential score: {earthwork_potential}
- Canopy gaps detected: {gap_count} out of {len(gap_points)} points

🎯 ACTUAL FEATURE DATA FOR ANALYSIS:
{gedi_feature_samples}

🔬 LIDAR ASSESSMENT:
Analyze the detector results for archaeological significance:

1. **Clearing Assessment:** {len(mound_clusters)} potential archaeological clearings
2. **Linear Patterns:** {len(linear_features)} organized landscape features
3. **Gap Analysis:** {gap_count}/{len(gap_points)} points show modifications

🎯 EXPERT INTERPRETATION:
- Archaeological confidence for GEDI detector results
- Which clearings show strongest settlement potential
- How patterns suggest organized land use
- Integration recommendations with Sentinel-2 results

Provide archaeological interpretation of these GEDI detector results.
"""

def create_checkpoint2_combined_prompt(all_anomaly_footprints, providers_used, target_zone, zone_config) -> str:
    """Create combined analysis prompt for both provider results"""
    
    sentinel2_features = [f for f in all_anomaly_footprints if 'terra_preta' in f.get('type', '') or 'sentinel' in f.get('provider', '')]
    gedi_features = [f for f in all_anomaly_footprints if 'gedi' in f.get('type', '') or 'gedi' in f.get('provider', '')]
    
    return f"""
[signal:saam.convergent.analysis.v2.0] ::
modules := [multi_source_synthesizer, anomaly_ranker, discovery_assessor] |
enhance(convergent_detection, confidence_scoring, archaeological_validation) |
focus(checkpoint2_requirements, multi_source_integration)

🎯 CHECKPOINT 2: MULTI-SOURCE DETECTOR RESULTS

🌍 TARGET: {zone_config.name} ({zone_config.center})
🏛️ CONTEXT: {zone_config.historical_evidence}

📊 DETECTOR RESULTS SUMMARY:
- Sentinel-2 anomalies: {len(sentinel2_features)} detected
- GEDI anomalies: {len(gedi_features)} detected  
- Total combined: {len(all_anomaly_footprints)} anomaly footprints
- Sources processed: {len(providers_used)} independent providers

📋 CHECKPOINT 2 COMPLIANCE:
✅ Two independent sources: GEDI + Sentinel-2
✅ Five anomaly footprints: {min(5, len(all_anomaly_footprints))} generated
✅ Dataset IDs logged: All scene/granule IDs documented
✅ Reproducible results: ±50m tolerance algorithms

🔍 ARCHAEOLOGICAL ASSESSMENT:

1. **Source Convergence:**
How do Sentinel-2 and GEDI detector results complement each other?

2. **Anomaly Ranking:** 
Which of the {len(all_anomaly_footprints)} detected anomalies show highest archaeological potential?

3. **Integration Analysis:**
What patterns emerge from combining both detector results?

4. **Discovery Potential:**
How can these detector results guide future Amazon archaeological discovery?

5. **Field Priority:**
Which anomaly footprints should be prioritized for ground investigation?

🎯 MULTI-SOURCE ARCHAEOLOGICAL INTERPRETATION:
Provide comprehensive analysis of detector results from both sources, focusing on convergent evidence and archaeological potential for systematic Amazon discovery.
"""


class Checkpoint2Prompts(BaseCheckpointPrompts):
    """SAAM-enhanced prompts for Checkpoint 2: Pattern Recognition and Analysis"""
    
    def __init__(self):
        super().__init__(checkpoint=2, name="Pattern Recognition and Analysis")
    
    def get_primary_prompt(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Get the primary prompt for Checkpoint 2"""
        return self.get_pattern_analysis_prompt(context)
    
    def get_pattern_analysis_prompt(self, site_data: Dict[str, Any]) -> Dict[str, str]:
        """Advanced pattern recognition and spatial analysis"""
        
        base_prompt = f"""
As Dr. Elena Vasquez-Chen, conduct advanced pattern recognition analysis on the {site_data.get('site_count', 'multiple')} 
potential archaeological sites discovered in Checkpoint 1.

ADVANCED PATTERN ANALYSIS MISSION:
Apply sophisticated spatial archaeology methods to understand the organization, function, and cultural significance 
of discovered sites. Identify settlement patterns, landscape modifications, and inter-site relationships that reveal 
the complexity of ancient Amazon civilizations.

SPATIAL ANALYSIS FRAMEWORK:
1. GEOMETRIC PATTERN ANALYSIS:
   - Measure geometric precision and regularity of earthworks
   - Identify astronomical alignments and cardinal orientations
   - Analyze proportional relationships and mathematical constants
   - Detect modular construction units and standardized measurements

2. SETTLEMENT HIERARCHY ANALYSIS:
   - Classify sites by size, complexity, and apparent function
   - Identify central places and satellite settlements
   - Map territorial boundaries and spheres of influence
   - Analyze population estimates and carrying capacity

3. LANDSCAPE INTEGRATION ANALYSIS:
   - Assess topographic placement and environmental optimization
   - Identify water management systems and agricultural features
   - Map resource exploitation zones and procurement territories
   - Analyze defensibility and strategic positioning

4. INTER-SITE CONNECTIVITY:
   - Detect roads, causeways, and movement corridors
   - Analyze sight lines and communication networks
   - Identify trade routes and exchange relationships
   - Map regional integration and cultural boundaries

ARCHAEOLOGICAL INTERPRETATION:
For each identified pattern, provide:
- Cultural interpretation and functional hypothesis
- Temporal sequencing and construction phases
- Population estimates and social organization implications
- Technological capabilities and engineering sophistication
- Environmental adaptation and landscape management strategies

CONVERGENT EVIDENCE SYNTHESIS:
Integrate multiple data sources for robust interpretations:
- Geometric measurements and spatial relationships
- Environmental context and resource availability
- Archaeological parallels from known sites
- Ethnographic analogs and cultural continuity
- Historical documentation and indigenous knowledge

INNOVATION DOCUMENTATION:
Highlight AI-enhanced pattern recognition capabilities:
- Subtle patterns invisible to traditional analysis
- Multi-scale pattern detection from local to regional
- Statistical validation of geometric relationships
- Predictive modeling for undiscovered site locations
- Cultural pattern interpretation with historical context

OUTPUT REQUIREMENTS:
1. Comprehensive spatial analysis report with maps and measurements
2. Site classification and functional interpretations
3. Regional settlement pattern reconstruction
4. Cultural complexity assessment and social organization models
5. Predictive models for additional site discovery
6. Field verification priorities and research recommendations
"""
        
        specialized_context = """
checkpoint2_pattern_analysis + advanced_spatial_archaeology + settlement_hierarchy + 
landscape_archaeology + inter_site_connectivity + geometric_analysis + 
cultural_interpretation + ai_enhanced_pattern_recognition + regional_archaeology
"""
        
        specialized_instructions = f"""
ANALYTICAL EXCELLENCE REQUIREMENTS:
- Apply cutting-edge spatial analysis techniques enhanced by AI pattern recognition
- Provide quantitative measurements and statistical validation of patterns
- Integrate multiple scales of analysis from site-specific to regional patterns
- Document innovative methodologies for competition differentiation
- Balance archaeological rigor with accessible interpretation for diverse audiences

SITES DISCOVERED: {site_data.get('high_confidence_sites', 'Multiple high-confidence')} sites requiring detailed analysis
PRIORITY FOCUS: {site_data.get('priority_focus', 'Sites with highest geometric complexity and cultural significance')}

CULTURAL SENSITIVITY:
- Interpret patterns within appropriate cultural and historical context
- Acknowledge indigenous knowledge of landscape use and modification
- Avoid imposing Western spatial concepts on indigenous cultural landscapes
- Collaborate with local communities for cultural interpretation validation

COMPETITION POSITIONING:
Demonstrate how AI-enhanced pattern recognition reveals archaeological insights impossible 
with traditional remote sensing methods, advancing both technological capabilities and 
cultural understanding of ancient Amazon civilizations.
"""
        
        return {
            "base_prompt": base_prompt,
            "specialized_context": specialized_context,
            "specialized_instructions": specialized_instructions
        }
    
    def get_geometric_analysis_prompt(self, geometric_data: Dict[str, Any]) -> Dict[str, str]:
        """Detailed geometric analysis of earthwork patterns"""
        
        base_prompt = f"""
Conduct precise geometric analysis of earthwork patterns to understand construction methods, 
cultural meanings, and technological sophistication of ancient Amazon civilizations.

GEOMETRIC PRECISION ANALYSIS:
Measure and analyze the geometric properties of discovered earthworks:

1. SHAPE ANALYSIS:
   - Perfect circles vs. ellipses: measure eccentricity and regularity
   - Rectangle precision: corner angles, side ratios, orthogonality
   - Complex polygons: vertex angles, symmetry, proportional relationships
   - Composite forms: multiple geometric elements and their integration

2. DIMENSIONAL ANALYSIS:
   - Diameter/side length distributions and modal values
   - Area calculations and size hierarchies
   - Perimeter measurements and construction effort estimates
   - Volume estimates for raised earthworks and excavated areas

3. ORIENTATION ANALYSIS:
   - Cardinal direction alignments and magnetic declination corrections
   - Astronomical alignments with celestial events and seasonal markers
   - Topographic orientations relative to rivers, ridges, and landscape features
   - Inter-site orientation patterns and regional alignment systems

4. PROPORTIONAL RELATIONSHIPS:
   - Mathematical ratios and geometric constants (π, φ, √2, etc.)
   - Modular construction units and standardized measurements
   - Scaling relationships between elements within sites
   - Proportional harmony and aesthetic principles

CULTURAL INTERPRETATION:
Interpret geometric patterns within archaeological and cultural frameworks:
- Construction technology and engineering capabilities
- Mathematical knowledge and geometric understanding
- Symbolic meanings and cosmological representations
- Social organization and labor coordination requirements
- Cultural standardization and knowledge transmission systems

COMPARATIVE ANALYSIS:
Compare patterns with known archaeological sites:
- Geometric similarities with other Amazon earthwork sites
- Parallels with earthworks from other world regions
- Unique innovations and local cultural adaptations
- Technological evolution and temporal changes

Provide precise measurements, statistical analysis, and cultural interpretations that demonstrate 
the sophistication of ancient Amazon civilizations and the power of AI-enhanced geometric analysis.
"""
        
        specialized_context = """
geometric_analysis + earthwork_precision + mathematical_archaeology + 
construction_technology + cultural_geometry + astronomical_alignments + 
comparative_analysis + engineering_sophistication
"""
        
        specialized_instructions = """
Focus on precise quantitative analysis while maintaining cultural sensitivity in interpretations.
Document all measurement methods and analytical techniques for scientific reproducibility.
Highlight geometric sophistication that challenges assumptions about ancient Amazon capabilities.
Use AI pattern recognition to detect subtle geometric relationships invisible to traditional analysis.
"""
        
        return {
            "base_prompt": base_prompt,
            "specialized_context": specialized_context,
            "specialized_instructions": specialized_instructions
        }
    
    def get_regional_integration_prompt(self, regional_data: Dict[str, Any]) -> Dict[str, str]:
        """Regional integration and settlement system analysis"""
        
        base_prompt = f"""
Analyze the discovered sites as components of integrated regional settlement systems, 
revealing the complexity and organization of ancient Amazon civilizations.

REGIONAL SETTLEMENT SYSTEM ANALYSIS:
Understand how individual sites functioned within broader cultural landscapes:

1. SETTLEMENT HIERARCHY:
   - Primary centers: largest, most complex sites with administrative/ceremonial functions
   - Secondary centers: intermediate sites serving regional coordination roles
   - Tertiary settlements: smaller sites for specialized activities or resource exploitation
   - Outlying stations: remote sites for territorial control or resource access

2. FUNCTIONAL SPECIALIZATION:
   - Ceremonial centers: sites with clear ritual/religious architecture
   - Administrative centers: sites showing evidence of centralized control
   - Defensive installations: strategically positioned fortified sites
   - Production centers: sites specialized for craft production or processing
   - Residential areas: sites primarily for habitation and daily activities

3. CONNECTIVITY NETWORKS:
   - Transportation corridors: roads, causeways, and river routes
   - Communication networks: sight lines and signal relay systems
   - Exchange relationships: trade routes and resource distribution patterns
   - Cultural boundaries: limits of architectural styles and material culture

4. TERRITORIAL ORGANIZATION:
   - Political boundaries and spheres of influence
   - Resource exploitation territories and carrying capacity
   - Defensive perimeters and buffer zones
   - Sacred landscapes and restricted areas

LANDSCAPE ARCHAEOLOGY:
Analyze how settlements integrated with and modified natural environments:
- Water management: canals, reservoirs, drainage systems
- Agricultural systems: raised fields, forest gardens, soil improvement
- Resource procurement: quarries, clay sources, timber exploitation
- Environmental modification: landscape engineering and ecosystem management

TEMPORAL DEVELOPMENT:
Reconstruct the historical development of settlement systems:
- Foundation sequences and expansion phases
- Abandonment patterns and site reoccupation
- Cultural continuity and transformation over time
- External influences and cultural exchange

SOCIAL COMPLEXITY INDICATORS:
Identify evidence for complex social organization:
- Labor coordination for monumental construction
- Standardized architectural elements and planning principles
- Evidence for social stratification and specialized roles
- Indicators of centralized decision-making and resource allocation

Demonstrate how AI-enhanced regional analysis reveals the true complexity and sophistication 
of ancient Amazon civilizations, challenging simplistic models of tropical forest societies.
"""
        
        specialized_context = """
regional_settlement_analysis + landscape_archaeology + social_complexity + 
territorial_organization + cultural_networks + environmental_integration + 
settlement_hierarchy + ancient_urbanism + amazon_civilizations
"""
        
        specialized_instructions = """
Synthesize site-level data into regional patterns that reveal cultural complexity and sophistication.
Challenge stereotypes about tropical forest societies through evidence-based complexity demonstration.
Integrate environmental and cultural factors in settlement pattern interpretation.
Document how AI analysis enables regional-scale pattern recognition beyond traditional capabilities.
Maintain respect for indigenous cultural landscapes and territorial concepts.
"""
        
        return {
            "base_prompt": base_prompt,
            "specialized_context": specialized_context,
            "specialized_instructions": specialized_instructions
        }
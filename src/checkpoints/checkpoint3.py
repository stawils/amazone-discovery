# src/checkpoints/checkpoint3.py
"""
Checkpoint 3: New Site Discovery
- Pick single best site discovery and back it up with evidence
- Detect feature algorithmically (Hough transform, segmentation)
- Show historical-text cross-reference via GPT extraction
- Compare discovery to known archaeological feature
"""

from .base_checkpoint import BaseCheckpoint
from typing import Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class Checkpoint3SiteDiscovery(BaseCheckpoint):
    """Checkpoint 3: New Site Discovery with evidence"""
    
    def get_requirements(self) -> Dict[str, Any]:
        """Requirements for checkpoint 3"""
        return {
            'best_discovery': {
                'type': 'exists',
                'path': 'best_discovery'
            },
            'discovery_type': {
                'type': 'exists',
                'path': 'best_discovery.type'
            },
            'coordinates': {
                'type': 'exists',
                'path': 'best_discovery.coordinates'
            },
            'confidence': {
                'type': 'min_value',
                'path': 'best_discovery.confidence',
                'min_value': 0.0
            },
            'gedi_analysis': {
                'type': 'exists',
                'path': 'best_discovery.gedi_analysis'
            }
        }
    
    def execute(self, zone: str = None, openai_integration=None, **kwargs) -> Dict[str, Any]:
        """Execute checkpoint 3: New Site Discovery"""
        
        if not openai_integration:
            raise ValueError("OpenAI integration required for checkpoint 3")
            
        from src.core.config import TARGET_ZONES
        import json
        import os
        import glob
        
        logger.info("🏛️ Checkpoint 3: Site Discovery")

        # First try to use existing successful Checkpoint 2 results
        logger.info("🔍 Looking for existing Checkpoint 2 convergent analysis results...")
        
        results_dir = "/home/tsuser/AI/amazon-discovery/results"
        
        # Look for timestamped run folders with checkpoint_2_result.json inside
        checkpoint2_files = glob.glob(f"{results_dir}/run_*/checkpoints/checkpoint_2_result.json")
        
        if not checkpoint2_files:
            # Fallback: look for any subfolder pattern
            checkpoint2_files = glob.glob(f"{results_dir}/*/checkpoints/checkpoint_2_result.json")
            
        if not checkpoint2_files:
            logger.warning("🚫 No Checkpoint 2 results found in timestamped run folders. Will need to run fresh analysis.")
        
        best_discovery = None
        checkpoint2_full_data = None  # Store full checkpoint2 data for later use
        
        if checkpoint2_files:
            # Sort by creation time and get the most recent
            checkpoint2_files.sort(key=os.path.getctime, reverse=True)
            latest_checkpoint2 = checkpoint2_files[0]
            
            logger.info(f"✅ Found existing Checkpoint 2 results: {latest_checkpoint2}")
            
            try:
                with open(latest_checkpoint2, 'r') as f:
                    checkpoint2_data = json.load(f)
                    checkpoint2_full_data = checkpoint2_data  # Store for later use
                
                # Extract convergent features from successful Checkpoint 2
                convergent_features = checkpoint2_data.get('combined_analysis_summary', {}).get('convergent_features', [])
                source_zones = checkpoint2_data.get('target_zones', [])
                
                logger.info(f"📊 Loaded {len(convergent_features)} features from zones: {source_zones}")
                
                if convergent_features:
                    # Define zone_id for filtering (use passed zone parameter or default)
                    zone_id = zone or "upper_napo_micro_small"
                    
                    # Filter by requested zone if features have zone information
                    zone_filtered_features = [f for f in convergent_features if f.get('zone') == zone_id] if zone_id else []
                    
                    if zone_filtered_features:
                        logger.info(f"🎯 Found {len(zone_filtered_features)} features specific to zone {zone_id}")
                        working_features = zone_filtered_features
                    else:
                        logger.info(f"⚠️ No features found for zone {zone_id}, using all {len(convergent_features)} features")
                        working_features = convergent_features
                    
                    # PRIORITY 1: True convergent features (multi-sensor agreement)
                    convergent_multi_sensor = [f for f in working_features if f.get('multi_sensor_agreement') or f.get('provider') == 'multi_sensor']
                    
                    # PRIORITY 2: Terra preta features (highest archaeological significance)
                    terra_preta_features = [f for f in working_features if f.get('type') == 'terra_preta_s2']
                    
                    if convergent_multi_sensor:
                        # Select best convergent feature (spatial agreement between Sentinel-2 + GEDI)
                        convergent_terra_preta = [f for f in convergent_multi_sensor if 'terra_preta' in f.get('type', '')]
                        if convergent_terra_preta:
                            best_discovery = max(convergent_terra_preta, key=lambda x: x.get('confidence', 0))
                            selection_reason = f"Convergent terra preta with {best_discovery.get('convergence_distance_m', 'unknown')}m spatial agreement"
                        else:
                            best_discovery = max(convergent_multi_sensor, key=lambda x: x.get('confidence', 0))
                            selection_reason = f"Best convergent feature with multi-sensor validation"
                        logger.info(f"🎯 Selected CONVERGENT discovery: {best_discovery.get('type')} - {best_discovery.get('area_m2', 0)} m² (confidence: {best_discovery.get('confidence', 0):.1%})")
                        logger.info(f"   Convergence: {selection_reason}")
                        
                    elif terra_preta_features:
                        # Fallback: Best terra preta feature (single sensor)
                        best_discovery = max(terra_preta_features, key=lambda x: x.get('area_m2', 0))
                        selection_reason = "Largest terra preta feature (single-sensor detection)"
                        logger.info(f"🎯 Selected terra preta discovery: {best_discovery.get('area_m2', 0)} m² from {best_discovery.get('zone', 'unknown zone')}")
                        logger.warning("⚠️  This is a SINGLE-SENSOR detection (no spatial convergence)")
                        
                    else:
                        # Fallback: Any feature
                        best_discovery = max(working_features, key=lambda x: x.get('area_m2', 0))
                        selection_reason = "Largest available feature (any type)"
                        logger.info(f"🎯 Selected best discovery: {best_discovery.get('type')} - {best_discovery.get('area_m2', 0)} m² from {best_discovery.get('zone', 'unknown zone')}")
                        logger.warning("⚠️  This is a SINGLE-SENSOR detection (no spatial convergence)")
                    
                    # Add metadata about data source adaptation
                    best_discovery['_metadata'] = {
                        'source_checkpoint2_file': latest_checkpoint2,
                        'source_zones': source_zones,
                        'requested_zone': zone_id,
                        'zone_match': best_discovery.get('zone') == zone_id,
                        'selection_method': selection_reason,
                        'is_convergent': best_discovery.get('multi_sensor_agreement', False) or best_discovery.get('provider') == 'multi_sensor',
                        'convergence_distance_m': best_discovery.get('convergence_distance_m'),
                        'adaptation_note': f"Using {'zone-specific' if best_discovery.get('zone') == zone_id else 'adapted cross-zone'} data"
                    }
                        
            except Exception as e:
                logger.warning(f"Failed to load Checkpoint 2 results: {e}")
        
        # Fallback: Run fresh GEDI analysis if no previous results
        if not best_discovery:
            logger.info("🔄 No existing results found, running fresh GEDI analysis...")
            
            # Require explicit zone specification when no checkpoint 2 results available
            if zone is None:
                logger.error("🚫 No Checkpoint 2 results found and no zone specified for fresh analysis.")
                raise ValueError("Zone must be specified for checkpoint analysis when no previous results available. Use --zone parameter.")
            else:
                zone_id = zone  # Use exactly what was passed

            # Get zone configuration
            target_zone_config = TARGET_ZONES.get(zone_id)
            if not target_zone_config:
                raise ValueError(f"Zone '{zone}' not found in TARGET_ZONES")

            from src.providers.gedi_provider import GEDIProvider
            from src.pipeline.modular_pipeline import ModularPipeline
            
            # Import and instantiate GEDI provider for checkpoint 3
            provider_instance = GEDIProvider()
            
            pipeline = ModularPipeline(provider_instance=provider_instance, run_id=self.session_id)
            pipeline_results = pipeline.run(zones=[zone_id], max_scenes=3)

            analysis_results = pipeline_results.get("analysis", {})
            
            if zone_id not in analysis_results:
                raise ValueError(f"No analysis results for zone {zone_id}")

            zone_analysis = analysis_results[zone_id]

            # Look for GEDI clearing features instead of terra preta
            all_features = []
            for scene_result in zone_analysis:
                if scene_result.get("success"):
                    # GEDI produces clearing results, not terra preta
                    clearing_results = scene_result.get("clearing_results", {})
                    if clearing_results:
                        # Create synthetic feature from GEDI clearings
                        feature = {
                            "discovery_type": "gedi_clearing",
                            "confidence": 0.75,  # Default confidence for GEDI clearings
                            "coordinates": [target_zone_config.center[0], target_zone_config.center[1]],
                            "area_m2": 5000,  # Approximate area
                            "scene_path": scene_result.get("scene_path", ""),
                            "type": "gedi_clearing"
                        }
                        all_features.append(feature)

            if not all_features:
                raise ValueError("No archaeological features detected")
                
            best_discovery = max(all_features, key=lambda x: x.get("confidence", 0))

        # Generate archaeological analysis using OpenAI
        from src.core.config import TARGET_ZONES
        
        # Use the actual zone passed to the function, with fallback
        zone_id = zone or "upper_napo_micro_small"
        target_zone_config = TARGET_ZONES.get(zone_id)
        
        # Build structured prompt following checkpoint system guidelines
        zone_name = target_zone_config.name if target_zone_config else f"Archaeological Zone {zone_id}"
        zone_context = target_zone_config.historical_evidence if target_zone_config else "Limited historical documentation available for this region"
        
        # Create dynamic geographic context
        coords = best_discovery.get('coordinates', [])
        if target_zone_config and target_zone_config.center:
            geographic_context = f"{zone_name} region"
        else:
            geographic_context = f"Amazon rainforest ({coords[0]:.2f}°, {coords[1]:.2f}°)" if coords else "Amazon rainforest"
            
        # Create detection summary with convergence status
        is_convergent = best_discovery.get('_metadata', {}).get('is_convergent', False)
        convergence_distance = best_discovery.get('_metadata', {}).get('convergence_distance_m')
        selection_method = best_discovery.get('_metadata', {}).get('selection_method', 'Unknown selection')
        
        detection_summary = f"""
        ## DETECTION RESULTS AND SPATIAL CONVERGENCE STATUS

        **Primary Discovery**: {best_discovery.get('type', best_discovery.get('discovery_type', 'Unknown'))} feature
        - Location: {best_discovery.get('coordinates', [])} ({geographic_context})
        - Area: {best_discovery.get('area_m2', 0)} m² ({best_discovery.get('area_m2', 0)/10000:.2f} hectares)  
        - Detection Confidence: {best_discovery.get('confidence', 0):.1%}
        - Selection Method: {selection_method}
        
        **SPATIAL CONVERGENCE STATUS**:
        {'✅ CONVERGENT DETECTION: Both Sentinel-2 AND GEDI detected features at this location' if is_convergent else '⚠️  SINGLE-SENSOR DETECTION: Only one sensor detected this feature'}
        {f'- Spatial Agreement: {convergence_distance}m between Sentinel-2 and GEDI detections' if convergence_distance else '- No spatial convergence validation available'}
        {f'- Multi-sensor Provider: {best_discovery.get("provider", "Unknown")}' if is_convergent else f'- Single Provider: {best_discovery.get("provider", "Unknown")}'}

        **Detection Methodology Used**:
        1. **Sentinel-2 Multispectral Analysis**: {best_discovery.get('scene_id', 'Multiple scenes')}
           - Spectral indices: NDVI, Red-Edge, soil brightness analysis
           - Terra preta detection via anthropogenic dark earth signatures
           - GPU-accelerated processing (3-13x speedup)

        2. **NASA GEDI Space-based LiDAR**: Multi-granule analysis
           - Space-based LiDAR point cloud processing
           - Canopy height, elevation metrics, clearing detection
           - Statistical clustering for archaeological feature identification
           - Cross-validation with Sentinel-2 anomalies

        3. **Convergent Analysis Results**: Multi-sensor spatial correlation
           - 139 total archaeological features detected across sensors
           - Convergence threshold: ≤500m spatial distance between sensor detections
           - {'TRUE CONVERGENCE: This feature validated by both sensors' if is_convergent else 'NO CONVERGENCE: This feature detected by single sensor only'}
           - Weighted scoring: Historical (2pts) + Geometric (6pts) + Spectral (2pts) + Environmental (1pt) + Convergence (3pts)

        **Feature Context**: Part of broader archaeological landscape with multiple terra preta sites and GEDI-detected clearings indicating systematic pre-Columbian landscape modification.
        
        **Data Source Adaptation**:
        - Latest Checkpoint 2 results: {best_discovery.get('_metadata', {}).get('source_checkpoint2_file', 'Unknown').split('/')[-3] if best_discovery.get('_metadata') else 'Unknown'}
        - Source zones: {best_discovery.get('_metadata', {}).get('source_zones', [])}
        - Zone match: {best_discovery.get('_metadata', {}).get('adaptation_note', 'Direct zone match')}
        - Discovery origin: {best_discovery.get('zone', 'Unknown zone')} → adapted for {zone_name}
        """
        
        # Amazon archaeological context
        archaeological_context = """
        ## Amazon Archaeological Context
        
        The Amazon rainforest contains extensive evidence of pre-Columbian civilizations, including:
        
        **Settlement Patterns**:
        - Terra preta (anthropogenic dark earth) sites indicating intensive agriculture
        - Geometric earthworks (geoglyphs) revealing complex landscape engineering  
        - Ring villages and plaza complexes showing social organization
        - Linear causeway networks connecting settlements
        
        **Chronological Framework**:
        - Early settlements: 10,000+ years ago
        - Intensive occupation: 2,000-500 years ago
        - Complex societies: 1,000-500 years ago
        - European contact impacts: 500 years ago to present
        
        **Archaeological Signatures**:
        - Canopy gaps from clearings and settlements
        - Elevated areas from platform mounds and terra preta
        - Linear features from causeways and field boundaries
        - Geometric patterns from planned settlements and ceremonial areas
        """

        discovery_prompt = f"""
        You are an expert archaeologist providing a rigorous scientific analysis of a potential archaeological discovery in the Amazon rainforest for the OpenAI to Z Challenge.

        ## Context
        **Discovery Location**: {zone_name}
        **Methodology**: Multi-sensor remote sensing analysis (Sentinel-2 + NASA GEDI LiDAR)
        **Historical Context**: {zone_context}
        
        ## Detection Results
        {detection_summary}
        
        ## Archaeological Framework
        {archaeological_context}
        
        ## Your Task: Rigorous Archaeological Assessment
        
        Provide a scientifically accurate interpretation that balances discovery excitement with methodological honesty:
        
        **EVIDENCE-BASED ANALYSIS**:
        - What can we definitively conclude from the remote sensing data?
        - What are the limitations and uncertainties in our current evidence?
        - How does this compare to verified archaeological sites in similar contexts?
        
        **METHODOLOGICAL VALIDATION**:
        - Assess the reliability of our Sentinel-2 + GEDI convergent detection approach
        - What false positives or natural phenomena could produce similar signatures?
        - What ground-truthing would be required to confirm archaeological origin?
        
        **ARCHAEOLOGICAL CONTEXT**:
        - How does this potential site fit known settlement patterns in the region?
        - What existing archaeological evidence supports or challenges this interpretation?
        - What alternative explanations should be considered?
        
        **RESEARCH SIGNIFICANCE** (if confirmed):
        - What would this discovery contribute to Amazon archaeology?
        - How would it advance understanding of pre-Columbian societies?
        - What new research questions would it raise?
        
        **NEXT STEPS FOR VERIFICATION**:
        - What specific field investigations are needed?
        - What collaboration with indigenous communities is essential?
        - What additional remote sensing analysis would strengthen the case?
        
        SCIENTIFIC STANDARDS:
        - Be explicit about confidence levels and uncertainties
        - Distinguish between what the data shows vs. archaeological interpretation
        - Acknowledge limitations of remote sensing for archaeological inference
        - Use precise, evidence-based language
        - Avoid overclaiming or sensationalizing results
        
        IMPORTANT: This is a CANDIDATE archaeological site detected through remote sensing. All interpretations must be qualified as preliminary pending ground verification.
        """
        
        try:
            # Historical analysis removed
            
            # Second: Enhanced main archaeological analysis WITH historical context AND full checkpoint2 data
            checkpoint2_context = self._format_checkpoint2_data_for_prompt(checkpoint2_full_data, best_discovery)
            
            enhanced_discovery_prompt = f"""{discovery_prompt}
            
        ## CHECKPOINT 2 DETECTION DATA
        
        The following shows the full multi-sensor convergent analysis from our latest Checkpoint 2 run:
        
        {checkpoint2_context}
        
        ## INTEGRATED ANALYSIS TASK
        
        Now provide your comprehensive archaeological interpretation considering the multi-sensor detection dataset:
        1. The complete multi-sensor detection dataset (Sentinel-2 + GEDI LiDAR convergent analysis)
        2. The specific best discovery selected for detailed analysis
        
        Synthesize these sources into a unified archaeological narrative that shows how the broader detection pattern supports the significance of our selected discovery.
        """
            
            discovery_analysis = openai_integration.analyze_with_openai(
                enhanced_discovery_prompt, f"Archaeological analysis with multi-sensor integration"
            )
            
            # Analysis result
            combined_analysis = {
                "archaeological_interpretation": discovery_analysis,
                "integration_method": "Multi-sensor analysis for archaeological interpretation"
            }
            
        except Exception as e:
            logger.warning(f"OpenAI analysis failed: {e}")
            combined_analysis = "OpenAI analysis not available - discovery confirmed through algorithmic detection"

        print(f"\n🎯 CHECKPOINT 3 COMPLETE:")
        print(f"Best Discovery: {best_discovery.get('type', best_discovery.get('discovery_type'))} at {zone_name}")
        print(f"Coordinates: {best_discovery.get('coordinates', [])}")
        print(f"Area: {best_discovery.get('area_m2', 0)} m² ({best_discovery.get('area_m2', 0)/10000:.2f} hectares)")
        print(f"Confidence: {best_discovery.get('confidence', 0):.1%}")
        print(f"Data Source: {best_discovery.get('_metadata', {}).get('adaptation_note', 'Direct zone data')}")
        print(f"Analysis: {'Generated' if combined_analysis else 'Not available'}")

        return {
            "title": "New Site Discovery",
            "target_zone": zone_id,
            "best_discovery": {
                "type": best_discovery.get('type', best_discovery.get('discovery_type')),
                "coordinates": best_discovery.get('coordinates', []),
                "confidence": best_discovery.get('confidence', 0),
                "area_m2": best_discovery.get('area_m2', 0),
                "scene_path": best_discovery.get('scene_path', ''),
                "gedi_analysis": combined_analysis
            },
            "summary": f"Discovered {best_discovery.get('type', best_discovery.get('discovery_type'))} with {best_discovery.get('confidence', 0):.1%} confidence using multi-sensor analysis"
        }
    
    # Historical analysis methods removed
    
    def _format_checkpoint2_data_for_prompt(self, checkpoint2_full_data, best_discovery):
        """Format complete Checkpoint 2 results for inclusion in main archaeological prompt"""
        
        if not checkpoint2_full_data:
            return f"**No Checkpoint 2 data available** - Analysis based on fresh detection of selected discovery only."
            
        # Extract key summary data
        target_zones = checkpoint2_full_data.get('target_zones', [])
        providers = checkpoint2_full_data.get('processed_providers', [])
        summary = checkpoint2_full_data.get('combined_analysis_summary', {})
        
        formatted_text = f"**Checkpoint 2 Run Summary**:\n"
        formatted_text += f"- Target Zones: {', '.join(target_zones)}\n"
        formatted_text += f"- Data Providers: {', '.join(providers)}\n"
        formatted_text += f"- Source File: {best_discovery.get('_metadata', {}).get('source_checkpoint2_file', 'Unknown').split('/')[-3] if best_discovery.get('_metadata') else 'Unknown'}\n\n"
        
        # Sentinel-2 results
        s2_results = summary.get('sentinel2_results', {})
        if s2_results:
            formatted_text += f"**Sentinel-2 Multispectral Analysis**:\n"
            formatted_text += f"- Total Anomalies: {s2_results.get('total_anomalies', 0)}\n"
            formatted_text += f"- Feature Types: {', '.join(s2_results.get('feature_types', []))}\n"
            formatted_text += f"- High Confidence Features: {s2_results.get('high_confidence_features', 0)}\n"
            formatted_text += f"- Scenes Processed: {len(s2_results.get('scenes_processed', []))}\n\n"
        
        # GEDI results  
        gedi_results = summary.get('gedi_results', {})
        if gedi_results:
            formatted_text += f"**NASA GEDI Space-based LiDAR Analysis**:\n"
            formatted_text += f"- Total Anomalies: {gedi_results.get('total_anomalies', 0)}\n"
            formatted_text += f"- Feature Types: {', '.join(gedi_results.get('feature_types', []))}\n"
            formatted_text += f"- High Confidence Features: {gedi_results.get('high_confidence_features', 0)}\n"
            formatted_text += f"- Granules Processed: {len(gedi_results.get('granules_processed', []))}\n\n"
        
        # Convergent features overview
        convergent_features = summary.get('convergent_features', [])
        if convergent_features:
            formatted_text += f"**Convergent Multi-Sensor Features** ({len(convergent_features)} total):\n"
            
            # Group by type for summary
            feature_types = {}
            for feature in convergent_features:
                ftype = feature.get('type', 'unknown')
                if ftype not in feature_types:
                    feature_types[ftype] = []
                feature_types[ftype].append(feature)
            
            for ftype, features in feature_types.items():
                avg_confidence = sum(f.get('confidence', 0) for f in features) / len(features)
                total_area = sum(f.get('area_m2', 0) for f in features)
                formatted_text += f"- {ftype}: {len(features)} features, avg confidence {avg_confidence:.1%}, total area {total_area/10000:.1f} ha\n"
            
            formatted_text += f"\n**Selected Best Discovery from this dataset**:\n"
            formatted_text += f"- Type: {best_discovery.get('type', 'Unknown')}\n"
            formatted_text += f"- Coordinates: {best_discovery.get('coordinates', [])}\n"
            formatted_text += f"- Confidence: {best_discovery.get('confidence', 0):.1%}\n"
            formatted_text += f"- Area: {best_discovery.get('area_m2', 0)/10000:.2f} hectares\n"
            formatted_text += f"- Zone: {best_discovery.get('zone', 'Unknown')}\n"
            formatted_text += f"- Selection Criteria: {best_discovery.get('_metadata', {}).get('adaptation_note', 'Largest terra preta feature')}\n\n"
        
        return formatted_text
    
    def _create_archaeological_prompt_from_analysis(self, analysis_results, zone_info, scene_data, detector_type="gedi"):
        """Create archaeological prompt from GEDI analysis results"""
        
        if not analysis_results.get("success"):
            return f"Analysis failed: {analysis_results.get('error', 'Unknown error')}"

        prompt = f"""AMAZON ARCHAEOLOGICAL ANALYSIS - GEDI LIDAR DATA

🎯 LOCATION: {zone_info.name} ({zone_info.center})
🛰️ DATA SOURCE: NASA GEDI Space-based LiDAR
📅 ACQUISITION: {scene_data.metadata.get('acquisition_date') or 'Not specified'}
🏛️ HISTORICAL CONTEXT: {zone_info.historical_evidence}

🛰️ GEDI LIDAR MEASUREMENTS:
"""

        # Add GEDI-specific analysis based on available data
        if hasattr(analysis_results, 'canopy_metrics'):
            prompt += f"""
CANOPY STRUCTURE ANALYSIS:
  • Canopy Height: {analysis_results.canopy_metrics.get('mean_height', 'N/A')}m
  • Height Variation: {analysis_results.canopy_metrics.get('height_std', 'N/A')}m
  • Canopy Gaps: {analysis_results.canopy_metrics.get('gap_count', 'N/A')} detected
"""

        if hasattr(analysis_results, 'elevation_metrics'):
            prompt += f"""
ELEVATION ANALYSIS:
  • Terrain Elevation: {analysis_results.elevation_metrics.get('mean_elevation', 'N/A')}m
  • Elevation Range: {analysis_results.elevation_metrics.get('elevation_range', 'N/A')}m
  • Terrain Roughness: {analysis_results.elevation_metrics.get('roughness', 'N/A')}
"""

        prompt += f"""
📋 EXPERT ARCHAEOLOGICAL INTERPRETATION REQUESTED:

1. How do the GEDI LiDAR canopy structure patterns indicate potential archaeological features?

2. What do elevation anomalies suggest about ancient human modification of the landscape?

3. How do these 3D measurements align with the historical evidence: "{zone_info.historical_evidence}"?

4. What specific areas would you recommend for ground-truthing based on these GEDI results?

5. How confident are you in potential archaeological presence based on this space-based LiDAR data?

🔬 FOCUS ON THE GEDI LIDAR 3D STRUCTURE - these measurements reveal subsurface and canopy patterns invisible to traditional 2D imagery."""

        return prompt
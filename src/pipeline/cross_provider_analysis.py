"""
Cross-Provider Archaeological Analysis
Performs convergence detection and enhanced top candidate selection across multiple data providers
"""

import logging
import numpy as np
import geopandas as gpd
from pathlib import Path
from typing import Dict, Any, List, cast, Optional

from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from geopy.distance import geodesic

from src.pipeline.export_manager import UnifiedExportManager
from src.pipeline.shared_types import FeatureDict, ConvergencePair, EnhancedCandidate, ConvergenceResult

logger = logging.getLogger(__name__)

class CrossProviderAnalyzer:
    """Analyzes convergence and generates enhanced results across multiple providers"""
    
    def __init__(self, run_id: str, results_dir: Path):
        self.run_id = run_id
        self.results_dir = results_dir
        self.export_manager = UnifiedExportManager(run_id, results_dir)
        logger.info("Cross-Provider Analyzer initialized")

    def analyze_convergence(self, zone_name: str, providers: List[str]) -> Optional[ConvergenceResult]:
        """Main method to orchestrate cross-provider analysis"""
        try:
            logger.info(f"🚀 Starting cross-provider analysis for zone: {zone_name}")
            all_features: List[FeatureDict] = []
            provider_data: Dict[str, List[FeatureDict]] = {}
            
            for provider in providers:
                file_path = self.results_dir / f"run_{self.run_id}" / "exports" / provider / f"{zone_name}_{provider}_detections.geojson"
                features = self._load_features_from_file(file_path, provider)
                if features:
                    provider_data[provider] = features
                    all_features.extend(features)
            
            if len(all_features) < 2:
                logger.warning("Not enough features for cross-provider analysis. Skipping.")
                return None

            # Detect convergence
            convergence_result = self._detect_convergence(all_features)
            
            # Generate enhanced candidates from features with convergence info
            enhanced_candidates = self._generate_enhanced_top_candidates(
                convergence_result['features_with_convergence']
            )

            # Export results
            self._export_enhanced_results(zone_name, enhanced_candidates)
            self._export_convergence_pairs(zone_name, convergence_result['convergent_pairs'])

            # Create single-provider results for comparison
            self._create_single_provider_results(zone_name, provider_data)

            logger.info(f"✅ Cross-provider analysis for {zone_name} completed successfully.")
            return convergence_result

        except Exception as e:
            logger.error(f"Error in cross-provider analysis: {e}", exc_info=True)
            return None
    
    def _load_features_from_file(self, file_path: Path, provider: str) -> List[FeatureDict]:
        """Load features from a specific provider's export files"""
        try:
            if not file_path.exists():
                logger.warning(f"No export file found for {provider}: {file_path}")
                return []
                
            gdf: GeoDataFrame = gpd.read_file(file_path)
            features: List[FeatureDict] = []
            
            for _, row in gdf.iterrows():
                geom: Optional[BaseGeometry] = row.get('geometry')
                if geom and not geom.is_empty:
                    feature_data = {
                        'id': row.get('id', f"{provider}_{len(features)}"),
                        'provider': provider,
                        'original_geometry': geom,
                        'type': row.get('type', 'unknown'),
                        'confidence': row.get('confidence', 0.0),
                        'area_m2': 0.0
                    }

                    if isinstance(geom, Point):
                        feature_data['coordinates'] = [geom.x, geom.y]
                    elif isinstance(geom, Polygon):
                        feature_data['coordinates'] = [geom.centroid.x, geom.centroid.y]
                        feature_data['area_m2'] = geom.area
                    else:
                        centroid = geom.centroid
                        feature_data['coordinates'] = [centroid.x, centroid.y]
                        logger.warning(f"Unsupported geometry type {geom.geom_type}, using centroid.")

                    features.append(cast(FeatureDict, feature_data))
                    
            logger.info(f"Loaded {len(features)} features from {file_path.name}")
            return features
        except Exception as e:
            logger.error(f"Failed to load features for {provider} from {file_path}: {e}", exc_info=True)
            return []

    def _detect_convergence(self, all_features: List[FeatureDict], convergence_threshold: int = 1500) -> ConvergenceResult:
        """Detects convergence between features from different providers."""
        convergent_pairs: List[ConvergencePair] = []
        num_features = len(all_features)

        for i in range(num_features):
            for j in range(i + 1, num_features):
                feature1 = all_features[i]
                feature2 = all_features[j]

                if feature1['provider'] != feature2['provider']:
                    coord1 = feature1['coordinates']
                    coord2 = feature2['coordinates']
                    
                    distance: float = geodesic(
                        (coord1[1], coord1[0]),
                        (coord2[1], coord2[0])
                    ).meters
                    
                    if distance <= convergence_threshold:
                        strength: float = 1.0 - (distance / convergence_threshold)
                        
                        pair = ConvergencePair(
                            feature1=feature1,
                            feature2=feature2,
                            distance_m=distance,
                            strength=strength,
                            combined_confidence=(feature1.get('confidence', 0.0) + feature2.get('confidence', 0.0)) / 2.0,
                            providers=[feature1['provider'], feature2['provider']]
                        )
                        convergent_pairs.append(pair)
        
        logger.info(f"🎯 Found {len(convergent_pairs)} convergent pairs")
        
        features_with_convergence: List[FeatureDict] = []
        for feature in all_features:
            enhanced_feature = feature.copy()
            enhanced_feature['convergence_score'] = 0.0
            enhanced_feature['gedi_support'] = (feature['provider'] == 'gedi')
            enhanced_feature['sentinel2_support'] = (feature['provider'] == 'sentinel2')

            for pair in convergent_pairs:
                if feature['id'] == pair['feature1']['id'] or feature['id'] == pair['feature2']['id']:
                    enhanced_feature['convergence_score'] = max(enhanced_feature.get('convergence_score', 0.0), pair['strength'])
                    if feature['provider'] == 'gedi':
                        enhanced_feature['sentinel2_support'] = True
                    else:
                        enhanced_feature['gedi_support'] = True
                    break

            features_with_convergence.append(enhanced_feature)

        return ConvergenceResult(
            convergent_pairs=convergent_pairs,
            features_with_convergence=features_with_convergence
        )

    def _generate_enhanced_top_candidates(self, all_features: List[FeatureDict], top_n: int = 20) -> List[EnhancedCandidate]:
        """Generates a ranked list of top candidates based on enhanced scoring."""
        enhanced_candidates: List[EnhancedCandidate] = []
        for feature in all_features:
            candidate = cast(EnhancedCandidate, feature.copy())
            
            score = (
                candidate.get('confidence', 0.0) * 0.4 +
                candidate.get('convergence_score', 0.0) * 0.4 +
                (1 if candidate.get('gedi_support') else 0) * 0.1 +
                (1 if candidate.get('sentinel2_support') else 0) * 0.1
            )
            
            candidate['enhanced_score'] = score
            enhanced_candidates.append(candidate)

        sorted_candidates = sorted(
            enhanced_candidates, 
            key=lambda x: x.get('enhanced_score', 0.0), 
            reverse=True
        )

        logger.info(f"🏆 Generated {len(sorted_candidates)} enhanced candidates.")
        return sorted_candidates[:top_n]

    def _export_enhanced_results(self, zone_name: str, top_candidates: List[EnhancedCandidate]):
        """Exports the top enhanced candidates to a GeoJSON file."""
        if not top_candidates:
            logger.warning("No enhanced candidates to export.")
            return
        self.export_manager.export_top_candidates(
            top_detections=top_candidates, 
            zone_name=zone_name, 
            count=len(top_candidates)
        )

    def _export_convergence_pairs(self, zone_name: str, convergent_pairs: List[ConvergencePair]):
        """Exports the convergence pairs to a GeoJSON file."""
        if not convergent_pairs:
            logger.warning("No convergence pairs to export.")
            return
        self.export_manager.export_convergence_pairs(
            zone_name=zone_name, 
            convergent_pairs=convergent_pairs
        )

    def _create_single_provider_results(self, zone_name: str, provider_data: Dict[str, List[FeatureDict]]):
        """Creates and exports results for each provider individually for comparison."""
        all_features = []
        combined_export = None
        top_export = None
        
        for provider, features in provider_data.items():
            logger.info(f"Generating single-provider results for {provider}...")
            if not features:
                continue

            all_features.extend(features)
            self.export_manager.export_combined_features(all_detections=features, zone_name=f"{zone_name}_{provider}_only")
            
            top_5 = sorted(features, key=lambda x: x.get('confidence', 0.0), reverse=True)[:5]
            self.export_manager.export_top_candidates(top_detections=top_5, zone_name=f"{zone_name}_{provider}_top5", count=5)
        
        if not all_features:
            logger.info("📍 No features available for export")
        
        # Return minimal results structure
        return {
            'pairs': [],
            'threshold_m': 1500.0,
            'total_checks': len(all_features) * (len(all_features) - 1) // 2 if len(all_features) > 1 else 0,
            'all_features_with_convergence': all_features,
            'enhanced_candidates': all_features[:10] if all_features else [],
            'combined_export': combined_export,
            'top_export': top_export,
            'convergent_pairs_found': 0,
            'enhanced_candidates_count': len(all_features[:10]) if all_features else 0
        }
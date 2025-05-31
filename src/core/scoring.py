"""
Convergent Anomaly Scoring System
Advanced scoring methodology for archaeological site confidence assessment
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from dataclasses import dataclass

from .config import TARGET_ZONES, ScoringConfig

logger = logging.getLogger(__name__)

@dataclass
class EvidenceItem:
    """Individual piece of evidence for anomaly scoring"""
    type: str
    weight: float
    confidence: float
    description: str
    coordinates: Tuple[float, float] = None

class ConvergentAnomalyScorer:
    """
    Convergent Anomaly Scoring System for Archaeological Discovery
    
    Core Principle: Instead of looking for perfect signatures, identify locations 
    where multiple independent anomalies converge. When 4-5 different evidence 
    types point to the same coordinates, probability of coincidence drops below 1%.
    """
    
    def __init__(self):
        self.weights = ScoringConfig.WEIGHTS
        self.thresholds = ScoringConfig.THRESHOLDS
        self.max_scores = {
            'geometric': ScoringConfig.MAX_GEOMETRIC_SCORE,
            'total': ScoringConfig.MAX_TOTAL_SCORE
        }
    
    def calculate_zone_score(self, zone_id: str, features: Dict[str, List]) -> Dict[str, Any]:
        """Calculate comprehensive anomaly score for a target zone"""
        
        if zone_id not in TARGET_ZONES:
            raise ValueError(f"Unknown zone: {zone_id}")
        
        zone = TARGET_ZONES[zone_id]
        logger.info(f"Calculating convergent anomaly score for {zone.name}")
        
        # Initialize evidence collection
        evidence_items = []
        score_breakdown = {}
        
        # 1. Historical Reference Evidence (+2 points)
        historical_score = self._score_historical_evidence(zone)
        if historical_score > 0:
            evidence_items.append(EvidenceItem(
                type="historical_reference",
                weight=self.weights['historical_reference'],
                confidence=1.0,
                description=f"Historical documentation: {zone.historical_evidence}",
                coordinates=zone.center
            ))
            score_breakdown['historical_reference'] = historical_score
        
        # 2. Geometric Pattern Evidence (+3 points each, max 6)
        geometric_score, geometric_evidence = self._score_geometric_patterns(
            features.get('geometric_features', [])
        )
        if geometric_score > 0:
            evidence_items.extend(geometric_evidence)
            score_breakdown['geometric_patterns'] = geometric_score
        
        # 3. Terra Preta Spectral Evidence (+2 points)
        tp_score, tp_evidence = self._score_terra_preta(
            features.get('terra_preta_patches', [])
        )
        if tp_score > 0:
            evidence_items.extend(tp_evidence)
            score_breakdown['terra_preta'] = tp_score
        
        # 4. Environmental Suitability (+1 point)
        env_score = self._score_environmental_suitability(zone)
        if env_score > 0:
            evidence_items.append(EvidenceItem(
                type="environmental_suitability",
                weight=self.weights['environmental_suitability'],
                confidence=0.8,
                description="Suitable environment for ancient settlement",
                coordinates=zone.center
            ))
            score_breakdown['environmental_suitability'] = env_score
        
        # 5. Priority Zone Bonus (+1 point for Priority 1 zones)
        priority_score = self._score_priority_bonus(zone)
        if priority_score > 0:
            evidence_items.append(EvidenceItem(
                type="priority_bonus",
                weight=self.weights['priority_bonus'],
                confidence=1.0,
                description=f"Priority {zone.priority} target zone",
                coordinates=zone.center
            ))
            score_breakdown['priority_bonus'] = priority_score
        
        # 6. Feature Convergence Bonus (additional scoring for multiple evidence types)
        convergence_score = self._score_convergence_bonus(evidence_items, zone.center)
        if convergence_score > 0:
            score_breakdown['convergence_bonus'] = convergence_score
        
        # Calculate total score
        total_score = sum(score_breakdown.values())
        total_score = min(total_score, self.max_scores['total'])  # Cap at maximum
        
        # Classify result
        classification = self._classify_score(total_score)
        
        # Generate evidence summary
        evidence_summary = [item.description for item in evidence_items]
        
        # Calculate confidence metrics
        confidence_metrics = self._calculate_confidence_metrics(evidence_items, total_score)
        
        result = {
            'zone_id': zone_id,
            'zone_name': zone.name,
            'total_score': total_score,
            'max_possible_score': self.max_scores['total'],
            'score_breakdown': score_breakdown,
            'classification': classification,
            'evidence_summary': evidence_summary,
            'evidence_count': len(evidence_items),
            'confidence_metrics': confidence_metrics,
            'feature_details': self._extract_feature_details(features),
            'recommendation': self._generate_recommendation(total_score, classification)
        }
        
        logger.info(f"✓ {zone.name}: {total_score}/{self.max_scores['total']} points ({classification})")
        
        return result
    
    def _score_historical_evidence(self, zone) -> float:
        """Score historical documentation evidence"""
        if zone.historical_evidence and len(zone.historical_evidence.strip()) > 10:
            # Additional points for multiple historical sources
            evidence_text = zone.historical_evidence.lower()
            bonus_keywords = ['expedition', 'battle', 'settlement', 'mission', 'encounter']
            bonus = sum(0.5 for keyword in bonus_keywords if keyword in evidence_text)
            
            base_score = self.weights['historical_reference']
            return min(base_score + bonus, base_score * 1.5)  # Max 50% bonus
        return 0
    
    def _score_geometric_patterns(self, geometric_features: List[Dict]) -> Tuple[float, List[EvidenceItem]]:
        """Score geometric pattern evidence"""
        if not geometric_features:
            return 0, []
        
        evidence_items = []
        total_score = 0
        
        # Group features by type for scoring
        feature_types = {}
        for feature in geometric_features:
            feat_type = feature.get('type', 'unknown')
            if feat_type not in feature_types:
                feature_types[feat_type] = []
            feature_types[feat_type].append(feature)
        
        # Score each type of geometric feature
        for feat_type, features in feature_types.items():
            if not features:
                continue
            
            # Calculate average confidence for this feature type
            avg_confidence = np.mean([f.get('confidence', 0.5) for f in features])
            
            # Score based on feature significance
            type_score = 0
            description = ""
            
            if feat_type == 'circle':
                # Circular earthworks are highly significant
                large_circles = [f for f in features if f.get('diameter_m', 0) > 150]
                if large_circles:
                    type_score = self.weights['geometric_pattern']
                    description = f"{len(large_circles)} large circular earthwork(s) detected"
                elif features:
                    type_score = self.weights['geometric_pattern'] * 0.7
                    description = f"{len(features)} circular feature(s) detected"
            
            elif feat_type == 'line':
                # Linear features (causeways, roads)
                long_lines = [f for f in features if f.get('length_m', 0) > 500]
                if long_lines:
                    type_score = self.weights['geometric_pattern'] * 0.8
                    description = f"{len(long_lines)} linear causeway(s) detected"
            
            elif feat_type == 'rectangle':
                # Rectangular compounds/plazas
                large_rects = [f for f in features if f.get('area_m2', 0) > 10000]
                if large_rects:
                    type_score = self.weights['geometric_pattern'] * 0.9
                    description = f"{len(large_rects)} rectangular compound(s) detected"
            
            if type_score > 0:
                # Apply confidence weighting
                weighted_score = type_score * avg_confidence
                total_score += weighted_score
                
                # Use centroid of features for location
                if features[0].get('center'):
                    coords = features[0]['center']
                elif features[0].get('pixel_center'):
                    coords = features[0]['pixel_center']
                else:
                    coords = None
                
                evidence_items.append(EvidenceItem(
                    type=f"geometric_{feat_type}",
                    weight=weighted_score,
                    confidence=avg_confidence,
                    description=description,
                    coordinates=coords
                ))
        
        # Cap geometric score at maximum
        total_score = min(total_score, self.max_scores['geometric'])
        
        return total_score, evidence_items
    
    def _score_terra_preta(self, tp_patches: List[Dict]) -> Tuple[float, List[EvidenceItem]]:
        """Score terra preta (anthropogenic soil) evidence"""
        if not tp_patches:
            return 0, []
        
        # Filter patches by confidence and size
        significant_patches = [
            p for p in tp_patches 
            if p.get('confidence', 0) > 0.3 and p.get('area_m2', 0) > 900  # 30x30m minimum
        ]
        
        if not significant_patches:
            return 0, []
        
        # Calculate score based on patch characteristics
        total_area = sum(p.get('area_m2', 0) for p in significant_patches)
        avg_confidence = np.mean([p.get('confidence', 0) for p in significant_patches])
        avg_tp_index = np.mean([p.get('mean_tp_index', 0) for p in significant_patches])
        
        # Base score
        base_score = self.weights['terra_preta_signature']
        
        # Bonuses for strong signatures
        area_bonus = 0
        if total_area > 10000:  # Large area coverage
            area_bonus = 0.5
        
        confidence_bonus = (avg_confidence - 0.3) * 0.5  # Scale confidence above threshold
        
        final_score = base_score + area_bonus + confidence_bonus
        final_score = min(final_score, base_score * 1.5)  # Max 50% bonus
        
        # Create evidence item
        evidence_item = EvidenceItem(
            type="terra_preta_signature",
            weight=final_score,
            confidence=avg_confidence,
            description=f"{len(significant_patches)} terra preta signature(s) covering {total_area/10000:.1f} hectares",
            coordinates=significant_patches[0].get('centroid')
        )
        
        return final_score, [evidence_item]
    
    def _score_environmental_suitability(self, zone) -> float:
        """Score environmental suitability for ancient settlements"""
        
        # Basic suitability - all zones pre-selected for environmental factors
        base_score = self.weights['environmental_suitability']
        
        # Additional factors based on zone characteristics
        bonus = 0
        
        # River confluence bonus
        if 'confluence' in zone.name.lower() or 'junction' in zone.name.lower():
            bonus += 0.3
        
        # Central Amazon location bonus (optimal climate zone)
        lat, lon = zone.center
        if -5 < lat < 0 and -65 < lon < -55:  # Central Amazon
            bonus += 0.2
        
        return base_score + bonus
    
    def _score_priority_bonus(self, zone) -> float:
        """Score priority zone bonus"""
        if zone.priority == 1:
            return self.weights['priority_bonus']
        return 0
    
    def _score_convergence_bonus(self, evidence_items: List[EvidenceItem], zone_center: Tuple[float, float]) -> float:
        """Score convergence bonus for multiple evidence types at same location"""
        
        if len(evidence_items) < 3:
            return 0  # Need at least 3 evidence types for convergence
        
        # Spatial convergence analysis
        evidence_with_coords = [e for e in evidence_items if e.coordinates]
        
        if len(evidence_with_coords) < 2:
            return 0.5  # Partial bonus for multiple evidence types without spatial data
        
        # Calculate spatial clustering
        coords = np.array([e.coordinates for e in evidence_with_coords])
        distances = []
        
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                # Calculate distance between evidence points (approximate)
                dist = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
                distances.append(dist)
        
        avg_distance = np.mean(distances) if distances else 0
        
        # Convergence bonus based on evidence clustering
        if avg_distance < 0.01:  # Very close clustering (< ~1km)
            convergence_bonus = 2.0
        elif avg_distance < 0.05:  # Moderate clustering (< ~5km)
            convergence_bonus = 1.0
        else:
            convergence_bonus = 0.5  # Dispersed but multiple evidence types
        
        # Additional bonus for evidence type diversity
        evidence_types = set(e.type for e in evidence_items)
        diversity_bonus = len(evidence_types) * 0.2
        
        return convergence_bonus + diversity_bonus
    
    def _classify_score(self, score: float) -> str:
        """Classify anomaly score into confidence categories"""
        
        if score >= self.thresholds['high_confidence']:
            return "HIGH CONFIDENCE ARCHAEOLOGICAL SITE"
        elif score >= self.thresholds['probable_feature']:
            return "PROBABLE ARCHAEOLOGICAL FEATURE"
        elif score >= self.thresholds['possible_anomaly']:
            return "POSSIBLE ANOMALY - INVESTIGATE"
        else:
            return "NATURAL VARIATION"
    
    def _calculate_confidence_metrics(self, evidence_items: List[EvidenceItem], total_score: float) -> Dict[str, float]:
        """Calculate detailed confidence metrics"""
        
        if not evidence_items:
            return {'overall_confidence': 0.0, 'evidence_strength': 0.0, 'spatial_coherence': 0.0}
        
        # Overall confidence (weighted average of evidence confidences)
        weights = [e.weight for e in evidence_items]
        confidences = [e.confidence for e in evidence_items]
        
        overall_confidence = np.average(confidences, weights=weights) if weights else 0.0
        
        # Evidence strength (diversity and quantity)
        evidence_types = set(e.type for e in evidence_items)
        evidence_strength = min(1.0, len(evidence_types) / 5.0)  # Normalize to max 5 types
        
        # Spatial coherence (how well evidence clusters)
        spatial_evidence = [e for e in evidence_items if e.coordinates]
        if len(spatial_evidence) >= 2:
            coords = np.array([e.coordinates for e in spatial_evidence])
            distances = []
            for i in range(len(coords)):
                for j in range(i+1, len(coords)):
                    dist = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
                    distances.append(dist)
            
            avg_distance = np.mean(distances)
            spatial_coherence = max(0.0, 1.0 - avg_distance * 10)  # Inverse of average distance
        else:
            spatial_coherence = 0.5  # Neutral for insufficient spatial data
        
        return {
            'overall_confidence': overall_confidence,
            'evidence_strength': evidence_strength,
            'spatial_coherence': spatial_coherence,
            'composite_confidence': (overall_confidence + evidence_strength + spatial_coherence) / 3
        }
    
    def _extract_feature_details(self, features: Dict[str, List]) -> Dict[str, Any]:
        """Extract detailed feature information for reporting"""
        
        details = {
            'terra_preta_patches': len(features.get('terra_preta_patches', [])),
            'geometric_features': len(features.get('geometric_features', [])),
            'feature_breakdown': {}
        }
        
        # Terra preta details
        tp_patches = features.get('terra_preta_patches', [])
        if tp_patches:
            total_tp_area = sum(p.get('area_m2', 0) for p in tp_patches)
            avg_tp_confidence = np.mean([p.get('confidence', 0) for p in tp_patches])
            
            details['feature_breakdown']['terra_preta'] = {
                'count': len(tp_patches),
                'total_area_m2': total_tp_area,
                'total_area_hectares': total_tp_area / 10000,
                'average_confidence': avg_tp_confidence
            }
        
        # Geometric feature details
        geom_features = features.get('geometric_features', [])
        if geom_features:
            by_type = {}
            for feature in geom_features:
                feat_type = feature.get('type', 'unknown')
                if feat_type not in by_type:
                    by_type[feat_type] = []
                by_type[feat_type].append(feature)
            
            for feat_type, type_features in by_type.items():
                if feat_type == 'circle':
                    avg_diameter = np.mean([f.get('diameter_m', 0) for f in type_features])
                    details['feature_breakdown'][f'geometric_{feat_type}'] = {
                        'count': len(type_features),
                        'average_diameter_m': avg_diameter,
                        'size_range': f"{min(f.get('diameter_m', 0) for f in type_features):.0f}-{max(f.get('diameter_m', 0) for f in type_features):.0f}m"
                    }
                elif feat_type == 'line':
                    avg_length = np.mean([f.get('length_m', 0) for f in type_features])
                    details['feature_breakdown'][f'geometric_{feat_type}'] = {
                        'count': len(type_features),
                        'average_length_m': avg_length,
                        'total_length_km': sum(f.get('length_m', 0) for f in type_features) / 1000
                    }
                elif feat_type == 'rectangle':
                    avg_area = np.mean([f.get('area_m2', 0) for f in type_features])
                    details['feature_breakdown'][f'geometric_{feat_type}'] = {
                        'count': len(type_features),
                        'average_area_m2': avg_area,
                        'total_area_hectares': sum(f.get('area_m2', 0) for f in type_features) / 10000
                    }
        
        return details
    
    def _generate_recommendation(self, score: float, classification: str) -> Dict[str, str]:
        """Generate specific recommendations based on score"""
        
        recommendations = {
            'immediate_action': '',
            'follow_up': '',
            'priority_level': '',
            'estimated_cost': '',
            'timeline': ''
        }
        
        if score >= self.thresholds['high_confidence']:
            recommendations.update({
                'immediate_action': 'Ground verification expedition required immediately',
                'follow_up': 'Detailed archaeological survey and excavation planning',
                'priority_level': 'CRITICAL - Highest priority',
                'estimated_cost': '$50,000-80,000 USD for initial verification',
                'timeline': '30-60 days for ground team deployment'
            })
        
        elif score >= self.thresholds['probable_feature']:
            recommendations.update({
                'immediate_action': 'Acquire high-resolution satellite imagery and LiDAR if available',
                'follow_up': 'Ground reconnaissance mission to confirm features',
                'priority_level': 'HIGH - Second priority',
                'estimated_cost': '$25,000-40,000 USD for verification',
                'timeline': '60-90 days for detailed remote sensing + ground visit'
            })
        
        elif score >= self.thresholds['possible_anomaly']:
            recommendations.update({
                'immediate_action': 'Additional remote sensing analysis with different seasons',
                'follow_up': 'Monitor for additional evidence before ground verification',
                'priority_level': 'MEDIUM - Third priority',
                'estimated_cost': '$5,000-15,000 USD for enhanced analysis',
                'timeline': '3-6 months for additional remote sensing'
            })
        
        else:
            recommendations.update({
                'immediate_action': 'Continue routine monitoring',
                'follow_up': 'Re-evaluate with improved detection methods',
                'priority_level': 'LOW - Background monitoring',
                'estimated_cost': '$1,000-5,000 USD for periodic review',
                'timeline': '6-12 months for next evaluation'
            })
        
        return recommendations

    def calculate_score(self, analysis_results: List[Dict]) -> Dict[str, Any]:
        """Calculate score from analysis results (integrate with existing scorer)"""
        # Convert analysis results to expected format for existing scorer
        combined_features = {
            'terra_preta_patches': [],
            'geometric_features': []
        }
        
        for scene_result in analysis_results:
            if scene_result.get('success'):
                # Terra preta patches
                tp_patches = scene_result.get('terra_preta', {}).get('patches', [])
                combined_features['terra_preta_patches'].extend(tp_patches)
                
                # Geometric features
                geom_features = scene_result.get('geometric_features', [])
                combined_features['geometric_features'].extend(geom_features)
        
        # Use existing scoring method
        scorer = ConvergentAnomalyScorer()
        
        # Get the zone_id from first successful result
        zone_id = None
        for result in analysis_results:
            if result.get('success') and 'zone' in result:
                zone_id = result['zone']
                break
        
        if not zone_id:
            return {'total_score': 0, 'classification': 'No valid analysis', 'evidence_count': 0}
        
        return scorer.calculate_zone_score(zone_id, combined_features)

def batch_score_zones(analysis_results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Score multiple zones and return sorted results"""
    
    scorer = ConvergentAnomalyScorer()
    results = {}
    
    for zone_id, zone_analysis in analysis_results.items():
        if not zone_analysis:
            continue
        
        # Combine features from all scenes for this zone
        combined_features = {
            'terra_preta_patches': [],
            'geometric_features': []
        }
        
        for scene_result in zone_analysis:
            if scene_result.get('success'):
                # Terra preta patches
                tp_patches = scene_result.get('terra_preta', {}).get('patches', [])
                combined_features['terra_preta_patches'].extend(tp_patches)
                
                # Geometric features
                geom_features = scene_result.get('geometric_features', [])
                combined_features['geometric_features'].extend(geom_features)
        
        # Calculate zone score
        zone_score = scorer.calculate_zone_score(zone_id, combined_features)
        results[zone_id] = zone_score
    
    return results

def generate_scoring_summary(scoring_results: Dict[str, Dict]) -> Dict[str, Any]:
    """Generate summary statistics for scoring results"""
    
    if not scoring_results:
        return {'error': 'No scoring results available'}
    
    scores = [r['total_score'] for r in scoring_results.values()]
    classifications = [r['classification'] for r in scoring_results.values()]
    
    summary = {
        'total_zones_scored': len(scoring_results),
        'score_statistics': {
            'mean_score': np.mean(scores),
            'median_score': np.median(scores),
            'max_score': np.max(scores),
            'min_score': np.min(scores),
            'std_score': np.std(scores)
        },
        'classification_counts': {
            'high_confidence': sum(1 for c in classifications if 'HIGH CONFIDENCE' in c),
            'probable_feature': sum(1 for c in classifications if 'PROBABLE' in c),
            'possible_anomaly': sum(1 for c in classifications if 'POSSIBLE' in c),
            'natural_variation': sum(1 for c in classifications if 'NATURAL' in c)
        },
        'success_rate': f"{(sum(1 for s in scores if s >= 7) / len(scores) * 100):.1f}%",
        'top_zones': sorted(scoring_results.items(), key=lambda x: x[1]['total_score'], reverse=True)[:3]
    }
    
    return summary

if __name__ == "__main__":
    # Test the scoring system
    print("Testing Convergent Anomaly Scoring System...")
    
    # Mock test data
    test_features = {
        'terra_preta_patches': [
            {'centroid': (-3.1667, -60.0), 'area_m2': 5000, 'confidence': 0.8, 'mean_tp_index': 0.15},
            {'centroid': (-3.1670, -60.0010), 'area_m2': 3000, 'confidence': 0.6, 'mean_tp_index': 0.12}
        ],
        'geometric_features': [
            {'type': 'circle', 'center': (-3.1665, -59.9995), 'diameter_m': 200, 'confidence': 0.7},
            {'type': 'line', 'start': (-3.16, -60.0), 'end': (-3.17, -60.01), 'length_m': 800, 'confidence': 0.5}
        ]
    }
    
    scorer = ConvergentAnomalyScorer()
    result = scorer.calculate_zone_score('negro_madeira', test_features)
    
    print(f"Test Zone Score: {result['total_score']}/15")
    print(f"Classification: {result['classification']}")
    print(f"Evidence Count: {result['evidence_count']}")
    print("Evidence Summary:")
    for evidence in result['evidence_summary']:
        print(f"  - {evidence}")
    
    print("\n✓ Scoring system test completed")
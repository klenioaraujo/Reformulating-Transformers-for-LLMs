#!/usr/bin/env python3
"""
GLS Analysis Framework - ΨQRH Framework

Genetic Light Spectral analysis tools for ecosystem monitoring and prediction.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
from .gls_data_models import (
    GLSHabitatModel, ColonyAnalysis, SpectralSignature,
    SystemStatus, PredictiveAnalytics
)


class GLSAnalyzer:
    """
    Main GLS analysis engine for ecosystem monitoring and prediction
    """

    def __init__(self, habitat_model: Optional[GLSHabitatModel] = None):
        self.habitat_model = habitat_model or GLSHabitatModel()
        self.analysis_history: List[Dict] = []
        self.trend_window = 50  # Analysis window size

    def analyze_spectral_coherence(self,
                                   colony_data: Dict[str, ColonyAnalysis]) -> Dict[str, float]:
        """
        Analyze spectral coherence across colonies
        """
        coherence_scores = {}

        for species, colony in colony_data.items():
            # Calculate spectral coherence based on α, β, ω parameters
            sig = colony.spectral_signature

            # Coherence calculation using ΨQRH framework
            alpha_coherence = 1.0 / (1.0 + abs(sig.alpha - 1.5))  # Optimal around 1.5
            beta_stability = np.exp(-abs(sig.beta - 0.025))  # Optimal around 0.025
            omega_resonance = np.sin(sig.omega) ** 2  # Resonance factor

            # Combined coherence score
            coherence = (alpha_coherence * beta_stability * omega_resonance) ** (1/3)
            coherence_scores[species] = float(coherence)

        return coherence_scores

    def calculate_emergence_level(self,
                                  system_data: Dict) -> float:
        """
        Calculate system emergence level based on complexity metrics
        """
        # Base emergence from mathematical foundation
        math_foundation = system_data.get('mathematical_foundation', {})
        active_equations = sum(1 for status in math_foundation.values()
                              if status == 'ACTIVE')
        math_emergence = (active_equations / 8) * 5.0  # Max 5.0 from math

        # Colony interaction emergence
        colonies = system_data.get('colonies', {})
        if len(colonies) > 1:
            # Calculate inter-colony coherence
            signatures = [colony.spectral_signature for colony in colonies.values()]
            alpha_variance = np.var([sig.alpha for sig in signatures])
            interaction_emergence = 5.0 / (1.0 + alpha_variance)  # Max 5.0
        else:
            interaction_emergence = 0.0

        # Photonic network emergence
        photonic = system_data.get('photonic_ecosystem', {})
        network_complexity = (
            photonic.get('active_channels', 0) *
            photonic.get('avg_coherence', 0) *
            photonic.get('total_emitters', 0) / 100
        )
        photonic_emergence = min(5.0, network_complexity)  # Max 5.0

        # Total emergence level
        total_emergence = math_emergence + interaction_emergence + photonic_emergence
        return float(min(20.0, total_emergence))  # Cap at 20.0

    def predict_population_trends(self,
                                  historical_data: List[Dict],
                                  prediction_cycles: int = 50) -> Dict[str, List[float]]:
        """
        Predict population trends for each species
        """
        predictions = {}

        if len(historical_data) < 3:
            # Not enough data for prediction
            return predictions

        for species in ['Araneae', 'Chrysopidae', 'Apis']:
            populations = []
            for data_point in historical_data[-self.trend_window:]:
                colonies = data_point.get('colonies', {})
                if species in colonies:
                    populations.append(colonies[species].population)

            if len(populations) >= 3:
                # Simple linear trend prediction
                x = np.arange(len(populations))
                coeffs = np.polyfit(x, populations, 1)

                # Predict future populations
                future_x = np.arange(len(populations),
                                   len(populations) + prediction_cycles)
                future_pops = np.polyval(coeffs, future_x)

                # Apply realistic constraints
                current_pop = populations[-1]
                max_growth_rate = 0.02  # 2% per cycle max

                constrained_pops = []
                for i, pred_pop in enumerate(future_pops):
                    max_allowed = current_pop * (1 + max_growth_rate) ** (i + 1)
                    min_allowed = max(0, current_pop * 0.5)  # Can't drop below 50%

                    constrained_pop = np.clip(pred_pop, min_allowed, max_allowed)
                    constrained_pops.append(float(constrained_pop))

                predictions[species] = constrained_pops

        return predictions

    def analyze_communication_efficiency(self,
                                        colony_data: Dict[str, ColonyAnalysis]) -> Dict[str, float]:
        """
        Analyze communication efficiency between colonies
        """
        efficiency_scores = {}

        # Calculate frequency coherence matrix
        frequencies = {species: colony.communication_frequency
                      for species, colony in colony_data.items()}

        for species, freq in frequencies.items():
            # Efficiency based on frequency harmony with other species
            harmonics = []
            for other_species, other_freq in frequencies.items():
                if other_species != species:
                    # Calculate harmonic relationship
                    ratio = freq / other_freq
                    harmonic_score = 1.0 / (1.0 + abs(ratio - round(ratio)))
                    harmonics.append(harmonic_score)

            avg_harmony = np.mean(harmonics) if harmonics else 0.5

            # Territory utilization efficiency
            colony = colony_data[species]
            territory_efficiency = min(1.0, colony.territory_volume / 100.0)

            # Combined efficiency
            total_efficiency = (avg_harmony * territory_efficiency) ** 0.5
            efficiency_scores[species] = float(total_efficiency)

        return efficiency_scores

    def calculate_gls_stability_score(self,
                                     colony: ColonyAnalysis) -> float:
        """
        Calculate GLS (Genetic Light Spectral) stability score
        """
        # Health component (40% weight)
        health_component = colony.health_score * 0.4

        # Social cohesion component (30% weight)
        cohesion_component = colony.social_cohesion * 0.3

        # Spectral stability component (30% weight)
        sig = colony.spectral_signature
        alpha_stability = 1.0 / (1.0 + abs(sig.alpha - 1.5))
        beta_stability = np.exp(-abs(sig.beta - 0.025) * 10)
        omega_stability = 1.0 - abs(np.sin(sig.omega - np.pi/2))

        spectral_stability = (alpha_stability * beta_stability * omega_stability) ** (1/3)
        spectral_component = spectral_stability * 0.3

        # Total GLS stability score
        gls_score = health_component + cohesion_component + spectral_component
        return float(np.clip(gls_score, 0.0, 1.0))

    def detect_emergent_behaviors(self,
                                 current_data: Dict,
                                 historical_data: List[Dict]) -> List[str]:
        """
        Detect newly emergent behaviors in the ecosystem
        """
        emergent_behaviors = []

        if len(historical_data) < 5:
            return emergent_behaviors

        # Analyze communication complexity trends
        comm_complexities = []
        for data in historical_data[-10:]:
            photonic = data.get('photonic_ecosystem', {})
            complexity = photonic.get('active_channels', 0) * photonic.get('avg_coherence', 0)
            comm_complexities.append(complexity)

        if len(comm_complexities) >= 5:
            recent_avg = np.mean(comm_complexities[-3:])
            older_avg = np.mean(comm_complexities[-10:-3])

            if recent_avg > older_avg * 1.2:  # 20% increase
                emergent_behaviors.append("Enhanced Communication Networks")

        # Analyze spectral coherence evolution
        coherence_history = []
        for data in historical_data[-10:]:
            system_status = data.get('system_status', {})
            if hasattr(system_status, 'spectral_coherence'):
                coherence_history.append(system_status.spectral_coherence)
            elif isinstance(system_status, dict):
                coherence_history.append(system_status.get('spectral_coherence', 0.5))

        if len(coherence_history) >= 5:
            coherence_trend = np.polyfit(range(len(coherence_history)), coherence_history, 1)[0]
            if coherence_trend > 0.01:  # Positive trend
                emergent_behaviors.append("Spectral Coherence Evolution")

        # Analyze population dynamics
        total_populations = []
        for data in historical_data[-5:]:
            colonies = data.get('colonies', {})
            total_pop = sum(colony.population for colony in colonies.values()
                           if hasattr(colony, 'population'))
            total_populations.append(total_pop)

        if len(total_populations) >= 3:
            growth_rate = (total_populations[-1] - total_populations[0]) / total_populations[0]
            if growth_rate > 0.1:  # 10% growth
                emergent_behaviors.append("Population Expansion Phase")
            elif growth_rate < -0.1:  # 10% decline
                emergent_behaviors.append("Population Stabilization Phase")

        return emergent_behaviors

    def generate_ecosystem_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive ecosystem analysis report
        """
        current_data = self.habitat_model.get_complete_status()

        # Perform various analyses
        coherence_analysis = self.analyze_spectral_coherence(current_data['colonies'])
        emergence_level = self.calculate_emergence_level(current_data)
        communication_efficiency = self.analyze_communication_efficiency(current_data['colonies'])

        # Calculate GLS scores for each colony
        gls_scores = {}
        for species, colony in current_data['colonies'].items():
            gls_scores[species] = self.calculate_gls_stability_score(colony)

        # Detect emergent behaviors
        emergent_behaviors = self.detect_emergent_behaviors(current_data, self.analysis_history)

        # Generate predictions
        population_predictions = self.predict_population_trends(self.analysis_history)

        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'emergence_level': emergence_level,
            'spectral_coherence_analysis': coherence_analysis,
            'communication_efficiency': communication_efficiency,
            'gls_stability_scores': gls_scores,
            'emergent_behaviors': emergent_behaviors,
            'population_predictions': population_predictions,
            'system_health': {
                'overall_stability': np.mean(list(gls_scores.values())) if gls_scores else 0.0,
                'mathematical_foundation': current_data['framework_validation']['fully_validated'],
                'photonic_network_status': current_data['photonic_ecosystem']['avg_coherence']
            },
            'recommendations': self._generate_recommendations(current_data, gls_scores)
        }

        # Store in history
        self.analysis_history.append({
            'timestamp': datetime.utcnow(),
            'data': current_data,
            'analysis': report
        })

        # Keep only recent history
        if len(self.analysis_history) > 100:
            self.analysis_history = self.analysis_history[-100:]

        return report

    def _generate_recommendations(self,
                                 current_data: Dict,
                                 gls_scores: Dict[str, float]) -> List[str]:
        """
        Generate recommendations based on analysis
        """
        recommendations = []

        # Check GLS scores
        for species, score in gls_scores.items():
            if score < 0.7:
                recommendations.append(
                    f"Monitor {species} colony health - GLS score below optimal threshold"
                )
            elif score > 0.95:
                recommendations.append(
                    f"{species} colony showing excellent stability - consider as model for others"
                )

        # Check photonic efficiency
        photonic = current_data.get('photonic_ecosystem', {})
        coherence = photonic.get('avg_coherence', 0.5)
        if coherence < 0.7:
            recommendations.append("Optimize photonic network coherence for better communication")
        elif coherence > 0.9:
            recommendations.append("Photonic network performing optimally - maintain current parameters")

        # Check mathematical foundation
        if not current_data['framework_validation']['fully_validated']:
            recommendations.append("Verify mathematical foundation - some equations may need attention")

        return recommendations

    def export_analysis_history(self, filename: str):
        """
        Export analysis history to JSON file
        """
        export_data = []
        for entry in self.analysis_history:
            export_entry = {
                'timestamp': entry['timestamp'].isoformat(),
                'analysis': entry['analysis']
            }
            export_data.append(export_entry)

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)


class GLSMonitor:
    """
    Real-time GLS ecosystem monitoring system
    """

    def __init__(self, analyzer: GLSAnalyzer):
        self.analyzer = analyzer
        self.alerts: List[Dict] = []
        self.thresholds = {
            'min_gls_score': 0.6,
            'min_coherence': 0.6,
            'max_population_change': 0.3,
            'min_emergence_level': 10.0
        }

    def check_system_health(self) -> Dict[str, Any]:
        """
        Check overall system health and generate alerts
        """
        report = self.analyzer.generate_ecosystem_report()
        alerts = []

        # Check GLS scores
        for species, score in report['gls_stability_scores'].items():
            if score < self.thresholds['min_gls_score']:
                alerts.append({
                    'type': 'GLS_LOW',
                    'species': species,
                    'score': score,
                    'message': f"{species} GLS score critically low: {score:.3f}"
                })

        # Check emergence level
        if report['emergence_level'] < self.thresholds['min_emergence_level']:
            alerts.append({
                'type': 'EMERGENCE_LOW',
                'level': report['emergence_level'],
                'message': f"System emergence level below threshold: {report['emergence_level']:.2f}"
            })

        # Check spectral coherence
        avg_coherence = np.mean(list(report['spectral_coherence_analysis'].values()))
        if avg_coherence < self.thresholds['min_coherence']:
            alerts.append({
                'type': 'COHERENCE_LOW',
                'coherence': avg_coherence,
                'message': f"Average spectral coherence low: {avg_coherence:.3f}"
            })

        # Store alerts
        for alert in alerts:
            alert['timestamp'] = datetime.utcnow().isoformat()
            self.alerts.append(alert)

        # Keep only recent alerts
        if len(self.alerts) > 50:
            self.alerts = self.alerts[-50:]

        return {
            'health_status': 'HEALTHY' if not alerts else 'ALERT',
            'active_alerts': alerts,
            'total_alerts': len(alerts),
            'system_report': report
        }
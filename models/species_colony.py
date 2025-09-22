#!/usr/bin/env python3
"""
Species Colony Model - ΨQRH Framework

Manages a colony of living specimens with population dynamics and social behaviors.
"""

import numpy as np
from typing import Dict, List, Any


class SpeciesColony:
    """Manages a colony of living specimens"""

    def __init__(self, species_name: str, max_population: int = 20):
        self.species_name = species_name
        self.max_population = max_population
        self.specimens: List[Any] = []
        self.colony_center = np.random.uniform([0, 0, 0, 0], [20, 15, 10, 6])
        self.social_cohesion = 0.0
        self.average_health = 0.0
        self.communication_network = []
        self.reproduction_events = 0

    def add_specimen(self, specimen):
        """Add a living specimen to the colony"""
        if len(self.specimens) < self.max_population:
            self.specimens.append(specimen)
            return True
        return False

    def update_colony_metrics(self):
        """Update colony-level metrics"""
        if not self.specimens:
            return

        # Calculate average health
        healths = [s.health for s in self.specimens if hasattr(s, 'health')]
        self.average_health = np.mean(healths) if healths else 0.0

        # Calculate social cohesion (distance-based)
        positions = []
        for specimen in self.specimens:
            if hasattr(specimen, 'position'):
                positions.append(specimen.position[:3])  # Use only spatial dimensions

        if len(positions) > 1:
            positions = np.array(positions)
            center = np.mean(positions, axis=0)
            distances = [np.linalg.norm(pos - center) for pos in positions]
            # Cohesion is inverse of average distance (normalized)
            avg_distance = np.mean(distances)
            self.social_cohesion = max(0.0, 1.0 - avg_distance / 10.0)
        else:
            self.social_cohesion = 1.0

    def get_status(self) -> Dict:
        """Get colony status summary"""
        alive_count = sum(1 for s in self.specimens if hasattr(s, 'health') and s.health > 0)

        return {
            'species': self.species_name,
            'population': len(self.specimens),
            'alive_count': alive_count,
            'average_health': float(self.average_health),
            'social_cohesion': float(self.social_cohesion),
            'colony_center': self.colony_center.tolist(),
            'reproduction_events': self.reproduction_events
        }
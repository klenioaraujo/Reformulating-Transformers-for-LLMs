#!/usr/bin/env python3
"""
Living Ecosystem Engine - ΨQRH Framework

Real-time simulation engine where each insect specimen is a complete living model
with its own characteristic behaviors, neural processing, and life cycles.

Each species represents a different solution from the ΨQRH solution space:
- Araneae: Complex web-building predators with quaternion spatial processing
- Chrysopidae: Agile flying prey with optical communication systems
- Apis: Social communicators with collective intelligence
- Scutigera Coleoptrata: Rapid multi-limb predators with enhanced sensory systems
"""

import numpy as np
import torch
import torch.nn as nn
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import threading
from queue import Queue
import sys
import os

# Import all living specimen models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .models.insect_specimens.araneae import Araneae_PsiQRH, AraneaeDNA
from .models.insect_specimens.chrysopidae import create_chrysopidae_specimen
from .models.insect_specimens.apis_mellifera import create_apis_specimen
from .models.insect_specimens.scutigera_coleoptrata import (
    ScutigeraColeoptrata_PsiQRH,
    ScutigeraColeoptrataDNA,
    create_fast_hunter,
    create_sensory_specialist,
    create_territorial_guardian
)
from .models.insect_specimens.communication import PadilhaWave
from .models.habitat_environment import HabitatEnvironment
from .models.species_colony import SpeciesColony






class LivingEcosystemEngine:
    """
    Main simulation engine for the living ΨQRH ecosystem

    Manages all living specimens, their interactions, and the environment.
    Each specimen is a complete living model with its own neural processing.
    """

    def __init__(self, habitat_config: Optional[Dict] = None):
        # Initialize habitat
        self.habitat = HabitatEnvironment(**(habitat_config or {}))

        # Species colonies
        self.colonies = {
            'Araneae': SpeciesColony('Araneae', max_population=15),
            'Chrysopidae': SpeciesColony('Chrysopidae', max_population=20),
            'Apis': SpeciesColony('Apis', max_population=25),
            'Scutigera': SpeciesColony('Scutigera', max_population=12)
        }

        # Global ecosystem state
        self.simulation_time = 0.0
        self.time_step = 0.1  # 100ms time steps
        self.running = False

        # Communication system
        self.global_communications = Queue()
        self.communication_log = []

        # Environmental dynamics
        self.environmental_changes = []
        self.chaos_factor = 0.1

        # Performance metrics
        self.total_specimens = 0
        self.total_interactions = 0
        self.total_reproductions = 0

        # Real-time data for visualization
        self.real_time_data = {
            'positions': [],
            'behaviors': [],
            'health_levels': [],
            'communication_events': [],
            'species_distribution': {},
            'environmental_state': {}
        }

        # Initialize ecosystem
        self._populate_initial_ecosystem()

    def _populate_initial_ecosystem(self):
        """Create initial population of living specimens"""
        print("🌱 Initializing Living Ecosystem...")

        # Create Araneae colony (spiders)
        for i in range(8):
            try:
                spider = Araneae_PsiQRH(dna=AraneaeDNA())
                if self.colonies['Araneae'].add_specimen(spider):
                    print(f"  🕷️  Created spider: {spider.specimen_id}")
            except Exception as e:
                print(f"  ⚠️  Failed to create spider {i}: {e}")

        # Create Chrysopidae colony (lacewings)
        for i in range(12):
            try:
                lacewing = create_chrysopidae_specimen()
                if self.colonies['Chrysopidae'].add_specimen(lacewing):
                    print(f"  🦋 Created lacewing: {getattr(lacewing, 'specimen_id', f'lacewing_{i}')}")
            except Exception as e:
                print(f"  ⚠️  Failed to create lacewing {i}: {e}")

        # Create Apis colony (bees)
        for i in range(15):
            try:
                bee = create_apis_specimen()
                if self.colonies['Apis'].add_specimen(bee):
                    print(f"  🐝 Created bee: {getattr(bee, 'specimen_id', f'bee_{i}')}")
            except Exception as e:
                print(f"  ⚠️  Failed to create bee {i}: {e}")

        # Create Scutigera colony (centipedes)
        centipede_types = [create_fast_hunter, create_sensory_specialist, create_territorial_guardian]
        for i in range(6):
            try:
                creator_func = centipede_types[i % len(centipede_types)]
                centipede = creator_func()
                if self.colonies['Scutigera'].add_specimen(centipede):
                    print(f"  🦶 Created centipede: {centipede.specimen_id}")
            except Exception as e:
                print(f"  ⚠️  Failed to create centipede {i}: {e}")

        # Calculate total population
        self.total_specimens = sum(len(colony.specimens) for colony in self.colonies.values())
        print(f"🌟 Living ecosystem initialized with {self.total_specimens} living specimens")

    def start_simulation(self, max_time: Optional[float] = None):
        """Start the living ecosystem simulation"""
        print("🚀 Starting Living Ecosystem Simulation...")
        self.running = True
        start_time = time.time()

        try:
            while self.running:
                # Check time limit
                if max_time and self.simulation_time >= max_time:
                    break

                # Update ecosystem
                self._update_ecosystem_step()

                # Sleep for real-time simulation
                time.sleep(self.time_step)

                # Update simulation time
                self.simulation_time += self.time_step

        except KeyboardInterrupt:
            print("\n⏹️  Simulation stopped by user")

        finally:
            self.running = False
            runtime = time.time() - start_time
            print(f"📊 Simulation completed. Runtime: {runtime:.1f}s, Sim time: {self.simulation_time:.1f}s")

    def _update_ecosystem_step(self):
        """Update all specimens and environment for one time step"""
        # Update environmental conditions
        self._update_environment()

        # Create environment state for specimens
        env_state = self._create_environment_state()

        # Update all living specimens
        for colony_name, colony in self.colonies.items():
            for specimen in colony.specimens:
                if hasattr(specimen, 'update_living_state'):
                    try:
                        specimen.update_living_state(env_state, self.time_step)
                    except Exception as e:
                        print(f"⚠️  Error updating {getattr(specimen, 'specimen_id', 'unknown')}: {e}")

        # Process inter-specimen interactions
        self._process_interactions()

        # Update colony metrics
        for colony in self.colonies.values():
            colony.update_colony_metrics()

        # Update real-time data for visualization
        self._update_real_time_data()

        # Process communications
        self._process_communications()

        # Environmental evolution
        self._evolve_environment()

    def _update_environment(self):
        """Update environmental conditions"""
        # Add small random fluctuations
        self.habitat.temperature += np.random.uniform(-0.1, 0.1)
        self.habitat.humidity += np.random.uniform(-0.01, 0.01)
        self.habitat.light_intensity += np.random.uniform(-0.05, 0.05)

        # Keep within realistic bounds
        self.habitat.temperature = np.clip(self.habitat.temperature, 15.0, 35.0)
        self.habitat.humidity = np.clip(self.habitat.humidity, 0.3, 0.9)
        self.habitat.light_intensity = np.clip(self.habitat.light_intensity, 0.1, 1.0)

        # Update dynamic fields
        self.habitat.vibrations += np.random.uniform(-0.02, 0.02, 12)
        self.habitat.vibrations = np.clip(self.habitat.vibrations, 0.0, 0.5)

        self.habitat.chemical_traces += np.random.uniform(-0.01, 0.01, 4)
        self.habitat.chemical_traces = np.clip(self.habitat.chemical_traces, 0.0, 1.0)

        # Update spectral fields (ΨQRH evolution)
        self.habitat.alpha_field += np.random.uniform(-0.05, 0.05, self.habitat.alpha_field.shape)
        self.habitat.alpha_field = np.clip(self.habitat.alpha_field, 0.1, 3.0)

        self.habitat.coherence_field += np.random.uniform(-0.02, 0.02, self.habitat.coherence_field.shape)
        self.habitat.coherence_field = np.clip(self.habitat.coherence_field, 0.0, 1.0)

    def _create_environment_state(self) -> Dict:
        """Create environment state dictionary for specimens"""
        # Collect prey positions for predators
        prey_positions = []
        for colony_name, colony in self.colonies.items():
            if colony_name in ['Chrysopidae', 'Apis']:  # Prey species
                for specimen in colony.specimens:
                    if hasattr(specimen, 'position') and hasattr(specimen, 'health'):
                        if specimen.health > 0:
                            prey_positions.append(specimen.position.tolist())

        return {
            'temperature': self.habitat.temperature,
            'humidity': self.habitat.humidity,
            'light_intensity': self.habitat.light_intensity,
            'vibrations': self.habitat.vibrations.copy(),
            'chemical_traces': self.habitat.chemical_traces.copy(),
            'air_currents': self.habitat.air_currents.copy(),
            'alpha_field': self.habitat.alpha_field,
            'coherence_field': self.habitat.coherence_field,
            'prey_positions': prey_positions,
            'chaos_factor': self.chaos_factor,
            'simulation_time': self.simulation_time,
            'communications': []  # Will be filled by specimens
        }

    def _process_interactions(self):
        """Process interactions between specimens"""
        all_specimens = []
        for colony in self.colonies.values():
            all_specimens.extend(colony.specimens)

        # Process predator-prey interactions
        predators = []
        prey = []

        for specimen in all_specimens:
            if hasattr(specimen, 'species'):
                if specimen.species in ['Araneae', 'Scutigera coleoptrata']:
                    predators.append(specimen)
                elif specimen.species in ['Chrysopidae', 'ApisMellifera']:
                    prey.append(specimen)

        # Check for predation events
        for predator in predators:
            if not hasattr(predator, 'position') or not hasattr(predator, 'hunting_mode'):
                continue

            if predator.hunting_mode and predator.health > 0.5:
                for prey_specimen in prey:
                    if not hasattr(prey_specimen, 'position') or prey_specimen.health <= 0:
                        continue

                    # Calculate distance
                    distance = np.linalg.norm(predator.position[:3] - prey_specimen.position[:3])

                    # Check if predation occurs
                    if distance < 0.5:  # Very close
                        predation_success = np.random.random() < 0.3  # 30% success rate

                        if predation_success:
                            # Prey loses health, predator gains energy
                            prey_specimen.health -= 0.5
                            predator.energy_level = min(1.0, predator.energy_level + 0.3)

                            print(f"🍽️  Predation: {predator.specimen_id} -> {getattr(prey_specimen, 'specimen_id', 'prey')}")
                            self.total_interactions += 1

    def _process_communications(self):
        """Process communication between specimens"""
        # Collect all communications from environment
        all_communications = []

        for colony in self.colonies.values():
            for specimen in colony.specimens:
                if hasattr(specimen, '_last_communication'):
                    all_communications.append(specimen._last_communication)

        # Process mating communications
        for comm in all_communications:
            if comm.get('mating_ready', False):
                # Find potential mates
                for colony in self.colonies.values():
                    for specimen in colony.specimens:
                        if (hasattr(specimen, 'analyze_wave') and
                            hasattr(specimen, 'mating_readiness') and
                            specimen.mating_readiness > 0.8):

                            # Create simple wave for analysis
                            wave = type('Wave', (), {
                                'frequency': comm.get('frequency', 200),
                                'amplitude': comm.get('amplitude', 0.5)
                            })()

                            compatibility = specimen.analyze_wave(wave)

                            if compatibility > 0.9:
                                # Reproduction event
                                print(f"💕 Mating event: {comm['sender_id']} + {getattr(specimen, 'specimen_id', 'unknown')}")
                                self.total_reproductions += 1

                                # Reset mating readiness
                                specimen.mating_readiness = 0.0

    def _update_real_time_data(self):
        """Update real-time data for visualization"""
        positions = []
        behaviors = []
        health_levels = []
        species_dist = {'Araneae': 0, 'Chrysopidae': 0, 'Apis': 0, 'Scutigera': 0}

        for colony_name, colony in self.colonies.items():
            alive_count = 0
            for specimen in colony.specimens:
                if hasattr(specimen, 'position') and hasattr(specimen, 'health'):
                    if specimen.health > 0:
                        positions.append({
                            'x': float(specimen.position[0]),
                            'y': float(specimen.position[1]),
                            'z': float(specimen.position[2]),
                            'w': float(specimen.position[3]) if len(specimen.position) > 3 else 0.0,
                            'species': colony_name,
                            'id': getattr(specimen, 'specimen_id', f'{colony_name}_{len(positions)}')
                        })

                        behaviors.append({
                            'id': getattr(specimen, 'specimen_id', f'{colony_name}_{len(behaviors)}'),
                            'behavior': getattr(specimen, 'behavior_state', 'unknown'),
                            'species': colony_name
                        })

                        health_levels.append({
                            'id': getattr(specimen, 'specimen_id', f'{colony_name}_{len(health_levels)}'),
                            'health': float(specimen.health),
                            'energy': float(getattr(specimen, 'energy_level', 0.5)),
                            'species': colony_name
                        })

                        alive_count += 1

            species_dist[colony_name] = alive_count

        self.real_time_data = {
            'positions': positions,
            'behaviors': behaviors,
            'health_levels': health_levels,
            'species_distribution': species_dist,
            'environmental_state': {
                'temperature': float(self.habitat.temperature),
                'humidity': float(self.habitat.humidity),
                'light_intensity': float(self.habitat.light_intensity),
                'chaos_factor': float(self.chaos_factor),
                'simulation_time': float(self.simulation_time)
            },
            'ecosystem_metrics': {
                'total_specimens': self.total_specimens,
                'alive_specimens': len(positions),
                'total_interactions': self.total_interactions,
                'total_reproductions': self.total_reproductions,
                'average_health': np.mean([h['health'] for h in health_levels]) if health_levels else 0.0
            }
        }

    def _evolve_environment(self):
        """Evolve environmental conditions over time"""
        # Gradual chaos factor evolution
        self.chaos_factor += np.random.uniform(-0.005, 0.005)
        self.chaos_factor = np.clip(self.chaos_factor, 0.05, 0.95)

        # Day/night cycle simulation
        day_cycle = np.sin(self.simulation_time * 2 * np.pi / 86400)  # 24-hour cycle
        self.habitat.light_intensity = 0.5 + 0.4 * day_cycle

    def get_ecosystem_status(self) -> Dict:
        """Get comprehensive ecosystem status"""
        colony_statuses = {}
        for name, colony in self.colonies.items():
            colony_statuses[name] = colony.get_status()

        return {
            'simulation_time': float(self.simulation_time),
            'running': self.running,
            'habitat': {
                'temperature': float(self.habitat.temperature),
                'humidity': float(self.habitat.humidity),
                'light_intensity': float(self.habitat.light_intensity),
                'chaos_factor': float(self.chaos_factor)
            },
            'colonies': colony_statuses,
            'metrics': {
                'total_specimens': self.total_specimens,
                'total_interactions': self.total_interactions,
                'total_reproductions': self.total_reproductions
            },
            'real_time_data': self.real_time_data
        }

    def stop_simulation(self):
        """Stop the simulation gracefully"""
        self.running = False
        print("🛑 Stopping ecosystem simulation...")

    def save_state(self, filename: str):
        """Save ecosystem state to file"""
        state = self.get_ecosystem_status()
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"💾 Ecosystem state saved to {filename}")

    def export_live_data(self) -> Dict:
        """Export current live data for web visualization"""
        return self.real_time_data


def create_living_ecosystem(config: Optional[Dict] = None) -> LivingEcosystemEngine:
    """Factory function to create a living ecosystem"""
    return LivingEcosystemEngine(config)


if __name__ == "__main__":
    # Demo: Create and run living ecosystem
    print("🌌 ΨQRH Living Ecosystem Engine Demo")
    print("=" * 50)

    # Create ecosystem
    ecosystem = create_living_ecosystem()

    # Run for 30 seconds
    print("🎬 Running ecosystem simulation for 30 seconds...")
    ecosystem.start_simulation(max_time=30.0)

    # Display final status
    status = ecosystem.get_ecosystem_status()
    print("\n📊 Final Ecosystem Status:")
    print(f"Total Specimens: {status['metrics']['total_specimens']}")
    print(f"Interactions: {status['metrics']['total_interactions']}")
    print(f"Reproductions: {status['metrics']['total_reproductions']}")

    for name, colony in status['colonies'].items():
        print(f"{name}: {colony['alive_count']}/{colony['population']} alive, health={colony['average_health']:.2f}")
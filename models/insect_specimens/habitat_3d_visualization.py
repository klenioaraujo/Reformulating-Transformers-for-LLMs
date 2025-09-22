"""
3D Natural Habitat Visualization with Species Evolution
Displays natural evolution within 4D spectral environment controlled by GLS framework
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import time
from typing import List, Dict, Tuple, Optional
import threading
import queue

from models.insect_specimens.gls_framework import (
    gls_stability_score,
    enhanced_dna_to_alpha_mapping,
    population_health_analysis
)
from models.insect_specimens.araneae import Araneae_PsiQRH
from models.insect_specimens.chrysopidae import Chrysopidae
from models.insect_specimens.dna import AraneaeDNA, ChrysopidaeDNA
from emergence_simulation import run_emergent_simulation


class NaturalHabitat3D:
    """
    3D Natural Habitat displaying species evolution within 4D spectral environment.
    Shows organic growth, territory formation, and natural behaviors emerging from ΨQRH dynamics.
    """

    def __init__(self, habitat_size: Tuple[float, float, float] = (10.0, 10.0, 8.0)):
        self.habitat_size = habitat_size
        self.specimens = []
        self.territories = {}
        self.history = {
            'populations': [],
            'genetic_diversity': [],
            'territorial_expansion': [],
            'species_interactions': [],
            'time_points': []
        }

        # 4D Spectral environment layers
        self.spectral_layers = {
            'alpha_field': np.zeros((50, 50, 30)),  # α parameter field
            'beta_field': np.zeros((50, 50, 30)),   # β parameter field
            'omega_field': np.zeros((50, 50, 30)),  # ω frequency field
            'gls_density': np.zeros((50, 50, 30))   # GLS interaction density
        }

        # Environmental factors
        self.environmental_gradients = self._generate_environmental_gradients()
        self.resource_distribution = self._generate_resource_distribution()

        # Natural emergence tracking
        self.emergence_metrics = {
            'behavioral_complexity': [],
            'territorial_stability': [],
            'mating_success_rates': [],
            'genetic_drift': []
        }

    def _generate_environmental_gradients(self) -> Dict[str, np.ndarray]:
        """Generate natural environmental gradients affecting species distribution."""
        x, y, z = np.meshgrid(
            np.linspace(0, self.habitat_size[0], 50),
            np.linspace(0, self.habitat_size[1], 50),
            np.linspace(0, self.habitat_size[2], 30)
        )

        # Natural terrain features
        elevation = 2 * np.sin(x/3) * np.cos(y/3) + 0.5 * z
        humidity = 0.8 - 0.3 * (z / self.habitat_size[2]) + 0.1 * np.sin(x/2)
        temperature = 25 + 5 * np.sin(y/4) - 2 * (z / self.habitat_size[2])
        light_intensity = 1.0 - 0.6 * (z / self.habitat_size[2]) + 0.2 * np.cos(x/2 + y/2)

        return {
            'elevation': elevation,
            'humidity': humidity,
            'temperature': temperature,
            'light_intensity': light_intensity
        }

    def _generate_resource_distribution(self) -> Dict[str, np.ndarray]:
        """Generate natural resource distribution patterns."""
        x, y, z = np.meshgrid(
            np.linspace(0, self.habitat_size[0], 50),
            np.linspace(0, self.habitat_size[1], 50),
            np.linspace(0, self.habitat_size[2], 30)
        )

        # Resource hotspots following natural patterns
        prey_density = (
            0.5 + 0.3 * np.exp(-((x-5)**2 + (y-5)**2)/10) +
            0.2 * np.exp(-((x-8)**2 + (y-3)**2)/8) +
            0.1 * np.random.normal(0, 0.1, x.shape)
        )

        nesting_sites = (
            self.environmental_gradients['elevation'] > 1.5
        ).astype(float) * (
            self.environmental_gradients['humidity'] > 0.6
        ).astype(float)

        water_sources = np.exp(-((x-3)**2 + (y-7)**2)/5) + np.exp(-((x-7)**2 + (y-2)**2)/4)

        return {
            'prey_density': prey_density,
            'nesting_sites': nesting_sites,
            'water_sources': water_sources
        }

    def add_specimen(self, specimen, position: Optional[Tuple[float, float, float]] = None):
        """Add specimen to habitat with natural positioning."""
        if position is None:
            # Natural placement based on species preferences
            position = self._find_natural_position(specimen)

        specimen_data = {
            'specimen': specimen,
            'position': np.array(position),
            'velocity': np.zeros(3),
            'territory_center': np.array(position),
            'territory_radius': 0.5,
            'energy': 1.0,
            'stress_level': 0.0,
            'behavioral_state': 'exploring',
            'interaction_history': [],
            'birth_time': time.time()
        }

        self.specimens.append(specimen_data)
        self._update_spectral_fields(specimen_data)

    def _find_natural_position(self, specimen) -> Tuple[float, float, float]:
        """Find natural starting position based on species characteristics."""
        # Different species prefer different habitat zones
        species_type = type(specimen).__name__

        if species_type == 'Araneae_PsiQRH':
            # Spiders prefer elevated areas with good prey density
            suitable_zones = (
                (self.environmental_gradients['elevation'] > 1.0) &
                (self.resource_distribution['prey_density'] > 0.4)
            )
        elif species_type == 'Chrysopidae':
            # Lacewings prefer areas with good light and moderate humidity
            suitable_zones = (
                (self.environmental_gradients['light_intensity'] > 0.5) &
                (self.environmental_gradients['humidity'] > 0.4) &
                (self.environmental_gradients['humidity'] < 0.8)
            )
        else:
            # Default positioning
            suitable_zones = self.resource_distribution['prey_density'] > 0.3

        # Find suitable coordinates
        indices = np.where(suitable_zones)
        if len(indices[0]) > 0:
            idx = np.random.randint(len(indices[0]))
            x_idx, y_idx, z_idx = indices[0][idx], indices[1][idx], indices[2][idx]

            x = (x_idx / 50) * self.habitat_size[0]
            y = (y_idx / 50) * self.habitat_size[1]
            z = (z_idx / 30) * self.habitat_size[2]

            return (x, y, z)
        else:
            # Random fallback
            return (
                np.random.uniform(0, self.habitat_size[0]),
                np.random.uniform(0, self.habitat_size[1]),
                np.random.uniform(0, self.habitat_size[2])
            )

    def _update_spectral_fields(self, specimen_data):
        """Update 4D spectral environment based on specimen's GLS properties."""
        specimen = specimen_data['specimen']
        position = specimen_data['position']

        if hasattr(specimen, 'gls_visual_layer') and specimen.gls_visual_layer:
            gls = specimen.gls_visual_layer
            spectral_features = gls.extract_spectral_features()

            # Map position to grid indices
            x_idx = int((position[0] / self.habitat_size[0]) * 49)
            y_idx = int((position[1] / self.habitat_size[1]) * 49)
            z_idx = int((position[2] / self.habitat_size[2]) * 29)

            # Update spectral fields with influence radius
            influence_radius = 3
            for dx in range(-influence_radius, influence_radius + 1):
                for dy in range(-influence_radius, influence_radius + 1):
                    for dz in range(-influence_radius, influence_radius + 1):
                        nx, ny, nz = x_idx + dx, y_idx + dy, z_idx + dz

                        if (0 <= nx < 50 and 0 <= ny < 50 and 0 <= nz < 30):
                            distance = np.sqrt(dx**2 + dy**2 + dz**2)
                            influence = np.exp(-distance / 2)

                            self.spectral_layers['alpha_field'][nx, ny, nz] += (
                                spectral_features['alpha'] * influence * 0.1
                            )
                            self.spectral_layers['beta_field'][nx, ny, nz] += (
                                spectral_features['beta'] * influence * 0.1
                            )
                            self.spectral_layers['omega_field'][nx, ny, nz] += (
                                spectral_features['omega'] * influence * 0.01
                            )
                            self.spectral_layers['gls_density'][nx, ny, nz] += influence * 0.1

    def simulate_natural_behaviors(self, dt: float = 0.1):
        """Simulate natural behaviors emerging from ΨQRH dynamics."""
        for specimen_data in self.specimens:
            specimen = specimen_data['specimen']

            # Natural movement based on environmental gradients and GLS interactions
            self._update_movement(specimen_data, dt)

            # Territory establishment and defense
            self._update_territory(specimen_data, dt)

            # Inter-species interactions
            self._process_interactions(specimen_data)

            # Energy and stress dynamics
            self._update_physiology(specimen_data, dt)

        # Process reproduction and genetic drift
        self._process_reproduction()

        # Update emergence metrics
        self._update_emergence_metrics()

    def _update_movement(self, specimen_data, dt):
        """Update specimen movement based on environmental gradients and GLS fields."""
        position = specimen_data['position']
        specimen = specimen_data['specimen']

        # Get local environmental conditions
        local_conditions = self._get_local_conditions(position)

        # Movement influenced by GLS spectral fields if available
        if hasattr(specimen, 'gls_visual_layer') and specimen.gls_visual_layer:
            gls_influence = self._calculate_gls_movement_influence(position, specimen.gls_visual_layer)
        else:
            gls_influence = np.zeros(3)

        # Environmental gradient forces
        gradient_force = self._calculate_environmental_force(position, specimen)

        # Territory attraction/repulsion
        territory_force = self._calculate_territory_force(specimen_data)

        # Combine forces for natural movement
        total_force = gradient_force + gls_influence + territory_force

        # Update velocity with damping
        damping = 0.8
        specimen_data['velocity'] = specimen_data['velocity'] * damping + total_force * dt

        # Update position with boundary constraints
        new_position = position + specimen_data['velocity'] * dt
        specimen_data['position'] = np.clip(
            new_position,
            [0, 0, 0],
            list(self.habitat_size)
        )

    def _get_local_conditions(self, position) -> Dict[str, float]:
        """Get environmental conditions at given position."""
        x_idx = int((position[0] / self.habitat_size[0]) * 49)
        y_idx = int((position[1] / self.habitat_size[1]) * 49)
        z_idx = int((position[2] / self.habitat_size[2]) * 29)

        x_idx = np.clip(x_idx, 0, 49)
        y_idx = np.clip(y_idx, 0, 49)
        z_idx = np.clip(z_idx, 0, 29)

        return {
            'elevation': self.environmental_gradients['elevation'][x_idx, y_idx, z_idx],
            'humidity': self.environmental_gradients['humidity'][x_idx, y_idx, z_idx],
            'temperature': self.environmental_gradients['temperature'][x_idx, y_idx, z_idx],
            'light_intensity': self.environmental_gradients['light_intensity'][x_idx, y_idx, z_idx],
            'prey_density': self.resource_distribution['prey_density'][x_idx, y_idx, z_idx]
        }

    def _calculate_gls_movement_influence(self, position, gls) -> np.ndarray:
        """Calculate movement influence from 4D spectral fields."""
        x_idx = int((position[0] / self.habitat_size[0]) * 49)
        y_idx = int((position[1] / self.habitat_size[1]) * 49)
        z_idx = int((position[2] / self.habitat_size[2]) * 29)

        x_idx = np.clip(x_idx, 0, 49)
        y_idx = np.clip(y_idx, 0, 49)
        z_idx = np.clip(z_idx, 0, 29)

        # Sample spectral gradients around current position
        alpha_gradient = np.zeros(3)
        beta_gradient = np.zeros(3)

        # Calculate gradients for movement direction
        if x_idx > 0 and x_idx < 49:
            alpha_gradient[0] = (
                self.spectral_layers['alpha_field'][x_idx+1, y_idx, z_idx] -
                self.spectral_layers['alpha_field'][x_idx-1, y_idx, z_idx]
            )

        if y_idx > 0 and y_idx < 49:
            alpha_gradient[1] = (
                self.spectral_layers['alpha_field'][x_idx, y_idx+1, z_idx] -
                self.spectral_layers['alpha_field'][x_idx, y_idx-1, z_idx]
            )

        if z_idx > 0 and z_idx < 29:
            alpha_gradient[2] = (
                self.spectral_layers['alpha_field'][x_idx, y_idx, z_idx+1] -
                self.spectral_layers['alpha_field'][x_idx, y_idx, z_idx-1]
            )

        # GLS influences movement toward spectral compatibility zones
        spectral_features = gls.extract_spectral_features()
        compatibility_factor = spectral_features['alpha'] / (1.0 + abs(spectral_features['alpha']))

        return alpha_gradient * compatibility_factor * 0.1

    def _calculate_environmental_force(self, position, specimen) -> np.ndarray:
        """Calculate environmental forces attracting specimens to suitable areas."""
        local_conditions = self._get_local_conditions(position)
        species_type = type(specimen).__name__

        force = np.zeros(3)

        # Species-specific environmental preferences
        if species_type == 'Araneae_PsiQRH':
            # Spiders prefer areas with good prey density and moderate elevation
            prey_preference = (local_conditions['prey_density'] - 0.5) * 2.0
            elevation_preference = np.tanh(local_conditions['elevation'] - 1.0)
            force += np.array([prey_preference, elevation_preference, 0]) * 0.2

        elif species_type == 'Chrysopidae':
            # Lacewings prefer well-lit areas with moderate humidity
            light_preference = (local_conditions['light_intensity'] - 0.6) * 2.0
            humidity_preference = (0.6 - abs(local_conditions['humidity'] - 0.6)) * 2.0
            force += np.array([light_preference, humidity_preference, 0.1]) * 0.15

        return force

    def _calculate_territory_force(self, specimen_data) -> np.ndarray:
        """Calculate territorial forces for natural territory formation."""
        position = specimen_data['position']
        territory_center = specimen_data['territory_center']
        territory_radius = specimen_data['territory_radius']

        # Attraction to territory center
        to_center = territory_center - position
        distance_to_center = np.linalg.norm(to_center)

        territory_force = np.zeros(3)

        if distance_to_center > territory_radius:
            # Outside territory - gentle attraction back
            territory_force = to_center / (distance_to_center + 0.1) * 0.1

        # Repulsion from other specimens' territories
        for other_data in self.specimens:
            if other_data is specimen_data:
                continue

            other_position = other_data['position']
            other_territory_center = other_data['territory_center']
            other_territory_radius = other_data['territory_radius']

            to_other = other_position - position
            distance_to_other = np.linalg.norm(to_other)

            if distance_to_other < (territory_radius + other_territory_radius):
                # Within combined territory radius - repulsion
                if distance_to_other > 0.1:
                    repulsion = -to_other / distance_to_other * 0.2
                    territory_force += repulsion

        return territory_force

    def _update_territory(self, specimen_data, dt):
        """Update territory boundaries based on resource availability and competition."""
        position = specimen_data['position']
        local_conditions = self._get_local_conditions(position)

        # Territory expands in areas with good resources
        resource_quality = (
            local_conditions['prey_density'] * 0.4 +
            local_conditions['light_intensity'] * 0.3 +
            (1.0 - abs(local_conditions['humidity'] - 0.6)) * 0.3
        )

        # Update territory center toward resource-rich areas
        if resource_quality > 0.6:
            territory_shift = (position - specimen_data['territory_center']) * 0.01
            specimen_data['territory_center'] += territory_shift

            # Expand territory in good areas
            specimen_data['territory_radius'] = min(
                specimen_data['territory_radius'] + 0.01 * resource_quality,
                2.0
            )
        else:
            # Contract territory in poor areas
            specimen_data['territory_radius'] = max(
                specimen_data['territory_radius'] - 0.005,
                0.3
            )

    def _process_interactions(self, specimen_data):
        """Process natural interactions between specimens."""
        specimen = specimen_data['specimen']
        position = specimen_data['position']

        # Find nearby specimens
        for other_data in self.specimens:
            if other_data is specimen_data:
                continue

            other_specimen = other_data['specimen']
            other_position = other_data['position']

            distance = np.linalg.norm(position - other_position)

            if distance < 1.0:  # Interaction range
                interaction_type = self._determine_interaction_type(
                    specimen, other_specimen, distance
                )

                self._process_interaction(specimen_data, other_data, interaction_type)

    def _determine_interaction_type(self, specimen1, specimen2, distance) -> str:
        """Determine type of interaction based on species and GLS compatibility."""
        type1 = type(specimen1).__name__
        type2 = type(specimen2).__name__

        # Same species interactions
        if type1 == type2:
            if distance < 0.3:
                return 'territorial_conflict'
            elif hasattr(specimen1, 'gls_visual_layer') and hasattr(specimen2, 'gls_visual_layer'):
                compatibility = specimen1.gls_visual_layer.compare(specimen2.gls_visual_layer)
                if compatibility > 0.8:
                    return 'potential_mating'
                else:
                    return 'neutral'
            else:
                return 'neutral'
        else:
            # Inter-species interactions
            if type1 == 'Araneae_PsiQRH' and type2 == 'Chrysopidae':
                return 'predator_prey'
            elif type1 == 'Chrysopidae' and type2 == 'Araneae_PsiQRH':
                return 'prey_predator'
            else:
                return 'neutral'

    def _process_interaction(self, specimen_data1, specimen_data2, interaction_type):
        """Process specific interaction between two specimens."""
        if interaction_type == 'territorial_conflict':
            # Increase stress for both
            specimen_data1['stress_level'] += 0.1
            specimen_data2['stress_level'] += 0.1

        elif interaction_type == 'potential_mating':
            # Record mating opportunity
            specimen_data1['interaction_history'].append({
                'type': 'mating_opportunity',
                'partner': id(specimen_data2['specimen']),
                'time': time.time()
            })

        elif interaction_type == 'predator_prey':
            # Energy transfer
            specimen_data1['energy'] += 0.1  # Predator gains energy
            specimen_data2['energy'] -= 0.2  # Prey loses energy
            specimen_data2['stress_level'] += 0.3

    def _update_physiology(self, specimen_data, dt):
        """Update energy, stress, and health based on environmental conditions."""
        local_conditions = self._get_local_conditions(specimen_data['position'])
        specimen = specimen_data['specimen']

        # Energy consumption and gain
        base_metabolism = 0.01 * dt
        foraging_success = local_conditions['prey_density'] * 0.02 * dt

        specimen_data['energy'] += foraging_success - base_metabolism
        specimen_data['energy'] = np.clip(specimen_data['energy'], 0.0, 2.0)

        # Stress reduction over time
        specimen_data['stress_level'] *= (1.0 - 0.05 * dt)

        # Environmental stress
        temp_stress = abs(local_conditions['temperature'] - 25) / 30.0
        humidity_stress = abs(local_conditions['humidity'] - 0.6) / 0.6

        environmental_stress = (temp_stress + humidity_stress) * 0.01 * dt
        specimen_data['stress_level'] += environmental_stress

        specimen_data['stress_level'] = np.clip(specimen_data['stress_level'], 0.0, 1.0)

        # Update specimen health based on GLS stability if available
        if hasattr(specimen, 'gls_visual_layer') and specimen.gls_visual_layer:
            gls_health = gls_stability_score(specimen.gls_visual_layer)
            overall_health = (
                0.4 * gls_health +
                0.3 * (1.0 - specimen_data['stress_level']) +
                0.3 * min(1.0, specimen_data['energy'])
            )
            specimen.health = overall_health

    def _process_reproduction(self):
        """Process natural reproduction based on GLS compatibility and environmental conditions."""
        # Find potential mating pairs based on recent interactions
        potential_pairs = []

        for specimen_data in self.specimens:
            for interaction in specimen_data['interaction_history']:
                if interaction['type'] == 'mating_opportunity':
                    # Find partner data
                    partner_data = next(
                        (s for s in self.specimens if id(s['specimen']) == interaction['partner']),
                        None
                    )
                    if partner_data:
                        potential_pairs.append((specimen_data, partner_data))

        # Process successful reproductions
        new_offspring = []
        for parent1_data, parent2_data in potential_pairs:
            if self._reproduction_success(parent1_data, parent2_data):
                offspring = self._create_offspring(parent1_data, parent2_data)
                if offspring:
                    new_offspring.append(offspring)

        # Add offspring to habitat
        for offspring_data in new_offspring:
            self.specimens.append(offspring_data)

    def _reproduction_success(self, parent1_data, parent2_data) -> bool:
        """Determine if reproduction is successful based on multiple factors."""
        # GLS compatibility
        if (hasattr(parent1_data['specimen'], 'gls_visual_layer') and
            hasattr(parent2_data['specimen'], 'gls_visual_layer')):
            compatibility = parent1_data['specimen'].gls_visual_layer.compare(
                parent2_data['specimen'].gls_visual_layer
            )
        else:
            compatibility = 0.5

        # Health and energy requirements
        health_factor = (parent1_data['specimen'].health + parent2_data['specimen'].health) / 2
        energy_factor = min(parent1_data['energy'], parent2_data['energy'])
        stress_factor = 1.0 - max(parent1_data['stress_level'], parent2_data['stress_level'])

        # Environmental suitability
        avg_position = (parent1_data['position'] + parent2_data['position']) / 2
        local_conditions = self._get_local_conditions(avg_position)
        environmental_factor = (
            local_conditions['prey_density'] * 0.4 +
            (1.0 - abs(local_conditions['temperature'] - 25) / 25) * 0.3 +
            local_conditions['humidity'] * 0.3
        )

        # Combined reproduction probability
        reproduction_probability = (
            compatibility * 0.3 +
            health_factor * 0.3 +
            energy_factor * 0.2 +
            stress_factor * 0.1 +
            environmental_factor * 0.1
        )

        return np.random.random() < reproduction_probability

    def _create_offspring(self, parent1_data, parent2_data):
        """Create offspring with natural positioning and inherited traits."""
        parent1 = parent1_data['specimen']
        parent2 = parent2_data['specimen']

        # Determine species type (assumes same species mating)
        species_type = type(parent1).__name__

        try:
            if species_type == 'Araneae_PsiQRH':
                offspring_specimen = Araneae_PsiQRH.reproduce(parent1, parent2)
            elif species_type == 'Chrysopidae':
                # Assuming similar reproduce method exists
                offspring_specimen = Chrysopidae.reproduce(parent1, parent2)
            else:
                return None
        except:
            return None

        # Natural positioning near parents
        parent_center = (parent1_data['position'] + parent2_data['position']) / 2
        offspring_position = parent_center + np.random.normal(0, 0.5, 3)
        offspring_position = np.clip(offspring_position, [0, 0, 0], list(self.habitat_size))

        offspring_data = {
            'specimen': offspring_specimen,
            'position': offspring_position,
            'velocity': np.zeros(3),
            'territory_center': offspring_position.copy(),
            'territory_radius': 0.2,  # Start with small territory
            'energy': 0.8,  # Start with good energy
            'stress_level': 0.1,
            'behavioral_state': 'juvenile',
            'interaction_history': [],
            'birth_time': time.time()
        }

        return offspring_data

    def _update_emergence_metrics(self):
        """Update metrics tracking natural emergence phenomena."""
        current_time = time.time()

        # Behavioral complexity (variety of behavioral states)
        behavioral_states = [s['behavioral_state'] for s in self.specimens]
        complexity = len(set(behavioral_states)) / max(1, len(behavioral_states))

        # Territorial stability (variation in territory sizes)
        territory_sizes = [s['territory_radius'] for s in self.specimens]
        if territory_sizes:
            stability = 1.0 - (np.std(territory_sizes) / (np.mean(territory_sizes) + 0.1))
        else:
            stability = 1.0

        # Genetic diversity through GLS variation
        if len(self.specimens) > 1:
            gls_specimens = [s for s in self.specimens
                           if hasattr(s['specimen'], 'gls_visual_layer')
                           and s['specimen'].gls_visual_layer]

            if len(gls_specimens) > 1:
                diversities = []
                for i in range(len(gls_specimens)):
                    for j in range(i+1, len(gls_specimens)):
                        similarity = gls_specimens[i]['specimen'].gls_visual_layer.compare(
                            gls_specimens[j]['specimen'].gls_visual_layer
                        )
                        diversities.append(1.0 - similarity)

                genetic_diversity = np.mean(diversities) if diversities else 0.5
            else:
                genetic_diversity = 0.5
        else:
            genetic_diversity = 1.0

        # Record metrics
        self.emergence_metrics['behavioral_complexity'].append(complexity)
        self.emergence_metrics['territorial_stability'].append(stability)
        self.emergence_metrics['genetic_drift'].append(genetic_diversity)

        # Update history
        self.history['time_points'].append(current_time)
        self.history['populations'].append(len(self.specimens))
        self.history['genetic_diversity'].append(genetic_diversity)

    def create_3d_visualization(self):
        """Create comprehensive 3D visualization of the natural habitat and species evolution."""
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "scatter3d", "colspan": 2}, None],
                   [{"type": "scatter"}, {"type": "scatter"}]],
            subplot_titles=(
                '3D Natural Habitat with Species Evolution',
                'Population Dynamics',
                'Emergence Metrics'
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        # Main 3D habitat visualization
        self._add_3d_habitat_plot(fig)

        # Population dynamics
        self._add_population_plot(fig)

        # Emergence metrics
        self._add_emergence_plot(fig)

        # Update layout
        fig.update_layout(
            title="🌿 Natural Habitat Evolution - ΨQRH 4D Spectral Environment",
            height=900,
            showlegend=True,
            template="plotly_white"
        )

        return fig

    def _add_3d_habitat_plot(self, fig):
        """Add 3D habitat plot showing specimens, territories, and environmental features."""
        # Environmental visualization (terrain elevation)
        x_env = np.linspace(0, self.habitat_size[0], 20)
        y_env = np.linspace(0, self.habitat_size[1], 20)
        X_env, Y_env = np.meshgrid(x_env, y_env)

        # Surface elevation
        Z_elevation = np.zeros_like(X_env)
        for i, x in enumerate(x_env):
            for j, y in enumerate(y_env):
                conditions = self._get_local_conditions([x, y, 2.0])
                Z_elevation[j, i] = conditions['elevation']

        # Add terrain surface
        fig.add_trace(
            go.Surface(
                x=X_env, y=Y_env, z=Z_elevation,
                colorscale='Earth',
                opacity=0.4,
                name='Terrain',
                showscale=False
            ),
            row=1, col=1
        )

        # Specimen positions colored by species and health
        species_colors = {'Araneae_PsiQRH': 'red', 'Chrysopidae': 'green'}

        for specimen_data in self.specimens:
            specimen = specimen_data['specimen']
            position = specimen_data['position']
            species_type = type(specimen).__name__

            # Size based on health/energy
            size = 5 + specimen_data['energy'] * 5

            # Color intensity based on health
            health = getattr(specimen, 'health', 0.5)

            fig.add_trace(
                go.Scatter3d(
                    x=[position[0]],
                    y=[position[1]],
                    z=[position[2]],
                    mode='markers',
                    marker=dict(
                        size=size,
                        color=species_colors.get(species_type, 'blue'),
                        opacity=0.3 + 0.7 * health,
                        line=dict(width=2, color='black')
                    ),
                    name=f'{species_type} (Health: {health:.2f})',
                    hovertemplate=(
                        f'Species: {species_type}<br>'
                        f'Position: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})<br>'
                        f'Health: {health:.3f}<br>'
                        f'Energy: {specimen_data["energy"]:.2f}<br>'
                        f'Territory: {specimen_data["territory_radius"]:.2f}<br>'
                        '<extra></extra>'
                    )
                ),
                row=1, col=1
            )

        # Territory boundaries (simplified as circles at ground level)
        for specimen_data in self.specimens:
            position = specimen_data['position']
            territory_center = specimen_data['territory_center']
            territory_radius = specimen_data['territory_radius']

            # Create territory circle
            theta = np.linspace(0, 2*np.pi, 20)
            circle_x = territory_center[0] + territory_radius * np.cos(theta)
            circle_y = territory_center[1] + territory_radius * np.sin(theta)
            circle_z = np.full_like(circle_x, territory_center[2])

            fig.add_trace(
                go.Scatter3d(
                    x=circle_x,
                    y=circle_y,
                    z=circle_z,
                    mode='lines',
                    line=dict(color='rgba(100,100,100,0.3)', width=2),
                    name='Territory',
                    showlegend=False
                ),
                row=1, col=1
            )

        # 4D Spectral field visualization (alpha field as isosurface)
        if np.any(self.spectral_layers['alpha_field'] > 0):
            x_spec = np.linspace(0, self.habitat_size[0], 50)
            y_spec = np.linspace(0, self.habitat_size[1], 50)
            z_spec = np.linspace(0, self.habitat_size[2], 30)

            # Sample spectral field for visualization
            alpha_field_sample = self.spectral_layers['alpha_field'][::5, ::5, ::3]
            x_sample = x_spec[::5]
            y_sample = y_spec[::5]
            z_sample = z_spec[::3]

            # Create volume visualization
            fig.add_trace(
                go.Volume(
                    x=x_sample, y=y_sample, z=z_sample,
                    value=alpha_field_sample.flatten(),
                    isomin=0.1,
                    isomax=alpha_field_sample.max(),
                    opacity=0.2,
                    surface_count=3,
                    colorscale='Viridis',
                    name='4D Spectral Field (α)',
                    showscale=False
                ),
                row=1, col=1
            )

        # Update 3D scene
        fig.update_scenes(
            xaxis_title="X (Habitat Space)",
            yaxis_title="Y (Habitat Space)",
            zaxis_title="Z (Height)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            row=1, col=1
        )

    def _add_population_plot(self, fig):
        """Add population dynamics plot."""
        if len(self.history['time_points']) > 1:
            time_points = [t - self.history['time_points'][0] for t in self.history['time_points']]

            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=self.history['populations'],
                    mode='lines+markers',
                    name='Population Size',
                    line=dict(color='blue', width=3)
                ),
                row=2, col=1
            )

            fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
            fig.update_yaxes(title_text="Population Count", row=2, col=1)

    def _add_emergence_plot(self, fig):
        """Add emergence metrics plot."""
        if len(self.emergence_metrics['behavioral_complexity']) > 1:
            indices = list(range(len(self.emergence_metrics['behavioral_complexity'])))

            fig.add_trace(
                go.Scatter(
                    x=indices,
                    y=self.emergence_metrics['behavioral_complexity'],
                    mode='lines',
                    name='Behavioral Complexity',
                    line=dict(color='orange')
                ),
                row=2, col=2
            )

            fig.add_trace(
                go.Scatter(
                    x=indices,
                    y=self.emergence_metrics['territorial_stability'],
                    mode='lines',
                    name='Territorial Stability',
                    line=dict(color='green')
                ),
                row=2, col=2
            )

            fig.add_trace(
                go.Scatter(
                    x=indices,
                    y=self.emergence_metrics['genetic_drift'],
                    mode='lines',
                    name='Genetic Diversity',
                    line=dict(color='purple')
                ),
                row=2, col=2
            )

            fig.update_xaxes(title_text="Time Steps", row=2, col=2)
            fig.update_yaxes(title_text="Metric Value", row=2, col=2)


def run_natural_habitat_simulation():
    """Run complete natural habitat simulation with 3D visualization."""
    print("🌿 Initializing Natural Habitat 3D Simulation")
    print("=" * 60)

    # Create habitat
    habitat = NaturalHabitat3D(habitat_size=(12.0, 10.0, 6.0))

    # Add initial population with natural diversity
    print("🦗 Adding initial species population...")

    # Add Araneae (spiders)
    for _ in range(3):
        spider = Araneae_PsiQRH(AraneaeDNA())
        habitat.add_specimen(spider)

    # Add Chrysopidae (lacewings)
    for _ in range(2):
        lacewing = Chrysopidae(ChrysopidaeDNA())
        habitat.add_specimen(lacewing)

    print(f"✅ Initial population: {len(habitat.specimens)} specimens")

    # Run simulation
    print("\n🔄 Running natural habitat simulation...")
    simulation_steps = 50
    dt = 0.2

    for step in range(simulation_steps):
        habitat.simulate_natural_behaviors(dt)

        if step % 10 == 0:
            print(f"   Step {step}/{simulation_steps}: "
                  f"Population = {len(habitat.specimens)}, "
                  f"Avg Energy = {np.mean([s['energy'] for s in habitat.specimens]):.2f}")

    print("✅ Simulation completed successfully!")

    # Create visualization
    print("\n📊 Generating 3D habitat visualization...")
    fig = habitat.create_3d_visualization()

    # Save visualization
    html_file = "natural_habitat_3d.html"
    fig.write_html(html_file)
    print(f"✅ 3D visualization saved: {html_file}")

    # Display summary
    print("\n📈 Natural Emergence Summary:")
    print(f"   Final Population: {len(habitat.specimens)} specimens")
    print(f"   Species Diversity: {len(set(type(s['specimen']).__name__ for s in habitat.specimens))} species")

    if habitat.emergence_metrics['behavioral_complexity']:
        print(f"   Behavioral Complexity: {habitat.emergence_metrics['behavioral_complexity'][-1]:.3f}")
        print(f"   Territorial Stability: {habitat.emergence_metrics['territorial_stability'][-1]:.3f}")
        print(f"   Genetic Diversity: {habitat.emergence_metrics['genetic_drift'][-1]:.3f}")

    # Display spectral environment status
    alpha_activity = np.sum(habitat.spectral_layers['alpha_field'])
    print(f"   4D Spectral Activity (α): {alpha_activity:.2f}")

    return habitat, fig


if __name__ == "__main__":
    run_natural_habitat_simulation()
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_specimen import PsiQRHBase, FractalGenerator, FractalGLS
from .dna import AraneaeDNA
from .communication import PadilhaWave
import sys
import os

# Ensure the root directory is in the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class ScutigeraColeoptrataDNA(AraneaeDNA):
    """
    Scutigera Coleoptrata DNA - House Centipede

    Highly specialized DNA for rapid movement, enhanced sensory perception,
    and multi-limb coordination through fractal neural networks.
    """

    def __init__(self):
        # Initialize base DNA structure
        super().__init__()

        # Centipede-specific genetic traits
        self.species_code = "SCUTIGERA_COLEOPTRATA"

        # Enhanced fractal dimension for complex movement patterns
        self.fractal_dimension = np.random.uniform(2.2, 2.8)  # Higher than spiders

        # Multi-limb coordination parameters
        self.limb_count = 15  # 15 pairs of legs = 30 total limbs
        self.coordination_matrix = self._generate_limb_coordination()

        # Enhanced sensory parameters
        self.antenna_sensitivity = np.random.uniform(0.8, 1.0)  # Very high
        self.visual_acuity = np.random.uniform(0.6, 0.9)  # Good but not exceptional
        self.vibration_sensitivity = np.random.uniform(0.9, 1.0)  # Excellent

        # Speed and agility parameters
        self.max_speed_factor = np.random.uniform(3.0, 5.0)  # Much faster than spiders
        self.acceleration_rate = np.random.uniform(2.5, 4.0)
        self.turning_agility = np.random.uniform(0.8, 1.0)

        # Predatory behavior parameters
        self.hunt_efficiency = np.random.uniform(0.7, 0.9)
        self.stealth_factor = np.random.uniform(0.4, 0.7)  # Less stealthy than spiders
        self.territorial_range = np.random.uniform(1.5, 3.0)  # Smaller territories

        # Communication preferences
        self.communication_preference = "vibration"  # Primary communication method

        # Update alpha with centipede-specific mapping
        self.alpha = self._map_centipede_fractal_to_alpha()

    def _generate_limb_coordination(self) -> np.ndarray:
        """Generate coordination matrix for 30 limbs (15 pairs)"""
        # Create wave-like coordination pattern
        coordination = np.zeros((30, 30))

        for i in range(30):
            for j in range(30):
                # Phase relationship between limbs
                phase_diff = abs(i - j) * (2 * np.pi / 30)
                coordination[i, j] = np.cos(phase_diff + self.fractal_dimension)

        return coordination

    def _map_centipede_fractal_to_alpha(self) -> float:
        """Map centipede fractal dimension to alpha parameter"""
        # Centipedes have more complex neural coordination
        # Higher fractal dimension -> higher alpha for complex processing
        D = self.fractal_dimension
        D_euclidean = 2.5  # Centipede reference dimension
        lambda_coupling = 1.2  # Stronger coupling than spiders

        alpha_base = 1.8  # Higher base than spiders
        complexity_ratio = (D - D_euclidean) / D_euclidean
        alpha = alpha_base * (1 + lambda_coupling * complexity_ratio)

        # Clamp to physical bounds with higher range for centipedes
        return np.clip(alpha, 1.0, 4.0)

    def generate_gls(self) -> FractalGLS:
        """Generate centipede-specific GLS with multi-limb patterns"""
        generator = FractalGenerator(dim=3)

        # Generate complex 3D fractal representing limb coordination
        transforms = self._generate_centipede_transforms()

        for transform in transforms:
            generator.add_transform(transform)

        generator.set_dna_signature(self.get_signature())

        return generator.generate_gls_spectrum(
            n_points=8000,  # More points for complex patterns
            preserve_dna_integrity=True
        )

    def _generate_centipede_transforms(self) -> List[List[float]]:
        """Generate IFS transforms for centipede movement patterns"""
        transforms = []

        # Base body segment transform
        transforms.append([
            0.5, 0.0, 0.0,    # Scale and rotate for main body
            0.0, 0.8, 0.0,
            0.0, 0.0, 0.7,
            0.1, 0.0, 0.0     # Translation
        ])

        # Limb coordination transforms (multiple for wave-like movement)
        for i in range(8):  # 8 coordination patterns
            angle = i * 2 * np.pi / 8
            scale = 0.3 + 0.2 * np.sin(angle + self.fractal_dimension)

            transform = [
                scale * np.cos(angle), -scale * np.sin(angle), 0.0,
                scale * np.sin(angle), scale * np.cos(angle), 0.0,
                0.0, 0.0, scale * 0.8,
                0.2 * np.cos(angle), 0.2 * np.sin(angle), 0.1 * i / 8
            ]
            transforms.append(transform)

        return transforms

    def get_signature(self) -> Tuple:
        """Get unique DNA signature for centipede"""
        return (
            self.species_code,
            round(self.fractal_dimension, 3),
            round(self.alpha, 3),
            self.limb_count,
            round(self.max_speed_factor, 2),
            round(self.antenna_sensitivity, 2),
            tuple(map(lambda x: round(x, 2), self.angles))
        )


class ScutigeraColeoptrata_PsiQRH(PsiQRHBase):
    """
    Scutigera Coleoptrata (House Centipede) - Living ΨQRH Specimen

    Fast, agile predator with 15 pairs of legs and exceptional sensory abilities.
    Specialized for rapid movement and multi-prey hunting in confined spaces.
    """

    def __init__(self, dna: Optional[ScutigeraColeoptrataDNA] = None):
        # Create DNA if not provided
        if dna is None:
            dna = ScutigeraColeoptrataDNA()

        super().__init__(dna)

        # Centipede-specific attributes
        self.species = "Scutigera coleoptrata"
        self.common_name = "House Centipede"
        self.specimen_id = self._generate_specimen_id()

        # Physical characteristics
        self.body_length = np.random.uniform(25, 50)  # mm
        self.leg_span = np.random.uniform(75, 100)    # mm with legs extended
        self.weight = np.random.uniform(0.5, 2.0)     # grams

        # Behavioral state
        self.current_speed = 0.0
        self.direction = np.random.uniform(0, 2 * np.pi)
        self.energy_level = np.random.uniform(0.7, 1.0)
        self.hunting_mode = False
        self.last_prey_detection = None

        # Position in 4D habitat
        self.position = np.random.uniform(0, 20, 4)  # x, y, z, w coordinates
        self.velocity = np.zeros(4)

        # Multi-limb state tracking
        self.limb_phases = np.random.uniform(0, 2*np.pi, 30)  # 30 limbs
        self.limb_coordination_active = True

        # Enhanced sensory systems
        self.sensory_range = 8.0  # meters
        self.vibration_sensors = self._initialize_vibration_sensors()
        self.antenna_state = np.random.uniform(-1, 1, 2)  # Left/right antenna

        # Predatory behavior
        self.prey_preferences = ["small_insects", "spiders", "silverfish", "flies"]
        self.hunt_success_rate = dna.hunt_efficiency
        self.last_successful_hunt = 0

        # Communication system
        self.communication_radius = 5.0
        self.vibration_signature = self._generate_vibration_signature()

        # Health and reproduction
        self.health = 1.0
        self.age = np.random.uniform(0, 365)  # days
        self.reproductive_maturity = 180  # days
        self.mating_readiness = 0.0

        # Living behavior parameters
        self.behavior_state = "exploring"  # exploring, hunting, resting, communicating
        self.behavior_timer = 0.0
        self.decision_threshold = 0.7

        # Initialize ΨQRH processing layers
        self._initialize_neural_layers()

    def _generate_specimen_id(self) -> str:
        """Generate unique specimen identifier"""
        import random
        import string
        return f"SCOL_{random.randint(100000, 999999)}"

    def _initialize_vibration_sensors(self) -> np.ndarray:
        """Initialize distributed vibration sensors along body"""
        # 12 vibration sensors along body segments
        return np.random.uniform(0.8, 1.0, 12)

    def _generate_vibration_signature(self) -> Dict:
        """Generate unique vibration signature for communication"""
        return {
            'base_frequency': 150 + hash(self.specimen_id) % 100,  # 150-250 Hz
            'pulse_pattern': [0.1, 0.05, 0.1, 0.05],  # Short pulses
            'amplitude_modulation': 0.8 + (hash(self.dna.get_signature()) % 100) / 500,
            'phase_signature': self.dna.fractal_dimension * np.pi
        }

    def _initialize_neural_layers(self):
        """Initialize ΨQRH neural processing layers"""
        embed_dim = 32  # Smaller than spiders but highly efficient

        # Import required ΨQRH components
        try:
            sys.path.append('/home/padilha/trabalhos/Reformulating Transformers')
            from qrh_layer import QRHLayer
            from quaternion_operations import QuaternionOperations

            # Multi-limb coordination layer
            self.limb_coordination_layer = QRHLayer(
                embed_dim=embed_dim,
                alpha=self.dna.alpha,
                theta_left=self.dna.angles[0],
                omega_left=self.dna.angles[1],
                phi_left=self.dna.angles[2],
                theta_right=self.dna.angles[3],
                omega_right=self.dna.angles[4],
                phi_right=self.dna.angles[5]
            )

            # Rapid decision processing layer
            self.decision_layer = QRHLayer(
                embed_dim=16,  # Fast processing
                alpha=self.dna.alpha * 1.5,  # Enhanced for quick decisions
                use_learned_rotation=True
            )

            # Sensory integration layer
            self.sensory_layer = QRHLayer(
                embed_dim=24,
                alpha=self.dna.alpha * 0.8,  # Stable sensory processing
                spatial_dims=(int(self.sensory_range), int(self.sensory_range))
            )

        except ImportError:
            print(f"Warning: Could not import ΨQRH layers for {self.specimen_id}")
            self.limb_coordination_layer = None
            self.decision_layer = None
            self.sensory_layer = None

    def update_living_state(self, environment: Dict, time_step: float = 0.1):
        """
        Update living state - this makes the centipede truly alive
        """
        self.behavior_timer += time_step

        # Update limb coordination
        self._update_limb_coordination(time_step)

        # Process sensory input from environment
        sensory_data = self._process_sensory_input(environment)

        # Make behavioral decisions
        self._make_behavioral_decisions(sensory_data, environment)

        # Update physical state
        self._update_physical_state(time_step)

        # Update health and aging
        self._update_health_and_aging(time_step)

        # Generate communication if needed
        self._generate_communication(environment)

    def _update_limb_coordination(self, time_step: float):
        """Update 30-limb coordination using wave patterns"""
        if not self.limb_coordination_active:
            return

        # Update limb phases based on movement speed
        speed_factor = self.current_speed / self.dna.max_speed_factor
        wave_frequency = 8.0 + speed_factor * 12.0  # 8-20 Hz coordination

        phase_increment = wave_frequency * time_step * 2 * np.pi

        for i in range(30):
            # Each limb has a phase offset based on position
            base_phase = i * (2 * np.pi / 30)
            self.limb_phases[i] += phase_increment

            # Apply coordination matrix influence
            coordination_influence = 0.0
            for j in range(30):
                coordination_influence += self.dna.coordination_matrix[i, j] * np.sin(self.limb_phases[j])

            self.limb_phases[i] += coordination_influence * 0.1 * time_step

    def _process_sensory_input(self, environment: Dict) -> torch.Tensor:
        """Process multi-modal sensory input"""
        # Create sensory input tensor
        sensory_features = []

        # Visual input (limited for centipedes)
        visual_input = environment.get('light_intensity', 0.5)
        sensory_features.extend([visual_input, visual_input * 0.8])

        # Vibration detection (primary sense)
        vibrations = environment.get('vibrations', np.zeros(12))
        if len(vibrations) != 12:
            vibrations = np.random.uniform(0, 0.1, 12)  # Background vibrations
        sensory_features.extend(vibrations.tolist())

        # Chemical detection (antenna)
        chemical_traces = environment.get('chemical_traces', np.zeros(4))
        if len(chemical_traces) != 4:
            chemical_traces = np.random.uniform(0, 0.2, 4)
        sensory_features.extend(chemical_traces.tolist())

        # Air movement detection
        air_currents = environment.get('air_currents', np.zeros(3))
        if len(air_currents) != 3:
            air_currents = np.random.uniform(-0.1, 0.1, 3)
        sensory_features.extend(air_currents.tolist())

        # Convert to tensor
        sensory_tensor = torch.tensor(sensory_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Process through GLS and sensory layer
        if self.gls_visual_layer and self.sensory_layer:
            try:
                gls_processed = self.gls_visual_layer.transform(sensory_tensor)
                neural_processed = self.sensory_layer(gls_processed)
                return neural_processed
            except Exception:
                return sensory_tensor

        return sensory_tensor

    def _make_behavioral_decisions(self, sensory_data: torch.Tensor, environment: Dict):
        """Make living behavioral decisions based on sensory input"""
        if self.decision_layer is None:
            self._fallback_decision_making(environment)
            return

        try:
            # Process through decision layer
            decision_output = self.decision_layer(sensory_data)
            decision_values = decision_output.squeeze().detach().numpy()

            # Extract decision parameters
            if len(decision_values) >= 4:
                exploration_drive = decision_values[0]
                hunting_drive = decision_values[1]
                fear_response = decision_values[2]
                social_drive = decision_values[3]
            else:
                # Fallback if output is wrong size
                exploration_drive = np.random.uniform(0.3, 0.7)
                hunting_drive = np.random.uniform(0.2, 0.8)
                fear_response = np.random.uniform(0.1, 0.4)
                social_drive = np.random.uniform(0.1, 0.3)

            # Make behavioral decisions
            if fear_response > 0.7:
                self.behavior_state = "fleeing"
                self.current_speed = self.dna.max_speed_factor * 0.9
                self._change_direction_randomly()

            elif hunting_drive > 0.6 and self.energy_level > 0.4:
                self.behavior_state = "hunting"
                self.hunting_mode = True
                self.current_speed = self.dna.max_speed_factor * 0.7
                self._hunt_for_prey(environment)

            elif exploration_drive > 0.5:
                self.behavior_state = "exploring"
                self.hunting_mode = False
                self.current_speed = self.dna.max_speed_factor * 0.3
                self._explore_territory()

            else:
                self.behavior_state = "resting"
                self.current_speed *= 0.9  # Slow down

        except Exception:
            self._fallback_decision_making(environment)

    def _fallback_decision_making(self, environment: Dict):
        """Fallback decision making without neural layers"""
        # Simple behavioral state machine
        if self.behavior_timer > 5.0:  # Change behavior every 5 seconds
            behaviors = ["exploring", "hunting", "resting"]
            weights = [0.5, 0.3, 0.2]

            if self.energy_level < 0.3:
                weights = [0.2, 0.1, 0.7]  # Rest more when tired
            elif self.hunting_mode:
                weights = [0.2, 0.7, 0.1]  # Continue hunting

            self.behavior_state = np.random.choice(behaviors, p=weights)
            self.behavior_timer = 0.0

    def _hunt_for_prey(self, environment: Dict):
        """Hunt for prey using enhanced sensory abilities"""
        # Look for prey in sensory range
        prey_detected = False
        prey_positions = environment.get('prey_positions', [])

        for prey_pos in prey_positions:
            distance = np.linalg.norm(np.array(prey_pos[:3]) - self.position[:3])
            if distance <= self.sensory_range:
                # Calculate approach vector
                approach_vector = np.array(prey_pos[:3]) - self.position[:3]
                approach_vector = approach_vector / np.linalg.norm(approach_vector)

                # Update direction toward prey
                self.direction = np.arctan2(approach_vector[1], approach_vector[0])
                self.current_speed = self.dna.max_speed_factor * 0.8
                prey_detected = True
                break

        if not prey_detected:
            # Search pattern - spiral movement
            self.direction += np.random.uniform(-0.5, 0.5)
            self.current_speed = self.dna.max_speed_factor * 0.4

    def _explore_territory(self):
        """Explore territory with characteristic centipede movement"""
        # Change direction occasionally
        if np.random.random() < 0.1:
            self.direction += np.random.uniform(-np.pi/4, np.pi/4)

        # Maintain moderate speed
        self.current_speed = self.dna.max_speed_factor * 0.3

    def _change_direction_randomly(self):
        """Change direction randomly (escape behavior)"""
        self.direction += np.random.uniform(-np.pi, np.pi)

    def _update_physical_state(self, time_step: float):
        """Update physical position and state"""
        # Update velocity based on direction and speed
        self.velocity[0] = self.current_speed * np.cos(self.direction)
        self.velocity[1] = self.current_speed * np.sin(self.direction)
        self.velocity[2] = 0.0  # Centipedes stay on surfaces
        self.velocity[3] = 0.1 * np.sin(self.position[0] * 0.1)  # 4D component

        # Update position
        self.position += self.velocity * time_step

        # Keep within habitat bounds (20x15x10x6)
        self.position[0] = np.clip(self.position[0], 0, 20)
        self.position[1] = np.clip(self.position[1], 0, 15)
        self.position[2] = np.clip(self.position[2], 0, 1)   # Stay near ground
        self.position[3] = np.clip(self.position[3], 0, 6)

        # Decrease speed gradually (friction)
        self.current_speed *= 0.95

    def _update_health_and_aging(self, time_step: float):
        """Update health, aging, and reproduction status"""
        # Age the centipede
        self.age += time_step / 86400  # Convert seconds to days

        # Energy consumption based on activity
        energy_consumption = 0.001 * time_step  # Base metabolism

        if self.behavior_state == "hunting":
            energy_consumption *= 3.0
        elif self.behavior_state == "fleeing":
            energy_consumption *= 5.0
        elif self.behavior_state == "resting":
            energy_consumption *= 0.5

        self.energy_level -= energy_consumption
        self.energy_level = np.clip(self.energy_level, 0.0, 1.0)

        # Health based on energy and age
        if self.energy_level < 0.2:
            self.health -= 0.001 * time_step
        else:
            self.health += 0.0005 * time_step  # Slow recovery

        self.health = np.clip(self.health, 0.0, 1.0)

        # Update mating readiness
        if self.age > self.reproductive_maturity and self.health > 0.7:
            self.mating_readiness = min(1.0, self.mating_readiness + 0.01 * time_step)
        else:
            self.mating_readiness = max(0.0, self.mating_readiness - 0.005 * time_step)

    def _generate_communication(self, environment: Dict):
        """Generate vibration-based communication"""
        # Communicate if in social mode or mating ready
        if (self.behavior_state == "communicating" or
            self.mating_readiness > 0.8 or
            np.random.random() < 0.05):  # Random communication

            # Generate vibration signal
            signal_strength = self.energy_level * self.dna.antenna_sensitivity

            communication_data = {
                'sender_id': self.specimen_id,
                'signal_type': 'vibration',
                'frequency': self.vibration_signature['base_frequency'],
                'amplitude': signal_strength,
                'position': self.position.copy(),
                'species': self.species,
                'mating_ready': self.mating_readiness > 0.8,
                'health': self.health
            }

            # Add to environment communication
            if 'communications' not in environment:
                environment['communications'] = []
            environment['communications'].append(communication_data)

    def analyze_wave(self, wave: 'PadilhaWave') -> float:
        """Analyze communication wave for compatibility"""
        if not hasattr(wave, 'frequency') or not hasattr(wave, 'amplitude'):
            return 0.0

        # Centipedes prefer similar frequencies
        freq_similarity = 1.0 / (1.0 + abs(wave.frequency - self.vibration_signature['base_frequency']) / 100.0)

        # Consider amplitude compatibility
        amp_compatibility = min(1.0, wave.amplitude / 0.8)

        # DNA-based compatibility through GLS
        if self.gls_visual_layer and hasattr(wave, 'emitter_gls'):
            try:
                gls_compatibility = self.gls_visual_layer.compare(wave.emitter_gls)
            except:
                gls_compatibility = 0.5
        else:
            gls_compatibility = 0.5

        # Combined compatibility score
        total_compatibility = (
            0.4 * freq_similarity +
            0.2 * amp_compatibility +
            0.4 * gls_compatibility
        )

        return np.clip(total_compatibility, 0.0, 1.0)

    def get_status(self) -> Dict:
        """Get current status of the living centipede"""
        return {
            'specimen_id': self.specimen_id,
            'species': self.species,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'behavior_state': self.behavior_state,
            'health': float(self.health),
            'energy_level': float(self.energy_level),
            'age_days': float(self.age),
            'mating_readiness': float(self.mating_readiness),
            'current_speed': float(self.current_speed),
            'direction_radians': float(self.direction),
            'hunting_mode': bool(self.hunting_mode),
            'limb_coordination_active': bool(self.limb_coordination_active),
            'sensory_range': float(self.sensory_range),
            'alpha': float(self.dna.alpha),
            'fractal_dimension': float(self.dna.fractal_dimension),
            'limb_count': int(self.dna.limb_count),
            'max_speed': float(self.dna.max_speed_factor),
            'communication_preference': self.dna.communication_preference,
            'alive': self.health > 0.0
        }

    def __repr__(self) -> str:
        return (f"ScutigeraColeoptrata_PsiQRH(id={self.specimen_id}, "
                f"pos=[{self.position[0]:.1f},{self.position[1]:.1f},{self.position[2]:.1f}], "
                f"behavior={self.behavior_state}, health={self.health:.2f}, "
                f"speed={self.current_speed:.1f}, alpha={self.dna.alpha:.2f})")


# Factory function for easy creation
def create_centipede(custom_dna: Optional[ScutigeraColeoptrataDNA] = None) -> ScutigeraColeoptrata_PsiQRH:
    """Create a new living Scutigera coleoptrata specimen"""
    return ScutigeraColeoptrata_PsiQRH(dna=custom_dna)


# Pre-configured centipede archetypes
def create_fast_hunter() -> ScutigeraColeoptrata_PsiQRH:
    """Create a centipede optimized for high-speed hunting"""
    dna = ScutigeraColeoptrataDNA()
    dna.max_speed_factor = 5.0
    dna.hunt_efficiency = 0.9
    dna.acceleration_rate = 4.0
    dna.alpha = dna._map_centipede_fractal_to_alpha()
    return ScutigeraColeoptrata_PsiQRH(dna)


def create_sensory_specialist() -> ScutigeraColeoptrata_PsiQRH:
    """Create a centipede with enhanced sensory abilities"""
    dna = ScutigeraColeoptrataDNA()
    dna.antenna_sensitivity = 1.0
    dna.vibration_sensitivity = 1.0
    dna.visual_acuity = 0.9
    dna.fractal_dimension = 2.8  # Higher complexity for sensory processing
    dna.alpha = dna._map_centipede_fractal_to_alpha()
    return ScutigeraColeoptrata_PsiQRH(dna)


def create_territorial_guardian() -> ScutigeraColeoptrata_PsiQRH:
    """Create a centipede focused on territorial control"""
    dna = ScutigeraColeoptrataDNA()
    dna.territorial_range = 3.0
    dna.stealth_factor = 0.7
    dna.turning_agility = 1.0
    dna.alpha = dna._map_centipede_fractal_to_alpha()
    return ScutigeraColeoptrata_PsiQRH(dna)
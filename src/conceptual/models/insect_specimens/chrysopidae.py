import torch
import torch.nn as nn
import numpy as np
import random
from .base_specimen import PsiQRHBase
from .dna import AraneaeDNA


class ChrysopidaeDNA(AraneaeDNA):
    """
    Specialized DNA for Chrysopidae (Green Lacewing) with predator-optimized genetics.
    Inherits fractal generation from AraneaeDNA but with predator-specific characteristics.
    """
    def __init__(self):
        # Initialize with predator-optimized IFS coefficients
        # These create sharper, more angular fractals suitable for prey detection
        predator_ifs = [
            # Sharp, angular transforms for movement detection
            [random.uniform(0.3, 0.7), random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2),
             random.uniform(0.3, 0.7), random.uniform(0, 1), random.uniform(0, 1)],
            # High-frequency transforms for fine detail detection
            [random.uniform(0.4, 0.6), random.uniform(0.1, 0.3), random.uniform(-0.1, 0.1),
             random.uniform(0.4, 0.6), random.uniform(0.2, 0.8), random.uniform(0.2, 0.8)]
        ]

        # Predator-optimized rotation angles for enhanced visual acuity
        predator_angles = [random.uniform(0, np.pi / 3) for _ in range(6)]  # Wider range for better detection

        super().__init__()
        self.ifs_coefficients = predator_ifs
        self.rotation_angles = predator_angles

        # Predator-specific attributes
        self.visual_acuity = random.uniform(0.7, 1.0)  # Higher acuity for predators
        self.attack_threshold = random.uniform(0.6, 0.8)  # Varied attack aggressiveness


class Chrysopidae(PsiQRHBase):
    """
    Chrysopidae (Green Lacewing) - GLS Predator Spectrum

    A sophisticated visual predator that uses GLS fractal projection to detect prey.
    The predator's visual system maps sensory input into fractal space, generating
    'prey fractals' that enable detection of movement and potential targets.
    """

    def __init__(self, dna: ChrysopidaeDNA = None, device: str = 'cpu'):
        # Use predator-specific DNA if not provided
        if dna is None:
            dna = ChrysopidaeDNA()

        super().__init__(dna=dna)
        self.device = device

        # Predator-specific attributes
        self.visual_acuity = dna.visual_acuity
        self.attack_threshold = dna.attack_threshold
        self.energy = 1.0
        self.hunt_state = "SEARCH"  # SEARCH, STALK, ATTACK, REST
        self.prey_memory = []  # Memory of recent prey encounters
        self.attack_cooldown = 0

        # Set the collapse function for GLS-driven prey detection
        self.collapse_function = self.prey_collapse

        # Heuristic: maximize successful prey capture while conserving energy
        self.heuristic = "hunt_and_conserve"

    def prey_collapse(self, sensory_input: torch.Tensor) -> dict:
        """
        GLS-driven prey collapse function that projects sensory input into fractal space
        to generate prey detection scores and behavioral decisions.

        Args:
            sensory_input: Environmental sensory data [B, T, F]

        Returns:
            Dictionary containing action and prey analysis
        """
        if self.gls_visual_layer is None:
            return {"action": "SEARCH", "prey_score": 0.0, "confidence": 0.0}

        # Projeta input no espaço GLS → gera "fractal de presa"
        gls_projection = self.gls_visual_layer.project(sensory_input)

        # Analyze the prey fractal for movement and target signatures
        prey_analysis = self._analyze_prey_fractal(gls_projection)

        # Calculate prey detection score using fractal characteristics
        score = torch.sigmoid(gls_projection.sum(dim=-1))
        mean_score = score.mean().item()

        # Enhanced scoring with predator-specific factors
        enhanced_score = mean_score * self.visual_acuity * self._energy_modifier()

        # Decision making based on GLS projection and predator state
        action = self._make_hunting_decision(enhanced_score, prey_analysis)

        # Update predator state
        self._update_predator_state(action, enhanced_score)

        return {
            "action": action,
            "prey_score": enhanced_score,
            "confidence": prey_analysis.get("confidence", 0.0),
            "hunt_state": self.hunt_state,
            "energy": self.energy,
            "fractal_dimension": prey_analysis.get("fractal_dimension", 0.0)
        }

    def _analyze_prey_fractal(self, gls_projection: torch.Tensor) -> dict:
        """
        Analyze the GLS projection to extract prey characteristics from fractal patterns.
        """
        batch_size, seq_len, feature_dim = gls_projection.shape

        # Calculate fractal complexity measures
        variance = torch.var(gls_projection, dim=-1).mean()
        gradient_magnitude = torch.abs(torch.diff(gls_projection, dim=1)).mean() if seq_len > 1 else torch.tensor(0.0)
        spectral_energy = torch.norm(gls_projection, dim=-1).mean()

        # Detect movement patterns in fractal space
        movement_score = gradient_magnitude.item()

        # Assess fractal dimension of detected patterns
        complexity_score = variance.item()

        # Calculate confidence based on fractal coherence
        coherence = 1.0 / (1.0 + torch.std(gls_projection).item())
        confidence = coherence * self.visual_acuity

        # Predator-specific pattern recognition
        # Look for characteristics typical of prey (irregular movement, medium complexity)
        prey_likelihood = self._calculate_prey_likelihood(movement_score, complexity_score, spectral_energy.item())

        return {
            "movement_score": movement_score,
            "complexity_score": complexity_score,
            "spectral_energy": spectral_energy.item(),
            "confidence": confidence,
            "prey_likelihood": prey_likelihood,
            "fractal_dimension": self.gls_visual_layer.fractal_dimension
        }

    def _calculate_prey_likelihood(self, movement: float, complexity: float, energy: float) -> float:
        """
        Calculate likelihood that detected fractal patterns represent prey.
        """
        # Prey typically shows:
        # - Moderate movement (not too fast, not stationary)
        # - Medium complexity (not too simple, not too chaotic)
        # - Appropriate energy levels

        # Optimal ranges for prey detection
        optimal_movement = 0.3
        optimal_complexity = 0.5
        optimal_energy = 1.0

        # Calculate distance from optimal prey characteristics
        movement_fitness = np.exp(-2 * (movement - optimal_movement)**2)
        complexity_fitness = np.exp(-2 * (complexity - optimal_complexity)**2)
        energy_fitness = np.exp(-1 * (energy - optimal_energy)**2)

        # Combined prey likelihood
        prey_likelihood = (movement_fitness * complexity_fitness * energy_fitness) ** (1/3)

        return float(prey_likelihood)

    def _energy_modifier(self) -> float:
        """Calculate energy-based modification to hunting efficiency."""
        if self.energy > 0.7:
            return 1.2  # High energy = enhanced hunting
        elif self.energy > 0.3:
            return 1.0  # Normal energy = normal hunting
        else:
            return 0.6  # Low energy = reduced hunting efficiency

    def _make_hunting_decision(self, prey_score: float, prey_analysis: dict) -> str:
        """
        Make hunting decision based on prey score and analysis.
        """
        # Update attack cooldown
        if self.attack_cooldown > 0:
            self.attack_cooldown -= 1

        # Energy-based decision modification
        energy_threshold_modifier = (2.0 - self.energy)  # Lower energy = higher threshold needed
        effective_threshold = self.attack_threshold * energy_threshold_modifier

        # Decision logic based on GLS projection analysis
        if prey_score > effective_threshold and self.attack_cooldown == 0:
            if prey_analysis["confidence"] > 0.7:
                return "ATTACK"
            else:
                return "STALK"  # High score but low confidence = stalk for better position
        elif prey_score > 0.3:
            return "SEARCH"  # Some potential = keep searching
        else:
            if self.energy < 0.4:
                return "REST"  # Low energy = rest
            else:
                return "SEARCH"

    def _update_predator_state(self, action: str, prey_score: float):
        """Update predator internal state based on action taken."""
        if action == "ATTACK":
            self.hunt_state = "ATTACK"
            self.energy -= 0.3  # Attacking costs energy
            self.attack_cooldown = 3  # Cooldown before next attack

            # Success probability based on prey score
            success_prob = min(0.9, prey_score)
            if random.random() < success_prob:
                self.energy += 0.5  # Successful hunt restores energy

        elif action == "STALK":
            self.hunt_state = "STALK"
            self.energy -= 0.1  # Stalking costs less energy

        elif action == "SEARCH":
            self.hunt_state = "SEARCH"
            self.energy -= 0.05  # Searching costs minimal energy

        elif action == "REST":
            self.hunt_state = "REST"
            self.energy += 0.2  # Resting restores energy

        # Energy bounds
        self.energy = np.clip(self.energy, 0.0, 1.0)

    def forward(self, sensory_input: torch.Tensor) -> dict:
        """
        Main forward pass for the Chrysopidae predator.
        Processes sensory input through GLS and returns hunting decision.
        """
        # Process through GLS visual system
        if self.gls_visual_layer is not None:
            transformed = self.gls_visual_layer.transform(sensory_input)

            # Apply prey collapse function
            if self.collapse_function is not None:
                result = self.collapse_function(transformed)
            else:
                result = {"action": "SEARCH", "prey_score": 0.0}
        else:
            result = {"action": "SEARCH", "prey_score": 0.0}

        # Add predator status information
        result.update({
            "species": "Chrysopidae",
            "visual_acuity": self.visual_acuity,
            "attack_threshold": self.attack_threshold,
            "gls_dimension": self.gls_visual_layer.fractal_dimension if self.gls_visual_layer else 0.0
        })

        return result

    def get_predator_stats(self) -> dict:
        """Get detailed predator statistics for analysis."""
        return {
            "species": "Chrysopidae",
            "energy": self.energy,
            "hunt_state": self.hunt_state,
            "visual_acuity": self.visual_acuity,
            "attack_threshold": self.attack_threshold,
            "attack_cooldown": self.attack_cooldown,
            "gls_fractal_dimension": self.gls_visual_layer.fractal_dimension if self.gls_visual_layer else 0.0,
            "gls_hash": self.gls_visual_layer.spectrum_hash % 10000 if self.gls_visual_layer else 0
        }

    def __repr__(self) -> str:
        return (f"Chrysopidae(energy={self.energy:.2f}, state={self.hunt_state}, "
                f"acuity={self.visual_acuity:.2f}, threshold={self.attack_threshold:.2f})")


# Backward compatibility
class Chrysopidae_PsiQRH(Chrysopidae):
    """Legacy class name for backward compatibility"""
    pass


def create_chrysopidae_specimen():
    """Factory function to create a Chrysopidae specimen"""
    return Chrysopidae()
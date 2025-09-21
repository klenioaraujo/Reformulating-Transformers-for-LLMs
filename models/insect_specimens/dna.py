import numpy as np
import random
import sys
import os
from dataclasses import dataclass, field

# Ensure the root directory is in the path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from needle_fractal_dimension import FractalGenerator
from qrh_layer import QRHConfig

@dataclass
class AraneaeDNA:
    """
    Represents the genetic code of a spider agent.
    This DNA is used to generate a specific QRHConfig for the agent's QRHLayer,
    making the layer's behavior an emergent property of the agent's genetics.
    """
    # Genetic sequence for the fractal signature (Iterated Function System coefficients)
    ifs_coefficients: list = field(default_factory=lambda: 
        [
            [random.uniform(0.4, 0.6), 0, 0, random.uniform(0.4, 0.6), random.uniform(0, 0.5), 0],
            [random.uniform(0.4, 0.6), 0, 0, random.uniform(0.4, 0.6), random.uniform(0.5, 1), random.uniform(0.3, 0.7)]
        ])

    # Genetic sequence for the 4D Unitary Rotation (6 angles)
    rotation_angles: list = field(default_factory=lambda: 
        [random.uniform(0, np.pi / 4) for _ in range(6)]) # Smaller initial rotations for stability

    def create_config(self, embed_dim: int, device: str) -> QRHConfig:
        """
        Translates this DNA into a QRHConfig object for the QRHLayer.
        This is the bridge between the agent's genetics and its processing physics.
        """
        # 1. Derive alpha from the fractal dimension
        fractal_dimension = self._calculate_fractal_dimension()
        alpha = self._map_dimension_to_alpha(fractal_dimension)

        # 2. Unpack the rotation angles
        theta_l, omega_l, phi_l, theta_r, omega_r, phi_r = self.rotation_angles

        # 3. Create and return the QRHConfig object
        return QRHConfig(
            embed_dim=embed_dim,
            alpha=alpha,
            theta_left=theta_l,
            omega_left=omega_l,
            phi_left=phi_l,
            theta_right=theta_r,
            omega_right=omega_r,
            phi_right=phi_r,
            device=device
        )

    def _calculate_fractal_dimension(self, n_points=5000) -> float:
        """Uses the FractalGenerator to calculate the dimension D from the IFS DNA."""
        try:
            generator = FractalGenerator(dim=2)
            for transform in self.ifs_coefficients:
                generator.add_transform(transform)
            
            points = generator.generate(n_points=n_points, warmup=100)
            dimension = generator.calculate_fractal_dimension(method='boxcount')
            
            return dimension if np.isfinite(dimension) and dimension > 0 else 1.5
        except Exception:
            return 1.5 # Fallback dimension

    def _map_dimension_to_alpha(self, dimension: float) -> float:
        """Maps the fractal dimension D to the alpha parameter."""
        alpha_0 = 1.5
        lambda_coupling = 0.8
        euclidean_dim = 2.0
        complexity_ratio = (dimension - euclidean_dim) / euclidean_dim
        alpha = alpha_0 * (1 + lambda_coupling * complexity_ratio)
        return np.clip(alpha, 0.1, 3.0)

    @staticmethod
    def crossover(dna1: 'AraneaeDNA', dna2: 'AraneaeDNA') -> 'AraneaeDNA':
        """Performs genetic crossover between two parents' DNA."""
        child_ifs = []
        for i in range(len(dna1.ifs_coefficients)):
            coeffs1 = np.array(dna1.ifs_coefficients[i])
            coeffs2 = np.array(dna2.ifs_coefficients[i])
            child_ifs.append(((coeffs1 + coeffs2) / 2).tolist())

        child_angles = []
        for i in range(len(dna1.rotation_angles)):
            child_angles.append(random.choice([dna1.rotation_angles[i], dna2.rotation_angles[i]]))

        return AraneaeDNA(ifs_coefficients=child_ifs, rotation_angles=child_angles)

    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.05):
        """Applies random mutations to the DNA."""
        for i in range(len(self.ifs_coefficients)):
            for j in range(len(self.ifs_coefficients[i])):
                if random.random() < mutation_rate:
                    self.ifs_coefficients[i][j] += random.uniform(-mutation_strength, mutation_strength)

        for i in range(len(self.rotation_angles)):
            if random.random() < mutation_rate:
                self.rotation_angles[i] += random.uniform(-mutation_strength * np.pi, mutation_strength * np.pi)
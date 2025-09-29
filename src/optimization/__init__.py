"""
Optimization modules for ΨQRH Transformer
"""

from .energy_normalizer import energy_preserve
from .advanced_energy_controller import AdvancedEnergyController
from .spectral_normalizer import (
    normalize_spectral_magnitude,
    spectral_operation_with_unitarity,
    SpectralUnitarityPreserver
)

__all__ = [
    "energy_preserve",
    "AdvancedEnergyController",
    "normalize_spectral_magnitude",
    "spectral_operation_with_unitarity",
    "SpectralUnitarityPreserver"
]
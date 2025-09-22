import torch
import torch.nn as nn
import numpy as np
import sys
import os
import pickle
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass

# Ensure the root directory is in the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class IFS:
    """
    Iterated Function System (IFS) - Individual transformation in the fractal generation.
    Each IFS represents a single affine transformation that contributes to the overall fractal pattern.
    """
    def __init__(self, coefficients: List[float]):
        """
        Initialize IFS with affine transformation coefficients.
        For 3D: coefficients = [a11, a12, a13, a21, a22, a23, a31, a32, a33, tx, ty, tz]
        Representing: [3x3 transformation matrix] + [3D translation vector]
        """
        self.coefficients = np.array(coefficients, dtype=np.float64)
        self.dim = self._infer_dimension()
        self.matrix, self.translation = self._parse_coefficients()

    def _infer_dimension(self) -> int:
        """Infer dimension from coefficient count"""
        n_coeffs = len(self.coefficients)
        if n_coeffs == 6:  # 2D: 4 matrix + 2 translation
            return 2
        elif n_coeffs == 12:  # 3D: 9 matrix + 3 translation
            return 3
        else:
            raise ValueError(f"Invalid coefficient count: {n_coeffs}. Expected 6 (2D) or 12 (3D)")

    def _parse_coefficients(self) -> Tuple[np.ndarray, np.ndarray]:
        """Parse coefficients into transformation matrix and translation vector"""
        if self.dim == 2:
            # 2D: [a, b, c, d, tx, ty] -> [[a, b], [c, d]] + [tx, ty]
            matrix = self.coefficients[:4].reshape(2, 2)
            translation = self.coefficients[4:6]
        elif self.dim == 3:
            # 3D: [a11, a12, a13, a21, a22, a23, a31, a32, a33, tx, ty, tz]
            matrix = self.coefficients[:9].reshape(3, 3)
            translation = self.coefficients[9:12]

        return matrix, translation

    def transform(self, point: np.ndarray) -> np.ndarray:
        """Apply the affine transformation to a point"""
        return self.matrix @ point + self.translation

    def __repr__(self) -> str:
        return f"IFS({self.dim}D, matrix={self.matrix.tolist()}, translation={self.translation.tolist()})"

class FractalGenerator:
    """
    Enhanced 3D Fractal Generator with GLS spectrum generation.
    Completely reusable and isolated - each instance is self-contained and serializable.
    """
    def __init__(self, dim: int = 3):
        """
        Initialize the fractal generator.
        Args:
            dim: Dimension (2 or 3) for fractal generation
        """
        if dim not in [2, 3]:
            raise ValueError("Dimension must be 2 or 3")

        self.dim = dim
        self.transforms: List[IFS] = []
        self.dna_signature: Optional[Tuple] = None
        self._cached_points: Optional[np.ndarray] = None
        self._cached_dimension: Optional[float] = None

    def add_transform(self, coeffs: List[float]) -> None:
        """
        Add an IFS transformation to the fractal generator.
        Args:
            coeffs: Transformation coefficients (6 for 2D, 12 for 3D)
        """
        ifs = IFS(coefficients=coeffs)
        if ifs.dim != self.dim:
            raise ValueError(f"IFS dimension {ifs.dim} doesn't match generator dimension {self.dim}")
        self.transforms.append(ifs)

    def set_dna_signature(self, signature: Tuple) -> None:
        """Set the DNA signature for integrity preservation"""
        self.dna_signature = signature

    def generate_gls_spectrum(self, n_points: int = 5000, preserve_dna_integrity: bool = True) -> 'FractalGLS':
        """
        Generate the complete GLS spectrum from the fractal transformations.

        Args:
            n_points: Number of points to generate for the fractal
            preserve_dna_integrity: Whether to normalize and preserve DNA integrity

        Returns:
            FractalGLS instance containing the complete visual spectrum
        """
        if not self.transforms:
            raise ValueError("No transformations defined. Add transforms first.")

        # Generate fractal points via IFS iteration
        points = self._iterate_ifs(n_points)

        if preserve_dna_integrity:
            points = self._normalize_and_preserve(points)

        # Calculate fractal dimension for spectrum scaling
        fractal_dimension = self._calculate_fractal_dimension(points)

        # Create complete GLS spectrum
        gls = FractalGLS(
            fractal_points=points,
            fractal_dimension=fractal_dimension,
            dna_signature=self.dna_signature or self._generate_signature()
        )

        return gls

    def _iterate_ifs(self, n_points: int) -> np.ndarray:
        """
        Generate fractal points using the chaos game algorithm with IFS transformations.
        """
        if not self.transforms:
            raise ValueError("No transformations available")

        points = np.zeros((n_points, self.dim))

        # Start with random initial point
        current_point = np.random.rand(self.dim)

        # Warmup iterations to reach the attractor
        warmup = min(1000, n_points // 10)
        for _ in range(warmup):
            # Randomly select a transformation
            transform = np.random.choice(self.transforms)
            current_point = transform.transform(current_point)

        # Generate the actual fractal points
        for i in range(n_points):
            # Randomly select a transformation (chaos game)
            transform = np.random.choice(self.transforms)
            current_point = transform.transform(current_point)
            points[i] = current_point

        self._cached_points = points
        return points

    def _normalize_and_preserve(self, points: np.ndarray) -> np.ndarray:
        """
        Normalize points while preserving DNA integrity and fractal structure.
        """
        # Calculate bounds
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        ranges = max_vals - min_vals

        # Avoid division by zero
        ranges = np.where(ranges < 1e-10, 1.0, ranges)

        # Normalize to [0, 1] while preserving relative structure
        normalized = (points - min_vals) / ranges

        # Apply DNA signature preservation if available
        if self.dna_signature is not None:
            # Use DNA signature to create consistent scaling
            signature_hash = hash(str(self.dna_signature)) % 1000000
            scale_factor = 0.8 + 0.4 * (signature_hash / 1000000)  # Scale factor between 0.8 and 1.2
            normalized = normalized * scale_factor

        return normalized

    def _calculate_fractal_dimension(self, points: np.ndarray) -> float:
        """
        Calculate fractal dimension using box-counting method.
        Optimized for 3D fractals with DNA-preserving normalization.
        """
        if self._cached_dimension is not None:
            return self._cached_dimension

        try:
            # Normalize points to [0, 1]^dim
            min_vals = np.min(points, axis=0)
            max_vals = np.max(points, axis=0)
            ranges = max_vals - min_vals
            ranges = np.where(ranges < 1e-10, 1.0, ranges)
            points_norm = (points - min_vals) / ranges

            # Box counting scales
            scales = np.logspace(-2, 0, 15)  # From 0.01 to 1.0
            counts = []

            for scale in scales:
                # Create grid
                grid_size = max(2, int(1/scale))

                if self.dim == 2:
                    grid = np.zeros((grid_size, grid_size), dtype=bool)
                    indices = (points_norm[:, :2] * (grid_size - 1)).astype(int)
                    indices = np.clip(indices, 0, grid_size - 1)
                    grid[indices[:, 0], indices[:, 1]] = True

                elif self.dim == 3:
                    grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
                    indices = (points_norm * (grid_size - 1)).astype(int)
                    indices = np.clip(indices, 0, grid_size - 1)
                    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True

                counts.append(np.sum(grid))

            # Linear fit in log-log space
            log_scales = np.log(1/scales)
            log_counts = np.log(np.array(counts) + 1)  # Add 1 to avoid log(0)

            # Filter valid points
            valid = np.isfinite(log_counts) & np.isfinite(log_scales) & (np.array(counts) > 0)

            if np.sum(valid) >= 2:
                coeffs = np.polyfit(log_scales[valid], log_counts[valid], 1)
                dimension = coeffs[0]
                # Clamp to reasonable range
                dimension = np.clip(dimension, 0.1, self.dim + 0.5)
            else:
                # Fallback dimension
                dimension = 1.5 if self.dim == 2 else 2.0

            self._cached_dimension = dimension
            return dimension

        except Exception:
            # Robust fallback
            return 1.5 if self.dim == 2 else 2.0

    def _generate_signature(self) -> Tuple:
        """Generate a signature from the current transforms for DNA integrity"""
        if not self.transforms:
            return ()

        signature = []
        for transform in self.transforms:
            signature.append(tuple(transform.coefficients.tolist()))

        return tuple(signature)

    def serialize(self) -> bytes:
        """Serialize the entire fractal generator for complete isolation"""
        data = {
            'dim': self.dim,
            'transforms': [ifs.coefficients.tolist() for ifs in self.transforms],
            'dna_signature': self.dna_signature,
            'cached_points': self._cached_points.tolist() if self._cached_points is not None else None,
            'cached_dimension': self._cached_dimension
        }
        return pickle.dumps(data)

    @classmethod
    def deserialize(cls, data: bytes) -> 'FractalGenerator':
        """Deserialize to recreate the exact fractal generator"""
        obj_data = pickle.loads(data)

        generator = cls(dim=obj_data['dim'])

        # Restore transforms
        for coeffs in obj_data['transforms']:
            generator.add_transform(coeffs)

        # Restore state
        generator.dna_signature = obj_data['dna_signature']
        generator._cached_dimension = obj_data['cached_dimension']

        if obj_data['cached_points'] is not None:
            generator._cached_points = np.array(obj_data['cached_points'])

        return generator

    def __repr__(self) -> str:
        return f"FractalGenerator({self.dim}D, {len(self.transforms)} transforms, signature={self.dna_signature is not None})"

class FractalGLS:
    """
    Enhanced Generalized Light Spectrum (GLS) - 3D fractal-generated visual spectrum.
    Completely isolated and self-contained - can be serialized, compared, combined
    without dependency on global state.
    """
    def __init__(self, fractal_points: np.ndarray, fractal_dimension: float, dna_signature: Tuple):
        # Core data (fully self-contained)
        self.fractal_points = np.array(fractal_points, copy=True)  # Defensive copy
        self.fractal_dimension = float(fractal_dimension)
        self.dna_signature = dna_signature

        # Derived visual data
        self.visual_spectrum = self._generate_spectrum()
        self.spectrum_resolution = self.visual_spectrum.shape[0]

        # Metadata for isolation and comparison
        self.creation_timestamp = self._get_timestamp()
        self.spectrum_hash = self._compute_spectrum_hash()

    def _get_timestamp(self) -> float:
        """Get creation timestamp for tracking"""
        import time
        return time.time()

    def _compute_spectrum_hash(self) -> int:
        """Compute hash of spectrum for fast comparison"""
        return hash(self.visual_spectrum.tobytes())

    def _generate_spectrum(self) -> np.ndarray:
        """Generate the visual spectrum from fractal points with enhanced robustness"""
        if len(self.fractal_points) == 0:
            # Fallback spectrum for empty fractal
            return np.ones((64, 64, 64)) * self.fractal_dimension

        # Handle different dimensions gracefully
        if self.fractal_points.shape[1] == 2:
            # 2D fractal -> extend to 3D spectrum
            points_3d = np.zeros((len(self.fractal_points), 3))
            points_3d[:, :2] = self.fractal_points
            points_3d[:, 2] = 0.5  # Fixed Z coordinate
            fractal_points = points_3d
        else:
            fractal_points = self.fractal_points

        # Robust bounds calculation
        min_vals = np.min(fractal_points, axis=0)
        max_vals = np.max(fractal_points, axis=0)
        ranges = max_vals - min_vals

        # Avoid division by zero and handle degenerate cases
        ranges = np.where(ranges < 1e-10, 1.0, ranges)

        # Normalize to [0, 1] range for spectrum generation
        normalized_points = (fractal_points - min_vals) / ranges

        # Dynamic spectrum resolution based on point density
        point_density = len(fractal_points) / 1000.0
        spectrum_resolution = int(np.clip(32 + 32 * point_density, 32, 128))

        # Initialize spectrum with DNA signature influence
        signature_influence = hash(str(self.dna_signature)) % 100 / 100.0
        spectrum = np.ones((spectrum_resolution, spectrum_resolution, spectrum_resolution)) * signature_influence * 0.1

        # Map fractal points to spectrum grid
        indices = (normalized_points * (spectrum_resolution - 1)).astype(int)
        indices = np.clip(indices, 0, spectrum_resolution - 1)

        # Create density-based spectrum with gradient effects
        for point_idx, (i, j, k) in enumerate(indices):
            # Main density contribution
            spectrum[i, j, k] += 1.0

            # Add gradient effects for smoothness
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    for dk in range(-1, 2):
                        ni, nj, nk = i + di, j + dj, k + dk
                        if 0 <= ni < spectrum_resolution and 0 <= nj < spectrum_resolution and 0 <= nk < spectrum_resolution:
                            distance = np.sqrt(di*di + dj*dj + dk*dk)
                            if distance > 0:
                                spectrum[ni, nj, nk] += 0.3 / distance

        # Apply fractal dimension scaling with DNA preservation
        spectrum = spectrum * self.fractal_dimension

        # Normalize spectrum to [0, 1] range while preserving structure
        if spectrum.max() > 0:
            spectrum = spectrum / spectrum.max()

        return spectrum

    def transform(self, sensory_input: torch.Tensor) -> torch.Tensor:
        """Enhanced transform with optimized spectral convolution"""
        batch_size, seq_len, feature_dim = sensory_input.shape

        # Convert spectrum to torch tensor with device matching
        spectrum_tensor = torch.from_numpy(self.visual_spectrum).float().to(sensory_input.device)

        # Optimized transformation using vectorized operations
        if feature_dim >= 3:
            # 3D feature mapping
            feature_coords = (torch.abs(sensory_input) * (self.spectrum_resolution - 1)).long().clamp(0, self.spectrum_resolution - 1)

            # Batch lookup in spectrum
            spectrum_values = spectrum_tensor[
                feature_coords[:, :, 0],
                feature_coords[:, :, 1],
                feature_coords[:, :, 2]
            ]

            # Apply transformation
            transformed = sensory_input * spectrum_values.unsqueeze(-1)

        else:
            # Lower dimensional inputs - use spectral sampling
            spectrum_mean = spectrum_tensor.mean()
            spectrum_std = spectrum_tensor.std()

            # Create position-dependent sampling
            position_factor = torch.arange(seq_len, device=sensory_input.device).float() / seq_len
            spectrum_sample = spectrum_mean + spectrum_std * torch.sin(2 * np.pi * position_factor)

            transformed = sensory_input * spectrum_sample.view(1, -1, 1)

        return transformed

    def compare(self, other: 'FractalGLS') -> float:
        """
        Compare two GLS instances for similarity (0.0 to 1.0).
        Enables GLS-based mate selection and similarity analysis.
        """
        if not isinstance(other, FractalGLS):
            return 0.0

        # Fast hash comparison
        if self.spectrum_hash == other.spectrum_hash:
            return 1.0

        # DNA signature similarity
        dna_similarity = self._compare_dna_signatures(other.dna_signature)

        # Spectral correlation
        spectral_similarity = self._compute_spectral_correlation(other.visual_spectrum)

        # Fractal dimension similarity
        dim_diff = abs(self.fractal_dimension - other.fractal_dimension)
        dim_similarity = np.exp(-dim_diff)

        # Weighted combination
        total_similarity = (
            0.4 * dna_similarity +
            0.4 * spectral_similarity +
            0.2 * dim_similarity
        )

        return np.clip(total_similarity, 0.0, 1.0)

    def _compare_dna_signatures(self, other_signature: Tuple) -> float:
        """Compare DNA signatures for genetic similarity"""
        if self.dna_signature == other_signature:
            return 1.0

        try:
            # Convert signatures to comparable format
            sig1_str = str(self.dna_signature)
            sig2_str = str(other_signature)

            # Simple string similarity
            common_chars = len(set(sig1_str) & set(sig2_str))
            total_chars = len(set(sig1_str) | set(sig2_str))

            return common_chars / total_chars if total_chars > 0 else 0.0

        except Exception:
            return 0.0

    def _compute_spectral_correlation(self, other_spectrum: np.ndarray) -> float:
        """Compute correlation between visual spectra"""
        try:
            # Flatten spectra for correlation
            spec1_flat = self.visual_spectrum.flatten()
            spec2_flat = other_spectrum.flatten()

            # Ensure same size
            min_size = min(len(spec1_flat), len(spec2_flat))
            spec1_flat = spec1_flat[:min_size]
            spec2_flat = spec2_flat[:min_size]

            # Compute correlation
            correlation = np.corrcoef(spec1_flat, spec2_flat)[0, 1]

            return max(0.0, correlation) if np.isfinite(correlation) else 0.0

        except Exception:
            return 0.0

    def combine(self, other: 'FractalGLS', ratio: float = 0.5) -> 'FractalGLS':
        """
        Combine two GLS instances for genetic crossover.
        Creates hybrid visual spectrum preserving both DNA signatures.
        """
        ratio = np.clip(ratio, 0.0, 1.0)

        # Combine fractal points
        n_points_1 = int(len(self.fractal_points) * ratio)
        n_points_2 = len(self.fractal_points) - n_points_1

        combined_points = np.vstack([
            self.fractal_points[:n_points_1],
            other.fractal_points[:n_points_2]
        ])

        # Combine fractal dimensions
        combined_dimension = ratio * self.fractal_dimension + (1 - ratio) * other.fractal_dimension

        # Combine DNA signatures
        combined_signature = (self.dna_signature, other.dna_signature, ratio)

        return FractalGLS(
            fractal_points=combined_points,
            fractal_dimension=combined_dimension,
            dna_signature=combined_signature
        )

    def serialize(self) -> bytes:
        """Serialize the complete GLS for storage and transmission"""
        data = {
            'fractal_points': self.fractal_points.tolist(),
            'fractal_dimension': self.fractal_dimension,
            'dna_signature': self.dna_signature,
            'visual_spectrum': self.visual_spectrum.tolist(),
            'creation_timestamp': self.creation_timestamp,
            'spectrum_hash': self.spectrum_hash
        }
        return pickle.dumps(data)

    @classmethod
    def deserialize(cls, data: bytes) -> 'FractalGLS':
        """Deserialize to recreate exact GLS instance"""
        obj_data = pickle.loads(data)

        # Create instance with restored data
        instance = cls.__new__(cls)  # Skip __init__
        instance.fractal_points = np.array(obj_data['fractal_points'])
        instance.fractal_dimension = obj_data['fractal_dimension']
        instance.dna_signature = obj_data['dna_signature']
        instance.visual_spectrum = np.array(obj_data['visual_spectrum'])
        instance.spectrum_resolution = instance.visual_spectrum.shape[0]
        instance.creation_timestamp = obj_data['creation_timestamp']
        instance.spectrum_hash = obj_data['spectrum_hash']

        return instance

    def __eq__(self, other) -> bool:
        """Equality comparison based on spectrum hash"""
        return isinstance(other, FractalGLS) and self.spectrum_hash == other.spectrum_hash

    def __hash__(self) -> int:
        """Hash based on spectrum for set operations"""
        return self.spectrum_hash

    def __repr__(self) -> str:
        return (f"FractalGLS(dim={self.fractal_dimension:.3f}, "
                f"points={len(self.fractal_points)}, "
                f"resolution={self.spectrum_resolution}, "
                f"hash={self.spectrum_hash % 10000})")

    def project(self, sensory_input: torch.Tensor) -> torch.Tensor:
        """
        Project sensory input into GLS fractal space to generate 'prey fractals'.
        This enables predators to detect prey by analyzing how input maps to fractal patterns.
        """
        batch_size, seq_len, feature_dim = sensory_input.shape

        # Convert spectrum to torch tensor
        spectrum_tensor = torch.from_numpy(self.visual_spectrum).float().to(sensory_input.device)

        # Create fractal projection by mapping input through 3D spectrum
        if feature_dim >= 3:
            # Multi-dimensional fractal projection
            # Map input features to 3D spectrum coordinates
            coords = (torch.abs(sensory_input) * (self.spectrum_resolution - 1)).long().clamp(0, self.spectrum_resolution - 1)

            # Sample spectrum values at input coordinates
            spectrum_samples = spectrum_tensor[
                coords[:, :, 0],
                coords[:, :, 1],
                coords[:, :, 2]
            ]

            # Generate prey fractal by combining input with spectrum response
            prey_fractal = sensory_input * spectrum_samples.unsqueeze(-1)

            # Add fractal dimension scaling for predator sensitivity
            prey_fractal = prey_fractal * self.fractal_dimension

        else:
            # Lower dimensional projection - use spectral decomposition
            # Create frequency-based mapping
            freq_indices = torch.arange(seq_len, device=sensory_input.device).float() / seq_len
            spectrum_slice = spectrum_tensor[
                (freq_indices * (self.spectrum_resolution - 1)).long().clamp(0, self.spectrum_resolution - 1),
                self.spectrum_resolution // 2,
                self.spectrum_resolution // 2
            ]

            # Generate prey pattern through spectral convolution
            prey_fractal = sensory_input * spectrum_slice.view(1, -1, 1)
            prey_fractal = prey_fractal * self.fractal_dimension

        # Apply fractal enhancement for predator vision
        # Emphasize high-frequency patterns that indicate movement/prey
        enhanced_fractal = self._enhance_predator_vision(prey_fractal)

        return enhanced_fractal

    def _enhance_predator_vision(self, fractal_projection: torch.Tensor) -> torch.Tensor:
        """
        Enhance fractal projection for predator vision by emphasizing movement patterns.
        """
        # Calculate temporal derivatives to detect movement
        if fractal_projection.size(1) > 1:
            temporal_gradient = torch.diff(fractal_projection, dim=1)
            # Pad to maintain sequence length
            temporal_gradient = torch.cat([temporal_gradient, temporal_gradient[:, -1:]], dim=1)
        else:
            temporal_gradient = fractal_projection

        # Calculate spatial patterns for prey detection
        spatial_variance = torch.var(fractal_projection, dim=-1, keepdim=True)

        # Combine fractal, movement, and spatial information
        enhanced = fractal_projection + 0.3 * temporal_gradient + 0.2 * spatial_variance

        # Apply predator-specific DNA signature enhancement
        if self.dna_signature:
            signature_influence = hash(str(self.dna_signature)) % 1000 / 1000.0
            enhanced = enhanced * (0.8 + 0.4 * signature_influence)

        return enhanced

    def extract_spectral_features(self) -> dict:
        """
        Extract spectral features from the GLS visual spectrum for wave generation.
        Maps fractal points to wave parameters (α, β, ω) while preserving DNA identity.
        """
        # Calculate core spectral characteristics
        spectrum_mean = np.mean(self.visual_spectrum)
        spectrum_std = np.std(self.visual_spectrum)
        spectrum_skewness = self._calculate_skewness(self.visual_spectrum)
        spectrum_kurtosis = self._calculate_kurtosis(self.visual_spectrum)

        # Extract fractal-based wave parameters
        # α (phase parameter) - derived from fractal dimension and DNA signature
        alpha = self.fractal_dimension * np.pi / 4  # Base phase from fractal complexity
        if self.dna_signature:
            signature_hash = hash(str(self.dna_signature))
            alpha += (signature_hash % 1000) / 1000.0 * np.pi / 2  # DNA-specific phase shift

        # β (dispersion parameter) - derived from spectral distribution
        beta = spectrum_std * 0.1  # Spectral spread influences dispersion

        # ω (frequency parameter) - derived from spectral energy and fractal structure
        spectral_energy = np.sum(self.visual_spectrum**2)
        omega = 2 * np.pi * (1.0 + spectral_energy / (spectrum_mean + 1e-6))

        # Additional harmonic parameters from fractal geometry
        # Extract spatial frequency components from fractal points
        spatial_frequencies = self._extract_spatial_frequencies()

        # DNA-preserved spectral signature
        dna_spectral_signature = self._generate_dna_spectral_signature()

        return {
            'alpha': float(alpha),
            'beta': float(beta),
            'omega': float(omega),
            'amplitude': float(spectrum_mean),
            'phase_shift': float(spectrum_skewness * np.pi / 6),
            'harmonics': spatial_frequencies,
            'dna_signature': dna_spectral_signature,
            'fractal_dimension': self.fractal_dimension,
            'spectral_energy': float(spectral_energy),
            'signature_hash': hash(str(self.dna_signature)) if self.dna_signature else 0
        }

    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of the spectral distribution"""
        if data.size == 0:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0

        normalized = (data - mean) / std
        skewness = np.mean(normalized**3)
        return float(skewness)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of the spectral distribution"""
        if data.size == 0:
            return 0.0

        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0

        normalized = (data - mean) / std
        kurtosis = np.mean(normalized**4) - 3  # Excess kurtosis
        return float(kurtosis)

    def _extract_spatial_frequencies(self) -> list:
        """Extract spatial frequency components from fractal points"""
        if len(self.fractal_points) == 0:
            return [1.0, 0.5, 0.25]  # Default harmonics

        # Calculate dominant frequencies in fractal point distribution
        try:
            # Use FFT to extract frequency components from each dimension
            frequencies = []
            for dim in range(min(3, self.fractal_points.shape[1])):
                points_1d = self.fractal_points[:, dim]

                # Ensure we have enough points for FFT
                if len(points_1d) >= 4:
                    fft_result = np.fft.fft(points_1d)
                    power_spectrum = np.abs(fft_result)**2

                    # Find dominant frequency indices
                    dominant_indices = np.argsort(power_spectrum)[-3:]  # Top 3 frequencies
                    dominant_freqs = dominant_indices / len(points_1d)
                    frequencies.extend(dominant_freqs.tolist())

            # Normalize and select top frequencies
            if frequencies:
                frequencies = sorted(set(frequencies))[:5]  # Top 5 unique frequencies
                # Ensure values are in reasonable range
                frequencies = [f for f in frequencies if 0.01 <= f <= 2.0]

            if not frequencies:
                frequencies = [1.0, 0.5, 0.25]  # Default if extraction fails

            return frequencies

        except Exception:
            return [1.0, 0.5, 0.25]  # Robust fallback

    def _generate_dna_spectral_signature(self) -> dict:
        """Generate spectral signature that preserves DNA identity"""
        if not self.dna_signature:
            return {'base_frequency': 1.0, 'harmonic_ratios': [1.0, 0.5], 'phase_pattern': [0.0]}

        # Convert DNA signature to spectral characteristics
        signature_str = str(self.dna_signature)
        signature_hash = hash(signature_str)

        # Generate DNA-specific spectral features
        base_frequency = 0.5 + (signature_hash % 1000) / 1000.0  # 0.5 to 1.5 Hz

        # Generate harmonic ratios from DNA structure
        harmonic_ratios = []
        for i, char in enumerate(signature_str[:5]):  # Use first 5 characters
            char_value = ord(str(char)) % 100
            ratio = 0.1 + (char_value / 100.0) * 0.8  # 0.1 to 0.9
            harmonic_ratios.append(ratio)

        # Generate phase patterns from DNA
        phase_pattern = []
        for i in range(0, len(signature_str), 2):
            if i+1 < len(signature_str):
                char_pair = signature_str[i:i+2]
                phase = (hash(char_pair) % 360) * np.pi / 180  # Phase in radians
                phase_pattern.append(phase)

        return {
            'base_frequency': base_frequency,
            'harmonic_ratios': harmonic_ratios[:3],  # Limit to 3 harmonics
            'phase_pattern': phase_pattern[:4],      # Limit to 4 phase components
            'genetic_complexity': len(signature_str),
            'dna_hash': signature_hash % 10000
        }

    def fractal_dimension_variance(self) -> float:
        """
        Calculate variance in fractal dimension across spectrum regions for stability assessment.
        High variance indicates unstable/chaotic GLS, low variance indicates stable GLS.
        """
        if self.visual_spectrum.size == 0:
            return 1.0  # Maximum instability for empty spectrum

        try:
            # Divide spectrum into regions and calculate local fractal dimensions
            resolution = self.spectrum_resolution
            region_size = max(8, resolution // 8)  # At least 8x8x8 regions

            local_dimensions = []

            # Sample multiple regions across the spectrum
            for i in range(0, resolution - region_size, region_size):
                for j in range(0, resolution - region_size, region_size):
                    for k in range(0, resolution - region_size, region_size):
                        # Extract local region
                        region = self.visual_spectrum[i:i+region_size, j:j+region_size, k:k+region_size]

                        # Calculate local fractal dimension using box-counting
                        local_dim = self._calculate_local_fractal_dimension(region)
                        local_dimensions.append(local_dim)

            if len(local_dimensions) < 2:
                return 0.1  # Stable if insufficient regions to measure variance

            # Calculate variance in local fractal dimensions
            dimension_variance = np.var(local_dimensions)

            # Normalize variance to [0, 1] range
            # Higher variance = more instability
            normalized_variance = min(1.0, dimension_variance)

            return float(normalized_variance)

        except Exception:
            # Robust fallback - assume moderate instability
            return 0.5

    def _calculate_local_fractal_dimension(self, region: np.ndarray) -> float:
        """Calculate fractal dimension for a local region of the spectrum"""
        try:
            if region.size == 0 or np.all(region == 0):
                return 1.0  # Default dimension for empty regions

            # Simplified box-counting for local regions
            region_size = region.shape[0]
            scales = [2, 4, 8, min(16, region_size)]

            counts = []
            for scale in scales:
                if scale >= region_size:
                    continue

                # Count non-empty boxes at this scale
                grid_size = region_size // scale
                count = 0

                for i in range(0, region_size - scale, scale):
                    for j in range(0, region_size - scale, scale):
                        for k in range(0, region_size - scale, scale):
                            box = region[i:i+scale, j:j+scale, k:k+scale]
                            if np.any(box > 0):
                                count += 1

                counts.append(count)

            if len(counts) < 2:
                return self.fractal_dimension  # Use global dimension as fallback

            # Linear fit to estimate local fractal dimension
            log_scales = np.log([s for s in scales[:len(counts)]])
            log_counts = np.log([c + 1 for c in counts])  # Add 1 to avoid log(0)

            if len(log_scales) >= 2:
                coeffs = np.polyfit(log_scales, log_counts, 1)
                local_dimension = -coeffs[0]  # Negative slope gives dimension

                # Clamp to reasonable range
                local_dimension = np.clip(local_dimension, 0.5, 3.0)
                return float(local_dimension)
            else:
                return self.fractal_dimension

        except Exception:
            return self.fractal_dimension  # Safe fallback

    def to_dict(self) -> dict:
        """Export GLS data as dictionary for analysis"""
        return {
            'fractal_dimension': self.fractal_dimension,
            'num_points': len(self.fractal_points),
            'spectrum_shape': self.visual_spectrum.shape,
            'dna_signature': self.dna_signature,
            'creation_timestamp': self.creation_timestamp,
            'spectrum_hash': self.spectrum_hash
        }

class PsiQRHBase(nn.Module):
    """
    Enhanced ΨQRH Base class with native GLS (Generalized Light Spectrum) integration.

    Each specimen is a distinct solution from the ΨQRH solution space,
    optimized by specific evolutionary pressures and grounded in visual fractal spectra.
    """
    def __init__(self, dna=None):
        super().__init__()
        self.dna = dna

        # GLS: Native fractal-generated visual spectrum preserving DNA
        self.gls_visual_layer = None
        if self.dna is not None:
            self.gls_visual_layer = self.dna.generate_gls()

        # ΨQRH components specialized for GLS-driven emergence
        self.collapse_function = None      # Wavefunction collapse through GLS
        self.quantum_basis = None          # Quantum processing of visual spectra
        self.relational_graph = []         # GLS-based interactions
        self.heuristic = None             # Optimization preserving GLS integrity

        # Architecture components
        self.sensory_input = []

    def forward(self, sensory_input):
        """
        Enhanced forward pass with GLS transformation.
        Processes input through the fractal visual spectrum before further processing.
        """
        if self.gls_visual_layer is not None:
            # Transform through GLS first
            transformed = self.gls_visual_layer.transform(sensory_input)

            # Apply collapse function if available
            if self.collapse_function is not None:
                collapsed = self.collapse_function(transformed)
                return collapsed
            else:
                return transformed
        else:
            # Fallback for specimens without DNA/GLS
            if self.collapse_function is not None:
                return self.collapse_function(sensory_input)
            else:
                return sensory_input


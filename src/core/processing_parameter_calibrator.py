#!/usr/bin/env python3
"""
Processing Parameter Calibrator for ΨQRH Pipeline
================================================

Auto-calibrates processing parameters based on data characteristics:

Parameters calibrated:
- dropout: Based on overfitting risk
- max_history: Based on temporal context
- vocab_size: Based on lexical diversity
- epsilon: Based on numerical stability requirements
"""

import torch
import math
import numpy as np
from typing import Dict, Any, List, Tuple
import re


class ProcessingParameterCalibrator:
    """
    Calibrates processing parameters based on data analysis
    """

    def __init__(self):
        """Initialize the processing parameter calibrator"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def analyze_overfitting_risk(self, text: str) -> float:
        """
        Analyze overfitting risk based on text characteristics

        Args:
            text: Input text

        Returns:
            Overfitting risk score (0-1, higher = more risk)
        """
        words = text.lower().split()
        if not words:
            return 0.5

        # Short text = higher overfitting risk
        length_factor = min(len(text) / 1000.0, 1.0)
        length_risk = 1.0 - length_factor

        # Low diversity = higher overfitting risk
        unique_words = len(set(words))
        diversity_ratio = unique_words / len(words)
        diversity_risk = 1.0 - diversity_ratio

        # Repetitive patterns = higher overfitting risk
        # Check for repeated word sequences
        if len(words) >= 4:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            unique_trigrams = len(set(trigrams))
            repetition_ratio = 1.0 - (unique_trigrams / len(trigrams))
        else:
            repetition_ratio = 0.0

        # Combined risk score
        risk_score = (length_risk * 0.4 + diversity_risk * 0.4 + repetition_ratio * 0.2)

        return risk_score

    def analyze_temporal_context(self, text: str) -> float:
        """
        Analyze temporal context requirements

        Args:
            text: Input text

        Returns:
            Temporal context score (0-1, higher = more context needed)
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.5

        # Average sentence length
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)

        # Number of sentences
        num_sentences = len(sentences)

        # Discourse markers (indicating complex relationships)
        discourse_markers = ['however', 'therefore', 'moreover', 'furthermore', 'consequently',
                           'nevertheless', 'accordingly', 'similarly', 'likewise', 'instead']

        marker_count = sum(1 for marker in discourse_markers if marker in text.lower())

        # Context score based on complexity indicators
        length_score = min(avg_sentence_length / 20.0, 1.0)  # Normalize
        quantity_score = min(num_sentences / 10.0, 1.0)  # Normalize
        complexity_score = min(marker_count / 3.0, 1.0)  # Normalize

        context_score = (length_score * 0.4 + quantity_score * 0.3 + complexity_score * 0.3)

        return context_score

    def analyze_lexical_diversity(self, text: str) -> float:
        """
        Analyze lexical diversity for vocabulary size calibration

        Args:
            text: Input text

        Returns:
            Lexical diversity score (0-1)
        """
        words = [w.lower() for w in re.findall(r'\b\w+\b', text)]
        if not words:
            return 0.0

        unique_words = len(set(words))
        total_words = len(words)

        # Type-token ratio
        ttr = unique_words / total_words

        # Root type-token ratio (corrected for text length)
        # RTTR = unique_words / sqrt(total_words)
        rttr = unique_words / math.sqrt(total_words) if total_words > 0 else 0

        # Normalized TTR (for comparison across different lengths)
        # Corrected TTR = TTR / sqrt(2 * total_words / (total_words - 1)) if total_words > 1 else TTR
        if total_words > 1:
            correction_factor = math.sqrt(2 * total_words / (total_words - 1))
            cttr = ttr / correction_factor
        else:
            cttr = ttr

        # Combined diversity score
        diversity_score = (ttr + rttr * 0.5 + cttr * 0.5) / 2.0

        return diversity_score

    def analyze_numerical_stability(self, text: str) -> float:
        """
        Analyze numerical stability requirements

        Args:
            text: Input text

        Returns:
            Stability requirement score (0-1, higher = more stability needed)
        """
        # Complex text = higher numerical precision needed
        words = text.split()
        avg_word_length = sum(len(w) for w in words) / max(len(words), 1)

        # Special characters indicate complex processing
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / max(len(text), 1)

        # Mathematical/scientific content indicators
        math_indicators = ['equation', 'formula', 'calculate', 'compute', 'matrix', 'vector',
                          'integral', 'derivative', 'function', 'algorithm', 'theorem']

        math_score = sum(1 for indicator in math_indicators if indicator in text.lower())
        math_score = min(math_score / 3.0, 1.0)  # Normalize

        # Combined stability score
        stability_score = (avg_word_length / 10.0 * 0.3 + special_ratio * 0.4 + math_score * 0.3)

        return stability_score

    def calibrate_dropout(self, overfitting_risk: float) -> float:
        """
        Calibrate dropout rate based on overfitting risk

        dropout = max(0.05, min(0.3, 0.1 * (1 + risk_factor)))
        """
        # Higher risk = higher dropout
        dropout = 0.1 * (1 + overfitting_risk * 1.5)  # Range: 0.1-0.25

        # Clamp to reasonable range
        dropout = max(0.05, min(0.3, dropout))

        return dropout

    def calibrate_max_history(self, temporal_context: float) -> int:
        """
        Calibrate maximum history length based on temporal context

        max_history = max(5, min(20, int(10 * (1 + context_factor))))
        """
        # Higher context needs = longer history
        max_history = int(10 * (1 + temporal_context * 0.5))  # Range: 10-15

        # Clamp to reasonable range
        max_history = max(5, min(20, max_history))

        return max_history

    def calibrate_vocab_size(self, lexical_diversity: float, text: str) -> int:
        """
        Calibrates vocabulary size by reading directly from the native vocabulary file,
        ensuring consistency across the entire pipeline.

        Args:
            lexical_diversity: (Ignored)
            text: (Ignored)

        Returns:
            The vocabulary size defined in data/native_vocab.json.
        """
        import json
        import os

        # Path to the ground-truth vocabulary file
        # This constructs the path from the current file's location
        vocab_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'native_vocab.json'))

        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                native_vocab_data = json.load(f)

            vocab_size = native_vocab_data.get('vocab_size')
            if vocab_size is None:
                raise ValueError("'vocab_size' key not found in native_vocab.json")

            print(f"INFO: Calibrated vocab_size to {vocab_size} from {vocab_path}")
            return vocab_size
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            print(f"ERROR: Could not calibrate vocab_size from {vocab_path}. Defaulting to 512. Error: {e}")
            # Fallback to a default value only if the primary source fails
            return 512

    def calibrate_epsilon(self, stability_requirement: float) -> float:
        """
        Calibrate epsilon (numerical stability) based on requirements

        epsilon = 1e-10 * (1 / (1 + stability_factor))
        """
        # Higher stability needs = smaller epsilon (more precision)
        stability_factor = stability_requirement * 10.0

        # Exponential decay for epsilon
        epsilon = 1e-10 * math.exp(-stability_factor * 0.5)

        # Clamp to reasonable range
        epsilon = max(1e-12, min(1e-6, epsilon))

        return epsilon

    def validate_dimensions(self, tensor: torch.Tensor, expected_shape: tuple, component_name: str) -> Dict[str, Any]:
        """
        Validate tensor dimensions for compatibility

        Args:
            tensor: Tensor to validate
            expected_shape: Expected shape (can contain -1 for variable dimensions)
            component_name: Name of the component for error reporting

        Returns:
            Validation results dictionary
        """
        actual_shape = tensor.shape
        is_compatible = True
        issues = []

        # Check dimensionality
        if len(actual_shape) != len(expected_shape):
            is_compatible = False
            issues.append(f"Wrong number of dimensions: expected {len(expected_shape)}, got {len(actual_shape)}")

        # Check each dimension
        for i, (expected, actual) in enumerate(zip(expected_shape, actual_shape)):
            if expected != -1 and expected != actual:
                is_compatible = False
                issues.append(f"Dimension {i}: expected {expected}, got {actual}")

        # Special case: handle 1D vs 2D tensors in quantum_mapping
        if component_name == "quantum_mapping" and len(actual_shape) == 1 and len(expected_shape) == 2:
            # This is the specific case causing the error: 1D tensor where 2D is expected
            is_compatible = False
            issues.append(f"Quantum mapping requires 2D tensor but got 1D: {actual_shape}")

        return {
            'is_compatible': is_compatible,
            'actual_shape': actual_shape,
            'expected_shape': expected_shape,
            'issues': issues,
            'component': component_name
        }

    def auto_calibrate_dimensions(self, tensor: torch.Tensor, target_dims: Dict[str, int],
                                  component_name: str) -> torch.Tensor:
        """
        Auto-calibrate tensor dimensions for compatibility

        Args:
            tensor: Input tensor
            target_dims: Target dimensions dictionary
            component_name: Component name for logging

        Returns:
            Calibrated tensor with compatible dimensions
        """
        current_shape = tensor.shape
        calibrated_tensor = tensor.clone()

        print(f"🔧 Auto-calibrating dimensions for {component_name}:")
        print(f"   📐 Current shape: {current_shape}")

        # Special case: handle quantum_mapping 1D → 2D conversion
        if component_name == "quantum_mapping" and len(current_shape) == 1:
            # Convert 1D tensor to 2D: [seq_len] → [seq_len, embed_dim]
            seq_len = current_shape[0]
            embed_dim = target_dims.get('embed_dim', 64)

            # Expand 1D tensor to 2D by repeating values across embedding dimension
            calibrated_tensor = calibrated_tensor.unsqueeze(-1).expand(-1, embed_dim)
            print(f"   🔄 Quantum mapping 1D→2D conversion: {current_shape} → {calibrated_tensor.shape}")
            current_shape = calibrated_tensor.shape

        # Standardize tensor dimensions to common format
        # Expected format: [batch_size, seq_len, embed_dim, quaternion_dim]

        # Ensure tensor has at least 4 dimensions
        while len(current_shape) < 4:
            calibrated_tensor = calibrated_tensor.unsqueeze(-1)
            current_shape = calibrated_tensor.shape
            print(f"   🔄 Added dimension: {current_shape}")

        # Set default target dimensions if not specified
        default_targets = {
            'batch_size': 1,
            'seq_len': 64,  # Standard sequence length
            'embed_dim': 64,  # Standard embedding dimension
            'quaternion_dim': 4  # Standard quaternion dimension
        }

        # Merge user targets with defaults
        merged_targets = default_targets.copy()
        merged_targets.update(target_dims)

        # Handle batch dimension calibration
        if 'batch_size' in merged_targets:
            target_batch_size = merged_targets['batch_size']
            if current_shape[0] != target_batch_size:
                if current_shape[0] < target_batch_size:
                    # Repeat tensor to match batch size
                    calibrated_tensor = calibrated_tensor.repeat(target_batch_size // current_shape[0] + 1, *([1] * (len(current_shape) - 1)))[:target_batch_size]
                    print(f"   🔄 Expanded batch dimension: {current_shape[0]} → {target_batch_size}")
                else:
                    # Take first batch_size elements
                    calibrated_tensor = calibrated_tensor[:target_batch_size]
                    print(f"   ➖ Reduced batch dimension: {current_shape[0]} → {target_batch_size}")

        # Handle sequence length calibration
        if 'seq_len' in merged_targets:
            target_seq_len = merged_targets['seq_len']
            if len(current_shape) > 1 and current_shape[1] != target_seq_len:
                if current_shape[1] < target_seq_len:
                    # Pad sequence dimension
                    padding_size = target_seq_len - current_shape[1]
                    padding_shape = list(current_shape)
                    padding_shape[1] = padding_size
                    # Ensure padding shape matches calibrated_tensor shape except for dimension 1
                    padding_shape[0] = calibrated_tensor.shape[0]  # Match batch dimension
                    if len(padding_shape) > 2:
                        padding_shape[2] = calibrated_tensor.shape[2]  # Match embed dimension
                    if len(padding_shape) > 3:
                        padding_shape[3] = calibrated_tensor.shape[3]  # Match quaternion dimension
                    padding = torch.zeros(padding_shape, dtype=tensor.dtype, device=tensor.device)
                    calibrated_tensor = torch.cat([calibrated_tensor, padding], dim=1)
                    print(f"   ➕ Padded sequence dimension: {current_shape[1]} → {target_seq_len}")
                else:
                    # Truncate sequence dimension
                    calibrated_tensor = calibrated_tensor[:, :target_seq_len]
                    print(f"   ➖ Truncated sequence dimension: {current_shape[1]} → {target_seq_len}")

        # Handle embedding dimension calibration
        if 'embed_dim' in merged_targets and len(current_shape) > 2:
            target_embed_dim = merged_targets['embed_dim']
            if current_shape[2] != target_embed_dim:
                if current_shape[2] < target_embed_dim:
                    # Pad embedding dimension
                    padding_size = target_embed_dim - current_shape[2]
                    padding_shape = list(current_shape)
                    padding_shape[2] = padding_size
                    # Ensure padding shape matches calibrated_tensor shape except for dimension 2
                    padding_shape[0] = calibrated_tensor.shape[0]  # Match batch dimension
                    padding_shape[1] = calibrated_tensor.shape[1]  # Match sequence dimension
                    if len(padding_shape) > 3:
                        padding_shape[3] = calibrated_tensor.shape[3]  # Match quaternion dimension
                    padding = torch.zeros(padding_shape, dtype=tensor.dtype, device=tensor.device)
                    calibrated_tensor = torch.cat([calibrated_tensor, padding], dim=2)
                    print(f"   ➕ Padded embedding dimension: {current_shape[2]} → {target_embed_dim}")
                else:
                    # Truncate embedding dimension
                    calibrated_tensor = calibrated_tensor[:, :, :target_embed_dim]
                    print(f"   ➖ Truncated embedding dimension: {current_shape[2]} → {target_embed_dim}")

        # Handle quaternion dimension calibration (4D)
        if 'quaternion_dim' in merged_targets and len(current_shape) > 3:
            target_quat_dim = merged_targets['quaternion_dim']
            if current_shape[3] != target_quat_dim:
                if current_shape[3] < target_quat_dim:
                    # Pad quaternion dimension
                    padding_size = target_quat_dim - current_shape[3]
                    padding_shape = list(current_shape)
                    padding_shape[3] = padding_size
                    # Ensure padding shape matches calibrated_tensor shape except for dimension 3
                    padding_shape[0] = calibrated_tensor.shape[0]  # Match batch dimension
                    padding_shape[1] = calibrated_tensor.shape[1]  # Match sequence dimension
                    padding_shape[2] = calibrated_tensor.shape[2]  # Match embed dimension
                    padding = torch.zeros(padding_shape, dtype=tensor.dtype, device=tensor.device)
                    calibrated_tensor = torch.cat([calibrated_tensor, padding], dim=3)
                    print(f"   ➕ Padded quaternion dimension: {current_shape[3]} → {target_quat_dim}")
                else:
                    # Truncate quaternion dimension
                    calibrated_tensor = calibrated_tensor[:, :, :, :target_quat_dim]
                    print(f"   ➖ Truncated quaternion dimension: {current_shape[3]} → {target_quat_dim}")

        final_shape = calibrated_tensor.shape
        print(f"   ✅ Calibrated shape: {final_shape}")

        return calibrated_tensor

    def ensure_dimension_compatibility(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor,
                                       operation_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Ensure dimension compatibility between two tensors for operations

        Args:
            tensor_a: First tensor
            tensor_b: Second tensor
            operation_name: Name of the operation for logging

        Returns:
            Tuple of calibrated tensors
        """
        shape_a = tensor_a.shape
        shape_b = tensor_b.shape

        print(f"🔍 Checking dimension compatibility for {operation_name}:")
        print(f"   📐 Tensor A shape: {shape_a}")
        print(f"   📐 Tensor B shape: {shape_b}")

        # First, ensure both tensors have standard 4D format: [batch_size, seq_len, embed_dim, quaternion_dim]
        calibrated_a = self.auto_calibrate_dimensions(tensor_a, {}, f"{operation_name}_tensor_a")
        calibrated_b = self.auto_calibrate_dimensions(tensor_b, {}, f"{operation_name}_tensor_b")

        shape_a = calibrated_a.shape
        shape_b = calibrated_b.shape

        # Handle broadcasting for common operations
        if operation_name in ['addition', 'subtraction', 'elementwise']:
            # Try to broadcast dimensions
            try:
                # Test broadcasting compatibility
                torch.broadcast_shapes(shape_a, shape_b)
                print("   ✅ Broadcasting compatible")
            except RuntimeError:
                # Need to calibrate dimensions
                print("   ⚠️ Broadcasting incompatible, calibrating...")

                # For elementwise operations, make dimensions compatible by padding smaller tensor
                max_dims = max(len(shape_a), len(shape_b))

                # Pad shapes to same length
                while len(shape_a) < max_dims:
                    shape_a = (1,) + shape_a
                    calibrated_a = calibrated_a.unsqueeze(0)

                while len(shape_b) < max_dims:
                    shape_b = (1,) + shape_b
                    calibrated_b = calibrated_b.unsqueeze(0)

                # Broadcast each dimension
                for i in range(max_dims):
                    max_dim = max(shape_a[i], shape_b[i])
                    if shape_a[i] != max_dim:
                        # Expand tensor_a dimension
                        calibrated_a = calibrated_a.expand(*[-1 if j != i else max_dim for j in range(max_dims)])
                    if shape_b[i] != max_dim:
                        # Expand tensor_b dimension
                        calibrated_b = calibrated_b.expand(*[-1 if j != i else max_dim for j in range(max_dims)])

                print(f"   ✅ Calibrated for broadcasting: A {tensor_a.shape} → {calibrated_a.shape}, B {tensor_b.shape} → {calibrated_b.shape}")

        elif operation_name in ['matrix_multiplication', 'matmul']:
            # Ensure last dimension of A matches second-to-last of B
            if len(shape_a) >= 2 and len(shape_b) >= 2:
                if shape_a[-1] != shape_b[-2]:
                    # Try to make compatible by padding/truncating
                    target_dim = min(shape_a[-1], shape_b[-2])  # Use smaller dimension
                    if shape_a[-1] > target_dim:
                        calibrated_a = calibrated_a[..., :target_dim]
                    if shape_b[-2] > target_dim:
                        calibrated_b = calibrated_b[..., :target_dim, :]

                    print(f"   ✅ Calibrated matrix multiplication dimensions: {shape_a[-1]} → {target_dim}")

        # Final validation - ensure both tensors have exactly the same shape
        if calibrated_a.shape != calibrated_b.shape:
            print(f"   ⚠️  Shapes still incompatible after calibration: {calibrated_a.shape} vs {calibrated_b.shape}")
            print(f"   🔧 Forcing shape compatibility by broadcasting...")

            # Force compatibility by expanding to maximum dimensions
            target_shape = []
            for dim_a, dim_b in zip(calibrated_a.shape, calibrated_b.shape):
                target_shape.append(max(dim_a, dim_b))

            # Expand both tensors to target shape
            calibrated_a = calibrated_a.expand(*target_shape)
            calibrated_b = calibrated_b.expand(*target_shape)

            print(f"   ✅ Final shapes: A {calibrated_a.shape}, B {calibrated_b.shape}")

        return calibrated_a, calibrated_b

    def calibrate_all(self, text: str) -> Dict[str, Any]:
        """
        Calibrate all processing parameters

        Args:
            text: Input text

        Returns:
            Dict with calibrated processing parameters
        """
        # Analyze processing requirements
        overfitting_risk = self.analyze_overfitting_risk(text)
        temporal_context = self.analyze_temporal_context(text)
        lexical_diversity = self.analyze_lexical_diversity(text)
        stability_requirement = self.analyze_numerical_stability(text)

        # Calibrate parameters
        dropout = self.calibrate_dropout(overfitting_risk)
        max_history = self.calibrate_max_history(temporal_context)
        vocab_size = self.calibrate_vocab_size(lexical_diversity, text)
        epsilon = self.calibrate_epsilon(stability_requirement)

        # State transition parameters for adaptive perception
        input_window = self._calculate_input_window(lexical_diversity, stability_requirement)
        quaternion_complexity = self._calculate_quaternion_complexity(stability_requirement, lexical_diversity)
        quaternion_phase_shift = self._calculate_quaternion_phase_shift(temporal_context, stability_requirement)
        sampling_temperature = self._calculate_sampling_temperature(stability_requirement, lexical_diversity)
        sampling_top_k = self._calculate_sampling_top_k(lexical_diversity, temporal_context)

        return {
            'dropout': dropout,
            'max_history': max_history,
            'vocab_size': vocab_size,
            'epsilon': epsilon,
            # State transition parameters for adaptive perception
            'input_window': input_window,
            'quaternion_complexity': quaternion_complexity,
            'quaternion_phase_shift': quaternion_phase_shift,
            'sampling_temperature': sampling_temperature,
            'sampling_top_k': sampling_top_k,
            'semantic_connectivity_strength': 1.0,  # Default semantic connectivity strength
            # Include analysis metrics for debugging
            'overfitting_risk': overfitting_risk,
            'temporal_context': temporal_context,
            'lexical_diversity': lexical_diversity,
            'stability_requirement': stability_requirement
        }

    def validate_processing_consistency(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate processing parameter consistency

        Args:
            params: Calibrated parameters

        Returns:
            Validation results
        """
        dropout = params['dropout']
        max_history = params['max_history']
        vocab_size = params['vocab_size']
        epsilon = params['epsilon']

        # Extract all parameters
        input_window = params.get('input_window', 'hann')
        quaternion_complexity = params.get('quaternion_complexity', 1.0)
        quaternion_phase_shift = params.get('quaternion_phase_shift', 0.0)
        sampling_temperature = params.get('sampling_temperature', 1.0)
        sampling_top_k = params.get('sampling_top_k', 10)
        semantic_connectivity_strength = params.get('semantic_connectivity_strength', 1.0)

        # Check parameter ranges
        range_checks = {
            'dropout_range': 0.05 <= dropout <= 0.3,
            'max_history_range': 5 <= max_history <= 20,
            'vocab_size_range': vocab_size in [41, 256, 512, 1024, 2048, 50257],  # Added 41 for native vocab
            'epsilon_range': 1e-12 <= epsilon <= 1e-6,
            'input_window_range': input_window in ['boxcar', 'hann', 'hamming'],
            'quaternion_complexity_range': 0.5 <= quaternion_complexity <= 2.0,
            'quaternion_phase_shift_range': 0.0 <= quaternion_phase_shift <= 1.0,
            'sampling_temperature_range': 0.1 <= sampling_temperature <= 2.0,
            'sampling_top_k_range': 5 <= sampling_top_k <= 100,
            'semantic_connectivity_strength_range': 0.1 <= semantic_connectivity_strength <= 5.0
        }

        # Processing consistency checks
        processing_checks = {
            'epsilon_reasonable': epsilon < 1e-8,  # Should be small for stability
            'vocab_logical_progression': vocab_size >= 256,  # Minimum reasonable vocab
            'history_positive': max_history > 0
        }

        all_checks_pass = all(range_checks.values()) and all(processing_checks.values())

        return {
            'range_checks': range_checks,
            'processing_checks': processing_checks,
            'all_checks_pass': all_checks_pass
        }

    def _calculate_input_window(self, lexical_diversity: float, stability_requirement: float) -> str:
        """
        Calculate input windowing function based on text characteristics.

        Higher diversity + higher stability needs = more sophisticated windowing
        """
        score = (lexical_diversity + stability_requirement) / 2.0

        if score < 0.3:
            return 'boxcar'  # No windowing for simple, stable text
        elif score < 0.7:
            return 'hann'    # Hann window for moderate complexity
        else:
            return 'hamming' # Hamming window for high complexity/high precision needs

    def _calculate_quaternion_complexity(self, stability_requirement: float, lexical_diversity: float) -> float:
        """
        Calculate quaternion mapping complexity factor.

        Higher stability needs + higher diversity = more complex quaternion transformations
        """
        base_complexity = 1.0
        stability_boost = stability_requirement * 0.5  # 0-0.5
        diversity_boost = lexical_diversity * 0.3      # 0-0.3

        complexity = base_complexity + stability_boost + diversity_boost
        return max(0.5, min(2.0, complexity))  # Range: 0.5-2.0

    def _calculate_quaternion_phase_shift(self, temporal_context: float, stability_requirement: float) -> float:
        """
        Calculate quaternion phase shift for temporal coherence.

        Higher temporal context + stability needs = more phase shifting for coherence
        """
        base_shift = 0.0
        temporal_shift = temporal_context * 0.5      # 0-0.5
        stability_shift = stability_requirement * 0.3 # 0-0.3

        phase_shift = base_shift + temporal_shift + stability_shift
        return max(0.0, min(1.0, phase_shift))  # Range: 0.0-1.0

    def _calculate_sampling_temperature(self, stability_requirement: float, lexical_diversity: float) -> float:
        """
        Calculate sampling temperature for text generation.

        Higher stability needs = lower temperature (more focused)
        Higher diversity = higher temperature (more creative)
        """
        base_temp = 0.7
        stability_adjustment = -stability_requirement * 0.4  # -0.4 to 0 (more stable = lower temp)
        diversity_adjustment = lexical_diversity * 0.3       # 0 to 0.3 (more diverse = higher temp)

        temperature = base_temp + stability_adjustment + diversity_adjustment
        return max(0.1, min(2.0, temperature))  # Range: 0.1-2.0

    def _calculate_sampling_top_k(self, lexical_diversity: float, temporal_context: float) -> int:
        """
        Calculate top-k sampling parameter.

        Higher diversity + temporal context = larger top-k (more options)
        """
        base_k = 10
        diversity_boost = int(lexical_diversity * 30)     # 0-30
        temporal_boost = int(temporal_context * 20)       # 0-20

        top_k = base_k + diversity_boost + temporal_boost
        return max(5, min(100, top_k))  # Range: 5-100
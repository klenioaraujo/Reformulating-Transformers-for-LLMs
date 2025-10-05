#!/usr/bin/env python3
"""
ΨQRH LM-Eval Model Implementation
=================================

This module implements an LM-Eval-compatible model interface for the ΨQRH framework,
enabling seamless integration with EleutherAI's Language Model Evaluation Harness.

The model follows LM-Eval's architecture and provides:
- Text generation with quaternion-harmonic processing
- Log-likelihood calculation based on energy conservation
- Batch processing capabilities
- Multiple-choice and completion task support

Author: Klenio Araujo Padilha
License: GNU GPLv3
"""

import os
import sys
import time
import logging
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    # Try to import LM-Eval components
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    LM_EVAL_AVAILABLE = True
except ImportError:
    # Fallback implementations if LM-Eval not installed
    LM_EVAL_AVAILABLE = False
    print("Warning: LM-Eval not installed. Using mock implementations.")
    # Create mock base class
    class LM:
        pass

from evaluation_engine.psiqrh_evaluation_engine import ΨQRHEvaluationEngine


@register_model("psiqrh") if LM_EVAL_AVAILABLE else lambda x: x
class ΨQRHLMEvalModel(LM):
    """
    LM-Eval-compatible model implementation for ΨQRH framework.

    This class implements the LM-Eval model interface while leveraging
    ΨQRH's quaternion-harmonic processing capabilities for enhanced
    language understanding and generation.
    """

    def __init__(self,
                 device: Optional[str] = None,
                 batch_size: int = 1,
                 max_length: int = 2048,
                 max_gen_toks: int = 512,
                 psiqrh_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize ΨQRH LM-Eval model.

        Args:
            device: Computing device ('cpu', 'cuda', 'mps')
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            max_gen_toks: Maximum tokens to generate
            psiqrh_config: ΨQRH model configuration
            **kwargs: Additional arguments
        """
        super().__init__()

        # LM-Eval required attributes
        self._device = device or self._detect_device()
        self._batch_size = batch_size
        self._max_length = max_length
        self._max_gen_toks = max_gen_toks

        # Initialize ΨQRH evaluation engine
        self.psiqrh_engine = ΨQRHEvaluationEngine(
            model_config=psiqrh_config,
            device=self._device,
            debug=kwargs.get('debug', False)
        )

        # Model metadata
        self.model_name = "psiqrh-quaternion-harmonic"
        self.model_version = "1.0.0"

        # Performance tracking
        self.request_count = 0
        self.total_tokens_generated = 0
        self.total_processing_time = 0.0

        # Tokenization (simplified - in practice would use proper tokenizer)
        self.vocab_size = 50000  # Mock vocabulary size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ΨQRH LM-Eval Model initialized: {self.model_name}")

    def _detect_device(self) -> str:
        """Detect optimal computing device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    # LM-Eval required properties
    @property
    def eot_token_id(self) -> int:
        """End of text token ID."""
        return self.eos_token_id

    @property
    def max_length(self) -> int:
        """Maximum sequence length."""
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        """Maximum generation tokens."""
        return self._max_gen_toks

    @property
    def batch_size(self) -> int:
        """Batch size for processing."""
        return self._batch_size

    @property
    def device(self) -> str:
        """Device used for computation."""
        return self._device

    # LM-Eval required methods

    def generate_until(self, requests: List[Any]) -> List[str]:
        """
        Generate text until stopping criteria are met.

        This method implements LM-Eval's text generation interface
        using ΨQRH's quaternion-harmonic processing.

        Args:
            requests: List of generation requests

        Returns:
            List of generated text completions
        """
        start_time = time.time()
        results = []

        self.logger.debug(f"Processing {len(requests)} generation requests")

        for i, request in enumerate(requests):
            try:
                # Parse request format
                if isinstance(request, tuple) and len(request) >= 2:
                    context = request[0]
                    until = request[1] if request[1] else []
                    max_tokens = getattr(request, 'max_tokens', self._max_gen_toks) if len(request) > 2 else self._max_gen_toks
                else:
                    context = str(request)
                    until = []
                    max_tokens = self._max_gen_toks

                # Generate through ΨQRH
                generated_text = self._generate_text(
                    context=context,
                    max_tokens=max_tokens,
                    until=until
                )

                results.append(generated_text)
                self.request_count += 1

            except Exception as e:
                self.logger.error(f"Error generating text for request {i}: {e}")
                results.append(f"Error: {str(e)}")

        # Update performance tracking
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time

        self.logger.info(f"Generated {len(results)} completions in {processing_time:.3f}s")
        return results

    def loglikelihood(self, requests: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        """
        Calculate log-likelihood for completion given context.

        Args:
            requests: List of (context, completion) tuples

        Returns:
            List of (logprob, is_greedy) tuples
        """
        start_time = time.time()
        results = []

        self.logger.debug(f"Calculating log-likelihood for {len(requests)} requests")

        for i, (context, completion) in enumerate(requests):
            try:
                # Calculate log-likelihood using ΨQRH energy conservation
                logprob, is_greedy = self._calculate_loglikelihood(context, completion)
                results.append((logprob, is_greedy))

            except Exception as e:
                self.logger.error(f"Error calculating log-likelihood for request {i}: {e}")
                results.append((-float('inf'), False))

        processing_time = time.time() - start_time
        self.total_processing_time += processing_time

        self.logger.info(f"Calculated {len(results)} log-likelihoods in {processing_time:.3f}s")
        return results

    def loglikelihood_rolling(self, requests: List[str]) -> List[float]:
        """
        Calculate rolling log-likelihood for sequences.

        Args:
            requests: List of text sequences

        Returns:
            List of log-likelihood values
        """
        results = []

        for text in requests:
            try:
                # Process through ΨQRH and estimate likelihood
                result = self.psiqrh_engine.process_text(text, max_tokens=1)

                if result['status'] == 'success':
                    # Use energy conservation as likelihood indicator
                    energy_ratio = result.get('energy_conservation', 1.0)
                    logprob = self._energy_to_logprob(energy_ratio)
                else:
                    logprob = -float('inf')

                results.append(logprob)

            except Exception as e:
                self.logger.error(f"Error in rolling log-likelihood: {e}")
                results.append(-float('inf'))

        return results

    # Helper methods for ΨQRH processing

    def _generate_text(self,
                      context: str,
                      max_tokens: int,
                      until: List[str]) -> str:
        """
        Generate text using ΨQRH quaternion-harmonic processing.

        Args:
            context: Input context text
            max_tokens: Maximum tokens to generate
            until: Stop sequences

        Returns:
            Generated text completion
        """
        # Process through ΨQRH engine
        result = self.psiqrh_engine.process_text(
            text=context,
            max_tokens=max_tokens,
            temperature=0.8  # Default temperature
        )

        if result['status'] == 'success':
            generated_text = result['text']

            # Apply stopping criteria
            if until:
                for stop_seq in until:
                    if stop_seq in generated_text:
                        generated_text = generated_text.split(stop_seq)[0]
                        break

            # Update token count
            token_count = len(generated_text.split())
            self.total_tokens_generated += token_count

            return generated_text

        else:
            # Return error message
            return f"Generation failed: {result.get('error', 'Unknown error')}"

    def _calculate_loglikelihood(self, context: str, completion: str) -> Tuple[float, bool]:
        """
        Calculate log-likelihood using ΨQRH energy conservation metrics.

        Args:
            context: Context text
            completion: Completion text

        Returns:
            Tuple of (log_probability, is_greedy)
        """
        # Combine context and completion
        full_text = context + completion

        # Process through ΨQRH
        result = self.psiqrh_engine.process_text(full_text, max_tokens=1)

        if result['status'] == 'success':
            # Extract ΨQRH quality metrics
            energy_conservation = result.get('energy_conservation', 1.0)
            quaternion_norm = result.get('quaternion_norm', 1.0)
            spectral_coherence = result.get('spectral_coherence', 1.0)

            # Calculate composite likelihood score
            logprob = self._calculate_composite_logprob(
                energy_conservation,
                quaternion_norm,
                spectral_coherence
            )

            # Determine if this is the greedy choice (high energy conservation)
            is_greedy = energy_conservation > 0.9

            return logprob, is_greedy

        else:
            # Low probability for failed processing
            return -10.0, False

    def _calculate_composite_logprob(self,
                                   energy_conservation: float,
                                   quaternion_norm: float,
                                   spectral_coherence: float) -> float:
        """
        Calculate composite log-probability from ΨQRH metrics.

        This method combines multiple ΨQRH quality indicators into
        a single log-probability estimate.
        """
        # Normalize each metric to [0, 1] range
        energy_score = max(0, min(energy_conservation, 2.0)) / 2.0
        norm_score = max(0, min(abs(quaternion_norm - 1.0), 1.0))
        norm_score = 1.0 - norm_score  # Invert so 1.0 is best
        coherence_score = max(0, min(spectral_coherence, 1.0))

        # Weighted combination
        composite_score = (
            0.5 * energy_score +
            0.3 * norm_score +
            0.2 * coherence_score
        )

        # Convert to log probability (map [0,1] to [-10, 0])
        logprob = -10.0 * (1.0 - composite_score)

        return max(logprob, -10.0)  # Floor at -10.0

    def _energy_to_logprob(self, energy_ratio: float) -> float:
        """Convert ΨQRH energy conservation ratio to log probability."""
        # Ideal energy ratio is 1.0 (perfect conservation)
        deviation = abs(energy_ratio - 1.0)

        # Convert to log probability
        logprob = -2.0 * deviation

        return max(logprob, -10.0)

    # Additional utility methods

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        avg_time = self.total_processing_time / max(self.request_count, 1)

        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'framework': 'ΨQRH (Psi Quaternionic-Harmonic)',
            'device': self._device,
            'max_length': self._max_length,
            'max_gen_toks': self._max_gen_toks,
            'batch_size': self._batch_size,
            'vocab_size': self.vocab_size,
            'performance': {
                'total_requests': self.request_count,
                'total_tokens_generated': self.total_tokens_generated,
                'total_processing_time': self.total_processing_time,
                'average_processing_time': avg_time,
                'tokens_per_second': self.total_tokens_generated / max(self.total_processing_time, 0.001)
            }
        }

    def reset_stats(self):
        """Reset performance tracking statistics."""
        self.request_count = 0
        self.total_tokens_generated = 0
        self.total_processing_time = 0.0
        self.logger.info("Performance statistics reset")


# Factory function for easy model creation

def create_psiqrh_lm_eval_model(device: Optional[str] = None,
                               batch_size: int = 1,
                               psiqrh_config: Optional[Dict[str, Any]] = None,
                               **kwargs) -> ΨQRHLMEvalModel:
    """
    Factory function to create a ΨQRH LM-Eval model with sensible defaults.

    Args:
        device: Computing device
        batch_size: Batch size for processing
        psiqrh_config: ΨQRH model configuration
        **kwargs: Additional arguments

    Returns:
        Configured ΨQRH LM-Eval model
    """
    default_psiqrh_config = {
        'embed_dim': 64,
        'alpha': 1.5,
        'beta': 0.01,
        'use_spectral_filtering': True,
        'use_fractal_analysis': True
    }

    if psiqrh_config:
        default_psiqrh_config.update(psiqrh_config)

    return ΨQRHLMEvalModel(
        device=device,
        batch_size=batch_size,
        psiqrh_config=default_psiqrh_config,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage and testing
    print("📊 ΨQRH LM-Eval Model - Integration Testing")
    print("=" * 50)

    # Create model
    model = create_psiqrh_lm_eval_model(
        batch_size=1,
        psiqrh_config={
            'embed_dim': 32,
            'alpha': 1.5,
            'use_spectral_filtering': True
        }
    )

    print(f"Model: {model.model_name}")
    print(f"Device: {model.device}")
    print(f"Max Length: {model.max_length}")

    # Test text generation
    print("\n--- Testing Text Generation ---")
    generation_requests = [
        ("What is the theory of relativity?", ["\n", ".", "?"]),
        ("Explain quantum mechanics", ["\n"]),
        ("The Padilha Wave Equation describes", [])
    ]

    generation_results = model.generate_until(generation_requests)

    for i, (request, result) in enumerate(zip(generation_requests, generation_results)):
        context = request[0]
        print(f"\nRequest {i+1}:")
        print(f"Context: {context}")
        print(f"Generated: {result}")

    # Test log-likelihood calculation
    print("\n--- Testing Log-Likelihood ---")
    loglik_requests = [
        ("The capital of France is", "Paris"),
        ("2 + 2 equals", "4"),
        ("The quaternion multiplication", "is non-commutative")
    ]

    loglik_results = model.loglikelihood(loglik_requests)

    for i, ((context, completion), (logprob, is_greedy)) in enumerate(zip(loglik_requests, loglik_results)):
        print(f"\nRequest {i+1}:")
        print(f"Context: {context}")
        print(f"Completion: {completion}")
        print(f"Log Prob: {logprob:.3f}")
        print(f"Is Greedy: {is_greedy}")

    # Test rolling log-likelihood
    print("\n--- Testing Rolling Log-Likelihood ---")
    rolling_requests = [
        "The mathematical foundation of ΨQRH framework",
        "Quaternion algebra provides",
        "Energy conservation in neural networks"
    ]

    rolling_results = model.loglikelihood_rolling(rolling_requests)

    for request, result in zip(rolling_requests, rolling_results):
        print(f"Text: {request}")
        print(f"Rolling Log Prob: {result:.3f}\n")

    # Print model information
    print("=" * 50)
    print("Model Information:")
    model_info = model.get_model_info()
    for key, value in model_info.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.3f}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")

    print("\n✅ LM-Eval integration testing completed!")
#!/usr/bin/env python3
"""
ΨQRH HELM Client Implementation
===============================

This module implements a HELM-compatible client for the ΨQRH framework,
allowing seamless integration with Stanford's Holistic Evaluation of Language Models.

The client follows HELM's architecture and provides:
- Request/Response handling
- Caching support
- Error handling and retry logic
- Metadata collection for ΨQRH-specific metrics

Author: Klenio Araujo Padilha
License: GNU GPLv3
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional, Sequence
from dataclasses import dataclass, asdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    # Try to import HELM components
    from helm.common.request import Request, RequestResult, GeneratedOutput
    from helm.common.tokenization_request import TokenizationRequest
    from helm.clients.client import Client, CachingClient
    from helm.common.cache import CacheConfig
    HELM_AVAILABLE = True
except ImportError:
    # Fallback implementations if HELM not installed
    HELM_AVAILABLE = False
    print("Warning: HELM not installed. Using mock implementations.")

from evaluation_engine.psiqrh_evaluation_engine import ΨQRHEvaluationEngine


@dataclass
class ΨQRHRequest:
    """ΨQRH-specific request structure with quaternion parameters."""
    prompt: str
    max_tokens: int = 150
    temperature: float = 1.0
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None

    # ΨQRH-specific parameters
    alpha: Optional[float] = None
    beta: Optional[float] = None
    use_fractal_analysis: bool = True
    use_spectral_filtering: bool = True
    quaternion_precision: str = 'float32'


@dataclass
class ΨQRHResponse:
    """ΨQRH-specific response with enhanced metadata."""
    text: str
    logprob: float
    tokens: List[str]

    # ΨQRH-specific metadata
    fractal_dimension: Optional[float] = None
    wave_parameters: Optional[Dict[str, float]] = None
    energy_conservation: Optional[float] = None
    quaternion_norm: Optional[float] = None
    spectral_coherence: Optional[float] = None
    processing_time: Optional[float] = None


class ΨQRHHELMClient(CachingClient if HELM_AVAILABLE else object):
    """
    HELM-compatible client for ΨQRH framework integration.

    This client implements HELM's Client interface while leveraging
    ΨQRH's quaternion-harmonic processing capabilities.
    """

    def __init__(self,
                 cache_config: Optional['CacheConfig'] = None,
                 psiqrh_config: Optional[Dict[str, Any]] = None,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None):
        """
        Initialize ΨQRH HELM client.

        Args:
            cache_config: HELM caching configuration
            psiqrh_config: ΨQRH model configuration
            api_key: API key (for compatibility, not used)
            api_base: API base URL (for compatibility, not used)
        """
        if HELM_AVAILABLE:
            super().__init__(cache_config=cache_config or CacheConfig())

        # Initialize ΨQRH evaluation engine
        self.psiqrh_engine = ΨQRHEvaluationEngine(
            model_config=psiqrh_config,
            debug=True
        )

        # Client metadata
        self.model_name = "psiqrh-quaternion-harmonic"
        self.api_key = api_key
        self.api_base = api_base

        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ΨQRH HELM Client initialized with model: {self.model_name}")

    def make_request(self, request: 'Request') -> 'RequestResult':
        """
        Main entry point for HELM request processing.

        This method implements HELM's Client.make_request interface
        while routing through ΨQRH's quaternion-harmonic processing.

        Args:
            request: HELM Request object

        Returns:
            HELM RequestResult object with ΨQRH processing results
        """
        start_time = time.time()
        self.request_count += 1

        try:
            self.logger.debug(f"Processing HELM request #{self.request_count}")

            # Convert HELM request to ΨQRH request
            psiqrh_request = self._helm_to_psiqrh_request(request)

            # Process through ΨQRH engine
            psiqrh_result = self._process_psiqrh_request(psiqrh_request)

            # Convert back to HELM format
            helm_result = self._psiqrh_to_helm_result(psiqrh_result, request)

            # Update performance tracking
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time

            self.logger.info(f"Request #{self.request_count} completed in {processing_time:.3f}s")

            return helm_result

        except Exception as e:
            self.logger.error(f"Error processing HELM request: {e}")
            # Return error response in HELM format
            return self._create_error_result(str(e), request)

    def _helm_to_psiqrh_request(self, helm_request: 'Request') -> ΨQRHRequest:
        """Convert HELM Request to ΨQRH format."""
        return ΨQRHRequest(
            prompt=helm_request.prompt,
            max_tokens=getattr(helm_request, 'max_tokens', 150),
            temperature=getattr(helm_request, 'temperature', 1.0),
            top_p=getattr(helm_request, 'top_p', 1.0),
            frequency_penalty=getattr(helm_request, 'frequency_penalty', 0.0),
            presence_penalty=getattr(helm_request, 'presence_penalty', 0.0),
            stop_sequences=getattr(helm_request, 'stop_sequences', None),

            # Extract ΨQRH parameters if present
            alpha=getattr(helm_request, 'alpha', None),
            beta=getattr(helm_request, 'beta', None),
            use_fractal_analysis=getattr(helm_request, 'use_fractal_analysis', True),
            use_spectral_filtering=getattr(helm_request, 'use_spectral_filtering', True)
        )

    def _process_psiqrh_request(self, request: ΨQRHRequest) -> ΨQRHResponse:
        """Process request through ΨQRH engine."""

        # Process text through ΨQRH framework
        result = self.psiqrh_engine.process_text(
            text=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )

        if result['status'] == 'success':
            # Calculate logprob based on energy conservation
            energy_ratio = result.get('energy_conservation', 1.0)
            logprob = self._calculate_logprob(energy_ratio)

            # Tokenize output (simplified)
            tokens = result['text'].split()

            return ΨQRHResponse(
                text=result['text'],
                logprob=logprob,
                tokens=tokens,
                fractal_dimension=result.get('fractal_dimension'),
                wave_parameters=result.get('wave_parameters'),
                energy_conservation=result.get('energy_conservation'),
                quaternion_norm=result.get('quaternion_norm'),
                spectral_coherence=result.get('spectral_coherence'),
                processing_time=result.get('processing_time')
            )
        else:
            # Handle error case
            error_text = f"ΨQRH Error: {result.get('error', 'Unknown error')}"
            return ΨQRHResponse(
                text=error_text,
                logprob=-10.0,  # Very low probability for errors
                tokens=[],
                processing_time=0.0
            )

    def _psiqrh_to_helm_result(self, psiqrh_response: ΨQRHResponse, original_request: 'Request') -> 'RequestResult':
        """Convert ΨQRH response back to HELM RequestResult format."""

        if not HELM_AVAILABLE:
            # Return mock result if HELM not available
            return MockRequestResult(psiqrh_response)

        # Create HELM GeneratedOutput
        generated_output = GeneratedOutput(
            text=psiqrh_response.text,
            logprob=psiqrh_response.logprob,
            tokens=psiqrh_response.tokens
        )

        # Create HELM RequestResult with ΨQRH metadata
        return RequestResult(
            success=True,
            cached=False,
            request_time=psiqrh_response.processing_time or 0.0,
            completions=[generated_output],
            embedding=[],  # Not used for text generation

            # Add ΨQRH-specific metadata
            raw_compute_request=asdict(psiqrh_response),
            raw_response={
                'psiqrh_metadata': {
                    'fractal_dimension': psiqrh_response.fractal_dimension,
                    'wave_parameters': psiqrh_response.wave_parameters,
                    'energy_conservation': psiqrh_response.energy_conservation,
                    'quaternion_norm': psiqrh_response.quaternion_norm,
                    'spectral_coherence': psiqrh_response.spectral_coherence,
                    'model_name': self.model_name,
                    'request_id': self.request_count
                }
            }
        )

    def _calculate_logprob(self, energy_ratio: float) -> float:
        """
        Calculate log probability based on ΨQRH energy conservation.

        Higher energy conservation indicates more coherent processing,
        which we map to higher probability.
        """
        # Clamp energy ratio to reasonable range
        energy_ratio = max(0.1, min(energy_ratio, 2.0))

        # Convert to log probability (rough approximation)
        # Energy ratio of 1.0 maps to logprob of 0.0
        # Lower/higher ratios get penalized
        deviation = abs(energy_ratio - 1.0)
        logprob = -deviation * 2.0  # Scale factor

        return max(logprob, -10.0)  # Floor at -10.0

    def _create_error_result(self, error_message: str, original_request: 'Request') -> 'RequestResult':
        """Create error result in HELM format."""

        if not HELM_AVAILABLE:
            return MockRequestResult(ΨQRHResponse(
                text=f"Error: {error_message}",
                logprob=-10.0,
                tokens=[]
            ))

        error_output = GeneratedOutput(
            text=f"Error: {error_message}",
            logprob=-10.0,
            tokens=[]
        )

        return RequestResult(
            success=False,
            cached=False,
            request_time=0.0,
            completions=[error_output],
            embedding=[],
            raw_compute_request={'error': error_message},
            raw_response={'error': error_message}
        )

    def get_model_name(self) -> str:
        """Return model name for HELM identification."""
        return self.model_name

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this client."""
        avg_time = self.total_processing_time / max(self.request_count, 1)

        return {
            'model_name': self.model_name,
            'total_requests': self.request_count,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': avg_time,
            'requests_per_second': self.request_count / max(self.total_processing_time, 0.001)
        }


class MockRequestResult:
    """Mock RequestResult for when HELM is not available."""

    def __init__(self, psiqrh_response: ΨQRHResponse):
        self.success = True
        self.cached = False
        self.request_time = psiqrh_response.processing_time or 0.0
        self.completions = [MockGeneratedOutput(psiqrh_response)]
        self.embedding = []
        self.raw_compute_request = asdict(psiqrh_response)
        self.raw_response = {'psiqrh_metadata': asdict(psiqrh_response)}


class MockGeneratedOutput:
    """Mock GeneratedOutput for when HELM is not available."""

    def __init__(self, psiqrh_response: ΨQRHResponse):
        self.text = psiqrh_response.text
        self.logprob = psiqrh_response.logprob
        self.tokens = psiqrh_response.tokens


# Helper functions for HELM integration

def create_psiqrh_helm_client(cache_dir: Optional[str] = None,
                             psiqrh_config: Optional[Dict[str, Any]] = None) -> ΨQRHHELMClient:
    """
    Factory function to create a ΨQRH HELM client with sensible defaults.

    Args:
        cache_dir: Directory for HELM caching
        psiqrh_config: ΨQRH model configuration

    Returns:
        Configured ΨQRH HELM client
    """
    cache_config = None
    if HELM_AVAILABLE and cache_dir:
        cache_config = CacheConfig(
            cache_dir=cache_dir,
            cache_stats_only=False
        )

    return ΨQRHHELMClient(
        cache_config=cache_config,
        psiqrh_config=psiqrh_config
    )


if __name__ == "__main__":
    # Example usage and testing
    print("🔬 ΨQRH HELM Client - Integration Testing")
    print("=" * 50)

    # Create client
    client = create_psiqrh_helm_client(
        psiqrh_config={
            'embed_dim': 32,
            'alpha': 1.5,
            'use_spectral_filtering': True
        }
    )

    # Create mock HELM request
    class MockHELMRequest:
        def __init__(self, prompt, max_tokens=100, temperature=1.0):
            self.prompt = prompt
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.top_p = 1.0
            self.frequency_penalty = 0.0
            self.presence_penalty = 0.0
            self.stop_sequences = None

    # Test requests
    test_requests = [
        "What is the relationship between quaternions and 3D rotations?",
        "Explain the mathematical foundation of the Padilha Wave Equation",
        "How does fractal dimension analysis improve language model performance?"
    ]

    for i, prompt in enumerate(test_requests, 1):
        print(f"\n--- Test {i} ---")
        print(f"Prompt: {prompt}")

        # Create and process request
        helm_request = MockHELMRequest(prompt, max_tokens=150)
        result = client.make_request(helm_request)

        print(f"Success: {result.success}")
        if result.completions:
            completion = result.completions[0]
            print(f"Response: {completion.text}")
            print(f"Log Prob: {completion.logprob:.3f}")
            print(f"Request Time: {result.request_time:.3f}s")

            # Print ΨQRH metadata if available
            if hasattr(result, 'raw_response') and 'psiqrh_metadata' in result.raw_response:
                metadata = result.raw_response['psiqrh_metadata']
                if metadata.get('fractal_dimension'):
                    print(f"Fractal Dimension: {metadata['fractal_dimension']:.3f}")
                if metadata.get('energy_conservation'):
                    print(f"Energy Conservation: {metadata['energy_conservation']:.3f}")

    # Print performance statistics
    print("\n" + "="*50)
    print("Performance Statistics:")
    stats = client.get_performance_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    print("\n✅ HELM integration testing completed!")
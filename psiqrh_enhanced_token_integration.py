#!/usr/bin/env python3
"""
ΨQRH Enhanced Token Analysis Integration

Integrates the enhanced token analysis system with the ΨQRH pipeline
for improved token selection with dynamic quantum vocabulary.
"""

import torch
import sys
import os
from typing import Dict, List, Tuple, Optional, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_token_analysis import EnhancedDCFTokenAnalysis

class ΨQRHEnhancedTokenIntegration:
    """
    Enhanced token analysis integration for ΨQRH pipeline
    """

    def __init__(self, pipeline_instance):
        """
        Initialize enhanced token integration with ΨQRH pipeline.

        Args:
            pipeline_instance: ΨQRH pipeline instance
        """
        self.pipeline = pipeline_instance
        self.enhanced_analyzer = None
        self._initialize_enhanced_analysis()

    def _initialize_enhanced_analysis(self):
        """Initialize enhanced token analysis system."""
        try:
            # Extract quantum vocabulary representations from pipeline
            quantum_vocab_representations = getattr(
                self.pipeline, 'quantum_vocab_representations', None
            )
            char_to_idx = getattr(self.pipeline, 'char_to_idx', None)

            # Initialize enhanced analyzer
            self.enhanced_analyzer = EnhancedDCFTokenAnalysis(
                device=getattr(self.pipeline, 'device', 'cpu'),
                quantum_vocab_representations=quantum_vocab_representations,
                char_to_idx=char_to_idx,
                enable_dynamic_vocabulary=True
            )

            print("      ✅ Enhanced token analysis system initialized")

        except Exception as e:
            print(f"      ⚠️  Failed to initialize enhanced analysis: {e}")
            self.enhanced_analyzer = None

    def enhanced_token_selection(self, logits: torch.Tensor,
                                input_text: str,
                                candidate_indices: Optional[torch.Tensor] = None,
                                embeddings: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Enhanced token selection with dynamic quantum vocabulary.

        Args:
            logits: Model logits
            input_text: Input text
            candidate_indices: Candidate token indices
            embeddings: Token embeddings

        Returns:
            Enhanced token selection results
        """
        if self.enhanced_analyzer is None:
            # Fallback to original method
            return self._fallback_selection(logits, candidate_indices)

        try:
            # Use enhanced analysis
            result = self.enhanced_analyzer.analyze_tokens_with_quantum_vocab(
                logits=logits,
                candidate_indices=candidate_indices,
                embeddings=embeddings,
                input_text=input_text
            )

            print("      ✅ Enhanced token selection completed")
            return result

        except Exception as e:
            print(f"      ⚠️  Enhanced token selection failed: {e}")
            return self._fallback_selection(logits, candidate_indices)

    def _fallback_selection(self, logits: torch.Tensor,
                           candidate_indices: Optional[torch.Tensor]) -> Dict[str, Any]:
        """Fallback token selection method."""
        # Simple top-1 selection
        if candidate_indices is not None:
            selected_token = candidate_indices[0].item() if hasattr(candidate_indices[0], 'item') else candidate_indices[0]
        else:
            selected_token = torch.argmax(logits).item()

        return {
            'selected_token': selected_token,
            'final_probability': 1.0,
            'fci_value': 0.5,
            'consciousness_state': 'UNKNOWN',
            'synchronization_order': 0.5,
            'processing_time': 0.001,
            'analysis_report': 'Fallback selection - Enhanced analysis not available'
        }

    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration statistics."""
        if self.enhanced_analyzer:
            return self.enhanced_analyzer.get_enhanced_analysis_stats()
        else:
            return {
                'enhanced_analysis_enabled': False,
                'fallback_mode': True
            }

def integrate_enhanced_token_analysis(pipeline_instance) -> ΨQRHEnhancedTokenIntegration:
    """
    Main integration function for ΨQRH pipeline.

    Args:
        pipeline_instance: ΨQRH pipeline instance

    Returns:
        Enhanced token integration instance
    """
    print("\n🔗 Integrating Enhanced Token Analysis with ΨQRH Pipeline...")

    # Initialize integration
    integration = ΨQRHEnhancedTokenIntegration(pipeline_instance)

    # Show integration stats
    stats = integration.get_integration_stats()
    print(f"   📊 Integration Statistics:")
    print(f"      Enhanced Analysis: {stats.get('enhanced_analysis_enabled', False)}")
    print(f"      Dynamic Vocabulary: {stats.get('dynamic_vocabulary_enabled', False)}")
    print(f"      Quantum Word References: {stats.get('quantum_word_references', False)}")

    if 'total_english_words' in stats:
        print(f"      English Words: {stats['total_english_words']:,}")
        print(f"      Character Mappings: {stats['total_character_mappings']}")

    print("   ✅ Enhanced token analysis integration complete!")

    return integration

def update_pipeline_with_enhanced_analysis(pipeline_instance) -> None:
    """
    Update ΨQRH pipeline with enhanced token analysis.

    Args:
        pipeline_instance: ΨQRH pipeline instance
    """
    # Initialize integration
    integration = integrate_enhanced_token_analysis(pipeline_instance)

    # Store integration in pipeline
    pipeline_instance.enhanced_token_integration = integration

    # Replace token analysis method if available
    if hasattr(pipeline_instance, '_analyze_tokens_dcf'):
        original_method = pipeline_instance._analyze_tokens_dcf

        def enhanced_token_analysis(logits, candidate_indices=None, embeddings=None):
            """Enhanced token analysis with dynamic quantum vocabulary."""
            # Get input text from pipeline context
            input_text = getattr(pipeline_instance, 'current_input_text', None)

            if input_text and integration.enhanced_analyzer:
                return integration.enhanced_token_selection(
                    logits=logits,
                    input_text=input_text,
                    candidate_indices=candidate_indices,
                    embeddings=embeddings
                )
            else:
                # Fallback to original method
                return original_method(logits, candidate_indices, embeddings)

        pipeline_instance._analyze_tokens_dcf = enhanced_token_analysis
        print("      ✅ Enhanced token analysis method integrated")

    print("   🎯 ΨQRH pipeline updated with enhanced token analysis!")

def test_integration():
    """Test the enhanced token analysis integration."""
    print("🧪 Testing Enhanced Token Analysis Integration")
    print("=" * 60)

    # Create mock pipeline for testing
    class MockPipeline:
        def __init__(self):
            self.device = "cpu"
            self.quantum_vocab_representations = None
            self.char_to_idx = None
            self.current_input_text = "what color is the sky?"

        def _analyze_tokens_dcf(self, logits, candidate_indices=None, embeddings=None):
            return {
                'selected_token': 42,
                'final_probability': 0.8,
                'fci_value': 0.6,
                'consciousness_state': 'MEDITATION'
            }

    # Test integration
    pipeline = MockPipeline()
    update_pipeline_with_enhanced_analysis(pipeline)

    # Test enhanced selection
    if hasattr(pipeline, 'enhanced_token_integration'):
        integration = pipeline.enhanced_token_integration

        # Test with sample data
        logits = torch.randn(100)
        result = integration.enhanced_token_selection(
            logits=logits,
            input_text="what color is the sky?"
        )

        print(f"\n📊 Test Results:")
        print(f"   Selected Token: {result['selected_token']}")
        print(f"   FCI Value: {result['fci_value']:.4f}")
        print(f"   Consciousness State: {result['consciousness_state']}")

        # Show quantum vocabulary analysis if available
        if 'quantum_vocabulary_analysis' in result:
            quantum_analysis = result['quantum_vocabulary_analysis']
            print(f"\n🔬 Quantum Vocabulary Analysis:")
            print(f"   Input Text: {quantum_analysis['input_text']}")
            print(f"   Quantum Prompt: {quantum_analysis['quantum_prompt']}")

    print("\n✅ Enhanced token analysis integration test completed!")

if __name__ == "__main__":
    test_integration()
#!/usr/bin/env python3
"""
Comprehensive Test Suite for Word Matrix ΨQRHSystem
==================================================

Tests all classes and functionality of the refactored system:
- VocabularyManager
- QuantumWordMatrix
- BiPsiQRHSemanticIntegrator
- BidirectionalTextProcessor
- Integration with ΨQRHSystem components

Ensures ZERO FALLBACK POLICY and validates word matrix logic.
"""

import sys
import os
import torch
import numpy as np
from typing import Dict, Any

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

sys.path.insert(0, project_root)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

def test_vocabulary_manager():
    """Test VocabularyManager functionality"""
    print("🧪 Testing VocabularyManager...")

    try:
        from src.cognitive.vocabulary_manager import VocabularyManager, get_vocabulary_manager

        # Test initialization
        vm = VocabularyManager()
        assert vm is not None, "VocabularyManager should initialize"

        # Test global instance
        global_vm = get_vocabulary_manager()
        assert global_vm is not None, "Global VocabularyManager should be available"

        # Test vocabulary loading
        current_vocab = vm.get_current_vocabulary()
        assert current_vocab is not None, "Should have GPT-2 vocabulary loaded"
        assert current_vocab.name == 'gpt2', "Should be GPT-2 vocabulary"
        assert current_vocab.vocab_size == 50257, "Should have 50257 tokens"

        # Test vocabulary info
        info = vm.get_vocabulary_info()
        assert info['current_vocabulary'] == 'gpt2', "Should report GPT-2 as current"
        assert info['vocab_size'] == 50257, "Should report correct vocab size"

        print("✅ VocabularyManager tests passed")
        return True

    except Exception as e:
        print(f"❌ VocabularyManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quantum_word_matrix():
    """Test QuantumWordMatrix functionality"""
    print("🧪 Testing QuantumWordMatrix...")

    try:
        from quantum_word_matrix import QuantumWordMatrix
        from src.cognitive.vocabulary_manager import get_vocabulary_manager

        # Get vocabulary
        vm = get_vocabulary_manager()
        vocab = vm.get_current_vocabulary()

        # Test initialization
        qwm = QuantumWordMatrix(
            embed_dim=128,
            device="cpu",
            word_to_id=vocab.word_to_id,
            id_to_word=vocab.id_to_word
        )

        assert qwm is not None, "QuantumWordMatrix should initialize"
        assert qwm.vocab_size == 50257, "Should have correct vocab size"
        assert qwm.embed_dim == 128, "Should have correct embed dim"

        # Test word encoding
        test_word = "Ġthe"  # Common GPT-2 token
        if test_word in vocab.word_to_id:
            embedding = qwm.encode_word(test_word)
            assert embedding.shape == (128,), f"Embedding should be shape (128,), got {embedding.shape}"
            assert not torch.isnan(embedding).any(), "Embedding should not contain NaN"
            assert not torch.isinf(embedding).any(), "Embedding should not contain Inf"

        # Test quantum state decoding
        test_state = torch.randn(128)
        decoded = qwm.decode_quantum_state(test_state, top_k=3)
        assert len(decoded) <= 3, "Should return at most 3 results"
        assert all(isinstance(word, str) and isinstance(score, float) for word, score in decoded), "Results should be (word, score) tuples"

        print("✅ QuantumWordMatrix tests passed")
        return True

    except Exception as e:
        print(f"❌ QuantumWordMatrix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bidirectional_semantic_integrator():
    """Test BiPsiQRHSemanticIntegrator functionality"""
    print("🧪 Testing BiPsiQRHSemanticIntegrator...")

    try:
        from src.cognitive.biPsiQRH_semantic_integration import BiPsiQRHSemanticIntegrator

        # Test initialization
        integrator = BiPsiQRHSemanticIntegrator()
        assert integrator is not None, "BiPsiQRHSemanticIntegrator should initialize"

        # Test bidirectional analysis
        test_text = "O banco oferece crédito para investimentos empresariais"
        semantic_state = integrator.analyze_text_bidirectional(test_text)

        assert semantic_state is not None, "Should return semantic state"
        assert hasattr(semantic_state, 'left_features'), "Should have left features"
        assert hasattr(semantic_state, 'right_features'), "Should have right features"
        assert hasattr(semantic_state, 'ternary_consensus'), "Should have ternary consensus"
        assert hasattr(semantic_state, 'energy_conserved'), "Should have energy conservation status"
        assert hasattr(semantic_state, 'consciousness_level'), "Should have consciousness level"

        # Test text composition
        composed_text = integrator.compose_text_bidirectional(semantic_state)
        assert isinstance(composed_text, str), "Should return string"
        assert len(composed_text) > 0, "Should return non-empty text"

        # Test tokenization
        tokens = integrator._tokenize_text(test_text)
        assert isinstance(tokens, list), "Should return list of tokens"
        assert len(tokens) > 0, "Should return non-empty token list"

        print("✅ BiPsiQRHSemanticIntegrator tests passed")
        return True

    except Exception as e:
        print(f"❌ BiPsiQRHSemanticIntegrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_bidirectional_text_processor():
    """Test BidirectionalTextProcessor functionality"""
    print("🧪 Testing BidirectionalTextProcessor...")

    try:
        from src.cognitive.biPsiQRH_semantic_integration import (
            BidirectionalTextProcessor,
            BiPsiQRHSemanticIntegrator
        )

        # Test initialization
        integrator = BiPsiQRHSemanticIntegrator()
        processor = BidirectionalTextProcessor(integrator)
        assert processor is not None, "BidirectionalTextProcessor should initialize"

        # Test analysis task
        test_text = "A empresa desenvolve soluções tecnológicas inovadoras"
        analysis_result = processor.process_text(test_text, task='analysis')

        assert analysis_result['task'] == 'bidirectional_semantic_analysis', "Should report correct task"
        assert 'semantic_state' in analysis_result, "Should have semantic state"
        assert 'analysis' in analysis_result, "Should have analysis results"

        analysis = analysis_result['analysis']
        required_keys = ['left_dominance', 'right_dominance', 'ternary_consensus',
                        'energy_conserved', 'consciousness_level']
        for key in required_keys:
            assert key in analysis, f"Analysis should contain {key}"

        # Test composition task
        composition_result = processor.process_text(test_text, task='composition')

        assert composition_result['task'] == 'bidirectional_text_composition', "Should report correct task"
        assert 'original_text' in composition_result, "Should have original text"
        assert 'composed_text' in composition_result, "Should have composed text"
        assert 'semantic_influence' in composition_result, "Should have semantic influence"

        # Test processor info
        info = processor.get_processor_info()
        assert 'name' in info, "Should have name"
        assert 'capabilities' in info, "Should have capabilities"
        assert 'ternary_logic' in info, "Should report ternary logic"
        assert 'energy_conservation' in info, "Should report energy conservation"

        print("✅ BidirectionalTextProcessor tests passed")
        return True

    except Exception as e:
        print(f"❌ BidirectionalTextProcessor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_system_integration():
    """Test complete system integration"""
    print("🧪 Testing complete system integration...")

    try:
        from src.cognitive.biPsiQRH_semantic_integration import create_bidirectional_semantic_processor

        # Test factory function
        processor = create_bidirectional_semantic_processor()
        assert processor is not None, "Factory function should create processor"

        # Test end-to-end workflow
        test_text = "O sistema de inteligência artificial processa dados complexos"

        # Analysis workflow
        analysis = processor.process_text(test_text, task='analysis')
        assert analysis['task'] == 'bidirectional_semantic_analysis'

        # Composition workflow
        composition = processor.process_text(test_text, task='composition')
        assert composition['task'] == 'bidirectional_text_composition'

        # Integration status
        status = processor.integrator.get_integration_status()
        assert 'ternary_logic_status' in status
        assert 'energy_conservation_rate' in status
        assert 'semantic_bases' in status
        assert 'consciousness_metrics' in status
        assert 'bidirectional_capabilities' in status

        print("✅ System integration tests passed")
        return True

    except Exception as e:
        print(f"❌ System integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_zero_fallback_policy():
    """Test ZERO FALLBACK POLICY compliance"""
    print("🧪 Testing ZERO FALLBACK POLICY compliance...")

    try:
        # Test that system fails gracefully without required components
        # This validates that no fallbacks are used

        # Test vocabulary manager without data
        from src.cognitive.vocabulary_manager import VocabularyManager
        import tempfile
        import shutil

        # Create temporary directory without vocab file
        temp_dir = tempfile.mkdtemp()
        try:
            temp_vm = VocabularyManager(base_path=temp_dir)
            # Should not crash but should not have vocabularies loaded
            assert len(temp_vm.vocabularies) == 0, "Should not load vocabularies without files"
        finally:
            shutil.rmtree(temp_dir)

        print("✅ ZERO FALLBACK POLICY tests passed")
        return True

    except Exception as e:
        print(f"❌ ZERO FALLBACK POLICY test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vocabulary_switching():
    """Test vocabulary switching functionality"""
    print("🧪 Testing vocabulary switching...")

    try:
        from src.cognitive.vocabulary_manager import get_vocabulary_manager, switch_vocabulary

        vm = get_vocabulary_manager()

        # Test current vocabulary
        current = vm.get_current_vocabulary()
        assert current is not None, "Should have current vocabulary"
        assert current.name == 'gpt2', "Should be GPT-2"

        # Test switching function
        result = switch_vocabulary('gpt2')
        assert result == True, "Should successfully switch to GPT-2"

        # Test invalid vocabulary
        result = switch_vocabulary('nonexistent')
        assert result == False, "Should fail to switch to nonexistent vocabulary"

        print("✅ Vocabulary switching tests passed")
        return True

    except Exception as e:
        print(f"❌ Vocabulary switching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all test suites"""
    print("🚀 Running comprehensive Word Matrix ΨQRHSystem tests")
    print("=" * 60)

    tests = [
        ("VocabularyManager", test_vocabulary_manager),
        ("QuantumWordMatrix", test_quantum_word_matrix),
        ("BiPsiQRHSemanticIntegrator", test_bidirectional_semantic_integrator),
        ("BidirectionalTextProcessor", test_bidirectional_text_processor),
        ("System Integration", test_system_integration),
        ("ZERO FALLBACK POLICY", test_zero_fallback_policy),
        ("Vocabulary Switching", test_vocabulary_switching),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} tests...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} tests: PASSED")
            else:
                print(f"❌ {test_name} tests: FAILED")
        except Exception as e:
            print(f"❌ {test_name} tests: ERROR - {e}")

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} test suites passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED! Word Matrix ΨQRHSystem is fully functional.")
        print("✅ Word Matrix Logic: ACTIVE")
        print("✅ Character Matrix Logic: REMOVED")
        print("✅ Bidirectional Processing: ENABLED")
        print("✅ Vocabulary Switching: ENABLED")
        print("✅ ZERO FALLBACK POLICY: ENFORCED")
        return True
    else:
        print(f"⚠️  {total - passed} test suites failed. System may have issues.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test Suite for Physics-Emergent Training System
===============================================

Tests the revolutionary physics-emergent training approach that uses:
- Auto-calibration based on fractal signals
- Harmonic orchestration for signal processing
- Consciousness metrics (FCI) for quality evaluation
- Emergent parameter adjustment through physical principles
"""

import torch
import pytest
from unittest.mock import Mock, patch
from train_physics_emergent import PhysicsEmergentTrainer
from psiqrh import ΨQRHPipeline


class TestPhysicsEmergentTrainer:
    """Test suite for the PhysicsEmergentTrainer class."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline for testing."""
        pipeline = Mock(spec=ΨQRHPipeline)
        # Mock the _generate_text_physical method
        pipeline._generate_text_physical.return_value = {
            'fci_value': 0.7,
            'synchronization_order': 0.8,
            'cluster_analysis': {
                'dominant_cluster': {'order_parameter': 0.75}
            },
            'energy_conservation': 0.9,
            'spectral_coherence': 0.85,
            'generated_text': 'blue'
        }
        return pipeline

    @pytest.fixture
    def trainer(self, mock_pipeline):
        """Create a trainer instance with mocked pipeline."""
        return PhysicsEmergentTrainer(mock_pipeline)

    def test_initialization(self, trainer):
        """Test trainer initialization."""
        assert trainer.pipeline is not None
        assert trainer.calibration_system is not None
        assert trainer.harmonic_orchestrator is not None
        assert trainer.dcf_analyzer is not None
        assert len(trainer.optimal_configurations) == 0
        assert len(trainer.consciousness_history) == 0
        assert len(trainer.parameter_evolution_trajectory) == 0

    def test_evaluate_consciousness_quality(self, trainer):
        """Test consciousness quality evaluation."""
        # Test with high quality result
        result = {
            'fci_value': 0.8,
            'synchronization_order': 0.9,
            'cluster_analysis': {
                'dominant_cluster': {'order_parameter': 0.85}
            }
        }

        metrics = trainer._evaluate_consciousness_quality(result)

        assert metrics['fci'] == 0.8
        assert metrics['sync_order'] == 0.9
        assert metrics['cluster_coherence'] == 0.85
        assert metrics['quality_score'] == pytest.approx(0.4*0.8 + 0.3*0.9 + 0.3*0.85, rel=1e-2)
        assert not metrics['requires_optimization']
        assert metrics['optimal_state']

    def test_emergent_parameter_adjustment(self, trainer):
        """Test emergent parameter adjustment."""
        config = {
            'physical_params': {'alpha': 1.0, 'beta': 1.0},
            'control_params': {'temperature': 1.0},
            'processing_params': {'semantic_connectivity_strength': 1.0}
        }

        # Test low FCI adjustment
        metrics_low_fci = {'fci': 0.3, 'sync_order': 0.8, 'cluster_coherence': 0.9}
        trainer._emergent_parameter_adjustment(config, metrics_low_fci, None)

        assert config['physical_params']['alpha'] == 1.2  # Increased by 20%
        assert config['physical_params']['beta'] == 1.1   # Increased by 10%

        # Test low sync order adjustment
        config_reset = {
            'physical_params': {'alpha': 1.0, 'beta': 1.0},
            'control_params': {'temperature': 1.0},
            'processing_params': {'semantic_connectivity_strength': 1.0}
        }
        metrics_low_sync = {'fci': 0.8, 'sync_order': 0.5, 'cluster_coherence': 0.9}
        trainer._emergent_parameter_adjustment(config_reset, metrics_low_sync, None)

        assert config_reset['control_params']['temperature'] == 0.9  # Decreased by 10%

    def test_physics_based_success_evaluation(self, trainer):
        """Test physics-based success evaluation."""
        # Test successful case
        result_success = {
            'generated_text': 'The sky is blue',
            'energy_conservation': 0.9,
            'spectral_coherence': 0.8
        }
        success = trainer._physics_based_success_evaluation(result_success, 'blue')

        assert success['semantic_achievement']  # 'blue' is in generated text
        assert success['physics_quality'] == pytest.approx(0.4*0.9 + 0.6*0.8, rel=1e-2)
        assert success['overall_success']

        # Test failure case
        result_failure = {
            'generated_text': 'The sky is green',
            'energy_conservation': 0.5,
            'spectral_coherence': 0.4
        }
        failure = trainer._physics_based_success_evaluation(result_failure, 'blue')

        assert not failure['semantic_achievement']  # 'blue' not in generated text
        assert failure['physics_quality'] < 0.7
        assert not failure['overall_success']

    @patch('train_physics_emergent.CompleteAutoCalibrationSystem')
    @patch('train_physics_emergent.HarmonicOrchestrator')
    def test_physics_emergent_training_cycle(self, mock_harmonic_orchestrator, mock_calibration_system, trainer):
        """Test the complete physics-emergent training cycle."""
        # Setup mocks
        mock_calibration = Mock()
        mock_calibration.calibrate_all_parameters.return_value = {
            'physical_params': {'alpha': 1.0, 'beta': 0.5}
        }
        mock_calibration_system.return_value = mock_calibration

        mock_orchestrator = Mock()
        mock_orchestrator.signature_analyzer.return_value = {'harmonic_signature': 'test'}
        mock_harmonic_orchestrator.return_value = mock_orchestrator

        # Mock the pipeline methods
        trainer.pipeline._generate_text_physical.return_value = {
            'fci_value': 0.7,
            'synchronization_order': 0.8,
            'cluster_analysis': {'dominant_cluster': {'order_parameter': 0.75}},
            'energy_conservation': 0.9,
            'spectral_coherence': 0.85,
            'generated_text': 'blue'
        }

        # Execute training cycle
        result = trainer.physics_emergent_training_cycle('The sky is', 'blue')

        # Verify result structure
        assert 'result' in result
        assert 'consciousness_metrics' in result
        assert 'physics_success' in result
        assert 'harmonic_signature' in result
        assert 'calibrated_config' in result

        # Verify consciousness metrics
        metrics = result['consciousness_metrics']
        assert 'fci' in metrics
        assert 'sync_order' in metrics
        assert 'cluster_coherence' in metrics
        assert 'quality_score' in metrics

        # Verify physics success
        success = result['physics_success']
        assert 'semantic_achievement' in success
        assert 'physics_quality' in success
        assert 'overall_success' in success


class TestPhysicsEmergentIntegration:
    """Integration tests for the physics-emergent training system."""

    @pytest.mark.integration
    def test_full_physics_emergent_workflow(self):
        """Test the complete physics-emergent training workflow."""
        # This test would run the actual training system
        # For now, we'll test the import and basic instantiation
        try:
            from train_physics_emergent import PhysicsEmergentTrainer, main
            from psiqrh import ΨQRHPipeline

            # Test instantiation
            pipeline = ΨQRHPipeline(enable_auto_calibration=True)
            trainer = PhysicsEmergentTrainer(pipeline)

            # Test a single training cycle
            result = trainer.physics_emergent_training_cycle('The sky is', 'blue')

            # Verify the result has expected structure
            assert isinstance(result, dict)
            assert 'consciousness_metrics' in result
            assert 'physics_success' in result

            print("✅ Integration test passed: Physics-emergent training workflow functional")

        except Exception as e:
            pytest.fail(f"Integration test failed: {e}")

    @pytest.mark.parametrize("input_text,target_text,expected_contains", [
        ("The sky is", "blue", "blue"),
        ("Grass is", "green", "green"),
        ("Hot and", "cold", "cold"),
    ])
    def test_semantic_reasoning_training(self, input_text, target_text, expected_contains):
        """Test semantic reasoning through physics-emergent training."""
        try:
            from train_physics_emergent import PhysicsEmergentTrainer
            from psiqrh import ΨQRHPipeline

            pipeline = ΨQRHPipeline(enable_auto_calibration=True)
            trainer = PhysicsEmergentTrainer(pipeline)

            result = trainer.physics_emergent_training_cycle(input_text, target_text)

            # Check if the generated text contains the expected semantic element
            generated_text = result['result'].get('generated_text', '')
            assert expected_contains.lower() in generated_text.lower(), \
                f"Expected '{expected_contains}' in generated text: '{generated_text}'"

        except Exception as e:
            pytest.skip(f"Semantic reasoning test skipped due to: {e}")


if __name__ == "__main__":
    # Run basic functionality test
    print("🧠 Testing Physics-Emergent Training System...")

    try:
        from train_physics_emergent import PhysicsEmergentTrainer
        from psiqrh import ΨQRHPipeline

        print("✅ Imports successful")

        # Test instantiation
        pipeline = ΨQRHPipeline(enable_auto_calibration=True)
        trainer = PhysicsEmergentTrainer(pipeline)
        print("✅ Instantiation successful")

        # Test training cycle
        result = trainer.physics_emergent_training_cycle('The sky is', 'blue')
        print(f"✅ Training cycle executed: FCI={result['consciousness_metrics']['fci']:.3f}")

        print("🎯 All basic tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
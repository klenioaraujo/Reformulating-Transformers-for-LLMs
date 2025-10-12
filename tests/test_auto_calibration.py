#!/usr/bin/env python3
"""
ΨQRH Auto-Calibration System Test Suite
========================================

Comprehensive tests for the refactored CompleteAutoCalibrationSystem integration.
Tests the clean data flow where calibrated parameters are passed explicitly through the pipeline.
"""

import pytest
import torch
import numpy as np
import json
import yaml
from unittest.mock import Mock, patch
import sys
import os
from pathlib import Path

# Add project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the pipeline and calibration system
from psiqrh import ΨQRHPipeline
from src.core.complete_auto_calibration_system import CompleteAutoCalibrationSystem


class TestAutoCalibrationSystem:
    """Test the CompleteAutoCalibrationSystem in isolation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calibration_system = CompleteAutoCalibrationSystem()

    def test_calibration_system_produces_valid_config(self):
        """Test that calibration system produces valid configuration dictionary."""
        test_text = "hello world"

        # Call calibration system
        calibrated_config = self.calibration_system.calibrate_all_parameters(
            text=test_text,
            fractal_signal=None,
            D_fractal=None
        )

        # Verify structure
        assert isinstance(calibrated_config, dict)
        assert 'physical_params' in calibrated_config
        assert 'architecture_params' in calibrated_config
        assert 'processing_params' in calibrated_config
        assert 'control_params' in calibrated_config

        # Verify physical parameters
        phys_params = calibrated_config['physical_params']
        required_phys_params = ['alpha', 'beta', 'I0', 'omega', 'k']
        for param in required_phys_params:
            assert param in phys_params
            assert isinstance(phys_params[param], (int, float))
            assert phys_params[param] > 0  # All physical params should be positive

        # Verify architecture parameters
        arch_params = calibrated_config['architecture_params']
        required_arch_params = ['embed_dim', 'num_heads', 'hidden_dim', 'num_layers']
        for param in required_arch_params:
            assert param in arch_params
            assert isinstance(arch_params[param], int)
            assert arch_params[param] > 0

        # Verify processing parameters
        proc_params = calibrated_config['processing_params']
        required_proc_params = ['dropout', 'max_history', 'vocab_size', 'epsilon']
        for param in required_proc_params:
            assert param in proc_params
            if param == 'dropout':
                assert 0 <= proc_params[param] <= 1
            else:
                assert proc_params[param] > 0

        # Verify control parameters
        ctrl_params = calibrated_config['control_params']
        required_ctrl_params = ['temperature', 'top_k', 'learning_rate']
        for param in required_ctrl_params:
            assert param in ctrl_params
            assert isinstance(ctrl_params[param], (int, float))
            assert ctrl_params[param] > 0

    def test_calibration_is_sensitive_to_input(self):
        """Test that calibration produces different results for different inputs."""
        simple_text = "hi"
        complex_text = "This is a much more complex sentence with many words and concepts to analyze."

        # Calibrate for simple text
        simple_config = self.calibration_system.calibrate_all_parameters(
            text=simple_text,
            fractal_signal=None,
            D_fractal=None
        )

        # Calibrate for complex text
        complex_config = self.calibration_system.calibrate_all_parameters(
            text=complex_text,
            fractal_signal=None,
            D_fractal=None
        )

        # Configurations should be different
        assert simple_config != complex_config

        # At least some physical parameters should differ
        simple_phys = simple_config['physical_params']
        complex_phys = complex_config['physical_params']

        # Check that at least one parameter differs significantly
        params_differ = False
        for param in ['alpha', 'beta', 'I0', 'omega', 'k']:
            if abs(simple_phys[param] - complex_phys[param]) > 1e-6:
                params_differ = True
                break

        assert params_differ, "Calibration should produce different parameters for different inputs"


class TestRefactoredPipeline:
    """Tests for the refactored pipeline using dependency injection."""

    def setup_method(self, method):
        """Set up a self-contained test environment using dependency injection."""
        import tempfile
        import os

        # Create a temporary directory for this test
        self.temp_dir = tempfile.mkdtemp()
        self.vocab_path = os.path.join(self.temp_dir, "native_vocab.json")

        # 1. Create a dummy native_vocab.json in the temporary path
        native_vocab = {
            "vocab_size": 5,
            "token_to_id": {"<pad>": 0, "<unk>": 1, "a": 2, "b": 3, "c": 4},
            "id_to_token": {"0": "<pad>", "1": "<unk>", "2": "a", "3": "b", "4": "c"}
        }
        with open(self.vocab_path, "w") as f:
            json.dump(native_vocab, f)

        # 2. Initialize the pipeline, injecting the path to the dummy vocab
        self.pipeline = ΨQRHPipeline(
            device='cpu',
            enable_auto_calibration=False,
            vocab_path=self.vocab_path  # Dependency Injection
        )

    def teardown_method(self, method):
        """Clean up temporary files."""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_pipeline_initializes_successfully(self):
        """Test that the pipeline initializes without the RuntimeError using dependency injection."""
        assert self.pipeline is not None
        assert self.pipeline.quantum_vocab_representations is not None
        assert self.pipeline.quantum_vocab_representations.shape[0] == 5  # vocab_size
        print("✅ Teste de Inicialização com Injeção de Dependência passou.")

    def test_full_pipeline_runs_with_mock_config(self):
        """Test if the full pipeline runs with dependency injection."""
        result = self.pipeline("a b c")
        assert isinstance(result, dict)
        assert 'response' in result
        assert isinstance(result['response'], str)
        print("✅ Teste de Execução com Injeção de Dependência passou.")


class TestCalibrationRobustness:
    """Test calibration system robustness and edge cases using dependency injection."""

    def setup_method(self, method):
        """Set up test environment with dependency injection."""
        import tempfile
        import os

        # Create a temporary directory for this test
        self.temp_dir = tempfile.mkdtemp()
        self.vocab_path = os.path.join(self.temp_dir, "native_vocab.json")

        # Create a dummy native_vocab.json in the temporary path
        native_vocab = {
            "vocab_size": 5,
            "token_to_id": {"<pad>": 0, "<unk>": 1, "a": 2, "b": 3, "c": 4},
            "id_to_token": {"0": "<pad>", "1": "<unk>", "2": "a", "3": "b", "4": "c"}
        }
        with open(self.vocab_path, "w") as f:
            json.dump(native_vocab, f)

        # Initialize the pipeline with dependency injection
        self.pipeline = ΨQRHPipeline(
            device='cpu',
            enable_auto_calibration=False,
            vocab_path=self.vocab_path
        )

    def teardown_method(self, method):
        """Clean up temporary files."""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_empty_text_calibration(self):
        """Test calibration with empty text input using dependency injection."""
        result = self.pipeline("")

        # Empty text should result in error status (expected behavior)
        assert result['status'] == 'error'
        assert 'error' in result

    def test_long_text_calibration(self):
        """Test calibration with very long text input using dependency injection."""
        long_text = "This is a very long text input. " * 100
        result = self.pipeline(long_text)

        assert result['status'] == 'success'
        assert 'calibration_config' in result
        assert 'physical_params' in result['calibration_config']

    def test_special_characters_calibration(self):
        """Test calibration with special characters and unicode using dependency injection."""
        special_text = "Hello 🌍! Ça va? ñoño αβγδε 123!@#"
        result = self.pipeline(special_text)

        assert result['status'] == 'success'
        assert 'calibration_config' in result
        assert 'physical_params' in result['calibration_config']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
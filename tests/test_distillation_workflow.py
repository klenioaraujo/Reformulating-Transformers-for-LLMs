#!/usr/bin/env python3
"""
Test Distillation Workflow - End-to-End Integration Test
=======================================================

Comprehensive test suite for the ΨQRH knowledge distillation workflow.
Tests the complete pipeline from model download to distilled model inference.
"""

import pytest
import torch
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import ΨQRH components
from src.architecture.psiqrh_transformer import PsiQRHTransformer


class TestDistillationWorkflow:
    """Test class for distillation workflow integration testing."""

    def setup_method(self):
        """Setup test fixtures."""
        self.test_model = "gpt2"
        self.source_dir = Path("models/source") / self.test_model.replace('/', '_')
        self.distilled_dir = Path("models/distilled")
        self.distilled_model_path = self.distilled_dir / f"psiqrh_distilled_{self.test_model.replace('/', '_')}.pt"

    def test_model_download_artifacts(self):
        """Test that model download creates expected artifacts."""
        # Check if source directory exists
        assert self.source_dir.exists(), f"Source model directory not found: {self.source_dir}"

        # Check for essential model files
        expected_files = [
            'config.json',
            'pytorch_model.bin',  # or pytorch_model-00001-of-00002.bin etc.
            'tokenizer.json',
            'tokenizer_config.json',
            'metadata.json'  # Our custom metadata
        ]

        found_files = list(self.source_dir.glob("*"))
        found_names = [f.name for f in found_files]

        # Check for at least some essential files
        essential_found = any(name in found_names for name in ['config.json', 'pytorch_model.bin'])
        assert essential_found, f"Essential model files not found in {self.source_dir}. Found: {found_names}"

        # Check for our metadata file
        metadata_path = self.source_dir / 'metadata.json'
        assert metadata_path.exists(), f"Metadata file not found: {metadata_path}"

        # Validate metadata content
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        required_keys = ['model_name', 'vocab_size', 'model_type', 'hidden_size']
        for key in required_keys:
            assert key in metadata, f"Required metadata key missing: {key}"

        print(f"✅ Model download artifacts validated for {self.test_model}")

    def test_distilled_model_creation(self):
        """Test that distilled model was created successfully."""
        # Check if distilled directory exists
        assert self.distilled_dir.exists(), f"Distilled models directory not found: {self.distilled_dir}"

        # Check if specific distilled model exists
        assert self.distilled_model_path.exists(), f"Distilled model not found: {self.distilled_model_path}"

        # Load and validate checkpoint structure
        checkpoint = torch.load(self.distilled_model_path, map_location='cpu')

        # Check for required keys in checkpoint
        required_keys = ['model_state_dict', 'config', 'distillation_info']
        for key in required_keys:
            assert key in checkpoint, f"Required checkpoint key missing: {key}"

        # Validate config structure
        config = checkpoint['config']
        required_config_keys = ['vocab_size', 'd_model', 'framework', 'conversion_method']
        for key in required_config_keys:
            assert key in config, f"Required config key missing: {key}"

        assert config['framework'] == 'ΨQRH', f"Unexpected framework: {config['framework']}"
        assert 'distillation' in config['conversion_method'], f"Unexpected conversion method: {config['conversion_method']}"

        # Validate distillation info
        dist_info = checkpoint['distillation_info']
        required_dist_keys = ['source_model', 'dynamic_dataset_generation', 'vocabulary_harmonization']
        for key in required_dist_keys:
            assert key in dist_info, f"Required distillation info key missing: {key}"

        print(f"✅ Distilled model validated: {self.distilled_model_path}")

    def test_distilled_model_inference(self):
        """Test that the distilled model can perform inference correctly."""
        # Load checkpoint
        checkpoint = torch.load(self.distilled_model_path, map_location='cpu')

        # Extract configuration
        config = checkpoint['config']

        # Create PsiQRH model instance
        model = PsiQRHTransformer(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            n_layers=config.get('n_layers', 6),
            n_heads=config.get('n_heads', 8),
            dim_feedforward=config.get('dim_feedforward', config['d_model'] * 4),
            max_seq_length=config.get('max_seq_length', 512),
            quaternion_multiplier=config.get('quaternion_multiplier', 4)
        )

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Test inference with a simple code snippet
        test_input = "def hello_world():"
        tokenizer_mock = MagicMock()
        tokenizer_mock.encode.return_value = [1, 2, 3, 4, 5]  # Mock token IDs
        tokenizer_mock.decode.return_value = "print('Hello, World!')"  # Mock output

        # Create input tensor (mock token IDs)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)  # [batch_size, seq_len]

        # Perform inference
        with torch.no_grad():
            outputs = model(input_ids)

        # Validate output structure
        assert isinstance(outputs, torch.Tensor), "Model output should be a tensor"
        assert outputs.dim() == 3, f"Expected 3D output tensor, got {outputs.dim()}D"

        batch_size, seq_len, vocab_size = outputs.shape
        assert batch_size == 1, f"Expected batch_size=1, got {batch_size}"
        assert seq_len == 5, f"Expected seq_len=5, got {seq_len}"
        assert vocab_size == config['vocab_size'], f"Expected vocab_size={config['vocab_size']}, got {vocab_size}"

        # Check that outputs are not NaN or Inf
        assert torch.isfinite(outputs).all(), "Model outputs contain NaN or Inf values"

        # Check that outputs have reasonable magnitude (not all zeros)
        assert outputs.abs().mean() > 1e-6, "Model outputs are suspiciously close to zero"

        print(f"✅ Distilled model inference validated")
        print(f"   📊 Output shape: {outputs.shape}")
        print(f"   📊 Output mean: {outputs.mean().item():.6f}")
        print(f"   📊 Output std: {outputs.std().item():.6f}")

    def test_evaluation_artifacts(self):
        """Test that evaluation artifacts were created (if evaluation was run)."""
        # Check for evaluation results directory
        eval_dir = Path("reports/evaluation")
        if eval_dir.exists():
            # Look for recent evaluation files
            eval_files = list(eval_dir.glob("*.json"))
            if eval_files:
                # Load most recent evaluation
                latest_eval = max(eval_files, key=lambda x: x.stat().st_mtime)

                with open(latest_eval, 'r') as f:
                    eval_data = json.load(f)

                # Check for expected evaluation metrics
                if 'metrics' in eval_data:
                    metrics = eval_data['metrics']
                    # Check for some common metrics
                    expected_metrics = ['fci_value', 'unitarity_score', 'energy_conservation']
                    found_metrics = [m for m in expected_metrics if m in metrics]

                    if found_metrics:
                        print(f"✅ Evaluation artifacts validated: {latest_eval}")
                        print(f"   📊 Found metrics: {found_metrics}")
                    else:
                        print(f"⚠️  Evaluation file found but expected metrics missing: {latest_eval}")
                else:
                    print(f"⚠️  Evaluation file found but no metrics section: {latest_eval}")
            else:
                print("⚠️  No evaluation files found - evaluation may not have been run")
        else:
            print("⚠️  Evaluation directory not found - evaluation may not have been run")

    @pytest.mark.parametrize("model_name", [
        "gpt2",
        "microsoft/DialoGPT-medium"
    ])
    def test_workflow_with_different_models(self, model_name):
        """Test the workflow can handle different model names."""
        # This test validates that the workflow can handle various model naming conventions
        normalized_name = model_name.replace('/', '_')
        expected_source_dir = Path("models/source") / normalized_name
        expected_distilled_path = Path("models/distilled") / f"psiqrh_distilled_{normalized_name}.pt"

        # Check that paths are constructed correctly
        assert str(expected_source_dir).replace('/', '_') == str(expected_source_dir), "Path normalization works"
        assert str(expected_distilled_path).endswith('.pt'), "Distilled model has correct extension"

        print(f"✅ Workflow path construction validated for model: {model_name}")


def test_full_distillation_process():
    """Main integration test that runs the complete distillation workflow."""
    # This test would be run by the Makefile target
    # It validates that all components work together end-to-end

    test_instance = TestDistillationWorkflow()

    # Run all validation steps
    test_instance.test_model_download_artifacts()
    test_instance.test_distilled_model_creation()
    test_instance.test_distilled_model_inference()
    test_instance.test_evaluation_artifacts()

    print("🎉 Full distillation workflow integration test PASSED!")


if __name__ == "__main__":
    # Allow running specific test with model name
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', default='gpt2',
                        help='Model name to test distillation for')

    args = parser.parse_args()

    # Update test instance with provided model name
    test_instance = TestDistillationWorkflow()
    test_instance.test_model = args.model_name
    test_instance.source_dir = Path("models/source") / args.model_name.replace('/', '_')
    test_instance.distilled_model_path = Path("models/distilled") / f"psiqrh_distilled_{args.model_name.replace('/', '_')}.pt"

    # Run tests
    test_full_distillation_process()
import pytest
import torch
import sys
import os

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ΨQRH import QuantumResonanceHarmonic
from quartz_light_prototype import QuartzLight
from qrh_layer import QRHLayer


class TestMultiDevice:
    """Test suite for multi-device compatibility (CPU, CUDA, MPS)"""

    @pytest.fixture(params=['cpu'])
    def device(self, request):
        """Fixture that provides available devices for testing"""
        device_name = request.param

        # Add CUDA if available
        if device_name == 'cuda' and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Add MPS if available
        if device_name == 'mps' and not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        return torch.device(device_name)

    @pytest.fixture(params=['cpu', 'cuda', 'mps'])
    def all_devices(self, request):
        """Fixture that tests all available devices"""
        device_name = request.param

        if device_name == 'cuda' and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        if device_name == 'mps' and not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        return torch.device(device_name)

    def test_qrh_device_compatibility(self, all_devices):
        """Test QuantumResonanceHarmonic works on all devices"""
        device = all_devices

        # Create QRH instance
        qrh = QuantumResonanceHarmonic(
            sequence_length=128,
            embedding_dim=64,
            frequency=1.0,
            phase_shift=0.0,
            amplitude=1.0,
            device=device
        )

        # Test input tensor
        batch_size = 2
        seq_len = 128
        embed_dim = 64
        x = torch.randn(batch_size, seq_len, embed_dim, device=device)

        # Forward pass
        output = qrh(x)

        # Assertions
        assert output.device == device
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_quartz_light_device_compatibility(self, all_devices):
        """Test QuartzLight model works on all devices"""
        device = all_devices

        # Create QuartzLight model
        model = QuartzLight(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            max_seq_len=64,
            device=device
        )

        # Test input
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        # Forward pass
        output = model(input_ids)

        # Assertions
        assert output.device == device
        assert output.shape == (batch_size, seq_len, 1000)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_qrh_layer_device_compatibility(self, all_devices):
        """Test QRHLayer works on all devices"""
        device = all_devices

        # Create QRH layer
        layer = QRHLayer(
            d_model=128,
            n_heads=4,
            dropout=0.1,
            device=device
        )

        # Test input
        batch_size = 2
        seq_len = 64
        d_model = 128
        x = torch.randn(batch_size, seq_len, d_model, device=device)

        # Forward pass
        output = layer(x)

        # Assertions
        assert output.device == device
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_device_transfer(self):
        """Test that models can be transferred between devices"""
        # Start on CPU
        qrh = QuantumResonanceHarmonic(
            sequence_length=64,
            embedding_dim=32,
            device=torch.device('cpu')
        )

        x = torch.randn(1, 64, 32)
        output_cpu = qrh(x)

        assert output_cpu.device.type == 'cpu'

        # Test CUDA transfer if available
        if torch.cuda.is_available():
            qrh_cuda = qrh.to('cuda')
            x_cuda = x.to('cuda')
            output_cuda = qrh_cuda(x_cuda)

            assert output_cuda.device.type == 'cuda'
            # Results should be similar (allowing for floating point differences)
            torch.testing.assert_close(
                output_cpu,
                output_cuda.cpu(),
                rtol=1e-4,
                atol=1e-4
            )

        # Test MPS transfer if available
        if torch.backends.mps.is_available():
            qrh_mps = qrh.to('mps')
            x_mps = x.to('mps')
            output_mps = qrh_mps(x_mps)

            assert output_mps.device.type == 'mps'
            # Results should be similar (allowing for floating point differences)
            torch.testing.assert_close(
                output_cpu,
                output_mps.cpu(),
                rtol=1e-4,
                atol=1e-4
            )

    def test_automatic_device_detection(self):
        """Test automatic device detection logic"""
        # Test the device detection from train_tiny_shakespeare.py
        expected_device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu'
        )

        # Create model with auto-detected device
        qrh = QuantumResonanceHarmonic(
            sequence_length=32,
            embedding_dim=16,
            device=expected_device
        )

        # Test that it works
        x = torch.randn(1, 32, 16, device=expected_device)
        output = qrh(x)

        assert output.device == expected_device
        assert not torch.isnan(output).any()

    def test_mixed_precision_compatibility(self, all_devices):
        """Test compatibility with mixed precision training"""
        device = all_devices

        # Skip MPS for mixed precision as it may not be fully supported
        if device.type == 'mps':
            pytest.skip("Mixed precision may not be fully supported on MPS")

        qrh = QuantumResonanceHarmonic(
            sequence_length=32,
            embedding_dim=64,
            device=device
        )

        x = torch.randn(1, 32, 64, device=device)

        # Test with autocast
        if device.type == 'cuda':
            with torch.autocast(device_type='cuda'):
                output = qrh(x)
                assert output.device == device
                assert not torch.isnan(output).any()
        else:
            # CPU autocast
            with torch.autocast(device_type='cpu'):
                output = qrh(x)
                assert output.device == device
                assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
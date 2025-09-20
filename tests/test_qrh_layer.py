import torch
import pytest
from qrh_layer import QRHLayer, QRHConfig

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_qrh_layer_device_agnostic(device):
    """Tests that the QRHLayer runs on different devices without crashing."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    try:
        # Configure and initialize the layer on the target device
        config = QRHConfig(embed_dim=16)
        layer = QRHLayer(config).to(device)
        
        # Create input tensor on the same device
        x = torch.randn(2, 32, 4 * config.embed_dim, device=device)
        
        # Forward pass
        output = layer(x)
        
        # Assertions
        assert output.device.type == device
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
        
    except Exception as e:
        pytest.fail(f"Layer failed on device '{device}' with exception: {e}")

@pytest.mark.parametrize("use_learned_rotation", [True, False])
def test_qrh_layer_grad_flow(use_learned_rotation):
    """Tests gradient flow for learnable parameters."""
    config = QRHConfig(embed_dim=8, use_learned_rotation=use_learned_rotation)
    layer = QRHLayer(config)
    
    x = torch.randn(2, 16, 4 * config.embed_dim)
    
    # Ensure input requires gradients for the test
    x.requires_grad = True
    
    # Forward pass
    output = layer(x)
    # Dummy loss
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check if project layers have gradients
    assert layer.v_proj.weight.grad is not None
    assert layer.out_proj.weight.grad is not None
    
    # Check if rotation parameters have gradients only when they are learnable
    if use_learned_rotation:
        assert layer.theta_left.grad is not None
        assert layer.omega_left.grad is not None
        assert layer.phi_left.grad is not None
        assert layer.theta_right.grad is not None
        assert layer.omega_right.grad is not None
        assert layer.phi_right.grad is not None
    else:
        # Buffers should not have gradients
        assert layer.theta_left.grad is None


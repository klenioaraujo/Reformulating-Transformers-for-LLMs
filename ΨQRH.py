import torch

# Import the classes from the new modules to maintain the public signature
from quaternion_operations import QuaternionOperations
from spectral_filter import SpectralFilter
from qrh_layer import QRHLayer
from gate_controller import GateController
from negentropy_transformer_block import NegentropyTransformerBlock

# The main execution block for usage examples and tests
if __name__ == "__main__":
    # Configuration
    embed_dim = 16
    batch_size = 2
    seq_len = 8

    # Input data
    x = torch.randn(batch_size, seq_len, 4 * embed_dim)

    # Layer with learnable rotation
    layer = QRHLayer(embed_dim, use_learned_rotation=True)

    # Forward test
    output = layer(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Rotation parameters - theta_left: {layer.theta_left.item():.4f}, "
          f"omega_left: {layer.omega_left.item():.4f}, phi_left: {layer.phi_left.item():.4f}")
    print(f"Rotation parameters - theta_right: {layer.theta_right.item():.4f}, "
          f"omega_right: {layer.omega_right.item():.4f}, phi_right: {layer.phi_right.item():.4f}")

    # Backward test
    loss = output.sum()
    loss.backward()

    print("Gradients computed successfully!")
    print("Layer implementation is working correctly.")

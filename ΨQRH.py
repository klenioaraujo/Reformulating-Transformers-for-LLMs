import torch
import yaml
import sys
from typing import Optional


# Import the classes from the new modules to maintain the public signature
from quaternion_operations import QuaternionOperations
from spectral_filter import SpectralFilter
from qrh_layer import QRHLayer, QRHConfig
from gate_controller import GateController
from negentropy_transformer_block import NegentropyTransformerBlock


class QRHFactory:
    @staticmethod
    def create_qrh_layer(config_path: str, device: Optional[str] = None) -> QRHLayer:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        config = QRHConfig(**config_dict['qrh_layer'])
        if device:
            config.device = device
        layer = QRHLayer(config)
        return layer.to(config.device)


def example_yaml_usage(config_path: str = "configs/qrh_config.yaml"):
    """Example: Loading config from YAML and running the layer."""
    print(f"--- Running YAML-based Usage Example from '{config_path}' ---")
    
    # 1. Load config from YAML file
    try:
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)['qrh_layer']
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: Could not load or parse '{config_path}'. {e}")
        return

    # 2. Create QRHConfig from dictionary
    config = QRHConfig(**config_dict)

    # 3. Handle device selection
    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU.")
        device = "cpu"

    # 4. Initialize layer and move to device
    layer = QRHLayer(config).to(device)
    
    # 5. Create dummy data and run forward pass
    x = torch.randn(2, 32, 4 * config.embed_dim, device=device)
    output = layer(x)

    print(f"Successfully ran layer configured from YAML on device: {output.device}")
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    print("----------------------------------------------------------\n")
    return output


# The main execution block for usage examples and tests
if __name__ == "__main__":
    # You can optionally pass a config file path as a command-line argument
    config_file = sys.argv[1] if len(sys.argv) > 1 else "configs/qrh_config.yaml"

    # Test QRHFactory
    print("--- Testing QRHFactory ---")
    layer = QRHFactory.create_qrh_layer(config_file, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"QRHFactory created layer on device: {layer.config.device}")

    example_yaml_usage(config_path=config_file)

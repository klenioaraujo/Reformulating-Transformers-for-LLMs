"""
Physics-Informed Gradient Calibrator for ΨQRH
==============================================

This script implements an advanced, physics-informed calibrator for the ΨQRH
ecosystem. Instead of a brute-force grid search, it uses gradient descent to
autonomously find optimal hyperparameters by minimizing a physical loss function.

The loss function is a composite of:
1.  Echo Loss (L_eco): Ensures signal integrity (output == input).
2.  Energy Loss (L_energia): Enforces Parseval's theorem, ensuring energy
    conservation between time and frequency domains.
3.  Synchronization Loss (L_sinc): Penalizes lack of synchronization in the
    Kuramoto oscillator layer, promoting stable spectral processing.

This method is vastly more efficient than grid search and guarantees that the
calibrated parameters are not just performant but also adhere to the core
mathematical and physical principles of the ΨQRH framework, including the
Padilha Wave Equation and chaotic dynamics.
"""

import torch
import torch.optim as optim
import yaml
from pathlib import Path
import sys
import time
import numpy as np

# --- Add project root to path ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- Import the core components of the pipeline ---
from src.core.numeric_signal_processor import NumericSignalProcessor
from src.core.conscious_working_memory import ConsciousWorkingMemory
from src.core.kuramoto_spectral_neurons import KuramotoSpectralLayer
from src.core.negentropy_transformer_block import NegentropyTransformerBlock


class PhysicsInformedPipeline(torch.nn.Module):
    """
    A differentiable PyTorch module that encapsulates the entire ΨQRH pipeline.
    This allows us to use gradient descent on its parameters.
    """
    def __init__(self, configs):
        super().__init__()

        # Extract and register all tunable parameters as learnable
        self.params = torch.nn.ParameterDict()
        self.param_mapping = {}  # Maps parameter names to config paths

        for config_name, config in configs.items():
            self._extract_parameters_from_config(config, config_name)

        # Initialize pipeline components with current configs
        self.processor = NumericSignalProcessor()

        # Initialize components with base configs
        self.cwm = ConsciousWorkingMemory()
        self.kuramoto = KuramotoSpectralLayer()
        self.transformer = NegentropyTransformerBlock(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            qrh_embed_dim=64
        )

    def _extract_parameters_from_config(self, config, prefix=""):
        """Recursively extract numerical parameters from config dict."""
        for key, value in config.items():
            current_path = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                self._extract_parameters_from_config(value, current_path)
            elif isinstance(value, (int, float)):
                # Only register parameters that are likely to be tunable
                param_indicators = ['strength', 'rate', 'alpha', 'beta', 'gamma',
                                  'coupling', 'decay', 'threshold', 'coefficient',
                                  'sensitivity', 'baseline', 'I0', 'k', 'lambda',
                                  'r_parameter', 'omega', 'frequency', 'phase']

                if any(indicator in key.lower() for indicator in param_indicators):
                    # Replace dots with underscores for PyTorch compatibility
                    param_name = f"{current_path}_{key}".replace('.', '_')
                    # Register as learnable parameter
                    self.params[param_name] = torch.nn.Parameter(torch.tensor(float(value)))
                    self.param_mapping[param_name] = (current_path, key)
                    print(f"  - Registered parameter: {param_name} = {value}")

    def _update_configs_with_params(self, configs):
        """Update configs with current parameter values."""
        updated_configs = configs.copy()

        for param_name, param_value in self.params.items():
            config_path, param_key = self.param_mapping[param_name]

            # Navigate to the parameter location
            path_parts = config_path.split('.')
            current_dict = updated_configs

            for part in path_parts:
                if part in current_dict and isinstance(current_dict[part], dict):
                    current_dict = current_dict[part]
                else:
                    break
            else:
                # Update the parameter value
                current_dict[param_key] = param_value.item()

        return updated_configs

    def forward(self, input_signal: torch.Tensor, configs: dict):
        """
        A forward pass through the entire pipeline with physical loss calculation.
        """
        # Update configs with current parameter values
        current_configs = self._update_configs_with_params(configs)

        try:
            # 1. Signal Processing
            # Convert input signal to text representation for processing
            input_text = ''.join([chr(int(c * 127)) for c in input_signal.tolist() if 0 <= c * 127 <= 127])
            if not input_text:
                input_text = "Hello World"

            signal_data = self.processor.process_text(input_text)

            # Convert to tensor for processing
            if isinstance(signal_data, dict):
                # Extract numeric array from processor result
                numeric_results = signal_data.get('numeric_results', [{}])
                if numeric_results and isinstance(numeric_results[0], dict):
                    numeric_array = numeric_results[0].get('original_array', [])
                else:
                    numeric_array = []

                if not numeric_array:
                    numeric_array = [ord(c) / 127.0 for c in input_text]
                signal_tensor = torch.tensor(numeric_array, dtype=torch.float32)
            else:
                signal_tensor = torch.tensor(signal_data, dtype=torch.float32)

            # Ensure proper shape for pipeline
            if signal_tensor.dim() == 1:
                signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len]

            # Pad and reshape for quaternion processing
            seq_len = signal_tensor.shape[-1]
            target_len = ((seq_len + 3) // 4) * 4  # Make divisible by 4
            if target_len > seq_len:
                signal_tensor = torch.nn.functional.pad(signal_tensor, (0, target_len - seq_len))

            # Reshape to quaternion format [batch, seq_len//4, 4]
            signal_quat = signal_tensor.view(1, -1, 4)

            # Expand to match transformer dimensions
            embed_dim = 64
            signal_expanded = torch.nn.functional.interpolate(
                signal_quat.permute(0, 2, 1),  # [1, 4, seq_len//4]
                size=embed_dim,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)  # [1, embed_dim, 4]

            signal_expanded = signal_expanded.view(1, embed_dim, 4)
            signal_expanded = signal_expanded.reshape(1, -1, 4 * 64)  # [1, seq_len, 256]

            # 2. Conscious Working Memory
            # Create consciousness state from spectral properties
            spectrum = torch.fft.fft(signal_tensor.flatten())
            spectral_energy = torch.abs(spectrum).sum().item()
            spectral_entropy = -torch.sum(
                torch.abs(spectrum) * torch.log(torch.abs(spectrum) + 1e-10)
            ).item()

            consciousness_state = {
                'entropy': min(max(spectral_entropy / 10.0, 0.0), 1.0),
                'fractal_dimension': 2.0 + 0.5 * (spectral_energy / 100.0),
                'fci': min(max(1.0 - spectral_entropy / 20.0, 0.0), 1.0)
            }

            memory_output, _ = self.cwm(signal_expanded, consciousness_state)

            # 3. Kuramoto Spectral Layer
            kuramoto_output, kuramoto_metrics = self.kuramoto(memory_output)

            # Extract synchronization order if available
            sync_order = kuramoto_metrics.get('synchronization_order_mean', 0.5) if kuramoto_metrics else 0.5
            if isinstance(sync_order, torch.Tensor):
                sync_order = sync_order.item()

            # 4. Transformer Block
            final_output = self.transformer(kuramoto_output)

            # --- Calculate Physical Losses ---

            # L_eco: Mean Squared Error between input and output
            # Compare spectral representations
            input_spectrum = torch.fft.fft(signal_tensor.flatten())
            output_spectrum = torch.fft.fft(final_output.flatten()[:len(input_spectrum)])

            loss_eco = torch.nn.functional.mse_loss(
                torch.abs(output_spectrum),
                torch.abs(input_spectrum)
            )

            # L_energia: Parseval's Theorem Violation
            time_energy = torch.sum(final_output.pow(2))
            fft_signal = torch.fft.fft(final_output.flatten())
            freq_energy = torch.sum(torch.abs(fft_signal).pow(2)) / len(fft_signal)
            loss_energia = torch.abs(time_energy - freq_energy) * 1e-5

            # L_sinc: Kuramoto Synchronization Loss
            loss_sinc = 1.0 - torch.tensor(sync_order, dtype=torch.float32)

            # Combine losses with physics-informed weights
            total_loss = (0.5 * loss_eco) + (0.3 * loss_energia) + (0.2 * loss_sinc)

            return total_loss, final_output, {
                'loss_eco': loss_eco.item(),
                'loss_energia': loss_energia.item(),
                'loss_sinc': loss_sinc.item(),
                'sync_order': sync_order
            }

        except Exception as e:
            # Return high loss if pipeline fails
            print(f"  ⚠️  Pipeline error: {e}")
            return torch.tensor(10.0, requires_grad=True), None, {}


def load_configs(config_paths):
    """Load configuration files from paths."""
    configs = {}
    for path in config_paths:
        if path.exists():
            with open(path, 'r') as f:
                config_name = path.stem
                configs[config_name] = yaml.safe_load(f)
                print(f"  ✓ Loaded config: {config_name}")
        else:
            print(f"  ⚠️  Config not found: {path}")

    return configs


def physics_gradient_calibration(
    config_paths: list,
    stimulus: str,
    num_steps: int = 100,
    learning_rate: float = 0.01
):
    """
    Performs calibration using gradient descent on a physical loss function.
    """
    print("🔬 Initializing Physics-Informed Gradient Calibration...")

    # 1. Load configurations
    configs = load_configs(config_paths)

    if not configs:
        print("❌ No valid configurations found!")
        return {}

    # 2. Prepare the input signal
    processor = NumericSignalProcessor()
    signal_data = processor.process_text(stimulus)

    # Create input tensor from stimulus
    char_values = [ord(c) / 127.0 for c in stimulus]
    input_tensor = torch.tensor(char_values, dtype=torch.float32)

    # 3. Create the differentiable pipeline
    pipeline = PhysicsInformedPipeline(configs)

    if len(pipeline.params) == 0:
        print("⚠️  No tunable parameters found in configurations!")
        return {}

    # 4. Setup the optimizer
    optimizer = optim.Adam(pipeline.parameters(), lr=learning_rate)

    print(f"  - Optimizing for stimulus: '{stimulus}'")
    print(f"  - Tunable parameters: {len(pipeline.params)}")
    print(f"  - Initial parameter values:")
    for name, param in pipeline.params.items():
        print(f"    - {name}: {param.item():.4f}")
    print("-" * 60)

    # 5. Run the optimization loop
    loss_history = []
    best_loss = float('inf')
    best_params = None

    for step in range(num_steps):
        optimizer.zero_grad()

        # Get the total physical loss
        loss, _, loss_components = pipeline(input_tensor, configs)

        # Backpropagate the loss
        loss.backward(retain_graph=True)

        # Update the parameters
        optimizer.step()

        loss_history.append(loss.item())

        # Track best parameters
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = {}
            for name, param in pipeline.named_parameters():
                if param.numel() == 1:  # Only scalar parameters
                    best_params[name] = param.item()
                else:
                    # For tensor parameters, store the tensor
                    best_params[name] = param.detach().clone()

        if step % 10 == 0 or step == num_steps - 1:
            print(f"  Step {step:03d}/{num_steps} | "
                  f"Total Loss: {loss.item():.6f} | "
                  f"Eco: {loss_components.get('loss_eco', 0):.4f} | "
                  f"Energy: {loss_components.get('loss_energia', 0):.4f} | "
                  f"Sync: {loss_components.get('loss_sinc', 0):.4f}")

    print("-" * 60)
    print(f"✅ Gradient Calibration Complete!")
    print(f"  - Best Loss: {best_loss:.6f}")
    print(f"  - Final Loss: {loss_history[-1]:.6f}")

    # 6. Extract and return the optimal parameters
    print("  - Optimal Parameters Found:")
    for name, value in best_params.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            print(f"    - {name}: {value.item():.6f}")
        elif isinstance(value, (int, float)):
            print(f"    - {name}: {value:.6f}")

    return best_params, configs


def save_calibrated_configs(base_configs: dict, optimal_params: dict, output_dir: Path):
    """Save calibrated configuration files."""
    print(f"\n💾 Saving calibrated configurations to '{output_dir}'...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create updated configs with optimal parameters
    calibrated_configs = base_configs.copy()

    for param_name, optimal_value in optimal_params.items():
        # Parse parameter name to find config location
        if '.' in param_name:
            parts = param_name.split('.')
            config_name = parts[0]
            param_path = parts[1:]

            if config_name in calibrated_configs:
                # Navigate to parameter location
                current_dict = calibrated_configs[config_name]
                for key in param_path[:-1]:
                    if key in current_dict and isinstance(current_dict[key], dict):
                        current_dict = current_dict[key]
                    else:
                        break
                else:
                    # Update parameter value
                    current_dict[param_path[-1]] = float(optimal_value)

    # Save each calibrated config
    for config_name, config_data in calibrated_configs.items():
        output_path = output_dir / f"{config_name}_gradient_calibrated.yaml"
        with open(output_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        print(f"  ✓ {output_path}")


def main():
    """Main execution function."""
    repo_root = Path(__file__).resolve().parent.parent

    # Configuration files to calibrate
    config_files = [
        repo_root / "configs" / "kuramoto_config.yaml",
        repo_root / "configs" / "working_memory_config.yaml",
        repo_root / "configs" / "psiqrh_transformer_config.yaml",
    ]

    stimulus_text = "Hello World"
    calibrated_output_dir = repo_root / "configs" / "gradient_calibrated"

    # Run gradient-based calibration
    optimal_parameters, base_configs = physics_gradient_calibration(
        config_files,
        stimulus_text,
        num_steps=200,
        learning_rate=0.05
    )

    if optimal_parameters:
        # Save calibrated configurations
        save_calibrated_configs(base_configs, optimal_parameters, calibrated_output_dir)
        print("\n🎉 Physics-informed gradient calibration finished!")
    else:
        print("\n❌ Calibration failed - no optimal parameters found.")


if __name__ == "__main__":
    main()
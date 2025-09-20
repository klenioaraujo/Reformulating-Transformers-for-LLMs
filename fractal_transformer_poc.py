import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft
import matplotlib.pyplot as plt
import math

# =============================================================================
# PART 1: CODE ADAPTED FROM ΨQRH.py AND needle_fractal_dimension.py
# =============================================================================

# --- Classes from ΨQRH.py ---
class QuaternionOperations:
    @staticmethod
    def multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
        w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z], dim=-1)

    @staticmethod
    def create_unit_quaternion(theta: torch.Tensor, omega: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        cos_theta_2, sin_theta_2 = torch.cos(theta / 2), torch.sin(theta / 2)
        cos_omega, sin_omega = torch.cos(omega), torch.sin(omega)
        cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
        return torch.stack([
            cos_theta_2, sin_theta_2 * cos_omega,
            sin_theta_2 * sin_omega * cos_phi, sin_theta_2 * sin_omega * sin_phi
        ], dim=-1)

class SpectralFilter(nn.Module):
    def __init__(self, alpha: float = 1.0, epsilon: float = 1e-10):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
    
    def forward(self, k: torch.Tensor) -> torch.Tensor:
        k_mag = torch.abs(k) + self.epsilon
        phase = self.alpha * torch.arctan(torch.log(k_mag))
        return torch.exp(1j * phase)

class QRHLayer(nn.Module):
    def __init__(self, embed_dim: int, alpha: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.total_dim = 4 * embed_dim
        self.spectral_filter = SpectralFilter(alpha)
        self.v_proj = nn.Linear(self.total_dim, self.total_dim)
        self.out_proj = nn.Linear(self.total_dim, self.total_dim)
        self.register_buffer('theta', torch.tensor(0.1))
        self.register_buffer('omega', torch.tensor(0.05))
        self.register_buffer('phi', torch.tensor(0.02))
        self.register_buffer('freqs', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device
        V = self.v_proj(x)
        D = self.embed_dim
        Ψ_w, Ψ_i, Ψ_j, Ψ_k = [V[:, :, i*D:(i+1)*D] for i in range(4)]
        Ψ_complex = Ψ_w + 1j * Ψ_i
        Ψ_fft = fft.fft(Ψ_complex, dim=1)
        if self.freqs is None or self.freqs.size(0) != seq_len:
            self.freqs = fft.fftfreq(seq_len, d=1.0, device=device)
        k = 2 * math.pi * self.freqs.view(1, seq_len, 1)
        F_k = self.spectral_filter(k)
        Ψ_filtered = Ψ_fft * F_k
        Ψ_ifft_complex = fft.ifft(Ψ_filtered, dim=1)
        Ψ_new = torch.cat([torch.real(Ψ_ifft_complex), torch.imag(Ψ_ifft_complex), Ψ_j, Ψ_k], dim=-1)
        Ψ_reshaped = Ψ_new.view(batch_size, seq_len, D, 4)
        R = QuaternionOperations.create_unit_quaternion(self.theta, self.omega, self.phi)
        rotated = QuaternionOperations.multiply(R.view(1, 1, 1, 4), Ψ_reshaped)
        Ψ_final = rotated.view(batch_size, seq_len, self.total_dim)
        return self.out_proj(Ψ_final) + x

# --- Class from needle_fractal_dimension.py (simplified version) ---
class FractalGenerator:
    def __init__(self):
        self.transforms = []
        self.points = None

    def add_transform(self, params):
        self.transforms.append(params)

    def generate(self, n_points=50000, warmup=100):
        points = np.zeros((n_points, 2))
        point = np.random.rand(2)
        for _ in range(warmup):
            t = np.array(self.transforms[np.random.randint(len(self.transforms))])
            A, b = t[:4].reshape(2, 2), t[4:]
            point = A.dot(point) + b
        for i in range(n_points):
            t = np.array(self.transforms[np.random.randint(len(self.transforms))])
            A, b = t[:4].reshape(2, 2), t[4:]
            point = A.dot(point) + b
            points[i] = point
        self.points = points
        return points

    def calculate_box_dimension(self):
        if self.points is None: self.generate()
        points_norm = (self.points - np.min(self.points, axis=0)) / (np.max(self.points, axis=0) - np.min(self.points, axis=0) + 1e-9)
        scales = np.logspace(-2.5, 0, 15, endpoint=False)
        counts = []
        for scale in scales:
            size = int(1/scale)
            grid = np.zeros((size, size), dtype=bool)
            indices = (points_norm * (size - 1)).astype(int)
            grid[indices[:, 0], indices[:, 1]] = True
            counts.append(np.sum(grid))
        valid = np.array(counts) > 0
        if np.sum(valid) < 2: return np.nan
        log_scales = np.log(1/scales[valid])
        log_counts = np.log(np.array(counts)[valid])
        coeffs = np.polyfit(log_scales, log_counts, 1)
        return coeffs[0]

# =============================================================================
# PART 2: PROOF OF CONCEPT FOR FRACTAL-LLM INTEGRATION
# =============================================================================

def map_dimension_to_alpha(D, D_min=1.0, D_max=2.0, alpha_min=0.5, alpha_max=2.5):
    """Maps the fractal dimension D to the alpha parameter."""
    if np.isnan(D): return 1.0 # Returns default alpha if D is not calculated
    # Linear mapping: (D - D_min) / (D_max - D_min) = (alpha - alpha_min) / (alpha_max - alpha_min)
    alpha = alpha_min + (D - D_min) * (alpha_max - alpha_min) / (D_max - D_min)
    return np.clip(alpha, alpha_min, alpha_max)

def main():
    """Main function that executes the proof of concept."""
    plt.style.use('seaborn-v0_8-paper')

    # --- STEP 1: Generate fractal and obtain its dimension ---
    print("1. Generating fractal (Sierpinski Triangle) and calculating its dimension...")
    sierpinski = FractalGenerator()
    s = 0.5
    transforms = [[s,0,0,s,0,0], [s,0,0,s,0.5,0], [s,0,0,s,0.25,0.5]]
    for t in transforms: sierpinski.add_transform(t)
    fractal_points = sierpinski.generate()
    D_fractal = sierpinski.calculate_box_dimension()
    D_theoretical = np.log(3)/np.log(2)

    # --- STEP 2: Map dimension to alpha parameter ---
    alpha_fractal = map_dimension_to_alpha(D_fractal)
    alpha_default = 1.0
    print(f"   - Theoretical Dimension (D): {D_theoretical:.4f}")
    print(f"   - Calculated Dimension (D): {D_fractal:.4f}")
    print("2. Mapping dimension to spectral filter alpha parameter...")
    print(f"   - Default Alpha: {alpha_default}")
    print(f"   - Fractal-Derived Alpha: {alpha_fractal:.4f}")

    # --- STEP 3: Configure layers and process data ---
    print("3. Configuring QRH layers and processing example data...")
    embed_dim = 32
    seq_len = 64
    batch_size = 1
    
    # Layer with default alpha
    layer_default = QRHLayer(embed_dim=embed_dim, alpha=alpha_default)
    # Layer with fractal-derived alpha
    layer_fractal = QRHLayer(embed_dim=embed_dim, alpha=alpha_fractal)

    # Input data (simulating an LLM state)
    input_tensor = torch.randn(batch_size, seq_len, 4 * embed_dim)

    # Process data
    output_default = layer_default(input_tensor)
    output_fractal = layer_fractal(input_tensor)

    # --- STEP 4: Analyze and visualize the impact ---
    print("4. Analyzing and visualizing the integration impact...")
    mse_diff = torch.mean((output_default - output_fractal)**2).item()
    print(f"   - Difference (MSE) between outputs: {mse_diff:.6f}")

    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('Proof of Concept: Fractal-LLM Integration (QRH Layer)', fontsize=18)

    # Plot 1: Fractal and its Dimension
    ax1 = axes[0]
    ax1.scatter(fractal_points[:, 0], fractal_points[:, 1], s=0.5, c='k', alpha=0.5)
    ax1.set_title(f'1. Generator Fractal (Sierpinski) | Calculated Dimension D ≈ {D_fractal:.3f}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')

    # Plot 2: Layer Outputs
    ax2 = axes[1]
    ax2.plot(output_default.detach().numpy().flatten()[::4*embed_dim], label='Default Output (α=1.0)', alpha=0.8)
    ax2.plot(output_fractal.detach().numpy().flatten()[::4*embed_dim], label=f'Fractal Output (α={alpha_fractal:.3f})', linestyle='--', alpha=0.8)
    ax2.set_title('2. Comparison of QRH Layer Outputs (one component)')
    ax2.set_xlabel('Position in Sequence')
    ax2.set_ylabel('Activation Value')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Absolute Difference
    ax3 = axes[2]
    diff = (output_default - output_fractal).abs().detach().numpy().flatten()[::4*embed_dim]
    ax3.plot(diff, color='red', label=f'Absolute Difference | MSE: {mse_diff:.6f}')
    ax3.set_title('3. Absolute Difference between Outputs')
    ax3.set_xlabel('Position in Sequence')
    ax3.set_ylabel('Absolute Difference')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('fractal_transformer_poc.png')
    print("\nProof of concept plot saved as 'fractal_transformer_poc.png'")

if __name__ == "__main__":
    main()

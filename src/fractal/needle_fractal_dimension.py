import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
import time

# =============================================================================
# 1. GERADOR DE FRACTAIS IFS PARAMÉTRICO
# =============================================================================
class FractalGenerator:
    def __init__(self, dim=2):
        self.dim = dim
        self.transforms = []
        self.points = None
        
    def add_transform(self, params):
        """Adds an affine transformation to the parameters"""
        self.transforms.append(params)
    
    def generate(self, n_points=100000, warmup=1000):
        """Generates fractal points using the chaos algorithm"""
        if not self.transforms:
            raise ValueError("No transformation defined")
            
        points = np.zeros((n_points, self.dim))
        point = np.random.rand(self.dim)
        
        # Warmup phase to converge to the attractor
        for _ in range(warmup):
            t_as_array = np.array(self.transforms[np.random.randint(len(self.transforms))])
            A = t_as_array[:self.dim**2].reshape(self.dim, self.dim)
            b = t_as_array[self.dim**2:]
            point = A.dot(point) + b
        
        # Point generation
        for i in range(n_points):
            t_as_array = np.array(self.transforms[np.random.randint(len(self.transforms))])
            A = t_as_array[:self.dim**2].reshape(self.dim, self.dim)
            b = t_as_array[self.dim**2:]
            point = A.dot(point) + b
            points[i] = point
            
        self.points = points
        return points
    
    def calculate_fractal_dimension(self, method='boxcount', extra_args={}):
        """Calculates the fractal dimension using various methods"""
        if self.points is None:
            raise ValueError("Generate the points first")
            
        if method == 'boxcount':
            return self._box_counting_dimension(**extra_args)
        elif method == 'spectral':
            return self._spectral_dimension(**extra_args)
        else:
            raise ValueError("Method not supported")
    
    def _box_counting_dimension(self, return_log_data=False):
        """Calculates dimension using box counting"""
        points = self.points
        
        # Normalize points to [0, 1]^dim
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        points_norm = (points - min_vals) / (max_vals - min_vals + 1e-9) # Add epsilon to avoid division by zero
        
        # Try various box scales
        scales = np.logspace(-2.5, 0, 20, endpoint=False) # Scales from 0.003 to 1
        counts = []
        
        for scale in scales:
            # Discretize the space
            size = int(1/scale)
            if self.dim == 2:
                grid = np.zeros((size, size), dtype=bool)
                indices = (points_norm[:, :2] * (size - 1)).astype(int) # Map to grid indices
                indices = np.clip(indices, 0, size-1) # Ensure indices are within bounds
                grid[indices[:, 0], indices[:, 1]] = True
            elif self.dim == 3:
                grid = np.zeros((size, size, size), dtype=bool)
                indices = (points_norm * (size - 1)).astype(int)
                indices = np.clip(indices, 0, size-1)
                grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
            counts.append(np.sum(grid))
        
        # Linear fit on log-log scale
        log_scales = np.log(1/scales)
        log_counts = np.log(counts)
        
        # Remove infinite or NaN values and zero counts
        valid = np.isfinite(log_counts) & np.isfinite(log_scales) & (np.array(counts) > 0)
        if np.sum(valid) < 2:
            return (np.nan, None) if return_log_data else np.nan
            
        coeffs = np.polyfit(log_scales[valid], log_counts[valid], 1)
        D = coeffs[0]
        
        if return_log_data:
            return D, (log_scales[valid], log_counts[valid], coeffs)
        return D
    
    def _spectral_dimension(self, grid_size=256, return_all=False):
        """Calculates dimension through power spectrum"""
        if self.dim != 2:
            raise NotImplementedError("Spectral dimension only implemented for 2D")
            
        # Create fractal image
        points = self.points
        # Use 2D histogram to create a dense fractal image
        grid, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=grid_size)
        grid = grid > 0  # Binarize the image
        
        # Transformada de Fourier 2D
        fft = np.fft.fft2(grid)
        power_spectrum = np.abs(fft)**2
        power_spectrum = np.fft.fftshift(power_spectrum) # Center the zero-frequency
        
        # Calculate radially averaged spectrum
        center = grid_size // 2
        y_idx, x_idx = np.indices((grid_size, grid_size))
        r = np.sqrt((x_idx - center)**2 + (y_idx - center)**2).astype(int) # Radial distance from center
        
        k_values = np.arange(1, center) # Radial frequencies (excluding center)
        spectrum_avg = np.array([np.mean(power_spectrum[r == k]) for k in k_values]) # Spectrum average for each k
        
        # Remove zeros and very small values for fitting
        valid = spectrum_avg > 1e-9
        if np.sum(valid) < 2: return (np.nan, *(None,)*5) if return_all else np.nan
        k_valid, spectrum_valid = k_values[valid], spectrum_avg[valid]
        
        # Power law fit P(k) = A * k^(-beta)
        def power_law(k, A, beta): return A * k**(-beta)
        
        try:
            params, _ = curve_fit(power_law, k_valid, spectrum_valid, p0=[spectrum_valid[0], 2], maxfev=5000)
            A, beta = params
            
            # Calculate fractal dimension (for 2D)
            D = (7 - beta) / 2  # Formula derived from spectrum-dimension relation
            
            if return_all:
                return D, beta, grid, power_spectrum, k_valid, spectrum_valid, params
            return D
        except:
            return (np.nan, *(None,)*5) if return_all else np.nan

# =============================================================================
# 2. SIMULADOR DE PULSO LASER
# =============================================================================
class LaserPulseSimulator:
    def __init__(self, I0=1.0, omega=2*np.pi, alpha=0.1, k=2*np.pi, beta=0.05):
        self.I0, self.omega, self.alpha, self.k, self.beta = I0, omega, alpha, k, beta

    def pulse(self, lambda_val, t):
        """Calculates the laser pulse at position lambda and time t"""
        return self.I0 * np.sin(self.omega * t + self.alpha * lambda_val) * \
               np.exp(1j * (self.omega * t - self.k * lambda_val + self.beta * lambda_val**2))

    def interact_with_fractal(self, fractal_points, scan_direction='x', t_range=(0, 1), n_t=100):
        """Simulates the pulse interaction with the fractal"""
        # Determine fractal bounds
        min_val, max_val = np.min(fractal_points), np.max(fractal_points)
        extent = max_val - min_val
        
        # Scan parameters
        t_values = np.linspace(t_range[0], t_range[1], n_t)
        lambda_values = np.linspace(min_val - 0.2*extent, max_val + 0.2*extent, n_t)
        
        # Matrix to store response
        response = np.zeros((n_t, n_t), dtype=complex)
        
        # For each point in space and time
        for i, t in enumerate(t_values):
            for j, l in enumerate(lambda_values):
                # Calculate pulse amplitude
                pulse_val = self.pulse(l, t)
                
                # Calculate interaction with fractal (simplified model)
                # The closer to the fractal, the greater the interaction
                if scan_direction == 'x':
                    dist = np.min(np.abs(fractal_points[:, 0] - l))
                else: # Assume 'y' or other direction, but use minimum distance to point set
                    dist = np.min(np.sqrt(np.sum((fractal_points - l)**2, axis=1)))
                
                # Interaction model: exponential decay with distance
                interaction = np.exp(-dist**2 / (0.1*extent)**2)
                response[i, j] = pulse_val * interaction
        return response, t_values, lambda_values

    def analyze_response(self, response):
        """Analyzes the response to extract fractal parameters"""
        # Power spectrum
        spectrum = np.abs(np.fft.fft2(response))**2
        spectrum = np.fft.fftshift(spectrum)
        
        # Calculate radially averaged spectrum
        n = spectrum.shape[0]
        center = n // 2
        y_idx, x_idx = np.indices((n, n))
        r = np.sqrt((x_idx - center)**2 + (y_idx - center)**2).astype(int)
        
        k_values = np.arange(1, center)
        spectrum_avg = np.array([np.mean(spectrum[r == k]) for k in k_values])
        
        # Remove zeros and very small values for fitting
        valid = spectrum_avg > 1e-9
        if np.sum(valid) < 2: return (np.nan,)*4
        k_valid, spectrum_valid = k_values[valid], spectrum_avg[valid]

        # Power law fit
        def power_law(k, A, beta): return A * k**(-beta)
        try:
            params, _ = curve_fit(power_law, k_valid, spectrum_valid, p0=[spectrum_valid[0], 2], maxfev=5000)
            A, beta = params
            
            # Calculate fractal dimension (approximation)
            D = (7 - beta) / 2  # For 2D structure
            
            return D, beta, k_valid, spectrum_valid
        except:
            return (np.nan,)*4

# =============================================================================
# 3. FUNÇÕES DE VISUALIZAÇÃO CONCEITUAL
# =============================================================================
def plot_box_counting_demo(points, scales_to_show):
    points_norm = (points - np.min(points, axis=0)) / (np.max(points, axis=0) - np.min(points, axis=0) + 1e-9)
    fig, axes = plt.subplots(1, len(scales_to_show), figsize=(5 * len(scales_to_show), 5))
    fig.suptitle('Conceptual Demonstration of Box-Counting Method', fontsize=16)
    
    for ax, scale in zip(axes, scales_to_show):
        size = int(1/scale)
        ax.scatter(points_norm[:, 0], points_norm[:, 1], s=1, c='blue', alpha=0.3)
        ax.set_xticks(np.linspace(0, 1, size + 1), minor=True)
        ax.set_yticks(np.linspace(0, 1, size + 1), minor=True)
        ax.grid(which='minor', color='red', linestyle='-', linewidth=0.5)
        ax.set_title(f'Scale: {scale:.3f} (Grid {size}x{size})')
        ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('needle_box_counting_demo.png')
    print("Box-Counting demonstration plot saved as 'needle_box_counting_demo.png'")

def plot_spectral_analysis_demo(grid, power_spectrum, k_valid, spectrum_valid, D_spec, beta_spec, fit_params):
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle('Density Spectral Analysis Demonstration', fontsize=16)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(grid, cmap='binary', origin='lower')
    ax1.set_title('1. Fractal Density Grid')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(np.log(power_spectrum + 1), cmap='viridis', origin='lower')
    ax2.set_title('2. 2D Power Spectrum (log)')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.loglog(k_valid, spectrum_valid, 'b.', label='Radial Spectrum Average')
    def power_law(k, A, beta): return A * k**(-beta)
    ax3.loglog(k_valid, power_law(k_valid, *fit_params), 'r-', label=f'Fit (β={beta_spec:.2f})')
    ax3.set_title(f'3. Power Law Fit (D={D_spec:.3f})')
    ax3.set_xlabel('Frequency (k)')
    ax3.set_ylabel('Power P(k)')
    ax3.legend()
    ax3.grid(True, which="both", linestyle='--')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('needle_spectral_analysis_demo.png')
    print("Spectral Analysis demonstration plot saved as 'needle_spectral_analysis_demo.png'")

# =============================================================================
# 4. EXECUÇÃO PRINCIPAL E DEMONSTRAÇÃO
# =============================================================================
def main():
    plt.style.use('seaborn-v0_8-paper')
    print("Generating Sierpinski Triangle (2D)...")
    sierpinski_2d = FractalGenerator(dim=2)
    s = 0.5
    transforms = [[s,0,0,s,0,0], [s,0,0,s,0.5,0], [s,0,0,s,0.25,0.5]]
    for t in transforms: sierpinski_2d.add_transform(t)
    points_2d = sierpinski_2d.generate(n_points=50000)

    # --- Analysis and Report ---
    D_box, log_data = sierpinski_2d.calculate_fractal_dimension('boxcount', extra_args={'return_log_data': True})
    D_spec, beta_spec, grid, ps, k_s, spec_s, fit_s = sierpinski_2d.calculate_fractal_dimension('spectral', extra_args={'return_all': True})
    D_theory = np.log(3)/np.log(2)

    print("\n" + "="*50)
    print("  FRACTAL ANALYSIS REPORT")
    print("="*50)
    print(f"Fractal: Sierpinski Triangle")
    print(f"Theoretical Dimension: {D_theory:.4f}")
    print(f"Dimension (Box-Counting): {D_box:.4f}")
    print(f"Dimension (Spectral Analysis): {D_spec:.4f}")
    print("="*50)

    # --- Conceptual Visualizations ---
    print("\nGenerating conceptual plots...")
    plot_box_counting_demo(points_2d, scales_to_show=[0.2, 0.1, 0.05])
    if D_spec is not np.nan:
        plot_spectral_analysis_demo(grid, ps, k_s, spec_s, D_spec, beta_spec, fit_s)

    # --- Results Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Sierpinski Triangle Analysis Results', fontsize=16)

    # Fractal Plot
    ax1.scatter(points_2d[:, 0], points_2d[:, 1], s=0.1, c='k', alpha=0.6)
    ax1.set_title('Generated Fractal Attractor')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Box-Counting Plot
    if log_data:
        log_scales, log_counts, coeffs = log_data
        ax2.plot(log_scales, log_counts, 'bo', label='Counting Data')
        fit_line = np.polyval(coeffs, log_scales)
        ax2.plot(log_scales, fit_line, 'r-', label=f'Linear Fit (D={coeffs[0]:.3f})')
        ax2.set_title('Box-Counting Analysis (Log-Log)')
        ax2.set_xlabel('log(1 / scale)')
        ax2.set_ylabel('log(N boxes)')
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('needle_results.png')
    print("Results plot saved as 'needle_results.png'")

if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
import time

# =============================================================================
# 1. GERADOR DE FRACTAIS IFS PARAMÉTRICO
# =============================================================================
class FractalGenerator:
    def __init__(self, dim=2):
        self.dim = dim
        self.transforms = []
        self.points = None
        
    def add_transform(self, params):
        """Adds an affine transformation to the parameters"""
        self.transforms.append(params)
    
    def generate(self, n_points=100000, warmup=1000):
        """Generates fractal points using the chaos algorithm"""
        if not self.transforms:
            raise ValueError("No transformation defined")
            
        points = np.zeros((n_points, self.dim))
        point = np.random.rand(self.dim)
        
        # Warmup phase to converge to the attractor
        for _ in range(warmup):
            t_as_array = np.array(self.transforms[np.random.randint(len(self.transforms))])
            A = t_as_array[:self.dim**2].reshape(self.dim, self.dim)
            b = t_as_array[self.dim**2:]
            point = A.dot(point) + b
        
        # Point generation
        for i in range(n_points):
            t_as_array = np.array(self.transforms[np.random.randint(len(self.transforms))])
            A = t_as_array[:self.dim**2].reshape(self.dim, self.dim)
            b = t_as_array[self.dim**2:]
            point = A.dot(point) + b
            points[i] = point
            
        self.points = points
        return points
    
    def calculate_fractal_dimension(self, method='boxcount', extra_args={}):
        """Calculates the fractal dimension using various methods"""
        if self.points is None:
            raise ValueError("Generate the points first")
            
        if method == 'boxcount':
            return self._box_counting_dimension(**extra_args)
        elif method == 'spectral':
            return self._spectral_dimension(**extra_args)
        else:
            raise ValueError("Method not supported")
    
    def _box_counting_dimension(self, return_log_data=False):
        """Calculates dimension using box counting"""
        points = self.points
        
        # Normalize points to [0, 1]^dim
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        points_norm = (points - min_vals) / (max_vals - min_vals + 1e-9) # Add epsilon to avoid division by zero
        
        # Try various box scales
        scales = np.logspace(-2.5, 0, 20, endpoint=False) # Scales from 0.003 to 1
        counts = []
        
        for scale in scales:
            # Discretize the space
            size = int(1/scale)
            if self.dim == 2:
                grid = np.zeros((size, size), dtype=bool)
                indices = (points_norm[:, :2] * (size - 1)).astype(int) # Map to grid indices
                indices = np.clip(indices, 0, size-1) # Ensure indices are within bounds
                grid[indices[:, 0], indices[:, 1]] = True
            elif self.dim == 3:
                grid = np.zeros((size, size, size), dtype=bool)
                indices = (points_norm * (size - 1)).astype(int)
                indices = np.clip(indices, 0, size-1)
                grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
            counts.append(np.sum(grid))
        
        # Linear fit on log-log scale
        log_scales = np.log(1/scales)
        log_counts = np.log(counts)
        
        # Remove infinite or NaN values and zero counts
        valid = np.isfinite(log_counts) & np.isfinite(log_scales) & (np.array(counts) > 0)
        if np.sum(valid) < 2:
            return (np.nan, None) if return_log_data else np.nan
            
        coeffs = np.polyfit(log_scales[valid], log_counts[valid], 1)
        D = coeffs[0]
        
        if return_log_data:
            return D, (log_scales[valid], log_counts[valid], coeffs)
        return D
    
    def _spectral_dimension(self, grid_size=256, return_all=False):
        """Calculates dimension through power spectrum"""
        if self.dim != 2:
            raise NotImplementedError("Spectral dimension only implemented for 2D")
            
        # Create fractal image
        points = self.points
        # Use 2D histogram to create a dense fractal image
        grid, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=grid_size)
        grid = grid > 0  # Binarize the image
        
        # Transformada de Fourier 2D
        fft = np.fft.fft2(grid)
        power_spectrum = np.abs(fft)**2
        power_spectrum = np.fft.fftshift(power_spectrum) # Center the zero-frequency
        
        # Calculate radially averaged spectrum
        center = grid_size // 2
        y_idx, x_idx = np.indices((grid_size, grid_size))
        r = np.sqrt((x_idx - center)**2 + (y_idx - center)**2).astype(int) # Radial distance from center
        
        k_values = np.arange(1, center) # Radial frequencies (excluding center)
        spectrum_avg = np.array([np.mean(power_spectrum[r == k]) for k in k_values]) # Spectrum average for each k
        
        # Remove zeros and very small values for fitting
        valid = spectrum_avg > 1e-9
        if np.sum(valid) < 2: return (np.nan, *(None,)*5) if return_all else np.nan
        k_valid, spectrum_valid = k_values[valid], spectrum_avg[valid]
        
        # Power law fit P(k) = A * k^(-beta)
        def power_law(k, A, beta): return A * k**(-beta)
        
        try:
            params, _ = curve_fit(power_law, k_valid, spectrum_valid, p0=[spectrum_valid[0], 2], maxfev=5000)
            A, beta = params
            
            # Calculate fractal dimension (for 2D)
            D = (7 - beta) / 2  # Formula derived from spectrum-dimension relation
            
            if return_all:
                return D, beta, grid, power_spectrum, k_valid, spectrum_valid, params
            return D
        except:
            return (np.nan, *(None,)*5) if return_all else np.nan

# =============================================================================
# 2. SIMULADOR DE PULSO LASER
# =============================================================================
class LaserPulseSimulator:
    def __init__(self, I0=1.0, omega=2*np.pi, alpha=0.1, k=2*np.pi, beta=0.05):
        self.I0, self.omega, self.alpha, self.k, self.beta = I0, omega, alpha, k, beta

    def pulse(self, lambda_val, t):
        """Calculates the laser pulse at position lambda and time t"""
        return self.I0 * np.sin(self.omega * t + self.alpha * lambda_val) * \
               np.exp(1j * (self.omega * t - self.k * lambda_val + self.beta * lambda_val**2))

    def interact_with_fractal(self, fractal_points, scan_direction='x', t_range=(0, 1), n_t=100):
        """Simulates the pulse interaction with the fractal"""
        # Determine fractal bounds
        min_val, max_val = np.min(fractal_points), np.max(fractal_points)
        extent = max_val - min_val
        
        # Scan parameters
        t_values = np.linspace(t_range[0], t_range[1], n_t)
        lambda_values = np.linspace(min_val - 0.2*extent, max_val + 0.2*extent, n_t)
        
        # Matrix to store response
        response = np.zeros((n_t, n_t), dtype=complex)
        
        # For each point in space and time
        for i, t in enumerate(t_values):
            for j, l in enumerate(lambda_values):
                # Calculate pulse amplitude
                pulse_val = self.pulse(l, t)
                
                # Calculate interaction with fractal (simplified model)
                # The closer to the fractal, the greater the interaction
                if scan_direction == 'x':
                    dist = np.min(np.abs(fractal_points[:, 0] - l))
                else: # Assume 'y' or other direction, but use minimum distance to point set
                    dist = np.min(np.sqrt(np.sum((fractal_points - l)**2, axis=1)))
                
                # Interaction model: exponential decay with distance
                interaction = np.exp(-dist**2 / (0.1*extent)**2)
                response[i, j] = pulse_val * interaction
        return response, t_values, lambda_values

    def analyze_response(self, response):
        """Analyzes the response to extract fractal parameters"""
        # Power spectrum
        spectrum = np.abs(np.fft.fft2(response))**2
        spectrum = np.fft.fftshift(spectrum)
        
        # Calculate radially averaged spectrum
        n = spectrum.shape[0]
        center = n // 2
        y_idx, x_idx = np.indices((n, n))
        r = np.sqrt((x_idx - center)**2 + (y_idx - center)**2).astype(int)
        
        k_values = np.arange(1, center)
        spectrum_avg = np.array([np.mean(spectrum[r == k]) for k in k_values])
        
        # Remove zeros and very small values for fitting
        valid = spectrum_avg > 1e-9
        if np.sum(valid) < 2: return (np.nan,)*4
        k_valid, spectrum_valid = k_values[valid], spectrum_avg[valid]

        # Power law fit
        def power_law(k, A, beta): return A * k**(-beta)
        try:
            params, _ = curve_fit(power_law, k_valid, spectrum_valid, p0=[spectrum_valid[0], 2], maxfev=5000)
            A, beta = params
            
            # Calculate fractal dimension (approximation)
            D = (7 - beta) / 2  # For 2D structure
            
            return D, beta, k_valid, spectrum_valid
        except:
            return (np.nan,)*4

# =============================================================================
# 3. FUNÇÕES DE VISUALIZAÇÃO CONCEITUAL
# =============================================================================
def plot_box_counting_demo(points, scales_to_show):
    points_norm = (points - np.min(points, axis=0)) / (np.max(points, axis=0) - np.min(points, axis=0) + 1e-9)
    fig, axes = plt.subplots(1, len(scales_to_show), figsize=(5 * len(scales_to_show), 5))
    fig.suptitle('Conceptual Demonstration of Box-Counting Method', fontsize=16)
    
    for ax, scale in zip(axes, scales_to_show):
        size = int(1/scale)
        ax.scatter(points_norm[:, 0], points_norm[:, 1], s=1, c='blue', alpha=0.3)
        ax.set_xticks(np.linspace(0, 1, size + 1), minor=True)
        ax.set_yticks(np.linspace(0, 1, size + 1), minor=True)
        ax.grid(which='minor', color='red', linestyle='-', linewidth=0.5)
        ax.set_title(f'Scale: {scale:.3f} (Grid {size}x{size})')
        ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('needle_box_counting_demo.png')
    print("Box-Counting demonstration plot saved as 'needle_box_counting_demo.png'")

def plot_spectral_analysis_demo(grid, power_spectrum, k_valid, spectrum_valid, D_spec, beta_spec, fit_params):
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle('Density Spectral Analysis Demonstration', fontsize=16)

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(grid, cmap='binary', origin='lower')
    ax1.set_title('1. Fractal Density Grid')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(np.log(power_spectrum + 1), cmap='viridis', origin='lower')
    ax2.set_title('2. 2D Power Spectrum (log)')

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.loglog(k_valid, spectrum_valid, 'b.', label='Radial Spectrum Average')
    def power_law(k, A, beta): return A * k**(-beta)
    ax3.loglog(k_valid, power_law(k_valid, *fit_params), 'r-', label=f'Fit (β={beta_spec:.2f})')
    ax3.set_title(f'3. Power Law Fit (D={D_spec:.3f})')
    ax3.set_xlabel('Frequency (k)')
    ax3.set_ylabel('Power P(k)')
    ax3.legend()
    ax3.grid(True, which="both", linestyle='--')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('needle_spectral_analysis_demo.png')
    print("Spectral Analysis demonstration plot saved as 'needle_spectral_analysis_demo.png'")

# =============================================================================
# 4. EXECUÇÃO PRINCIPAL E DEMONSTRAÇÃO
# =============================================================================
def main():
    plt.style.use('seaborn-v0_8-paper')
    print("Generating Sierpinski Triangle (2D)...")
    sierpinski_2d = FractalGenerator(dim=2)
    s = 0.5
    transforms = [[s,0,0,s,0,0], [s,0,0,s,0.5,0], [s,0,0,s,0.25,0.5]]
    for t in transforms: sierpinski_2d.add_transform(t)
    points_2d = sierpinski_2d.generate(n_points=50000)

    # --- Analysis and Report ---
    D_box, log_data = sierpinski_2d.calculate_fractal_dimension('boxcount', extra_args={'return_log_data': True})
    D_spec, beta_spec, grid, ps, k_s, spec_s, fit_s = sierpinski_2d.calculate_fractal_dimension('spectral', extra_args={'return_all': True})
    D_theory = np.log(3)/np.log(2)

    print("\n" + "="*50)
    print("  FRACTAL ANALYSIS REPORT")
    print("="*50)
    print(f"Fractal: Sierpinski Triangle")
    print(f"Theoretical Dimension: {D_theory:.4f}")
    print(f"Dimension (Box-Counting): {D_box:.4f}")
    print(f"Dimension (Spectral Analysis): {D_spec:.4f}")
    print("="*50)

    # --- Conceptual Visualizations ---
    print("\nGenerating conceptual plots...")
    plot_box_counting_demo(points_2d, scales_to_show=[0.2, 0.1, 0.05])
    if D_spec is not np.nan:
        plot_spectral_analysis_demo(grid, ps, k_s, spec_s, D_spec, beta_spec, fit_s)

    # --- Results Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Sierpinski Triangle Analysis Results', fontsize=16)

    # Fractal Plot
    ax1.scatter(points_2d[:, 0], points_2d[:, 1], s=0.1, c='k', alpha=0.6)
    ax1.set_title('Generated Fractal Attractor')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Box-Counting Plot
    if log_data:
        log_scales, log_counts, coeffs = log_data
        ax2.plot(log_scales, log_counts, 'bo', label='Counting Data')
        fit_line = np.polyval(coeffs, log_scales)
        ax2.plot(log_scales, fit_line, 'r-', label=f'Linear Fit (D={coeffs[0]:.3f})')
        ax2.set_title('Box-Counting Analysis (Log-Log)')
        ax2.set_xlabel('log(1 / scale)')
        ax2.set_ylabel('log(N boxes)')
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('needle_results.png')
    print("Results plot saved as 'needle_results.png'")

if __name__ == "__main__":
    main()
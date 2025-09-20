import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def generate_cantor_set(iterations, n_points):
    """
    Generates a deterministic representation of the 1D Cantor set.
    """
    signal = np.ones(n_points)
    
    for i in range(iterations):
        segment_len = n_points // (3**i)
        for j in range(3**i):
            # Identify the central third of each segment
            start = j * segment_len
            middle_start = start + segment_len // 3
            middle_end = start + 2 * segment_len // 3
            if middle_start < middle_end: # Ensure there is an interval to be removed
                signal[middle_start:middle_end] = 0
            
    return signal

def calculate_power_spectrum(signal):
    """
    Calculates the 1D power spectrum of a signal.
    """
    fft_result = np.fft.fft(signal)
    power_spectrum = np.abs(fft_result)**2
    # Only the first half of the spectrum is needed due to symmetry
    half_len = len(signal) // 2
    return power_spectrum[1:half_len]

def fit_power_law(spectrum):
    """
    Fits a power law P(k) = A * k**(-beta) to the spectrum.
    """
    k = np.arange(1, len(spectrum) + 1)
    
    # Remove zeros to avoid log problems and ensure sufficient points
    valid = spectrum > 1e-10
    if np.sum(valid) < 10:
        return np.nan, np.nan, None
        
    k_valid = k[valid]
    spectrum_valid = spectrum[valid]

    log_k = np.log(k_valid)
    log_spectrum = np.log(spectrum_valid)
    
    # Linear fit in log-log space
    try:
        coeffs, cov = np.polyfit(log_k, log_spectrum, 1, cov=True)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, None

    beta = -coeffs[0]
    A_log = coeffs[1]
    
    # Estimate beta error
    beta_err = np.sqrt(cov[0, 0]) if cov is not None and cov[0, 0] > 0 else np.nan
    
    fit_fn = lambda k_val: np.exp(A_log) * k_val**(-beta)
    
    return beta, beta_err, fit_fn

def main():
    """
    Main function to execute the proof of concept.
    """
    # --- Parameters ---
    iterations = 4
    n_points = 3**8  # Use a power of 3 for perfect alignment

    # --- Theoretical Calculations ---
    D_theoretical = np.log(2) / np.log(3)
    # For a 1D fractal signal/set (stationary), the relationship is β = 3 - 2D.
    # The formula β = 5 - 2D applies to fractional Brownian motion profiles (non-stationary).
    beta_theoretical = 3 - 2 * D_theoretical

    # --- Simulation ---
    # 1. Generate the Cantor set
    cantor_signal = generate_cantor_set(iterations, n_points)
    
    # 2. Calculate the power spectrum
    spectrum = calculate_power_spectrum(cantor_signal)
    
    # 3. Measure the beta exponent
    beta_measured, beta_error, fit_function = fit_power_law(spectrum)

    # --- Report ---
    print("="*60)
    print("  Proof of Concept: Fractal Dimension via Power Spectrum")
    print("="*60)
    print(f"Fractal: 1D Cantor Set")
    print(f"Iterations: {iterations}, Grid points: {n_points}")
    print("-" * 60)
    print(f"Theoretical Dimension (D = log(2)/log(3)): {D_theoretical:.4f}")
    print(f"Theoretical Spectral Exponent (β = 5 - 2D): {beta_theoretical:.4f}")
    if not np.isnan(beta_measured):
        print(f"Measured Spectral Exponent (β): {beta_measured:.4f} ± {beta_error:.4f}")
    else:
        print("Measured Spectral Exponent (β): Fit failed")
    print("-" * 60)
    
    if not np.isnan(beta_measured):
        error_percent = abs(beta_measured - beta_theoretical) / beta_theoretical * 100
        print(f"Relative measurement error: {error_percent:.2f}%")
    print("="*60)

    # --- Visualization ---
    plt.style.use('seaborn-v0_8-paper')
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle('Spectrum-Dimension Relationship Demonstration for Cantor Set', fontsize=16)
    
    # 1. Cantor Set Plot
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(np.arange(n_points) / n_points, cantor_signal, 'k-', linewidth=1)
    ax1.set_title(f'Cantor Set Signal (Iterations: {iterations})', fontsize=14)
    ax1.set_xlabel('Normalized Position')
    ax1.set_ylabel('Signal')
    ax1.set_yticks([0, 1])
    ax1.set_ylim([-0.1, 1.1])
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 2. Power Spectrum Plot (Log-Log)
    ax2 = fig.add_subplot(2, 1, 2)
    k = np.arange(1, len(spectrum) + 1)
    ax2.loglog(k, spectrum, 'b.', label='Measured Power Spectrum', alpha=0.5)
    
    # Fit line
    if fit_function is not None:
        k_fit = np.logspace(np.log10(k[k>0][0]), np.log10(k[-1]), 200)
        ax2.loglog(k_fit, fit_function(k_fit), 'r-', 
                   label=f'Power Law Fit (β = {beta_measured:.3f})', linewidth=2.5)

    ax2.set_title('Power Spectrum on Log-Log Scale', fontsize=14)
    ax2.set_xlabel('Wave Number (k)')
    ax2.set_ylabel('Power P(k)')
    ax2.legend()
    ax2.grid(True, which="both", linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('proof_of_concept.png')
    print("\nPlot saved as 'proof_of_concept.png'")

if __name__ == "__main__":
    main()

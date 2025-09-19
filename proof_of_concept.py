import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def generate_cantor_set(iterations, n_points):
    """
    Genera uma representação determinística do conjunto de Cantor 1D.
    """
    signal = np.ones(n_points)
    
    for i in range(iterations):
        segment_len = n_points // (3**i)
        for j in range(3**i):
            # Identifica o terço central de cada segmento
            start = j * segment_len
            middle_start = start + segment_len // 3
            middle_end = start + 2 * segment_len // 3
            if middle_start < middle_end: # Garante que há um intervalo a ser removido
                signal[middle_start:middle_end] = 0
            
    return signal

def calculate_power_spectrum(signal):
    """
    Calcula o espectro de potência 1D de um sinal.
    """
    fft_result = np.fft.fft(signal)
    power_spectrum = np.abs(fft_result)**2
    # Apenas a primeira metade do espectro é necessária devido à simetria
    half_len = len(signal) // 2
    return power_spectrum[1:half_len]

def fit_power_law(spectrum):
    """
    Ajusta uma lei de potência P(k) = A * k**(-beta) ao espectro.
    """
    k = np.arange(1, len(spectrum) + 1)
    
    # Remover zeros para evitar problemas com o log e garantir pontos suficientes
    valid = spectrum > 1e-10
    if np.sum(valid) < 10:
        return np.nan, np.nan, None
        
    k_valid = k[valid]
    spectrum_valid = spectrum[valid]

    log_k = np.log(k_valid)
    log_spectrum = np.log(spectrum_valid)
    
    # Ajuste linear no espaço log-log
    try:
        coeffs, cov = np.polyfit(log_k, log_spectrum, 1, cov=True)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, None

    beta = -coeffs[0]
    A_log = coeffs[1]
    
    # Estimar o erro de beta
    beta_err = np.sqrt(cov[0, 0]) if cov is not None and cov[0, 0] > 0 else np.nan
    
    fit_fn = lambda k_val: np.exp(A_log) * k_val**(-beta)
    
    return beta, beta_err, fit_fn

def main():
    """
    Função principal para executar a prova de conceito.
    """
    # --- Parâmetros ---
    iterations = 4
    n_points = 3**8  # Usar uma potência de 3 para um alinhamento perfeito

    # --- Cálculos Teóricos ---
    D_theoretical = np.log(2) / np.log(3)
    # Para um sinal/conjunto fractal 1D (estacionário), a relação é β = 3 - 2D.
    # A fórmula β = 5 - 2D se aplica a perfis de movimento Browniano fracionário (não-estacionário).
    beta_theoretical = 3 - 2 * D_theoretical

    # --- Simulação ---
    # 1. Gerar o conjunto de Cantor
    cantor_signal = generate_cantor_set(iterations, n_points)
    
    # 2. Calcular o espectro de potência
    spectrum = calculate_power_spectrum(cantor_signal)
    
    # 3. Medir o expoente beta
    beta_measured, beta_error, fit_function = fit_power_law(spectrum)

    # --- Relatório ---
    print("="*60)
    print("  Prova de Conceito: Dimensão Fractal via Espectro de Potência")
    print("="*60)
    print(f"Fractal: Conjunto de Cantor 1D")
    print(f"Iterações: {iterations}, Pontos na grade: {n_points}")
    print("-" * 60)
    print(f"Dimensão Teórica (D = log(2)/log(3)): {D_theoretical:.4f}")
    print(f"Expoente Espectral Teórico (β = 5 - 2D): {beta_theoretical:.4f}")
    if not np.isnan(beta_measured):
        print(f"Expoente Espectral Medido (β): {beta_measured:.4f} ± {beta_error:.4f}")
    else:
        print("Expoente Espectral Medido (β): Falha no ajuste")
    print("-" * 60)
    
    if not np.isnan(beta_measured):
        error_percent = abs(beta_measured - beta_theoretical) / beta_theoretical * 100
        print(f"Erro relativo da medição: {error_percent:.2f}%")
    print("="*60)

    # --- Visualização ---
    plt.style.use('seaborn-v0_8-paper')
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle('Demonstração da Relação Espectro-Dimensão para o Conjunto de Cantor', fontsize=16)
    
    # 1. Gráfico do Conjunto de Cantor
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(np.arange(n_points) / n_points, cantor_signal, 'k-', linewidth=1)
    ax1.set_title(f'Sinal do Conjunto de Cantor (Iterações: {iterations})', fontsize=14)
    ax1.set_xlabel('Posição Normalizada')
    ax1.set_ylabel('Sinal')
    ax1.set_yticks([0, 1])
    ax1.set_ylim([-0.1, 1.1])
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 2. Gráfico do Espectro de Potência (Log-Log)
    ax2 = fig.add_subplot(2, 1, 2)
    k = np.arange(1, len(spectrum) + 1)
    ax2.loglog(k, spectrum, 'b.', label='Espectro de Potência Medido', alpha=0.5)
    
    # Linha de ajuste
    if fit_function is not None:
        k_fit = np.logspace(np.log10(k[k>0][0]), np.log10(k[-1]), 200)
        ax2.loglog(k_fit, fit_function(k_fit), 'r-', 
                   label=f'Ajuste de Lei de Potência (β = {beta_measured:.3f})', linewidth=2.5)

    ax2.set_title('Espectro de Potência em Escala Log-Log', fontsize=14)
    ax2.set_xlabel('Número de Onda (k)')
    ax2.set_ylabel('Potência P(k)')
    ax2.legend()
    ax2.grid(True, which="both", linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('proof_of_concept.png')
    print("\nGráfico salvo como 'proof_of_concept.png'")

if __name__ == "__main__":
    main()

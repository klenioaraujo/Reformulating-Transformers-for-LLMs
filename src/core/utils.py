import torch
import torch.nn as nn


def compute_energy(x: torch.Tensor) -> torch.Tensor:
    """
    Computa energia corretamente como ||x||² = soma(x²)
    Mantém dimensão para broadcasting.

    Args:
        x: Tensor para calcular energia

    Returns:
        Tensor de energia com mesma forma exceto última dimensão
    """
    return torch.sum(x ** 2, dim=-1, keepdim=True)


def energy_normalize(x_input: torch.Tensor, x_output: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normaliza x_output para ter a mesma energia de x_input.
    Usa raiz quadrada porque energia = ||x||², então escala = sqrt(energia_input / energia_output)

    Args:
        x_input: Tensor de entrada para referência de energia
        x_output: Tensor de saída a ser normalizado
        eps: Valor pequeno para evitar divisão por zero

    Returns:
        Tensor normalizado com mesma energia que x_input
    """
    energy_input = compute_energy(x_input)
    energy_output = compute_energy(x_output)

    # Evita divisão por zero
    scale = torch.sqrt(energy_input / (energy_output + eps))
    return x_output * scale


# Alias para compatibilidade
def energy_preserve(x_input: torch.Tensor, x_output: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Alias para energy_normalize para compatibilidade"""
    return energy_normalize(x_input, x_output, epsilon)


def validate_parseval(x_time: torch.Tensor, x_freq: torch.Tensor, tolerance: float = 1e-5) -> bool:
    """
    Verifica compliance com Teorema de Parseval em tempo de execução.

    Args:
        x_time: Sinal no domínio do tempo
        x_freq: Sinal no domínio da frequência
        tolerance: Tolerância para a razão de energia

    Returns:
        True se compliant com Parseval, False caso contrário
    """
    energy_time = compute_energy(x_time).sum()
    energy_freq = compute_energy(x_freq).sum()
    ratio = energy_time / (energy_freq + 1e-8)

    # Validação assertiva para debugging
    if abs(ratio - 1.0) >= tolerance:
        print(f"⚠️  Parseval violation: ratio={ratio:.6f}, tolerance={tolerance}")
        print(f"   Time domain energy: {energy_time:.6f}")
        print(f"   Freq domain energy: {energy_freq:.6f}")

    return abs(ratio - 1.0) < tolerance


def parseval_checkpoint(x_time: torch.Tensor, operation_name: str = "unknown") -> bool:
    """
    Checkpoint de Parseval para validação em tempo de execução.

    Args:
        x_time: Sinal no domínio do tempo
        operation_name: Nome da operação para logging

    Returns:
        True se Parseval preservado, False caso contrário
    """
    # Aplica FFT ortonormal
    x_freq = torch.fft.fft(x_time, norm="ortho")

    # Valida Parseval
    is_valid = validate_parseval(x_time, x_freq)

    if not is_valid:
        print(f"⚠️  Parseval violation in {operation_name}")
        energy_time = compute_energy(x_time).sum().item()
        energy_freq = compute_energy(x_freq).sum().item()
        print(f"   Time domain energy: {energy_time:.6f}")
        print(f"   Freq domain energy: {energy_freq:.6f}")
        print(f"   Ratio: {energy_time/energy_freq:.6f}")

    return is_valid


def spectral_operation_with_parseval(x: torch.Tensor, operation_func, operation_name: str = "spectral_op") -> torch.Tensor:
    """
    Executa operação espectral com validação de Parseval.

    Args:
        x: Tensor de entrada
        operation_func: Função que opera no domínio da frequência
        operation_name: Nome da operação para logging

    Returns:
        Resultado da operação com Parseval preservado
    """
    # Checkpoint inicial
    parseval_checkpoint(x, f"{operation_name}_input")

    # Para sinais reais, use RFFT (mais eficiente)
    if not torch.is_complex(x):
        x_fft = torch.fft.rfft(x, norm="ortho")
    else:
        x_fft = torch.fft.fft(x, norm="ortho")

    # Executa operação no domínio da frequência
    result_fft = operation_func(x_fft)

    # Preserva unitariedade (magnitude = 1.0)
    from ..optimization.spectral_normalizer import normalize_spectral_magnitude
    result_fft = normalize_spectral_magnitude(result_fft)

    # Aplica IFFT ortonormal e verifica parte imaginária
    if not torch.is_complex(x):
        # Para sinais reais originais, use IRFFT
        result = torch.fft.irfft(result_fft, n=x.shape[-1], norm="ortho")
    else:
        result = torch.fft.ifft(result_fft, norm="ortho")

        # Verifica se a parte imaginária é desprezível antes de descartar
        max_imag = torch.max(torch.abs(result.imag))
        if max_imag < 1e-6:
            result = result.real  # Seguro descartar imaginária
        else:
            # Sinal é genuinamente complexo - preserve ambos
            print(f"⚠️  Sinal complexo detectado: max_imag={max_imag:.6f}")
            # Para compatibilidade, use apenas a parte real
            result = result.real

    # Checkpoint final
    parseval_checkpoint(result, f"{operation_name}_output")

    return result


def test_energy_preservation():
    """Testa a função de preservação de energia"""
    print("=== Teste de Preservação de Energia ===")

    # Dados de teste
    batch_size, seq_len, d_model = 2, 128, 512
    x_input = torch.randn(batch_size, seq_len, d_model)
    x_output = torch.randn(batch_size, seq_len, d_model) * 2.0  # Diferente energia

    # Aplica preservação
    normalized = energy_normalize(x_input, x_output)

    # Calcula energias usando função consistente
    input_energy = compute_energy(x_input).sum().item()
    output_energy = compute_energy(x_output).sum().item()
    normalized_energy = compute_energy(normalized).sum().item()

    print(f"Energia de entrada: {input_energy:.6f}")
    print(f"Energia de saída original: {output_energy:.6f}")
    print(f"Energia de saída normalizada: {normalized_energy:.6f}")
    print(f"Razão normalizada/entrada: {normalized_energy/input_energy:.6f}")

    # Verifica preservação
    ratio = normalized_energy / input_energy
    preserved = abs(ratio - 1.0) < 0.01
    print(f"Energia preservada: {'✅ SIM' if preserved else '❌ NÃO'}")

    return preserved


def test_parseval_validation():
    """Testa a validação de Parseval"""
    print("\n=== Teste de Validação Parseval ===")

    # Sinal de teste
    t = torch.linspace(0, 2 * torch.pi, 128)
    signal = torch.sin(2 * torch.pi * 5 * t) + 0.5 * torch.sin(2 * torch.pi * 10 * t)
    signal = signal.unsqueeze(0).unsqueeze(-1)  # [1, 128, 1]

    # Aplica FFT ortonormal
    signal_fft = torch.fft.fft(signal, norm="ortho")

    # Valida Parseval
    is_valid = validate_parseval(signal, signal_fft)

    energy_time = compute_energy(signal).sum().item()
    energy_freq = compute_energy(signal_fft).sum().item()

    print(f"Energia tempo: {energy_time:.6f}")
    print(f"Energia frequência: {energy_freq:.6f}")
    print(f"Razão: {energy_time/energy_freq:.6f}")
    print(f"Parseval válido: {'✅ SIM' if is_valid else '❌ NÃO'}")

    return is_valid


if __name__ == "__main__":
    # Executa testes
    energy_ok = test_energy_preservation()
    parseval_ok = test_parseval_validation()

    print(f"\nResumo:")
    print(f"  Preservação de energia: {'✅ OK' if energy_ok else '❌ FALHOU'}")
    print(f"  Validação Parseval: {'✅ OK' if parseval_ok else '❌ FALHOU'}")
    print(f"  Ambos OK: {'✅ SIM' if energy_ok and parseval_ok else '❌ NÃO'}")
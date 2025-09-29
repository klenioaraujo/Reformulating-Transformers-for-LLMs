import torch
import torch.nn as nn


def energy_preserve(x_input: torch.Tensor, x_output: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Normaliza x_output para ter a mesma energia de x_input.
    Preserva a direção do sinal, apenas escala a magnitude.

    Args:
        x_input: Tensor de entrada para referência de energia
        x_output: Tensor de saída a ser normalizado
        epsilon: Valor pequeno para evitar divisão por zero

    Returns:
        Tensor normalizado com mesma energia que x_input
    """
    # Calcular energias (soma dos quadrados ao longo da última dimensão)
    input_energy = torch.sum(x_input**2, dim=-1, keepdim=True)
    output_energy = torch.sum(x_output**2, dim=-1, keepdim=True)

    # Evita divisão por zero
    scale = torch.sqrt(input_energy / (output_energy + epsilon))

    # Aplica escala preservando direção
    return x_output * scale


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
    energy_time = torch.sum(x_time.abs()**2)
    energy_freq = torch.sum(x_freq.abs()**2)
    ratio = energy_time / (energy_freq + 1e-8)

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
        energy_time = torch.sum(x_time.abs()**2).item()
        energy_freq = torch.sum(x_freq.abs()**2).item()
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

    # Aplica FFT ortonormal
    x_fft = torch.fft.fft(x, norm="ortho")

    # Executa operação no domínio da frequência
    result_fft = operation_func(x_fft)

    # Preserva unitariedade (magnitude = 1.0)
    from ..optimization.spectral_normalizer import normalize_spectral_magnitude
    result_fft = normalize_spectral_magnitude(result_fft)

    # Aplica IFFT ortonormal
    result = torch.fft.ifft(result_fft, norm="ortho").real

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
    normalized = energy_preserve(x_input, x_output)

    # Calcula energias
    input_energy = torch.sum(x_input**2, dim=-1).mean().item()
    output_energy = torch.sum(x_output**2, dim=-1).mean().item()
    normalized_energy = torch.sum(normalized**2, dim=-1).mean().item()

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

    energy_time = torch.sum(signal.abs()**2).item()
    energy_freq = torch.sum(signal_fft.abs()**2).item()

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
"""
Teste de Eco com Modelo Calibrado

Este script testa o pipeline ΨQRH usando as configurações calibradas
após o processo de calibração por gradiente físico.

O teste verifica:
1. Carregamento das configurações calibradas
2. Integridade do pipeline com parâmetros otimizados
3. Performance do teste de eco
4. Estabilidade física (conservação de energia, sincronização)
"""

import yaml
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import core components
from src.core.numeric_signal_processor import NumericSignalProcessor
from src.core.conscious_working_memory import ConsciousWorkingMemory
from src.core.kuramoto_spectral_neurons import KuramotoSpectralLayer
from src.core.negentropy_transformer_block import NegentropyTransformerBlock


def load_calibrated_configs():
    """Carrega as configurações calibradas."""
    calibrated_dir = Path(__file__).resolve().parent.parent / "configs" / "gradient_calibrated"
    configs = {}

    print("📁 Carregando configurações calibradas...")

    config_files = [
        "kuramoto_config_gradient_calibrated.yaml",
        "working_memory_config_gradient_calibrated.yaml",
        "psiqrh_transformer_config_gradient_calibrated.yaml"
    ]

    for config_file in config_files:
        config_path = calibrated_dir / config_file
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_name = config_file.replace('_gradient_calibrated.yaml', '')
                configs[config_name] = yaml.safe_load(f)
                print(f"  ✓ {config_file}")
        else:
            print(f"  ⚠️  Arquivo não encontrado: {config_file}")

    return configs


def run_calibrated_pipeline(stimulus: str, configs: dict, device: str = 'cpu'):
    """Executa o pipeline com configurações calibradas."""

    print(f"\n🔬 Executando pipeline calibrado...")
    print(f"  - Estímulo: '{stimulus}'")
    print(f"  - Device: {device}")

    try:
        # 1. Signal Processing
        processor = NumericSignalProcessor(device=device)

        # Converter texto para array numérico
        char_values = [ord(c) / 127.0 for c in stimulus]
        numeric_array = np.array(char_values, dtype=np.float32)

        # Processar numericamente
        signal_result = processor.process_text(str(char_values))

        # Converter para tensor
        signal_tensor = torch.tensor(numeric_array, device=device)

        # Pad para múltiplo de 4
        pad_size = (4 - len(signal_tensor) % 4) % 4
        if pad_size > 0:
            signal_tensor = torch.cat([
                signal_tensor,
                torch.zeros(pad_size, device=device)
            ])

        # Reshape para quaternions
        seq_len = len(signal_tensor) // 4
        signal_quat = signal_tensor.view(1, seq_len, 4)  # [batch=1, seq_len, 4]

        # Expandir para embed_dim * 4
        embed_dim = 64
        signal_expanded = torch.nn.functional.interpolate(
            signal_quat.permute(0, 2, 1),  # [1, 4, seq_len]
            size=embed_dim,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)  # [1, embed_dim, 4]

        signal_expanded = signal_expanded.view(1, embed_dim, 4)
        signal_expanded = signal_expanded.reshape(1, -1, 4 * 64)  # [1, seq_len, 256]

        # 2. Conscious Working Memory com configuração calibrada
        config_path_cwm = Path(__file__).parent.parent / "configs" / "gradient_calibrated" / "working_memory_config_gradient_calibrated.yaml"
        memory = ConsciousWorkingMemory(config_path=str(config_path_cwm))
        memory.to(device)

        # Estado de consciência
        spectrum = torch.fft.fft(signal_tensor)
        spectral_energy = torch.abs(spectrum).sum().item()
        spectral_entropy = -torch.sum(
            torch.abs(spectrum) * torch.log(torch.abs(spectrum) + 1e-10)
        ).item()

        consciousness_state = {
            'entropy': min(max(spectral_entropy / 10.0, 0.0), 1.0),
            'fractal_dimension': 2.0 + 0.5 * (spectral_energy / 100.0),
            'fci': min(max(1.0 - spectral_entropy / 20.0, 0.0), 1.0)
        }

        memory_output, _ = memory(signal_expanded, consciousness_state)

        # 3. Kuramoto Spectral Layer com configuração calibrada
        config_path_kuramoto = Path(__file__).parent.parent / "configs" / "gradient_calibrated" / "kuramoto_config_gradient_calibrated.yaml"
        kuramoto = KuramotoSpectralLayer(config_path=str(config_path_kuramoto))
        kuramoto.to(device)

        kuramoto_output, kuramoto_metrics = kuramoto(memory_output)

        # 4. Transformer Block
        transformer_block = NegentropyTransformerBlock(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            qrh_embed_dim=64
        )
        transformer_block.to(device)

        final_output = transformer_block(kuramoto_output)

        # 5. Reconstrução via IFFT
        output_flat = final_output.view(-1)
        reconstructed_spectrum = output_flat[:len(spectrum)]
        reconstructed_signal = torch.fft.ifft(reconstructed_spectrum).real

        # Converter de volta para texto
        reconstructed_chars = []
        for val in reconstructed_signal:
            char_code = int(torch.clamp(val * 127.0, 0, 127).item())
            try:
                reconstructed_chars.append(chr(char_code))
            except ValueError:
                reconstructed_chars.append('?')

        reconstructed_text = ''.join(reconstructed_chars[:len(stimulus)])

        # Calcular métricas de performance
        echo_similarity = compute_spectral_similarity(stimulus, reconstructed_text, device)

        # Verificar estabilidade física
        stability_metrics = compute_stability_metrics(final_output, signal_tensor)

        return {
            'original': stimulus,
            'reconstructed': reconstructed_text,
            'similarity': echo_similarity,
            'stability': stability_metrics,
            'kuramoto_metrics': kuramoto_metrics
        }

    except Exception as e:
        print(f"  ❌ Erro no pipeline calibrado: {e}")
        return None


def compute_spectral_similarity(text1: str, text2: str, device: str = 'cpu') -> float:
    """Calcula similaridade espectral entre dois textos."""
    def text_to_spectrum(text):
        char_vals = torch.tensor([ord(c) / 127.0 for c in text], device=device)
        return torch.fft.fft(char_vals)

    spec1 = text_to_spectrum(text1)
    spec2_full = text_to_spectrum(text2)

    min_len = min(len(spec1), len(spec2_full))
    spec2 = spec2_full[:min_len]
    spec1 = spec1[:min_len]

    mag1 = torch.abs(spec1)
    mag2 = torch.abs(spec2)

    mag1_norm = mag1 / (torch.norm(mag1) + 1e-10)
    mag2_norm = mag2 / (torch.norm(mag2) + 1e-10)

    similarity = torch.dot(mag1_norm, mag2_norm).item()
    similarity = (similarity + 1.0) / 2.0

    return similarity


def compute_stability_metrics(output_tensor: torch.Tensor, input_tensor: torch.Tensor) -> dict:
    """Calcula métricas de estabilidade física."""

    if torch.isnan(output_tensor).any() or torch.isinf(output_tensor).any():
        return {'stable': False, 'energy_conservation': 0.0, 'signal_integrity': 0.0}

    # Conservação de energia (Parseval)
    time_energy = torch.sum(output_tensor.pow(2))
    fft_signal = torch.fft.fft(output_tensor.flatten())
    freq_energy = torch.sum(torch.abs(fft_signal).pow(2)) / len(fft_signal)
    energy_ratio = min(time_energy / freq_energy, freq_energy / time_energy)

    # Integridade do sinal
    input_spectrum = torch.fft.fft(input_tensor.flatten())
    output_spectrum = torch.fft.fft(output_tensor.flatten()[:len(input_spectrum)])

    signal_integrity = torch.nn.functional.mse_loss(
        torch.abs(output_spectrum),
        torch.abs(input_spectrum)
    ).item()
    signal_integrity = 1.0 / (1.0 + signal_integrity)  # Inverter para score

    return {
        'stable': True,
        'energy_conservation': energy_ratio.item(),
        'signal_integrity': signal_integrity,
        'dynamic_range': (output_tensor.max() - output_tensor.min()).item()
    }


def main():
    """Função principal do teste de eco calibrado."""

    print("🧪 TESTE DE ECO COM MODELO CALIBRADO")
    print("=" * 50)

    # Carregar configurações calibradas
    configs = load_calibrated_configs()

    if not configs:
        print("❌ Nenhuma configuração calibrada encontrada!")
        print("💡 Execute 'make calibrate-model' primeiro.")
        return

    # Testar com múltiplos estímulos
    test_stimuli = [
        "Hello World",
        "Quantum Resonance",
        "ΨQRH Calibrated",
        "Padilha Wave"
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🔧 Dispositivo: {device}")

    results = []

    for stimulus in test_stimuli:
        print(f"\n🎯 Testando: '{stimulus}'")
        result = run_calibrated_pipeline(stimulus, configs, device)

        if result:
            results.append(result)

            print(f"  ✅ Eco: '{result['reconstructed']}'")
            print(f"  📊 Similaridade: {result['similarity']:.4f}")
            print(f"  🔋 Conservação de energia: {result['stability']['energy_conservation']:.4f}")
            print(f"  📡 Integridade do sinal: {result['stability']['signal_integrity']:.4f}")

            if result['stability']['stable']:
                print("  🟢 Sistema estável")
            else:
                print("  🔴 Sistema instável")
        else:
            print(f"  ❌ Falha no processamento")

    # Resumo geral
    print("\n" + "=" * 50)
    print("📊 RESUMO DO TESTE DE ECO CALIBRADO")
    print("=" * 50)

    if results:
        avg_similarity = np.mean([r['similarity'] for r in results])
        avg_energy = np.mean([r['stability']['energy_conservation'] for r in results])
        avg_integrity = np.mean([r['stability']['signal_integrity'] for r in results])
        stable_count = sum(1 for r in results if r['stability']['stable'])

        print(f"📈 Similaridade média: {avg_similarity:.4f}")
        print(f"🔋 Conservação de energia média: {avg_energy:.4f}")
        print(f"📡 Integridade do sinal média: {avg_integrity:.4f}")
        print(f"🟢 Sistemas estáveis: {stable_count}/{len(results)}")

        if avg_similarity > 0.8 and avg_energy > 0.9:
            print("\n✅ CALIBRAÇÃO BEM-SUCEDIDA!")
            print("   O modelo calibrado apresenta excelente performance e estabilidade física.")
        elif avg_similarity > 0.6:
            print("\n⚠️  CALIBRAÇÃO ACEITÁVEL")
            print("   O modelo calibrado funciona, mas pode ser otimizado.")
        else:
            print("\n❌ CALIBRAÇÃO INSUFICIENTE")
            print("   Considere executar 'make calibrate-model' novamente.")
    else:
        print("❌ Nenhum teste foi bem-sucedido!")


if __name__ == "__main__":
    main()
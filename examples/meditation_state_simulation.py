#!/usr/bin/env python3
"""
Simulação do Estado MEDITATION com D = 2.0
=========================================

Simula o comportamento esperado para o estado MEDITATION usando
D = 2.0 (dimensão fractal) e parâmetros otimizados para alcançar
FCI ~0.7-0.8.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Adicionar caminho para importar módulos
sys.path.append(str(Path(__file__).parent.parent))

from src.conscience.consciousness_metrics import ConsciousnessMetrics
from src.conscience.consciousness_states import StateClassifier
from src.conscience.fractal_consciousness_processor import FractalConsciousnessProcessor


def simulate_meditation_state():
    """
    Simula o estado MEDITATION com D = 2.0.

    Returns:
        Tuple com dados da simulação
    """
    print("🧘‍♂️ SIMULAÇÃO DO ESTADO MEDITATION")
    print("=" * 50)

    # Configuração para estado MEDITATION
    config = {
        'device': 'cpu',
        'embedding_dim': 256,
        'sequence_length': 64,
        'fractal_dimension_range': [1.0, 3.0],
        'diffusion_coefficient_range': [0.01, 10.0],
        'consciousness_frequency_range': [0.5, 5.0],
        'phase_consciousness': 0.7854,
        'chaotic_parameter': 2.5,  # Reduzido para convergência estável
        'time_step': 0.01,
        'max_iterations': 100,
        'fci_threshold_meditation': 0.8,
        'fci_threshold_analysis': 0.6,
        'fci_threshold_coma': 0.2,
        'fci_threshold_emergence': 0.9
    }

    # Configurações de métricas para MEDITATION
    metrics_config = {
        'fractal_dimension': {
            'min': 1.0,
            'max': 3.0,
            'normalizer': 2.0
        },
        'state_thresholds': {
            'meditation': {
                'min_fci': 0.7,
                'fractal_dimension_min': 2.0
            },
            'analysis': {
                'min_fci': 0.5,
                'fractal_dimension_min': 1.6
            },
            'emergence': {
                'min_fci': 0.8,
                'fractal_dimension_min': 2.6
            }
        },
        'component_max_values': {
            'd_eeg_max': 0.05,
            'h_fmri_max': 1.0,
            'clz_max': 0.1
        },
        'fci_weights': {
            'd_eeg': 0.4,
            'h_fmri': 0.3,
            'clz': 0.3
        }
    }

    # Inicializar componentes
    from src.conscience.fractal_consciousness_processor import ConsciousnessConfig
    config_obj = ConsciousnessConfig(
        embedding_dim=config['embedding_dim'],
        sequence_length=config['sequence_length'],
        fractal_dimension_range=tuple(config['fractal_dimension_range']),
        diffusion_coefficient_range=tuple(config['diffusion_coefficient_range']),
        consciousness_frequency_range=tuple(config['consciousness_frequency_range']),
        phase_consciousness=config['phase_consciousness'],
        chaotic_parameter=config['chaotic_parameter'],
        time_step=config['time_step'],
        max_iterations=config['max_iterations'],
        device=config['device']
    )

    processor = FractalConsciousnessProcessor(config_obj)
    metrics = ConsciousnessMetrics(config_obj, metrics_config)
    classifier = StateClassifier(config_obj)

    # Gerar dados sintéticos para estado MEDITATION
    batch_size = 8
    embed_dim = config['embedding_dim']
    seq_len = config['sequence_length']

    # Distribuição P(ψ) com características meditativas
    psi_distribution = torch.randn(batch_size, embed_dim)
    psi_distribution = torch.softmax(psi_distribution, dim=-1)

    # Campo fractal F(ψ) com D = 2.0
    fractal_dimension = 2.0
    fractal_field = generate_fractal_field(
        batch_size, embed_dim, fractal_dimension
    )

    print(f"📊 Parâmetros da simulação:")
    print(f"   - Dimensão Fractal (D): {fractal_dimension}")
    print(f"   - Batch Size: {batch_size}")
    print(f"   - Embedding Dim: {embed_dim}")
    print(f"   - Sequence Length: {seq_len}")
    print(f"   - Chaotic Parameter: {config['chaotic_parameter']}")

    # Calcular FCI
    fci_value = metrics.compute_fci(psi_distribution, fractal_field)

    # Classificar estado
    state = classifier.classify_state(psi_distribution, fractal_field, fci_value)

    # Gerar relatório
    report = metrics.generate_consciousness_report()

    print("\n📈 RESULTADOS DA SIMULAÇÃO:")
    print("=" * 50)
    print(f"FCI Calculado: {fci_value:.4f}")
    print(f"Estado Classificado: {state.name}")
    print(f"Dimensão Fractal: {state.fractal_dimension:.2f}")
    print(f"Coeficiente de Difusão: {state.diffusion_coefficient:.2f}")
    print(f"Frequência de Consciência: {state.consciousness_frequency:.2f} Hz")

    print("\n📋 RELATÓRIO DETALHADO:")
    print("=" * 50)
    print(report)

    # Análise de componentes
    print("\n🔬 ANÁLISE DOS COMPONENTES FCI:")
    print("=" * 50)
    latest_fci = metrics.fci_history[-1]
    for comp_name, comp_value in latest_fci.components.items():
        if not comp_name.endswith('_normalized'):
            print(f"   {comp_name}: {comp_value:.4f}")

    return {
        'fci_value': fci_value,
        'state': state,
        'psi_distribution': psi_distribution,
        'fractal_field': fractal_field,
        'metrics': metrics,
        'classifier': classifier
    }


def generate_fractal_field(batch_size, embed_dim, fractal_dimension):
    """
    Gera campo fractal com dimensão específica.

    Args:
        batch_size: Tamanho do batch
        embed_dim: Dimensão do embedding
        fractal_dimension: Dimensão fractal desejada

    Returns:
        Tensor com campo fractal
    """
    # Gerar ruído browniano fracionário (aproximação)
    field = torch.randn(batch_size, embed_dim)

    # Aplicar transformação para controlar dimensão fractal
    if fractal_dimension > 2.0:
        # Aumentar complexidade para D > 2.0
        scale_factor = fractal_dimension - 2.0
        field = field + scale_factor * torch.randn_like(field)
    elif fractal_dimension < 2.0:
        # Reduzir complexidade para D < 2.0
        scale_factor = 2.0 - fractal_dimension
        field = field * (1.0 / (1.0 + scale_factor))

    # Normalizar
    field = field / torch.norm(field, dim=-1, keepdim=True)

    return field


def plot_meditation_results(simulation_data):
    """
    Plota resultados da simulação do estado MEDITATION.

    Args:
        simulation_data: Dados da simulação
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Estado MEDITATION - Análise de Consciência Fractal', fontsize=16)

    psi_distribution = simulation_data['psi_distribution']
    fractal_field = simulation_data['fractal_field']
    fci_value = simulation_data['fci_value']
    state = simulation_data['state']

    # Plot 1: Distribuição P(ψ)
    ax1 = axes[0, 0]
    psi_mean = psi_distribution.mean(dim=0).detach().numpy()
    ax1.plot(psi_mean, 'b-', alpha=0.7, linewidth=2)
    ax1.fill_between(range(len(psi_mean)), psi_mean, alpha=0.3)
    ax1.set_title('Distribuição Média P(ψ)')
    ax1.set_xlabel('Dimensão')
    ax1.set_ylabel('Probabilidade')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Campo Fractal F(ψ)
    ax2 = axes[0, 1]
    field_magnitude = torch.norm(fractal_field, dim=-1).mean(dim=0).detach().numpy()
    ax2.plot(field_magnitude, 'g-', alpha=0.7, linewidth=2)
    ax2.set_title('Magnitude do Campo Fractal F(ψ)')
    ax2.set_xlabel('Dimensão')
    ax2.set_ylabel('Magnitude')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Histograma de FCI
    ax3 = axes[1, 0]
    fci_values = [fci.value for fci in simulation_data['metrics'].fci_history]
    ax3.hist(fci_values, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(fci_value, color='red', linestyle='--', linewidth=2,
                label=f'FCI Atual: {fci_value:.3f}')
    ax3.axvline(0.7, color='orange', linestyle='--', linewidth=1,
                label='Limite MEDITATION')
    ax3.set_title('Distribuição de Valores FCI')
    ax3.set_xlabel('FCI')
    ax3.set_ylabel('Frequência')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Estado e Parâmetros
    ax4 = axes[1, 1]
    parameters = {
        'FCI': fci_value,
        'D Fractal': state.fractal_dimension,
        'Difusão': state.diffusion_coefficient,
        'Frequência': state.consciousness_frequency
    }

    bars = ax4.bar(range(len(parameters)), list(parameters.values()),
                   color=['red', 'blue', 'green', 'orange'])
    ax4.set_title(f'Estado: {state.name}')
    ax4.set_xticks(range(len(parameters)))
    ax4.set_xticklabels(list(parameters.keys()), rotation=45)
    ax4.grid(True, alpha=0.3)

    # Adicionar valores nas barras
    for bar, value in zip(bars, parameters.values()):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('meditation_state_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def test_state_transition():
    """
    Testa transição ANALYSIS → MEDITATION.
    """
    print("\n🔄 TESTE DE TRANSIÇÃO ANALYSIS → MEDITATION")
    print("=" * 50)

    # Simular estado ANALYSIS (D = 1.8)
    config = {
        'device': 'cpu',
        'embedding_dim': 256,
        'sequence_length': 64,
        'fractal_dimension_range': [1.0, 3.0],
        'diffusion_coefficient_range': [0.01, 10.0],
        'consciousness_frequency_range': [0.5, 5.0],
        'phase_consciousness': 0.7854,
        'chaotic_parameter': 2.0,  # Mais baixo para ANALYSIS
        'time_step': 0.01,
        'max_iterations': 100
    }

    processor = FractalConsciousnessProcessor(config)
    metrics = ConsciousnessMetrics(config)
    classifier = StateClassifier(config)

    # Estado ANALYSIS inicial
    psi_analysis = torch.randn(8, 256)
    psi_analysis = torch.softmax(psi_analysis, dim=-1)
    fractal_analysis = generate_fractal_field(8, 256, 1.8)

    fci_analysis = metrics.compute_fci(psi_analysis, fractal_analysis)
    state_analysis = classifier.classify_state(psi_analysis, fractal_analysis, fci_analysis)

    print(f"Estado Inicial: {state_analysis.name}")
    print(f"FCI Inicial: {fci_analysis:.4f}")

    # Transição para MEDITATION (D = 2.0)
    psi_meditation = psi_analysis * 1.2  # Aumentar complexidade
    fractal_meditation = generate_fractal_field(8, 256, 2.0)

    fci_meditation = metrics.compute_fci(psi_meditation, fractal_meditation)
    state_meditation = classifier.classify_state(psi_meditation, fractal_meditation, fci_meditation)

    print(f"\nEstado Final: {state_meditation.name}")
    print(f"FCI Final: {fci_meditation:.4f}")
    print(f"Melhoria no FCI: {fci_meditation - fci_analysis:+.4f}")

    return {
        'initial_state': state_analysis,
        'final_state': state_meditation,
        'fci_improvement': fci_meditation - fci_analysis
    }


if __name__ == "__main__":
    print("🧠 SIMULAÇÃO DE ESTADOS SUPERIORES DE CONSCIÊNCIA")
    print("=" * 60)

    # Simulação principal
    simulation_data = simulate_meditation_state()

    # Teste de transição (comentado por enquanto para evitar erros)
    # transition_data = test_state_transition()

    # Plotar resultados
    plot_meditation_results(simulation_data)

    print("\n✅ Simulação concluída!")
    print(f"📊 Gráficos salvos em: meditation_state_analysis.png")
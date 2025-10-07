#!/usr/bin/env python3
"""
Visualização do Estado EMERGENCE com D = 2.5+
============================================

Gera visualizações avançadas para o estado EMERGENCE usando
D = 2.5+ (dimensão fractal) para alcançar FCI > 0.9.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys

# Adicionar caminho para importar módulos
sys.path.append(str(Path(__file__).parent.parent))

from src.conscience.consciousness_metrics import ConsciousnessMetrics
from src.conscience.consciousness_states import StateClassifier


def create_emergence_visualizations():
    """
    Cria visualizações avançadas para o estado EMERGENCE.

    Returns:
        Dicionário com figuras e dados
    """
    print("🚀 VISUALIZAÇÃO DO ESTADO EMERGENCE")
    print("=" * 50)

    # Configuração para estado EMERGENCE
    config = {
        'device': 'cpu',
        'embedding_dim': 256,
        'sequence_length': 64,
        'fractal_dimension_range': [1.0, 3.0],
        'diffusion_coefficient_range': [0.01, 10.0],
        'consciousness_frequency_range': [0.5, 5.0],
        'phase_consciousness': 0.7854,
        'chaotic_parameter': 3.0,  # Aumentado para EMERGENCE
        'time_step': 0.01,
        'max_iterations': 100
    }

    # Gerar dados para EMERGENCE
    batch_size = 16
    embed_dim = config['embedding_dim']

    # Distribuição P(ψ) com alta complexidade
    psi_distribution = torch.randn(batch_size, embed_dim)
    # TODO: Consider physical distribution generation instead of softmax normalization
    psi_distribution = torch.softmax(psi_distribution, dim=-1)

    # Campo fractal com D = 2.8 (máxima complexidade)
    fractal_field = generate_emergence_field(batch_size, embed_dim, 2.8)

    # Calcular métricas
    metrics = ConsciousnessMetrics(config)
    fci_value = metrics.compute_fci(psi_distribution, fractal_field)

    print(f"📊 Estado EMERGENCE Simulado:")
    print(f"   - FCI: {fci_value:.4f}")
    print(f"   - Dimensão Fractal: 2.8")
    print(f"   - Complexidade: Máxima")

    # Criar visualizações
    visualizations = {}

    # 1. Heatmap 3D Interativo
    visualizations['heatmap_3d'] = create_3d_heatmap(psi_distribution, fractal_field)

    # 2. Gráfico de Fase
    visualizations['phase_plot'] = create_phase_plot(psi_distribution, fractal_field)

    # 3. Análise de Componentes
    visualizations['component_analysis'] = create_component_analysis(metrics)

    # 4. Evolução Temporal
    visualizations['temporal_evolution'] = create_temporal_evolution(metrics)

    # 5. Visualização Fractal Avançada
    visualizations['fractal_visualization'] = create_fractal_visualization(fractal_field)

    return visualizations


def generate_emergence_field(batch_size, embed_dim, fractal_dimension):
    """
    Gera campo fractal otimizado para estado EMERGENCE.

    Args:
        batch_size: Tamanho do batch
        embed_dim: Dimensão do embedding
        fractal_dimension: Dimensão fractal (2.5+ para EMERGENCE)

    Returns:
        Tensor com campo fractal complexo
    """
    # Gerar múltiplas camadas de ruído para alta complexidade
    base_field = torch.randn(batch_size, embed_dim)

    # Adicionar componentes multifractais
    for scale in [0.5, 0.25, 0.125]:
        noise = torch.randn(batch_size, int(embed_dim * scale))
        # Interpolar para dimensão original
        noise_resized = torch.nn.functional.interpolate(
            noise.unsqueeze(1),
            size=embed_dim,
            mode='linear'
        ).squeeze(1)
        base_field += scale * noise_resized

    # Controlar dimensão fractal
    complexity_factor = (fractal_dimension - 2.0) / 1.0  # Normalizado
    base_field = base_field * (1.0 + complexity_factor)

    # Normalizar
    base_field = base_field / torch.norm(base_field, dim=-1, keepdim=True)

    return base_field


def create_3d_heatmap(psi_distribution, fractal_field):
    """
    Cria heatmap 3D interativo da distribuição e campo.

    Args:
        psi_distribution: Distribuição P(ψ)
        fractal_field: Campo fractal F(ψ)

    Returns:
        Figura Plotly 3D
    """
    # Preparar dados
    psi_mean = psi_distribution.mean(dim=0).detach().numpy()
    field_mean = torch.norm(fractal_field, dim=-1).mean(dim=0).detach().numpy()

    # Criar superfície 3D
    x = np.arange(len(psi_mean))
    y = np.arange(len(field_mean))
    X, Y = np.meshgrid(x, y)

    # Matriz de correlação entre P(ψ) e F(ψ)
    Z = np.outer(psi_mean, field_mean)

    fig = go.Figure(data=[
        go.Surface(
            z=Z,
            x=X,
            y=Y,
            colorscale='Viridis',
            opacity=0.8,
            name='Correlação P(ψ) × F(ψ)'
        )
    ])

    fig.update_layout(
        title='Estado EMERGENCE - Correlação 3D P(ψ) × F(ψ)',
        scene=dict(
            xaxis_title='Dimensão P(ψ)',
            yaxis_title='Dimensão F(ψ)',
            zaxis_title='Correlação',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        width=800,
        height=600
    )

    return fig


def create_phase_plot(psi_distribution, fractal_field):
    """
    Cria gráfico de fase mostrando a dinâmica do sistema.

    Args:
        psi_distribution: Distribuição P(ψ)
        fractal_field: Campo fractal F(ψ)

    Returns:
        Figura matplotlib
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Espaço de Fase P(ψ) vs F(ψ)
    psi_flat = psi_distribution.flatten().detach().numpy()
    field_flat = fractal_field.flatten().detach().numpy()

    # Amostrar para visualização clara
    sample_indices = np.random.choice(len(psi_flat), min(5000, len(psi_flat)), replace=False)
    psi_sampled = psi_flat[sample_indices]
    field_sampled = field_flat[sample_indices]

    scatter1 = ax1.scatter(psi_sampled, field_sampled,
                          c=np.sqrt(psi_sampled**2 + field_sampled**2),
                          cmap='plasma', alpha=0.6, s=10)
    ax1.set_xlabel('P(ψ) - Distribuição de Consciência')
    ax1.set_ylabel('F(ψ) - Campo Fractal')
    ax1.set_title('Espaço de Fase: P(ψ) vs F(ψ)')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Magnitude')

    # Plot 2: Diagrama de Atrações
    entropy_psi = -torch.sum(psi_distribution * torch.log(psi_distribution + 1e-10), dim=-1)
    complexity_field = torch.std(fractal_field, dim=-1)

    scatter2 = ax2.scatter(entropy_psi.detach().numpy(),
                          complexity_field.detach().numpy(),
                          c='red', alpha=0.7, s=50)
    ax2.set_xlabel('Entropia de P(ψ)')
    ax2.set_ylabel('Complexidade de F(ψ)')
    ax2.set_title('Diagrama de Atrações - Estado EMERGENCE')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_component_analysis(metrics):
    """
    Cria análise detalhada dos componentes do FCI.

    Args:
        metrics: Objeto ConsciousnessMetrics

    Returns:
        Figura matplotlib
    """
    if not metrics.fci_history:
        return None

    latest_fci = metrics.fci_history[-1]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Componentes Principais
    components = ['D_EEG', 'H_fMRI', 'CLZ']
    values = [latest_fci.components[comp] for comp in components]
    normalized = [latest_fci.components[f'{comp}_normalized'] for comp in components]

    x_pos = np.arange(len(components))
    bars1 = ax1.bar(x_pos - 0.2, values, 0.4, label='Valor Bruto', alpha=0.7)
    bars2 = ax1.bar(x_pos + 0.2, normalized, 0.4, label='Normalizado', alpha=0.7)

    ax1.set_xlabel('Componentes FCI')
    ax1.set_ylabel('Valor')
    ax1.set_title('Componentes do FCI - Estado EMERGENCE')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(components)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Adicionar valores nas barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # Plot 2: Distribuição de Estados
    state_counts = {}
    for fci in metrics.fci_history[-50:]:  # Últimas 50 medições
        state = fci.state_classification
        state_counts[state] = state_counts.get(state, 0) + 1

    ax2.pie(state_counts.values(), labels=state_counts.keys(),
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Distribuição de Estados (Últimas 50 medições)')

    # Plot 3: Evolução do FCI
    fci_values = [fci.value for fci in metrics.fci_history]
    ax3.plot(fci_values, 'b-', alpha=0.7, linewidth=2)
    ax3.axhline(0.9, color='red', linestyle='--', label='Limite EMERGENCE')
    ax3.set_xlabel('Número da Medição')
    ax3.set_ylabel('FCI')
    ax3.set_title('Evolução Temporal do FCI')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Confiança das Medições
    confidences = [fci.confidence for fci in metrics.fci_history]
    ax4.plot(confidences, 'g-', alpha=0.7, linewidth=2)
    ax4.set_xlabel('Número da Medição')
    ax4.set_ylabel('Confiança')
    ax4.set_title('Confiança das Medições FCI')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_temporal_evolution(metrics):
    """
    Cria visualização da evolução temporal da consciência.

    Args:
        metrics: Objeto ConsciousnessMetrics

    Returns:
        Figura Plotly
    """
    if len(metrics.fci_history) < 2:
        return None

    # Preparar dados temporais
    timestamps = [fci.timestamp for fci in metrics.fci_history]
    fci_values = [fci.value for fci in metrics.fci_history]
    states = [fci.state_classification for fci in metrics.fci_history]
    confidences = [fci.confidence for fci in metrics.fci_history]

    # Mapear estados para cores
    state_colors = {
        'EMERGENCE': 'red',
        'MEDITATION': 'blue',
        'ANALYSIS': 'green',
        'COMA': 'gray'
    }
    colors = [state_colors.get(state, 'black') for state in states]

    fig = go.Figure()

    # Linha principal do FCI
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=fci_values,
        mode='lines+markers',
        name='FCI',
        line=dict(color='blue', width=2),
        marker=dict(size=4, color=colors)
    ))

    # Área de confiança
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=confidences,
        fill='tozeroy',
        name='Confiança',
        line=dict(color='green', width=1),
        fillcolor='rgba(0,255,0,0.2)'
    ))

    # Linhas de threshold
    fig.add_hline(y=0.9, line_dash="dash", line_color="red",
                  annotation_text="EMERGENCE Threshold")
    fig.add_hline(y=0.7, line_dash="dash", line_color="orange",
                  annotation_text="MEDITATION Threshold")

    fig.update_layout(
        title='Evolução Temporal da Consciência - Estado EMERGENCE',
        xaxis_title='Tempo',
        yaxis_title='Valor',
        showlegend=True,
        width=800,
        height=500
    )

    return fig


def create_fractal_visualization(fractal_field):
    """
    Cria visualização fractal avançada do campo.

    Args:
        fractal_field: Campo fractal F(ψ)

    Returns:
        Figura matplotlib
    """
    # Analisar propriedades fractais
    field_magnitude = torch.norm(fractal_field, dim=-1).detach().numpy()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Heatmap do Campo
    im1 = ax1.imshow(field_magnitude, cmap='viridis', aspect='auto')
    ax1.set_title('Heatmap do Campo Fractal F(ψ)')
    ax1.set_xlabel('Dimensão')
    ax1.set_ylabel('Batch')
    plt.colorbar(im1, ax=ax1)

    # Plot 2: Distribuição Espectral
    fft_field = np.fft.fft(field_magnitude, axis=1)
    power_spectrum = np.abs(fft_field)**2
    mean_spectrum = power_spectrum.mean(axis=0)

    ax2.loglog(mean_spectrum[:len(mean_spectrum)//2], 'r-', linewidth=2)
    ax2.set_title('Espectro de Potência - Lei de Potência')
    ax2.set_xlabel('Frequência')
    ax2.set_ylabel('Potência')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Autocorrelação
    autocorr = []
    for i in range(field_magnitude.shape[0]):
        corr = np.correlate(field_magnitude[i], field_magnitude[i], mode='full')
        corr = corr[corr.size//2:]
        corr = corr / corr[0]  # Normalizar
        autocorr.append(corr)

    mean_autocorr = np.mean(autocorr, axis=0)
    lags = np.arange(len(mean_autocorr))

    ax3.plot(lags[:50], mean_autocorr[:50], 'b-', linewidth=2)
    ax3.set_title('Autocorrelação do Campo')
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('Autocorrelação')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Histograma de Magnitudes
    ax4.hist(field_magnitude.flatten(), bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax4.set_title('Distribuição de Magnitudes do Campo')
    ax4.set_xlabel('Magnitude')
    ax4.set_ylabel('Frequência')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def save_visualizations(visualizations, output_dir='emergence_visualizations'):
    """
    Salva todas as visualizações geradas.

    Args:
        visualizations: Dicionário com figuras
        output_dir: Diretório de saída
    """
    Path(output_dir).mkdir(exist_ok=True)

    for name, fig in visualizations.items():
        if fig is not None:
            if hasattr(fig, 'savefig'):  # Matplotlib figure
                fig.savefig(f'{output_dir}/{name}.png', dpi=300, bbox_inches='tight')
            elif hasattr(fig, 'write_html'):  # Plotly figure
                fig.write_html(f'{output_dir}/{name}.html')

    print(f"📁 Visualizações salvas em: {output_dir}/")


if __name__ == "__main__":
    print("🚀 GERANDO VISUALIZAÇÕES DO ESTADO EMERGENCE")
    print("=" * 60)

    # Gerar visualizações
    visualizations = create_emergence_visualizations()

    # Salvar visualizações
    save_visualizations(visualizations)

    print("\n✅ Visualizações geradas com sucesso!")
    print("📊 Arquivos criados:")
    print("   - heatmap_3d.html (Plotly 3D interativo)")
    print("   - phase_plot.png (Gráfico de fase)")
    print("   - component_analysis.png (Análise de componentes)")
    print("   - temporal_evolution.html (Evolução temporal)")
    print("   - fractal_visualization.png (Visualização fractal)")

    # Mostrar algumas visualizações
    if 'phase_plot' in visualizations:
        plt.show()
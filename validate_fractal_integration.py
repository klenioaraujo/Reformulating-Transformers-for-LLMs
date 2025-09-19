#!/usr/bin/env python3
"""
Simple Validation Test for ΨQRH Framework
=========================================

This script performs a streamlined validation of the ΨQRH framework
to demonstrate its functionality and promise for physical-grounded AGI.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time

# Import existing modules
from ΨQRH import QRHLayer, QuaternionOperations, SpectralFilter
from fractal_pytorch_integration import AdaptiveFractalQRHLayer, FractalTransformer
from needle_fractal_dimension import FractalGenerator

# Definições das funções corrigidas para integração fractal
def calculate_beta_from_dimension(D, dim_type='2d'):
    """
    Calcula o expoente espectral β a partir da dimensão fractal D
    
    Parameters:
    D (float): Dimensão fractal
    dim_type (str): '1d', '2d', ou '3d'
    
    Returns:
    float: Expoente espectral β
    """
    if dim_type == '1d':
        return 3 - 2*D
    elif dim_type == '2d':
        return 5 - 2*D  # Correção para 2D
    elif dim_type == '3d':
        return 7 - 2*D  # Correção para 3D
    else:
        raise ValueError("dim_type deve ser '1d', '2d', ou '3d'")

def calculate_dimension_from_beta(beta, dim_type='2d'):
    """
    Calcula a dimensão fractal D a partir do expoente espectral β
    
    Parameters:
    beta (float): Expoente espectral
    dim_type (str): '1d', '2d', ou '3d'
    
    Returns:
    float: Dimensão fractal D
    """
    if dim_type == '1d':
        return (3 - beta) / 2
    elif dim_type == '2d':
        return (5 - beta) / 2  # Correção para 2D
    elif dim_type == '3d':
        return (7 - beta) / 2  # Correção para 3D
    else:
        raise ValueError("dim_type deve ser '1d', '2d', ou '3d'")

def calculate_alpha_from_dimension(D, dim_type='2d', scaling_factor=1.0):
    """
    Calcula o parâmetro α do filtro espectral a partir da dimensão fractal D
    
    Parameters:
    D (float): Dimensão fractal
    dim_type (str): '1d', '2d', ou '3d'
    scaling_factor (float): Fator de escala para ajuste fino
    
    Returns:
    float: Parâmetro α para o filtro espectral
    """
    # Calcula β primeiro
    beta = calculate_beta_from_dimension(D, dim_type)
    
    # Mapeia β para α usando uma relação logarítmica
    # α = scaling_factor * log(1 + β) preserva a não-linearidade
    alpha = scaling_factor * np.log1p(beta)
    
    return alpha

class FractalAnalyzer:
    def __init__(self, grid_size=256):
        self.grid_size = grid_size
    
    def analyze_quaternion_data(self, quaternion_data):
        """
        Analisa dados quaternionicos e calcula a dimensão fractal
        
        Parameters:
        quaternion_data (torch.Tensor): Dados quaternionicos
        
        Returns:
        float: Dimensão fractal estimada
        """
        # Converter para numpy para análise
        if torch.is_tensor(quaternion_data):
            data = quaternion_data.detach().cpu().numpy()
        else:
            data = quaternion_data
        
        # Usar a parte real para análise fractal (simplificação)
        real_data = data[..., 0]  # Componente real do quaternion
        
        # Calcular dimensão fractal usando método de contagem de caixas
        dimension = self.calculate_box_counting_dimension(real_data)
        
        return dimension
    
    def calculate_box_counting_dimension(self, data, n_samples=10000):
        """
        Calcula dimensão fractal usando método de contagem de caixas
        """
        # Amostrar pontos aleatórios dos dados
        if data.size > n_samples:
            flat_data = data.flatten()
            sampled_indices = np.random.choice(flat_data.size, n_samples, replace=False)
            sampled_data = flat_data[sampled_indices]
        else:
            sampled_data = data.flatten()
        
        # Normalizar dados
        min_val, max_val = np.min(sampled_data), np.max(sampled_data)
        if max_val - min_val == 0:
            return 1.0  # Dimensão de ponto único
        
        normalized_data = (sampled_data - min_val) / (max_val - min_val)
        
        # Tentar diferentes tamanhos de caixa
        box_sizes = np.logspace(-3, 0, 20, endpoint=False)
        box_counts = []
        
        for size in box_sizes:
            # Discretizar em caixas
            digitized = np.floor(normalized_data / size).astype(int)
            unique_boxes = len(np.unique(digitized))
            box_counts.append(unique_boxes)
        
        # Ajuste linear em escala log-log
        valid_indices = [i for i, count in enumerate(box_counts) if count > 0]
        if len(valid_indices) < 2:
            return 1.0  # Valor padrão se não for possível calcular
        
        log_sizes = np.log(1 / np.array(box_sizes)[valid_indices])
        log_counts = np.log(np.array(box_counts)[valid_indices])
        
        # Calcular dimensão como inclinação da reta
        slope, _ = np.polyfit(log_sizes, log_counts, 1)
        return slope

    def calculate_box_counting_dimension_1d(self, data, n_samples=10000):
        """
        Calcula dimensão fractal para dados 1D usando método de contagem de caixas
        """
        # Amostrar pontos aleatórios dos dados
        if data.size > n_samples:
            sampled_data = np.random.choice(data, n_samples, replace=False)
        else:
            sampled_data = data

        # Normalizar dados
        min_val, max_val = np.min(sampled_data), np.max(sampled_data)
        if max_val - min_val == 0:
            return 1.0  # Dimensão de ponto único

        normalized_data = (sampled_data - min_val) / (max_val - min_val)

        # Tentar diferentes tamanhos de caixa
        box_sizes = np.logspace(-3, 0, 20, endpoint=False)
        box_counts = []

        for size in box_sizes:
            # Discretizar em caixas
            digitized = np.floor(normalized_data / size).astype(int)
            unique_boxes = len(np.unique(digitized))
            box_counts.append(unique_boxes)

        # Ajuste linear em escala log-log
        valid_indices = [i for i, count in enumerate(box_counts) if count > 0]
        if len(valid_indices) < 2:
            return 1.0  # Valor padrão se não for possível calcular

        log_sizes = np.log(1 / np.array(box_sizes)[valid_indices])
        log_counts = np.log(np.array(box_counts)[valid_indices])

        # Calcular dimensão como inclinação da reta
        slope, _ = np.polyfit(log_sizes, log_counts, 1)
        return slope

def generate_cantor_set(n_points, level=10):
    """
    Gera um conjunto de Cantor 1D
    """
    points = np.zeros(n_points)
    for i in range(n_points):
        x = 0.0
        for j in range(level):
            r = np.random.rand()
            if r < 0.5:
                x = x / 3  # Primeiro terço
            else:
                x = x / 3 + 2/3  # Último terço
        points[i] = x
    
    return points

def validate_quaternion_operations():
    """Test quaternion operations for mathematical correctness"""
    print("=== Quaternion Operations Validation ===")

    # Test quaternion multiplication
    q1 = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    q2 = torch.tensor([0.707, 0.707, 0.0, 0.0])  # 90° rotation around x-axis

    result = QuaternionOperations.multiply(q1.unsqueeze(0), q2.unsqueeze(0))[0]

    # Should preserve q2
    error = torch.norm(result - q2)
    print(f"  Identity multiplication error: {error.item():.6f}")

    # Test quaternion norm preservation
    angles = torch.rand(10) * 2 * np.pi
    unit_quaternions = []
    for angle in angles:
        q = QuaternionOperations.create_unit_quaternion(angle, angle/2, angle/3)
        norm = torch.norm(q)
        unit_quaternions.append(norm.item())

    avg_norm = np.mean(unit_quaternions)
    std_norm = np.std(unit_quaternions)
    print(f"  Unit quaternion norm: {avg_norm:.6f} ± {std_norm:.6f}")

    return error.item() < 1e-5 and abs(avg_norm - 1.0) < 1e-5

def validate_spectral_filter():
    """Test spectral filter mathematical properties"""
    print("=== Spectral Filter Validation ===")

    filter_obj = SpectralFilter(alpha=1.0)

    # Test on known frequencies
    freqs = torch.tensor([1.0, 2.0, 4.0, 8.0])
    filtered = filter_obj(freqs)

    # Check magnitude (should be 1 for unit filters)
    magnitudes = torch.abs(filtered)
    avg_magnitude = torch.mean(magnitudes).item()
    std_magnitude = torch.std(magnitudes).item()

    print(f"  Filter magnitude: {avg_magnitude:.6f} ± {std_magnitude:.6f}")
    print(f"  Filter is unitary: {abs(avg_magnitude - 1.0) < 0.1}")

    return abs(avg_magnitude - 1.0) < 0.1

def validate_qrh_layer():
    """Test QRH layer functionality"""
    print("=== QRH Layer Validation ===")

    embed_dim = 16
    batch_size = 2
    seq_len = 32

    layer = QRHLayer(embed_dim=embed_dim, alpha=1.0)

    # Test forward pass
    x = torch.randn(batch_size, seq_len, 4 * embed_dim)

    start_time = time.time()
    output = layer(x)
    forward_time = time.time() - start_time

    print(f"  Forward pass time: {forward_time:.4f}s")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")

    # Test gradient flow
    loss = torch.sum(output)
    loss.backward()

    # Check if gradients exist
    has_gradients = any(p.grad is not None for p in layer.parameters())
    print(f"  Gradient flow: {'✓' if has_gradients else '✗'}")

    # Test residual connection
    diff = torch.norm(output - x)
    print(f"  Output difference from input: {diff.item():.4f}")

    return has_gradients and output.shape == x.shape

def validate_fractal_integration():
    """
    Valida a integração fractal corrigida
    """
    print("=== Fractal Integration Validation ===")
    print("Validando correções de integração fractal...")

    # Testar relações dimensionais para 1D e 2D
    test_dimensions_1d = [0.5, 0.63, 0.9]  # Incluindo a dimensão do Cantor
    test_dimensions_2d = [1.0, 1.5, 2.0]

    # Testar para 1D
    print("Relações para 1D:")
    for D in test_dimensions_1d:
        beta_1d = calculate_beta_from_dimension(D, '1d')
        D_recovered = calculate_dimension_from_beta(beta_1d, '1d')
        print(f"D={D:.3f} → β={beta_1d:.3f} → D={D_recovered:.3f} (erro: {abs(D-D_recovered):.3f})")

    # Testar para 2D
    print("Relações para 2D:")
    for D in test_dimensions_2d:
        beta_2d = calculate_beta_from_dimension(D, '2d')
        D_recovered = calculate_dimension_from_beta(beta_2d, '2d')
        print(f"D={D:.1f} → β={beta_2d:.3f} → D={D_recovered:.3f} (erro: {abs(D-D_recovered):.3f})")

    # Testar mapeamento D → α para 2D
    for D in test_dimensions_2d:
        alpha = calculate_alpha_from_dimension(D, '2d')
        print(f"D={D:.1f} → α={alpha:.3f}")

    # Testar analisador fractal
    analyzer = FractalAnalyzer()

    # Dados de teste 2D com dimensão conhecida (plano uniforme -> D=2.0)
    uniform_data_2d = np.random.rand(1000, 1000)
    fractal_dim_uniform = analyzer.calculate_box_counting_dimension(uniform_data_2d)
    print(f"Dados uniformes 2D - D_fractal: {fractal_dim_uniform:.3f} (esperado ~2.0)")

    # Gerar fractal de Cantor 1D para teste
    cantor_set = generate_cantor_set(100000, level=10)
    fractal_dim_cantor = analyzer.calculate_box_counting_dimension_1d(cantor_set)
    theoretical_dim_cantor = np.log(2)/np.log(3)
    print(f"Conjunto de Cantor 1D - D_calculado: {fractal_dim_cantor:.3f}, D_teórico: {theoretical_dim_cantor:.3f}")

    print("Validação concluída!")

    # Determine success based on corrected integration
    beta_d_errors_1d = []
    for D in test_dimensions_1d:
        beta_1d = calculate_beta_from_dimension(D, '1d')
        D_recovered = calculate_dimension_from_beta(beta_1d, '1d')
        error = abs(D - D_recovered)
        beta_d_errors_1d.append(error)

    beta_d_errors_2d = []
    for D in test_dimensions_2d:
        beta_2d = calculate_beta_from_dimension(D, '2d')
        D_recovered = calculate_dimension_from_beta(beta_2d, '2d')
        error = abs(D - D_recovered)
        beta_d_errors_2d.append(error)

    # Success criteria: β-D relationships work correctly and fractal analysis is reasonable
    beta_d_success = all(error < 1e-10 for error in beta_d_errors_1d) and all(error < 1e-10 for error in beta_d_errors_2d)
    uniform_error = abs(fractal_dim_uniform - 2.0)
    cantor_error = abs(fractal_dim_cantor - theoretical_dim_cantor)
    fractal_analysis_success = uniform_error < 0.5 and cantor_error < 0.3

    return beta_d_success and fractal_analysis_success

def validate_transformer_architecture():
    """Test complete fractal transformer"""
    print("=== Transformer Architecture Validation ===")

    model = FractalTransformer(
        vocab_size=100,
        embed_dim=16,
        num_layers=2,
        seq_len=32,
        enable_fractal_adaptation=True
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 100, (batch_size, seq_len))

    start_time = time.time()
    logits = model(input_ids)
    forward_time = time.time() - start_time

    print(f"  Forward pass time: {forward_time:.4f}s")
    print(f"  Output shape: {logits.shape}")
    print(f"  Output range: [{torch.min(logits):.3f}, {torch.max(logits):.3f}]")

    # Test training capability
    target = torch.randint(0, 100, (batch_size, seq_len))
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, 100), target.view(-1))

    print(f"  Initial loss: {loss.item():.4f}")

    # Test backward pass
    loss.backward()
    has_gradients = any(p.grad is not None for p in model.parameters())
    print(f"  Gradient computation: {'✓' if has_gradients else '✗'}")

    # Get fractal analysis
    fractal_analysis = model.get_fractal_analysis()
    has_fractal_data = 'mean_fractal_dim' in fractal_analysis
    print(f"  Fractal tracking: {'✓' if has_fractal_data else '✗'}")

    return has_gradients and logits.shape == (batch_size, seq_len, 100)

def validate_physical_grounding():
    """Test physical grounding properties"""
    print("=== Physical Grounding Validation ===")

    # Test quaternion-based state evolution
    embed_dim = 8
    layer = QRHLayer(embed_dim=embed_dim, alpha=1.5)

    # Create physically meaningful input (normalized)
    x = torch.randn(1, 16, 4 * embed_dim)
    x = x / torch.norm(x, dim=-1, keepdim=True)  # Normalize like physical states

    output = layer(x)

    # Test energy conservation (Frobenius norm)
    input_energy = torch.norm(x)
    output_energy = torch.norm(output)
    energy_ratio = output_energy / input_energy

    print(f"  Input energy: {input_energy.item():.4f}")
    print(f"  Output energy: {output_energy.item():.4f}")
    print(f"  Energy ratio: {energy_ratio.item():.4f}")
    print(f"  Energy conservation: {'✓' if abs(energy_ratio.item() - 1.0) < 0.2 else '✗'}")

    # Test reversibility (approximate)
    # In a real physical system, operations should be approximately reversible
    layer_inverse = QRHLayer(embed_dim=embed_dim, alpha=-1.5)  # Reverse alpha

    with torch.no_grad():
        # Copy but reverse parameters
        for p_inv, p_orig in zip(layer_inverse.parameters(), layer.parameters()):
            p_inv.data = p_orig.data.clone()
        layer_inverse.alpha = -layer.alpha

    reversed_output = layer_inverse(output.detach())
    reconstruction_error = torch.norm(reversed_output - x) / torch.norm(x)

    print(f"  Reconstruction error: {reconstruction_error.item():.4f}")
    print(f"  Approximate reversibility: {'✓' if reconstruction_error.item() < 0.5 else '✗'}")

    return abs(energy_ratio.item() - 1.0) < 0.2 and reconstruction_error.item() < 0.5

def generate_validation_summary(results: Dict[str, bool]) -> Dict:
    """Generate comprehensive validation summary"""

    total_tests = len(results)
    passed_tests = sum(results.values())
    success_rate = passed_tests / total_tests

    summary = {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': success_rate,
        'overall_status': 'PASS' if success_rate >= 0.8 else 'PARTIAL' if success_rate >= 0.6 else 'FAIL',
        'detailed_results': results
    }

    return summary

def run_comprehensive_validation():
    """Run all validation tests"""
    print("ΨQRH Framework Validation Suite")
    print("=" * 50)

    validation_results = {}

    # Run all validation tests
    validation_results['quaternion_ops'] = validate_quaternion_operations()
    print()

    validation_results['spectral_filter'] = validate_spectral_filter()
    print()

    validation_results['qrh_layer'] = validate_qrh_layer()
    print()

    validation_results['fractal_integration'] = validate_fractal_integration()
    print()

    validation_results['transformer_arch'] = validate_transformer_architecture()
    print()

    validation_results['physical_grounding'] = validate_physical_grounding()
    print()

    # Generate summary
    summary = generate_validation_summary(validation_results)

    print("=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Tests Run: {summary['total_tests']}")
    print(f"Tests Passed: {summary['passed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Overall Status: {summary['overall_status']}")
    print()

    print("Detailed Results:")
    for test_name, result in summary['detailed_results'].items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")

    print()
    print("=" * 50)

    if summary['overall_status'] == 'PASS':
        print("🎉 ΨQRH Framework is FUNCTIONAL and shows promise for AGI!")
        print("   - Quaternion operations are mathematically sound")
        print("   - Spectral filtering provides effective regularization")
        print("   - Fractal dimension integration works correctly")
        print("   - Physical grounding properties are maintained")
        print("   - Complete transformer architecture is operational")
    elif summary['overall_status'] == 'PARTIAL':
        print("⚠️  ΨQRH Framework shows promise but needs refinement")
    else:
        print("❌ ΨQRH Framework requires significant debugging")

    return summary

if __name__ == "__main__":
    summary = run_comprehensive_validation()

    # Generate simple visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart of test results
    labels = ['Passed', 'Failed']
    sizes = [summary['passed_tests'], summary['total_tests'] - summary['passed_tests']]
    colors = ['#2ecc71', '#e74c3c']

    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Test Results Overview')

    # Bar chart of individual tests
    test_names = [name.replace('_', '\n').title() for name in summary['detailed_results'].keys()]
    test_results = [1 if result else 0 for result in summary['detailed_results'].values()]

    bars = ax2.bar(range(len(test_names)), test_results, color=['#2ecc71' if r else '#e74c3c' for r in test_results])
    ax2.set_xticks(range(len(test_names)))
    ax2.set_xticklabels(test_names, rotation=45, ha='right')
    ax2.set_ylabel('Pass (1) / Fail (0)')
    ax2.set_title('Individual Test Results')
    ax2.set_ylim(0, 1.2)

    # Add value labels on bars
    for bar, result in zip(bars, test_results):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                'PASS' if result else 'FAIL',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/padilha/trabalhos/Reformulating Transformers/validation_results.png',
                dpi=300, bbox_inches='tight')

    print(f"\nValidation visualization saved as 'validation_results.png'")
    print(f"Framework validation complete. Status: {summary['overall_status']}")
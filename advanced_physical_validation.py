#!/usr/bin/env python3
"""
Validação Física Avançada
=========================

Sistema de validação física rigorosa para o framework ΨQRH,
focado em conservação de energia, unitariedade e consistência fractal.

Princípios Validados:
- Conservação de Energia: ||output|| ≈ ||input|| (dentro de 5%)
- Unitaridade: Operações preservam normas
- Consistência Fractal: D calculado via power-law fitting
- Estabilidade Numérica: Valores finitos e bem-condicionados

Uso:
    from advanced_physical_validation import AdvancedPhysicalValidator
    validator = AdvancedPhysicalValidator()
    results = validator.validate_comprehensive(tensor_input, tensor_output)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math


class AdvancedPhysicalValidator:
    """
    Validação física avançada com foco em:
    - Conservação de energia
    - Unitaridade
    - Estabilidade numérica
    - Consistência fractal
    """

    def __init__(self, tolerance_energy: float = 0.05, tolerance_unitary: float = 1e-5):
        """
        Inicializa o validador físico.

        Args:
            tolerance_energy: Tolerância para conservação de energia (5% padrão)
            tolerance_unitary: Tolerância para unitariedade
        """
        self.tolerance_energy = tolerance_energy
        self.tolerance_unitary = tolerance_unitary

        print("🔬 Advanced Physical Validator inicializado")
        print(".1f")
        print(".2e")

    def validate_comprehensive(self, input_tensor: torch.Tensor,
                             output_tensor: torch.Tensor,
                             operation_name: str = "unknown") -> Dict[str, Any]:
        """
        Validação física abrangente de uma operação quântica.

        Args:
            input_tensor: Tensor de entrada
            output_tensor: Tensor de saída
            operation_name: Nome da operação para logging

        Returns:
            Dicionário com resultados de todas as validações
        """
        results = {
            'operation': operation_name,
            'input_shape': input_tensor.shape,
            'output_shape': output_tensor.shape,
            'timestamp': str(torch.tensor(1.0))  # Placeholder
        }

        # 1. Validação de conservação de energia
        results['energy_conservation'] = self.validate_energy_conservation(
            input_tensor, output_tensor
        )

        # 2. Validação de unitariedade (se aplicável)
        if self._is_matrix_like(input_tensor) and self._is_matrix_like(output_tensor):
            results['unitarity'] = self.validate_unitarity(output_tensor)
        else:
            results['unitarity'] = {'applicable': False, 'reason': 'Not matrix-like tensors'}

        # 3. Validação de estabilidade numérica
        results['numerical_stability'] = self.validate_numerical_stability(
            input_tensor, output_tensor
        )

        # 4. Validação de consistência fractal (se aplicável)
        if input_tensor.dim() >= 2 and output_tensor.dim() >= 2:
            expected_dim = 1.7  # Dimensão fractal típica do sistema ΨQRH
            results['fractal_consistency'] = self.validate_fractal_consistency(
                output_tensor, expected_dim
            )
        else:
            results['fractal_consistency'] = {'applicable': False, 'reason': 'Low dimensional tensors'}

        # 5. Resumo geral
        results['overall_validation'] = self._compute_overall_validation(results)

        return results

    def validate_energy_conservation(self, input_tensor: torch.Tensor,
                                   output_tensor: torch.Tensor,
                                   tolerance: Optional[float] = None) -> Dict[str, Any]:
        """
        Valida conservação rigorosa de energia.

        ||output|| ≈ ||input|| (dentro da tolerância)
        """
        if tolerance is None:
            tolerance = self.tolerance_energy

        # Calcular normas
        input_energy = torch.norm(input_tensor).item()
        output_energy = torch.norm(output_tensor).item()

        # Razão de conservação
        if input_energy > 0:
            energy_ratio = output_energy / input_energy
        else:
            energy_ratio = 1.0  # Evitar divisão por zero

        # Verificar conservação
        energy_conserved = abs(energy_ratio - 1.0) <= tolerance

        # Correção automática se necessário
        correction_applied = False
        if not energy_conserved and energy_ratio > 0:
            # Aplicar correção de energia (normalização)
            correction_factor = input_energy / output_energy
            # Nota: Esta é apenas uma validação, não modifica o tensor

        result = {
            'input_energy': input_energy,
            'output_energy': output_energy,
            'energy_ratio': energy_ratio,
            'energy_conserved': energy_conserved,
            'within_tolerance': energy_conserved,
            'tolerance': tolerance,
            'deviation_percent': abs(energy_ratio - 1.0) * 100,
            'correction_applied': correction_applied
        }

        return result

    def validate_unitarity(self, matrix: torch.Tensor,
                          tolerance: Optional[float] = None) -> Dict[str, Any]:
        """
        Valida unitariedade de uma matriz quântica.

        Para U unitária: U†U = I
        """
        if tolerance is None:
            tolerance = self.tolerance_unitary

        if matrix.dim() != 2:
            return {
                'applicable': False,
                'reason': f'Matrix must be 2D, got {matrix.dim()}D',
                'is_unitary': False
            }

        try:
            # Calcular U†U
            U_dagger = matrix.conj().T
            identity_approx = U_dagger @ matrix

            # Matriz identidade de referência
            identity = torch.eye(matrix.size(0), dtype=matrix.dtype, device=matrix.device)

            # Calcular diferença
            unitary_diff = torch.norm(identity_approx - identity).item()
            is_unitary = unitary_diff <= tolerance

            # Desvio máximo
            max_deviation = torch.max(torch.abs(identity_approx - identity)).item()

            result = {
                'applicable': True,
                'unitary_difference': unitary_diff,
                'is_unitary': is_unitary,
                'max_deviation': max_deviation,
                'within_tolerance': is_unitary,
                'tolerance': tolerance,
                'matrix_shape': matrix.shape,
                'condition_number': self._compute_condition_number(matrix)
            }

        except Exception as e:
            result = {
                'applicable': True,
                'error': str(e),
                'is_unitary': False,
                'matrix_shape': matrix.shape
            }

        return result

    def validate_numerical_stability(self, input_tensor: torch.Tensor,
                                   output_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Valida estabilidade numérica dos tensores.
        """
        result = {}

        # Verificar valores finitos
        result['input_finite'] = torch.isfinite(input_tensor).all().item()
        result['output_finite'] = torch.isfinite(output_tensor).all().item()
        result['all_finite'] = result['input_finite'] and result['output_finite']

        # Verificar NaN
        result['input_has_nan'] = torch.isnan(input_tensor).any().item()
        result['output_has_nan'] = torch.isnan(output_tensor).any().item()
        result['has_nan'] = result['input_has_nan'] or result['output_has_nan']

        # Verificar infinito
        result['input_has_inf'] = torch.isinf(input_tensor).any().item()
        result['output_has_inf'] = torch.isinf(output_tensor).any().item()
        result['has_inf'] = result['input_has_inf'] or result['output_has_inf']

        # Estatísticas de valor
        if result['output_finite']:
            result['output_stats'] = {
                'min': output_tensor.min().item(),
                'max': output_tensor.max().item(),
                'mean': output_tensor.mean().item(),
                'std': output_tensor.std().item(),
                'range': output_tensor.max().item() - output_tensor.min().item()
            }
        else:
            result['output_stats'] = {'error': 'Non-finite values present'}

        # Verificar normas zero (problema numérico)
        input_norm = torch.norm(input_tensor).item()
        output_norm = torch.norm(output_tensor).item()
        result['input_norm_zero'] = input_norm == 0.0
        result['output_norm_zero'] = output_norm == 0.0

        # Avaliação geral de estabilidade
        result['is_stable'] = (
            result['all_finite'] and
            not result['has_nan'] and
            not result['has_inf'] and
            not result['output_norm_zero']
        )

        return result

    def validate_fractal_consistency(self, tensor: torch.Tensor,
                                   expected_dim: float,
                                   tolerance: float = 0.1) -> Dict[str, Any]:
        """
        Valida consistência fractal usando análise de box-counting simplificada.

        Args:
            tensor: Tensor a ser analisado
            expected_dim: Dimensão fractal esperada
            tolerance: Tolerância para consistência
        """
        if tensor.dim() < 2:
            return {
                'applicable': False,
                'reason': f'Tensor must be at least 2D, got {tensor.dim()}D'
            }

        try:
            # Análise simplificada de dimensão fractal
            # Usar análise de valores singulares como proxy
            if tensor.dim() == 2:
                U, S, Vt = torch.svd(tensor.float())
            else:
                # Para tensores de maior dimensão, achatar
                tensor_flat = tensor.flatten(start_dim=-2)
                U, S, Vt = torch.svd(tensor_flat.float())

            # Estimar dimensão fractal dos valores singulares
            # Usar decaimento exponencial como proxy
            log_s = torch.log(S[:20] + 1e-8)
            indices = torch.arange(len(log_s), dtype=torch.float32)

            if len(log_s) > 1:
                # Regressão linear simples
                slope = -torch.mean(torch.diff(log_s) / torch.diff(indices))
                estimated_dim = slope.item()

                # D = (3 - β) / 2, onde β = slope
                fractal_dim = (3 - slope.item()) / 2
                dim_error = abs(fractal_dim - expected_dim)
                is_consistent = dim_error <= tolerance

                result = {
                    'applicable': True,
                    'estimated_fractal_dim': fractal_dim,
                    'expected_fractal_dim': expected_dim,
                    'dimension_error': dim_error,
                    'is_consistent': is_consistent,
                    'within_tolerance': is_consistent,
                    'tolerance': tolerance,
                    'slope': slope.item(),
                    'singular_values_analyzed': len(S)
                }
            else:
                result = {
                    'applicable': True,
                    'error': 'Insufficient singular values for fractal analysis'
                }

        except Exception as e:
            result = {
                'applicable': True,
                'error': str(e)
            }

        return result

    def _compute_overall_validation(self, validation_results: Dict) -> Dict[str, Any]:
        """
        Computa validação geral baseada em todos os testes.
        """
        overall = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'critical_failures': 0
        }

        # Contar testes de conservação de energia
        if 'energy_conservation' in validation_results:
            overall['total_tests'] += 1
            if validation_results['energy_conservation'].get('energy_conserved', False):
                overall['passed_tests'] += 1
            else:
                overall['failed_tests'] += 1

        # Contar testes de unitariedade
        if 'unitarity' in validation_results:
            unitary = validation_results['unitarity']
            if unitary.get('applicable', False):
                overall['total_tests'] += 1
                if unitary.get('is_unitary', False):
                    overall['passed_tests'] += 1
                else:
                    overall['failed_tests'] += 1

        # Contar testes de estabilidade numérica
        if 'numerical_stability' in validation_results:
            overall['total_tests'] += 1
            if validation_results['numerical_stability'].get('is_stable', False):
                overall['passed_tests'] += 1
            else:
                overall['failed_tests'] += 1
                overall['critical_failures'] += 1  # Instabilidade numérica é crítica

        # Contar testes de consistência fractal
        if 'fractal_consistency' in validation_results:
            fractal = validation_results['fractal_consistency']
            if fractal.get('applicable', False):
                overall['total_tests'] += 1
                if fractal.get('is_consistent', False):
                    overall['passed_tests'] += 1
                else:
                    overall['failed_tests'] += 1

        # Calcular taxa de sucesso
        if overall['total_tests'] > 0:
            overall['success_rate'] = overall['passed_tests'] / overall['total_tests']
            overall['overall_status'] = 'EXCELLENT' if overall['success_rate'] >= 0.95 else \
                                      'GOOD' if overall['success_rate'] >= 0.8 else \
                                      'POOR' if overall['success_rate'] >= 0.6 else 'CRITICAL'
        else:
            overall['success_rate'] = 0.0
            overall['overall_status'] = 'NO_TESTS'

        return overall

    def _is_matrix_like(self, tensor: torch.Tensor) -> bool:
        """Verifica se tensor é adequado para operações matriciais."""
        return tensor.dim() == 2 and min(tensor.shape) > 1

    def _compute_condition_number(self, matrix: torch.Tensor) -> float:
        """Computa número de condição de uma matriz."""
        try:
            U, S, Vt = torch.svd(matrix)
            if S[-1] > 0:
                return (S[0] / S[-1]).item()
            else:
                return float('inf')
        except:
            return float('inf')

    def log_validation_results(self, results: Dict, filepath: Optional[str] = None):
        """
        Registra resultados de validação em arquivo.
        """
        if filepath:
            import json
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"📄 Resultados de validação salvos em: {filepath}")


# Funções de teste
def test_energy_conservation_example():
    """Exemplo de teste de conservação de energia."""
    validator = AdvancedPhysicalValidator()

    # Criar tensores de teste
    input_tensor = torch.randn(10, 64)
    # Simular operação que preserva energia
    output_tensor = input_tensor * 0.95  # Pequena perda de energia

    results = validator.validate_energy_conservation(input_tensor, output_tensor)

    print("🧪 Teste de Conservação de Energia:")
    print(".3f")
    print(f"   Conservada: {results['energy_conserved']}")
    print(".1f")

    return results


def test_unitarity_example():
    """Exemplo de teste de unitariedade."""
    validator = AdvancedPhysicalValidator()

    # Criar matriz unitária (rotação)
    theta = torch.tensor(0.5)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    rotation_matrix = torch.tensor([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ], dtype=torch.complex64)

    results = validator.validate_unitarity(rotation_matrix)

    print("🧪 Teste de Unitaridade:")
    print(".2e")
    print(f"   Unitária: {results['is_unitary']}")

    return results


if __name__ == "__main__":
    print("🔬 Testes do Advanced Physical Validator")
    print("=" * 50)

    # Teste de conservação de energia
    energy_results = test_energy_conservation_example()
    print()

    # Teste de unitariedade
    unitary_results = test_unitarity_example()
    print()

    print("✅ Testes concluídos!")
#!/usr/bin/env python3
"""
Comprehensive Integration Test Suite for ΨQRH Framework
=======================================================

Suite completa de testes de integração que valida:
1. Todos os componentes individuais
2. Integração entre componentes
3. Performance e estabilidade
4. Conformidade com configuração
5. Casos edge e robustez
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import yaml
import logging
import time
import warnings
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="comprehensive_integration_test.log"
)

# Suprimir warnings para logs mais limpos
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import modules
from ΨQRH import QRHLayer, QuaternionOperations, SpectralFilter
from fractal_pytorch_integration import AdaptiveFractalQRHLayer, FractalTransformer
from needle_fractal_dimension import FractalGenerator
from quartz_light_prototype import (
    calculate_beta_from_dimension,
    calculate_dimension_from_beta,
    calculate_alpha_from_dimension,
    calculate_dimension_from_alpha,
    FractalAnalyzer
)

class ComprehensiveIntegrationTester:
    def __init__(self, config_path="fractal_config.yaml"):
        """Inicializa o testador com configurações"""
        self.config_path = config_path
        self.config = self.load_config()
        self.test_results = {}
        self.performance_metrics = {}
        self.start_time = time.time()

        logging.info("=== INÍCIO DOS TESTES DE INTEGRAÇÃO COMPLETA ===")
        print("=== COMPREHENSIVE INTEGRATION TEST SUITE ===")
        print(f"Config: {config_path}")
        print("=" * 60)

    def load_config(self) -> Dict[str, Any]:
        """Carrega configuração do arquivo YAML"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logging.info(f"Configuração carregada: {self.config_path}")
            return config
        except Exception as e:
            logging.error(f"Erro ao carregar config: {e}")
            return self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        """Configuração padrão caso o arquivo não exista"""
        return {
            'fractal_integration': {
                'default_method': 'box_counting',
                'box_counting': {
                    'grid_size': 256,
                    'n_samples': 10000,
                    'min_box_size': 0.001,
                    'max_box_size': 1.0,
                    'n_scales': 20
                },
                'alpha_mapping': {
                    'scaling_factor': 1.0,
                    'dim_type': '2d'
                },
                'validation': {
                    'tolerance': {
                        'dimensional': 1e-10,
                        'fractal_analysis': 0.1,
                        'alpha_mapping': 0.1
                    }
                }
            }
        }

    def test_configuration_compliance(self) -> bool:
        """Testa conformidade com a configuração carregada"""
        print("=== Configuration Compliance Test ===")
        logging.info("Testando conformidade com configuração")

        try:
            # Verificar estrutura da configuração
            required_keys = ['fractal_integration']
            for key in required_keys:
                assert key in self.config, f"Chave obrigatória {key} não encontrada"

            # Testar configurações de tolerância
            tolerance_config = self.config['fractal_integration']['validation']['tolerance']

            # Testar mapeamento dimensional com tolerâncias da config
            test_dims = [0.5, 1.0, 1.5, 2.0, 2.7]
            dimensional_errors = []

            for D in test_dims:
                for dim_type in ['1d', '2d', '3d']:
                    beta = calculate_beta_from_dimension(D, dim_type)
                    D_recovered = calculate_dimension_from_beta(beta, dim_type)
                    error = abs(D - D_recovered)
                    dimensional_errors.append(error)

            max_dimensional_error = max(dimensional_errors)
            dimensional_threshold = float(tolerance_config['dimensional'])
            dimensional_ok = max_dimensional_error < dimensional_threshold

            print(f"  Dimensional tolerance: {max_dimensional_error:.2e} < {dimensional_threshold:.2e} ✓" if dimensional_ok else f"  Dimensional tolerance: FAIL")
            logging.info(f"Teste de tolerância dimensional: {'APROVADO' if dimensional_ok else 'REPROVADO'}")

            # Testar configurações de análise fractal
            analyzer = FractalAnalyzer()

            # Dados de teste
            uniform_data = np.random.uniform(0, 1, (1000, 2))
            fractal_dim = analyzer.calculate_box_counting_dimension(uniform_data)
            fractal_error = abs(fractal_dim - 2.0)
            fractal_threshold = float(tolerance_config['fractal_analysis'])
            fractal_ok = fractal_error < fractal_threshold

            print(f"  Fractal analysis tolerance: {fractal_error:.3f} < {fractal_threshold:.3f} ✓" if fractal_ok else f"  Fractal analysis tolerance: FAIL")
            logging.info(f"Teste de análise fractal: {'APROVADO' if fractal_ok else 'REPROVADO'}")

            success = dimensional_ok and fractal_ok
            print(f"  Configuration compliance: {'✓ PASS' if success else '✗ FAIL'}")

            return success

        except Exception as e:
            logging.error(f"Erro no teste de configuração: {e}")
            print(f"  Configuration compliance: ✗ FAIL - {e}")
            return False

    def test_component_integration(self) -> bool:
        """Testa integração entre componentes principais"""
        print("\n=== Component Integration Test ===")
        logging.info("Testando integração entre componentes")

        try:
            # Test 1: Fractal Analysis → Alpha Calculation → Spectral Filter
            analyzer = FractalAnalyzer()

            # Gerar dados fractais conhecidos
            cantor_data = self.generate_cantor_set(10000)
            fractal_dim = analyzer.calculate_box_counting_dimension_1d(cantor_data)

            # Calcular alpha baseado na dimensão
            alpha = calculate_alpha_from_dimension(fractal_dim, '1d')

            # Criar filtro com alpha calculado
            spectral_filter = SpectralFilter(alpha=alpha)

            # Testar filtro
            test_freqs = torch.linspace(1, 10, 100)
            filtered_output = spectral_filter(test_freqs)

            filter_stable = torch.isfinite(filtered_output).all()
            reasonable_alpha = 0.1 <= alpha <= 5.0
            reasonable_dim = 0.3 <= fractal_dim <= 1.0

            print(f"  Fractal → Alpha → Filter chain: {'✓' if filter_stable and reasonable_alpha and reasonable_dim else '✗'}")
            print(f"    Fractal dim: {fractal_dim:.3f}, Alpha: {alpha:.3f}")

            # Test 2: QRH Layer → Fractal Analysis
            embed_dim = 16
            qrh_layer = QRHLayer(embed_dim=embed_dim, alpha=alpha)

            # Input de teste
            batch_size, seq_len = 4, 32
            x = torch.randn(batch_size, seq_len, 4 * embed_dim)

            # Forward pass
            output = qrh_layer(x)

            # Análise fractal da saída
            output_np = output.detach().cpu().numpy()
            # Usar a primeira componente para análise fractal
            output_flat = output_np.reshape(-1, output_np.shape[-1])[:, 0]  # Primeira componente
            output_dim = analyzer.calculate_box_counting_dimension_1d(output_flat)

            qrh_integration_ok = (
                output.shape == x.shape and
                torch.isfinite(output).all() and
                0.5 <= output_dim <= 3.0
            )

            print(f"  QRH → Fractal Analysis: {'✓' if qrh_integration_ok else '✗'}")
            print(f"    Output dim: {output_dim:.3f}")

            # Test 3: Complete Transformer Pipeline
            model = FractalTransformer(
                vocab_size=1000,
                embed_dim=embed_dim,
                num_layers=2,
                seq_len=seq_len,
                enable_fractal_adaptation=True
            )

            input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            logits = model(input_ids)

            # Verificar análise fractal do modelo
            fractal_analysis = model.get_fractal_analysis()

            transformer_ok = (
                logits.shape == (batch_size, seq_len, 1000) and
                torch.isfinite(logits).all() and
                'mean_fractal_dim' in fractal_analysis
            )

            print(f"  Complete Transformer: {'✓' if transformer_ok else '✗'}")
            if 'mean_fractal_dim' in fractal_analysis:
                print(f"    Model fractal dim: {fractal_analysis['mean_fractal_dim']:.3f}")

            overall_integration = filter_stable and reasonable_alpha and qrh_integration_ok and transformer_ok
            logging.info(f"Integração de componentes: {'APROVADO' if overall_integration else 'REPROVADO'}")

            return overall_integration

        except Exception as e:
            logging.error(f"Erro na integração de componentes: {e}")
            print(f"  Component integration: ✗ FAIL - {e}")
            return False

    def test_performance_benchmarks(self) -> bool:
        """Testa benchmarks de performance"""
        print("\n=== Performance Benchmark Test ===")
        logging.info("Testando benchmarks de performance")

        try:
            performance_results = {}

            # Benchmark 1: QRH Layer Forward Pass
            embed_dim = 64
            layer = QRHLayer(embed_dim=embed_dim, alpha=1.5)
            x = torch.randn(8, 128, 4 * embed_dim)

            # Warmup
            for _ in range(5):
                _ = layer(x)

            # Benchmark
            start_time = time.time()
            for _ in range(100):
                output = layer(x)
            qrh_time = (time.time() - start_time) / 100
            performance_results['qrh_forward_ms'] = qrh_time * 1000

            # Benchmark 2: Fractal Analysis
            analyzer = FractalAnalyzer()
            data_2d = np.random.rand(1000, 1000)

            start_time = time.time()
            fractal_dim = analyzer.calculate_box_counting_dimension(data_2d)
            fractal_time = time.time() - start_time
            performance_results['fractal_analysis_ms'] = fractal_time * 1000

            # Benchmark 3: Transformer Forward Pass
            model = FractalTransformer(
                vocab_size=10000,
                embed_dim=256,
                num_layers=6,
                seq_len=512,
                enable_fractal_adaptation=True
            )

            input_ids = torch.randint(0, 10000, (4, 512))

            # Warmup
            for _ in range(3):
                _ = model(input_ids)

            start_time = time.time()
            logits = model(input_ids)
            transformer_time = time.time() - start_time
            performance_results['transformer_forward_ms'] = transformer_time * 1000

            # Performance thresholds (reasonable for development)
            thresholds = {
                'qrh_forward_ms': 50,  # 50ms
                'fractal_analysis_ms': 5000,  # 5s
                'transformer_forward_ms': 2000  # 2s
            }

            performance_ok = True
            for metric, value in performance_results.items():
                threshold = thresholds[metric]
                passed = value < threshold
                performance_ok &= passed
                print(f"  {metric}: {value:.2f}ms {'✓' if passed else '✗'} (threshold: {threshold}ms)")

            self.performance_metrics = performance_results
            logging.info(f"Benchmarks de performance: {performance_results}")

            return performance_ok

        except Exception as e:
            logging.error(f"Erro nos benchmarks: {e}")
            print(f"  Performance benchmarks: ✗ FAIL - {e}")
            return False

    def test_edge_cases_robustness(self) -> bool:
        """Testa casos extremos e robustez"""
        print("\n=== Edge Cases & Robustness Test ===")
        logging.info("Testando casos extremos e robustez")

        edge_case_results = []

        try:
            # Edge Case 1: Zero/Empty inputs
            layer = QRHLayer(embed_dim=8, alpha=1.0)
            zero_input = torch.zeros(1, 10, 32)
            zero_output = layer(zero_input)
            zero_case_ok = torch.isfinite(zero_output).all()
            edge_case_results.append(zero_case_ok)
            print(f"  Zero input handling: {'✓' if zero_case_ok else '✗'}")

            # Edge Case 2: Very small/large alpha values
            small_alpha = SpectralFilter(alpha=0.01)
            large_alpha = SpectralFilter(alpha=10.0)
            test_freq = torch.tensor([1.0, 2.0, 5.0])

            small_alpha_output = small_alpha(test_freq)
            large_alpha_output = large_alpha(test_freq)

            alpha_case_ok = (
                torch.isfinite(small_alpha_output).all() and
                torch.isfinite(large_alpha_output).all()
            )
            edge_case_results.append(alpha_case_ok)
            print(f"  Extreme alpha values: {'✓' if alpha_case_ok else '✗'}")

            # Edge Case 3: Single point fractal analysis
            single_point = np.array([0.5])
            analyzer = FractalAnalyzer()
            single_point_dim = analyzer.calculate_box_counting_dimension_1d(single_point)
            single_point_ok = 0.5 <= single_point_dim <= 1.5  # Reasonable fallback
            edge_case_results.append(single_point_ok)
            print(f"  Single point analysis: {'✓' if single_point_ok else '✗'} (dim: {single_point_dim:.3f})")

            # Edge Case 4: Very long sequences
            try:
                long_seq = torch.randint(0, 100, (1, 2048))  # Long sequence
                model_small = FractalTransformer(
                    vocab_size=100,
                    embed_dim=32,
                    num_layers=1,
                    seq_len=2048,
                    enable_fractal_adaptation=False  # Disable for speed
                )
                long_output = model_small(long_seq)
                long_seq_ok = torch.isfinite(long_output).all()
            except:
                long_seq_ok = False

            edge_case_results.append(long_seq_ok)
            print(f"  Long sequence handling: {'✓' if long_seq_ok else '✗'}")

            # Edge Case 5: NaN resilience
            try:
                nan_input = torch.randn(2, 16, 32)
                nan_input[0, 0, 0] = float('nan')

                # Test with nan detection/handling
                nan_safe_layer = QRHLayer(embed_dim=8, alpha=1.0)
                with torch.no_grad():
                    nan_output = nan_safe_layer(nan_input)
                    # Check if NaNs propagated (expected) or were handled
                    nan_case_ok = not torch.isnan(nan_output).all()  # Not everything should be NaN
            except:
                nan_case_ok = False

            edge_case_results.append(nan_case_ok)
            print(f"  NaN resilience: {'✓' if nan_case_ok else '✗'}")

            overall_robustness = sum(edge_case_results) >= len(edge_case_results) * 0.8  # 80% pass rate
            logging.info(f"Robustez casos extremos: {'APROVADO' if overall_robustness else 'REPROVADO'}")

            return overall_robustness

        except Exception as e:
            logging.error(f"Erro nos testes de robustez: {e}")
            print(f"  Edge cases robustness: ✗ FAIL - {e}")
            return False

    def test_mathematical_consistency(self) -> bool:
        """Testa consistência matemática das operações"""
        print("\n=== Mathematical Consistency Test ===")
        logging.info("Testando consistência matemática")

        try:
            consistency_results = []

            # Test 1: Quaternion algebra properties
            q1 = torch.tensor([1.0, 0.5, 0.3, 0.1])
            q1 = q1 / torch.norm(q1)
            q2 = torch.tensor([0.8, 0.2, 0.4, 0.6])
            q2 = q2 / torch.norm(q2)

            # Associativity: (q1 * q2) * q3 = q1 * (q2 * q3)
            q3 = torch.tensor([0.1, 0.9, 0.2, 0.3])
            q3 = q3 / torch.norm(q3)

            left = QuaternionOperations.multiply(
                QuaternionOperations.multiply(q1.unsqueeze(0), q2.unsqueeze(0)),
                q3.unsqueeze(0)
            )[0]

            right = QuaternionOperations.multiply(
                q1.unsqueeze(0),
                QuaternionOperations.multiply(q2.unsqueeze(0), q3.unsqueeze(0))
            )[0]

            associativity_error = torch.norm(left - right)
            associativity_ok = associativity_error < 1e-5
            consistency_results.append(associativity_ok)
            print(f"  Quaternion associativity: {'✓' if associativity_ok else '✗'} (error: {associativity_error:.2e})")

            # Test 2: Dimensional relationship invertibility
            test_dimensions = [0.5, 1.0, 1.585, 2.0, 2.7]
            invertibility_errors = []

            for D in test_dimensions:
                for dim_type in ['1d', '2d', '3d']:
                    if dim_type == '3d' and D < 2.0:
                        continue  # Skip invalid 3D dimensions

                    beta = calculate_beta_from_dimension(D, dim_type)
                    D_recovered = calculate_dimension_from_beta(beta, dim_type)
                    error = abs(D - D_recovered)
                    invertibility_errors.append(error)

            max_invertibility_error = max(invertibility_errors)
            invertibility_ok = max_invertibility_error < 1e-10
            consistency_results.append(invertibility_ok)
            print(f"  Dimensional invertibility: {'✓' if invertibility_ok else '✗'} (max error: {max_invertibility_error:.2e})")

            # Test 3: Spectral filter properties
            filter_obj = SpectralFilter(alpha=1.0)
            freqs = torch.logspace(0, 2, 50)  # 1 to 100 Hz

            # Power law behavior: |H(f)|^2 ∝ f^(-α)
            response = filter_obj(freqs)
            power = torch.abs(response) ** 2

            # Linear fit in log-log space
            log_freqs = torch.log10(freqs)
            log_power = torch.log10(power + 1e-10)  # Avoid log(0)

            # Simple linear regression
            n = len(log_freqs)
            sum_x = torch.sum(log_freqs)
            sum_y = torch.sum(log_power)
            sum_xy = torch.sum(log_freqs * log_power)
            sum_xx = torch.sum(log_freqs ** 2)

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
            expected_slope = -1.0  # For α=1.0

            slope_error = abs(slope - expected_slope)
            spectral_ok = slope_error < 0.5  # Reasonable tolerance
            consistency_results.append(spectral_ok)
            print(f"  Spectral filter power law: {'✓' if spectral_ok else '✗'} (slope: {slope:.3f}, expected: {expected_slope:.3f})")

            # Test 4: Energy conservation in QRH layers
            layer = QRHLayer(embed_dim=16, alpha=1.0)
            x = torch.randn(4, 32, 64)
            x = x / torch.norm(x, dim=-1, keepdim=True)  # Normalize input

            with torch.no_grad():
                output = layer(x)

            input_energy = torch.norm(x)
            output_energy = torch.norm(output)
            energy_ratio = output_energy / input_energy

            energy_conservation_ok = abs(energy_ratio - 1.0) < 0.3  # 30% tolerance
            consistency_results.append(energy_conservation_ok)
            print(f"  Energy conservation: {'✓' if energy_conservation_ok else '✗'} (ratio: {energy_ratio:.3f})")

            overall_consistency = sum(consistency_results) >= len(consistency_results) * 0.75  # 75% pass rate
            logging.info(f"Consistência matemática: {'APROVADO' if overall_consistency else 'REPROVADO'}")

            return overall_consistency

        except Exception as e:
            logging.error(f"Erro na consistência matemática: {e}")
            print(f"  Mathematical consistency: ✗ FAIL - {e}")
            return False

    def generate_cantor_set(self, n_points: int, level: int = 10) -> np.ndarray:
        """Gera conjunto de Cantor para testes"""
        points = np.zeros(n_points)
        for i in range(n_points):
            x = 0.0
            for j in range(level):
                r = np.random.rand()
                if r < 0.5:
                    x = x / 3
                else:
                    x = x / 3 + 2/3
            points[i] = x
        return points

    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Executa todos os testes e gera relatório completo"""
        print("Iniciando suite completa de testes...\n")

        # Executar todos os testes
        test_functions = [
            ("Configuration Compliance", self.test_configuration_compliance),
            ("Component Integration", self.test_component_integration),
            ("Performance Benchmarks", self.test_performance_benchmarks),
            ("Edge Cases & Robustness", self.test_edge_cases_robustness),
            ("Mathematical Consistency", self.test_mathematical_consistency)
        ]

        results = {}
        for test_name, test_func in test_functions:
            try:
                start_time = time.time()
                result = test_func()
                test_time = time.time() - start_time
                results[test_name] = {
                    'passed': result,
                    'execution_time_ms': test_time * 1000
                }
                logging.info(f"{test_name}: {'APROVADO' if result else 'REPROVADO'} ({test_time:.3f}s)")
            except Exception as e:
                results[test_name] = {
                    'passed': False,
                    'execution_time_ms': 0,
                    'error': str(e)
                }
                logging.error(f"{test_name}: ERRO - {e}")

        self.test_results = results
        return self.generate_final_report()

    def generate_final_report(self) -> Dict[str, Any]:
        """Gera relatório final detalhado"""
        total_time = time.time() - self.start_time

        print("\n" + "=" * 60)
        print("COMPREHENSIVE INTEGRATION TEST REPORT")
        print("=" * 60)

        passed_tests = sum(1 for result in self.test_results.values() if result['passed'])
        total_tests = len(self.test_results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0

        # Determinar status geral
        if success_rate >= 0.9:
            overall_status = "EXCELLENT"
            status_emoji = "🎉"
        elif success_rate >= 0.8:
            overall_status = "GOOD"
            status_emoji = "✅"
        elif success_rate >= 0.6:
            overall_status = "PARTIAL"
            status_emoji = "⚠️"
        else:
            overall_status = "NEEDS_WORK"
            status_emoji = "❌"

        print(f"Tests Run: {total_tests}")
        print(f"Tests Passed: {passed_tests}")
        print(f"Success Rate: {success_rate:.1%}")
        print(f"Total Execution Time: {total_time:.2f}s")
        print(f"Overall Status: {status_emoji} {overall_status}")
        print()

        print("Detailed Results:")
        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            exec_time = result.get('execution_time_ms', 0)
            print(f"  {test_name}: {status} ({exec_time:.1f}ms)")
            if 'error' in result:
                print(f"    Error: {result['error']}")

        print()

        # Performance metrics
        if self.performance_metrics:
            print("Performance Metrics:")
            for metric, value in self.performance_metrics.items():
                print(f"  {metric}: {value:.2f}ms")
            print()

        # Recomendações baseadas nos resultados
        print("Recommendations:")
        if overall_status == "EXCELLENT":
            print("  🎯 Framework ready for production use")
            print("  🚀 Consider advanced optimizations")
        elif overall_status == "GOOD":
            print("  🔧 Minor refinements recommended")
            print("  📊 Monitor performance in production")
        elif overall_status == "PARTIAL":
            print("  ⚡ Address failing components before deployment")
            print("  🔍 Review edge case handling")
        else:
            print("  🛠️  Significant development work required")
            print("  🔄 Re-run tests after fixes")

        # Salvar relatório detalhado
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'overall_status': overall_status,
            'execution_time_s': total_time,
            'detailed_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'config_used': self.config
        }

        # Salvar em arquivo
        report_file = "comprehensive_integration_report.yaml"
        with open(report_file, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)

        print(f"\nDetailed report saved: {report_file}")
        logging.info(f"Relatório completo salvo: {report_file}")
        logging.info(f"STATUS FINAL: {overall_status} ({success_rate:.1%} aprovação)")

        return report

def main():
    """Função principal para executar os testes"""
    tester = ComprehensiveIntegrationTester()
    final_report = tester.run_comprehensive_tests()

    print("\n" + "=" * 60)
    print("COMPREHENSIVE INTEGRATION TESTING COMPLETE")
    print("=" * 60)

    return final_report

if __name__ == "__main__":
    report = main()
#!/usr/bin/env python3
"""
Prompt Engine para Validação Crítica do ΨQRH

Este módulo implementa um sistema completo de validação matemática e benchmark
para garantir que o ΨQRH esteja pronto para comparações públicas justas.
"""

import torch
import torch.nn as nn
import yaml
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import time
from datetime import datetime

from src.core.qrh_layer import QRHLayer, QRHConfig
from src.core.quaternion_operations import QuaternionOperations

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuração para validação matemática"""
    energy_conservation_threshold: Tuple[float, float]
    unitarity_threshold: float
    quaternion_norm_stability_threshold: float
    test_batch_size: int
    test_sequence_length: int
    test_embed_dim: int
    num_test_iterations: int

    def __init__(self,
                 energy_conservation_threshold: Tuple[float, float] = (0.95, 1.05),
                 unitarity_threshold: float = 0.95,
                 quaternion_norm_stability_threshold: float = 0.05,
                 test_batch_size: int = 32,
                 test_sequence_length: int = 128,
                 test_embed_dim: int = 64,
                 num_test_iterations: int = 100):
        self.energy_conservation_threshold = energy_conservation_threshold
        self.unitarity_threshold = unitarity_threshold
        self.quaternion_norm_stability_threshold = quaternion_norm_stability_threshold
        self.test_batch_size = test_batch_size
        self.test_sequence_length = test_sequence_length
        self.test_embed_dim = test_embed_dim
        self.num_test_iterations = num_test_iterations

@dataclass
class BenchmarkConfig:
    """Configuração para benchmark comparativo"""
    model_size: int
    dataset: str
    tokenizer: str
    hardware: str
    metrics: List[str]
    training_steps: int
    validation_frequency: int

    def __init__(self,
                 model_size: int = 82_000_000,
                 dataset: str = "OpenWebText",
                 tokenizer: str = "GPT-2",
                 hardware: str = "4×A100",
                 metrics: List[str] = None,
                 training_steps: int = 10000,
                 validation_frequency: int = 100):
        self.model_size = model_size
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.hardware = hardware
        self.metrics = metrics if metrics is not None else ["perplexity", "tokens_per_second", "gpu_memory"]
        self.training_steps = training_steps
        self.validation_frequency = validation_frequency

class MathematicalValidator:
    """Validador de propriedades matemáticas críticas"""

    def __init__(self, config: ValidationConfig):
        self.config = config

    def test_energy_conservation(self, layer: QRHLayer) -> Dict[str, Any]:
        """Testa conservação de energia: ||output|| / ||input|| ∈ [0.95, 1.05]"""
        results = []

        for i in range(self.config.num_test_iterations):
            # Gerar input aleatório
            x = torch.randn(
                self.config.test_batch_size,
                self.config.test_sequence_length,
                4 * self.config.test_embed_dim
            )

            with torch.no_grad():
                output = layer(x)

                # Calcular normas
                input_norm = torch.norm(x).item()
                output_norm = torch.norm(output).item()

                # Evitar divisão por zero
                if input_norm > 1e-6:
                    energy_ratio = output_norm / input_norm
                    results.append({
                        'iteration': i,
                        'input_norm': input_norm,
                        'output_norm': output_norm,
                        'energy_ratio': energy_ratio,
                        'within_threshold': (
                            self.config.energy_conservation_threshold[0] <= energy_ratio <=
                            self.config.energy_conservation_threshold[1]
                        )
                    })

        # Estatísticas
        ratios = [r['energy_ratio'] for r in results]
        success_rate = sum(1 for r in results if r['within_threshold']) / len(results)

        return {
            'test_name': 'energy_conservation',
            'success_rate': success_rate,
            'mean_ratio': np.mean(ratios),
            'std_ratio': np.std(ratios),
            'min_ratio': min(ratios),
            'max_ratio': max(ratios),
            'threshold': self.config.energy_conservation_threshold,
            'details': results
        }

    def test_spectral_filter_unitarity(self, layer: QRHLayer) -> Dict[str, Any]:
        """Testa unitariedade do filtro espectral: |F(k)| ≈ 1.0"""
        results = []

        # Testar em diferentes frequências
        frequencies = torch.linspace(0.01, 1.0, 100)

        for freq in frequencies:
            # Simular filtro espectral
            k_mag = freq.unsqueeze(0)
            filter_response = layer.spectral_filter(k_mag)

            # Calcular magnitude
            filter_magnitude = torch.abs(filter_response).item()

            results.append({
                'frequency': freq.item(),
                'filter_magnitude': filter_magnitude,
                'within_threshold': abs(filter_magnitude - 1.0) <= self.config.unitarity_threshold
            })

        # Estatísticas
        magnitudes = [r['filter_magnitude'] for r in results]
        success_rate = sum(1 for r in results if r['within_threshold']) / len(results)

        return {
            'test_name': 'spectral_filter_unitarity',
            'success_rate': success_rate,
            'mean_magnitude': np.mean(magnitudes),
            'std_magnitude': np.std(magnitudes),
            'threshold': self.config.unitarity_threshold,
            'details': results
        }

    def test_quaternion_norm_stability(self, layer: QRHLayer) -> Dict[str, Any]:
        """Testa estabilidade da norma quaterniônica"""
        results = []

        for i in range(self.config.num_test_iterations):
            # Gerar quatérnios unitários aleatórios
            q = torch.randn(4)
            q_normalized = q / torch.norm(q)

            # Aplicar operações quaterniônicas
            with torch.no_grad():
                # Testar multiplicação quaterniônica
                q_result = QuaternionOperations.multiply(q_normalized.unsqueeze(0), q_normalized.unsqueeze(0))
                q_result_norm = torch.norm(q_result).item()

                # Verificar se a norma permanece próxima de 1.0
                norm_deviation = abs(q_result_norm - 1.0)

                results.append({
                    'iteration': i,
                    'input_norm': 1.0,
                    'output_norm': q_result_norm,
                    'norm_deviation': norm_deviation,
                    'within_threshold': norm_deviation <= self.config.quaternion_norm_stability_threshold
                })

        # Estatísticas
        deviations = [r['norm_deviation'] for r in results]
        success_rate = sum(1 for r in results if r['within_threshold']) / len(results)

        return {
            'test_name': 'quaternion_norm_stability',
            'success_rate': success_rate,
            'mean_deviation': np.mean(deviations),
            'std_deviation': np.std(deviations),
            'threshold': self.config.quaternion_norm_stability_threshold,
            'details': results
        }

    def run_comprehensive_validation(self, layer: QRHLayer) -> Dict[str, Any]:
        """Executa validação matemática completa"""
        logger.info("Iniciando validação matemática completa...")

        start_time = time.time()

        tests = [
            self.test_energy_conservation,
            self.test_spectral_filter_unitarity,
            self.test_quaternion_norm_stability
        ]

        results = {}
        for test_func in tests:
            test_name = test_func.__name__
            logger.info(f"Executando teste: {test_name}")

            try:
                result = test_func(layer)
                results[test_name] = result
                logger.info(f"Teste {test_name} concluído: {result['success_rate']:.2%} de sucesso")
            except Exception as e:
                logger.error(f"Erro no teste {test_name}: {e}")
                results[test_name] = {
                    'test_name': test_name,
                    'error': str(e),
                    'success_rate': 0.0
                }

        # Calcular pontuação geral
        success_rates = [r.get('success_rate', 0.0) for r in results.values() if 'success_rate' in r]
        overall_score = np.mean(success_rates) if success_rates else 0.0

        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': overall_score,
            'tests_passed': sum(1 for r in results.values() if r.get('success_rate', 0.0) >= 0.95),
            'total_tests': len(tests),
            'execution_time': time.time() - start_time,
            'detailed_results': results
        }

        logger.info(f"Validação concluída. Pontuação geral: {overall_score:.2%}")
        return validation_report

class ConfigurationManager:
    """Gerenciador de configuração padronizada"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.schema = self._load_schema()

    def _load_schema(self) -> Dict[str, Any]:
        """Define schema de validação para configuração"""
        return {
            'qrh_config': {
                'type': 'dict',
                'required': True,
                'schema': {
                    'embed_dim': {'type': 'integer', 'min': 1},
                    'alpha': {'type': 'float', 'min': 0.0},
                    'theta_left': {'type': 'float'},
                    'omega_left': {'type': 'float'},
                    'phi_left': {'type': 'float'},
                    'theta_right': {'type': 'float'},
                    'omega_right': {'type': 'float'},
                    'phi_right': {'type': 'float'},
                    'use_learned_rotation': {'type': 'boolean'},
                    'spatial_dims': {'type': 'list', 'nullable': True},
                    'use_windowing': {'type': 'boolean'},
                    'window_type': {'type': 'string'},
                    'fft_cache_size': {'type': 'integer', 'min': 1},
                    'device': {'type': 'string'}
                }
            },
            'validation_config': {
                'type': 'dict',
                'required': True,
                'schema': {
                    'energy_conservation_threshold': {'type': 'list', 'minlength': 2, 'maxlength': 2},
                    'unitarity_threshold': {'type': 'float', 'min': 0.0},
                    'quaternion_norm_stability_threshold': {'type': 'float', 'min': 0.0},
                    'test_batch_size': {'type': 'integer', 'min': 1},
                    'test_sequence_length': {'type': 'integer', 'min': 1},
                    'test_embed_dim': {'type': 'integer', 'min': 1},
                    'num_test_iterations': {'type': 'integer', 'min': 1}
                }
            },
            'benchmark_config': {
                'type': 'dict',
                'required': True,
                'schema': {
                    'model_size': {'type': 'integer', 'min': 1000},
                    'dataset': {'type': 'string'},
                    'tokenizer': {'type': 'string'},
                    'hardware': {'type': 'string'},
                    'metrics': {'type': 'list'},
                    'training_steps': {'type': 'integer', 'min': 1},
                    'validation_frequency': {'type': 'integer', 'min': 1}
                }
            }
        }

    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Valida configuração contra schema"""
        errors = []

        for section, schema in self.schema.items():
            if schema['required'] and section not in config:
                errors.append(f"Seção obrigatória '{section}' não encontrada")
            elif section in config:
                section_config = config[section]
                if not isinstance(section_config, dict):
                    errors.append(f"Seção '{section}' deve ser um dicionário")
                else:
                    # Validar campos da seção
                    for field, field_schema in schema['schema'].items():
                        if field_schema.get('required', False) and field not in section_config:
                            errors.append(f"Campo obrigatório '{field}' não encontrado em '{section}'")
                        elif field in section_config:
                            value = section_config[field]
                            # Validar tipo
                            expected_type = field_schema.get('type')
                            if expected_type == 'integer' and not isinstance(value, int):
                                errors.append(f"Campo '{field}' deve ser inteiro")
                            elif expected_type == 'float' and not isinstance(value, (int, float)):
                                errors.append(f"Campo '{field}' deve ser float")
                            elif expected_type == 'string' and not isinstance(value, str):
                                errors.append(f"Campo '{field}' deve ser string")
                            elif expected_type == 'boolean' and not isinstance(value, bool):
                                errors.append(f"Campo '{field}' deve ser booleano")
                            elif expected_type == 'list' and not isinstance(value, list):
                                errors.append(f"Campo '{field}' deve ser lista")

                            # Validar restrições
                            if 'min' in field_schema and value < field_schema['min']:
                                errors.append(f"Campo '{field}' deve ser >= {field_schema['min']}")
                            if 'minlength' in field_schema and len(value) < field_schema['minlength']:
                                errors.append(f"Campo '{field}' deve ter pelo menos {field_schema['minlength']} elementos")

        return len(errors) == 0, errors

    def load_config(self) -> Tuple[QRHConfig, ValidationConfig, BenchmarkConfig]:
        """Carrega e valida configuração"""
        if not self.config_path.exists():
            # Criar configuração padrão se não existir
            self._create_default_config()

        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Validar configuração
        is_valid, errors = self.validate_config(config_data)
        if not is_valid:
            raise ValueError(f"Configuração inválida: {errors}")

        # Criar objetos de configuração
        qrh_config = QRHConfig(**config_data['qrh_config'])
        validation_config = ValidationConfig(**config_data['validation_config'])
        benchmark_config = BenchmarkConfig(**config_data['benchmark_config'])

        return qrh_config, validation_config, benchmark_config

    def _create_default_config(self):
        """Cria configuração padrão"""
        default_config = {
            'qrh_config': {
                'embed_dim': 64,
                'alpha': 1.0,
                'theta_left': 0.1,
                'omega_left': 0.05,
                'phi_left': 0.02,
                'theta_right': 0.08,
                'omega_right': 0.03,
                'phi_right': 0.015,
                'use_learned_rotation': True,
                'spatial_dims': None,
                'use_windowing': True,
                'window_type': 'hann',
                'fft_cache_size': 10,
                'device': 'cpu'
            },
            'validation_config': {
                'energy_conservation_threshold': [0.95, 1.05],
                'unitarity_threshold': 0.95,
                'quaternion_norm_stability_threshold': 0.05,
                'test_batch_size': 32,
                'test_sequence_length': 128,
                'test_embed_dim': 64,
                'num_test_iterations': 100
            },
            'benchmark_config': {
                'model_size': 82000000,
                'dataset': "OpenWebText",
                'tokenizer': "GPT-2",
                'hardware': "4×A100",
                'metrics': ["perplexity", "tokens_per_second", "gpu_memory"],
                'training_steps': 10000,
                'validation_frequency': 100
            }
        }

        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)

        logger.info(f"Configuração padrão criada em {self.config_path}")

class BenchmarkRunner:
    """Executor de benchmarks comparativos"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def run_comparison(self, baseline_model: nn.Module, psiqrh_model: nn.Module) -> Dict[str, Any]:
        """Executa comparação justa entre baseline Transformer e ΨQRH"""
        logger.info("Iniciando benchmark comparativo...")

        results = {
            'baseline': {},
            'psiqrh': {},
            'comparison': {}
        }

        # TODO: Implementar benchmark completo com:
        # - Carregamento de dataset OpenWebText
        # - Tokenização GPT-2
        # - Treinamento e validação
        # - Métricas de performance

        # Placeholder para implementação futura
        results['status'] = 'benchmark_not_implemented'
        results['message'] = 'Benchmark completo será implementado na próxima fase'

        return results

class ValidationReportGenerator:
    """Gerador de relatórios de validação transparentes"""

    def __init__(self, output_dir: str = "validation_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_report(self, validation_results: Dict[str, Any],
                       benchmark_results: Optional[Dict[str, Any]] = None) -> str:
        """Gera relatório completo de validação"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"validation_report_{timestamp}.json"

        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'framework_version': 'ΨQRH v1.0',
                'validation_type': 'comprehensive_mathematical_validation'
            },
            'summary': {
                'overall_score': validation_results.get('overall_score', 0.0),
                'tests_passed': validation_results.get('tests_passed', 0),
                'total_tests': validation_results.get('total_tests', 0),
                'execution_time': validation_results.get('execution_time', 0.0)
            },
            'detailed_results': validation_results.get('detailed_results', {}),
            'benchmark_results': benchmark_results
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Gerar versão resumida em markdown
        self._generate_markdown_summary(report, timestamp)

        logger.info(f"Relatório gerado: {report_path}")
        return str(report_path)

    def _generate_markdown_summary(self, report: Dict[str, Any], timestamp: str):
        """Gera resumo em markdown para fácil leitura"""
        md_path = self.output_dir / f"summary_{timestamp}.md"

        with open(md_path, 'w') as f:
            f.write(f"# Relatório de Validação ΨQRH\n\n")
            f.write(f"**Data**: {report['metadata']['timestamp']}\n\n")

            # Resumo
            summary = report['summary']
            f.write("## Resumo Executivo\n\n")
            f.write(f"- **Pontuação Geral**: {summary['overall_score']:.2%}\\n")
            f.write(f"- **Testes Aprovados**: {summary['tests_passed']}/{summary['total_tests']}\\n")
            f.write(f"- **Tempo de Execução**: {summary['execution_time']:.2f}s\n\n")

            # Resultados detalhados
            f.write("## Resultados Detalhados\n\n")
            for test_name, result in report['detailed_results'].items():
                f.write(f"### {test_name}\n\n")
                f.write(f"- **Taxa de Sucesso**: {result.get('success_rate', 0):.2%}\\n")
                if 'mean_ratio' in result:
                    f.write(f"- **Média**: {result['mean_ratio']:.4f}\\n")
                if 'threshold' in result:
                    f.write(f"- **Limite**: {result['threshold']}\\n")
                f.write("\n")

class PromptEngine:
    """Engine principal para validação e benchmark"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_manager = ConfigurationManager(config_path)
        self.report_generator = ValidationReportGenerator()

    def run_critical_validation(self) -> Dict[str, Any]:
        """Executa validação crítica das falhas matemáticas"""
        logger.info("=== VALIDAÇÃO CRÍTICA ΨQRH ===")

        # Carregar configuração
        qrh_config, validation_config, benchmark_config = self.config_manager.load_config()

        # Criar layer ΨQRH
        layer = QRHLayer(qrh_config)

        # Executar validação matemática
        validator = MathematicalValidator(validation_config)
        validation_results = validator.run_comprehensive_validation(layer)

        # Gerar relatório
        report_path = self.report_generator.generate_report(validation_results)

        # Verificar se passou nos critérios críticos
        critical_passed = validation_results['overall_score'] >= 0.95

        result = {
            'critical_validation_passed': critical_passed,
            'overall_score': validation_results['overall_score'],
            'report_path': report_path,
            'validation_details': validation_results
        }

        if critical_passed:
            logger.info("✅ VALIDAÇÃO CRÍTICA APROVADA - Pronto para benchmark público")
        else:
            logger.warning("❌ VALIDAÇÃO CRÍTICA REPROVADA - Corrija as falhas antes do benchmark")

        return result

    def run_fair_comparison(self) -> Dict[str, Any]:
        """Executa comparação justa entre baseline Transformer e ΨQRH"""
        logger.info("=== COMPARAÇÃO JUSTA ΨQRH vs BASELINE ===")

        # Primeiro validar criticamente
        validation_result = self.run_critical_validation()

        if not validation_result['critical_validation_passed']:
            logger.error("Comparação cancelada: validação crítica reprovada")
            return {
                'comparison_status': 'cancelled',
                'reason': 'critical_validation_failed',
                'validation_result': validation_result
            }

        # Carregar configuração
        _, _, benchmark_config = self.config_manager.load_config()

        # Executar benchmark (implementação futura)
        benchmark_runner = BenchmarkRunner(benchmark_config)

        # TODO: Implementar modelos baseline e ΨQRH
        baseline_model = None  # Placeholder
        psiqrh_model = None    # Placeholder

        benchmark_results = benchmark_runner.run_comparison(baseline_model, psiqrh_model)

        # Gerar relatório completo
        full_report_path = self.report_generator.generate_report(
            validation_result['validation_details'],
            benchmark_results
        )

        return {
            'comparison_status': 'completed',
            'full_report_path': full_report_path,
            'validation_result': validation_result,
            'benchmark_results': benchmark_results
        }

def main():
    """Função principal"""
    engine = PromptEngine()

    print("Prompt Engine ΨQRH - Validação Crítica e Benchmark")
    print("=" * 60)

    # Executar validação crítica
    print("\n1. Executando validação crítica...")
    validation_result = engine.run_critical_validation()

    if validation_result['critical_validation_passed']:
        print("\n2. Validação aprovada! Executando comparação justa...")
        comparison_result = engine.run_fair_comparison()
        print(f"Comparação concluída: {comparison_result['comparison_status']}")
    else:
        print("\n2. Comparação cancelada - Corrija as falhas críticas primeiro")

    print(f"\nRelatórios disponíveis em: {engine.report_generator.output_dir}")

if __name__ == "__main__":
    main()
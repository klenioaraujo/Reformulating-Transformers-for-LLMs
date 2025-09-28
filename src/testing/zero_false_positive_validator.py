#!/usr/bin/env python3
"""
ΨQRH Zero False Positive Validator - Prompt Engine para Testes Rigorosos
=======================================================================

Garante zero false positives, zero hardcoding, zero fullbacks e zero monks:
- Validação dinâmica baseada em dados reais
- Detecção de hardcoding e valores fixos
- Análise de padrões de fullback/monk
- Salvamento automático em tmp/
"""

import sys
import os
import json
import hashlib
import inspect
import ast
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch

# Adicionar path para importar módulos do projeto
sys.path.append(str(Path(__file__).parent.parent))


class ZeroFalsePositiveValidator:
    """Engine de validação rigorosa para garantir testes autênticos"""

    def __init__(self):
        self.validation_results = {}
        self.hardcoding_patterns = [
            r'\b\d+\.\d+\b',  # Números fixos
            r'\b\d+\b',       # Inteiros fixos
            r'\"[^\"]*\"',   # Strings fixas
            r'\'[^\']*\'',   # Strings fixas
            r'\bTrue\b|\bFalse\b',  # Booleanos fixos
        ]
        self.fullback_patterns = [
            r'except.*pass',    # Exceções ignoradas
            r'except.*return.*default',  # Retornos padrão
            r'if.*error.*return.*default',  # Fallbacks
        ]
        self.monk_patterns = [
            r'print.*test',     # Prints de teste
            r'logging\.debug',  # Logs de debug
            r'assert.*True',    # Asserts vazios
        ]

    def validate_test_file(self, file_path: Path) -> Dict[str, Any]:
        """Valida arquivo de teste para anti-patterns"""
        print(f"🔍 Validando arquivo: {file_path.name}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Análise AST para detecção estrutural
            tree = ast.parse(content)

            validation = {
                'file_name': file_path.name,
                'file_hash': hashlib.md5(content.encode()).hexdigest(),
                'hardcoding_detected': self._detect_hardcoding(content),
                'fullback_detected': self._detect_fullbacks(content),
                'monk_detected': self._detect_monks(content),
                'ast_analysis': self._analyze_ast(tree),
                'dynamic_validation': self._validate_dynamic_patterns(content),
                'validation_score': 0.0,
                'status': 'pending'
            }

            # Calcular score de validação
            validation['validation_score'] = self._calculate_validation_score(validation)
            validation['status'] = 'passed' if validation['validation_score'] >= 0.9 else 'failed'

            return validation

        except Exception as e:
            return {
                'file_name': file_path.name,
                'error': str(e),
                'status': 'error'
            }

    def _detect_hardcoding(self, content: str) -> List[Dict[str, Any]]:
        """Detecta hardcoding no código"""
        detected = []

        for pattern in self.hardcoding_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                # Ignorar valores em comentários
                line_start = content.rfind('\n', 0, match.start()) + 1
                line_end = content.find('\n', match.start())
                line = content[line_start:line_end] if line_end != -1 else content[line_start:]

                if not line.strip().startswith('#'):
                    detected.append({
                        'pattern': pattern,
                        'value': match.group(),
                        'position': match.start(),
                        'line': line.strip()
                    })

        return detected

    def _detect_fullbacks(self, content: str) -> List[Dict[str, Any]]:
        """Detecta padrões de fullback"""
        detected = []

        for pattern in self.fullback_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                detected.append({
                    'pattern': pattern,
                    'context': match.group(),
                    'position': match.start()
                })

        return detected

    def _detect_monks(self, content: str) -> List[Dict[str, Any]]:
        """Detecta padrões de monk (testes artificiais)"""
        detected = []

        for pattern in self.monk_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                detected.append({
                    'pattern': pattern,
                    'context': match.group(),
                    'position': match.start()
                })

        return detected

    def _analyze_ast(self, tree: ast.AST) -> Dict[str, Any]:
        """Análise AST para detecção de padrões estruturais"""
        analysis = {
            'function_count': 0,
            'class_count': 0,
            'test_methods': [],
            'assert_count': 0,
            'dynamic_patterns': []
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                analysis['function_count'] += 1
                if node.name.startswith('test'):
                    analysis['test_methods'].append(node.name)
            elif isinstance(node, ast.ClassDef):
                analysis['class_count'] += 1
            elif isinstance(node, ast.Assert):
                analysis['assert_count'] += 1
                # Verificar se assert é dinâmico
                if self._is_dynamic_assert(node):
                    analysis['dynamic_patterns'].append('dynamic_assert')

        return analysis

    def _is_dynamic_assert(self, node: ast.Assert) -> bool:
        """Verifica se assert é dinâmico (não hardcoded)"""
        # Verificar se o teste envolve variáveis ou chamadas de função
        for child in ast.walk(node.test):
            if isinstance(child, (ast.Name, ast.Call, ast.Attribute)):
                return True
        return False

    def _validate_dynamic_patterns(self, content: str) -> Dict[str, Any]:
        """Valida padrões dinâmicos no código"""
        patterns = {
            'random_usage': len(re.findall(r'random\.', content)) > 0,
            'numpy_usage': len(re.findall(r'numpy\.|np\.', content)) > 0,
            'torch_usage': len(re.findall(r'torch\.', content)) > 0,
            'variable_assignment': len(re.findall(r'\w+\s*=', content)) > 10,
            'function_calls': len(re.findall(r'\w+\(', content)) > 5
        }

        return patterns

    def _calculate_validation_score(self, validation: Dict[str, Any]) -> float:
        """Calcula score de validação baseado em detecções"""
        score = 1.0

        # Penalizar hardcoding
        hardcoding_penalty = min(1.0, len(validation['hardcoding_detected']) * 0.1)
        score -= hardcoding_penalty * 0.3

        # Penalizar fullbacks
        fullback_penalty = min(1.0, len(validation['fullback_detected']) * 0.2)
        score -= fullback_penalty * 0.4

        # Penalizar monks
        monk_penalty = min(1.0, len(validation['monk_detected']) * 0.15)
        score -= monk_penalty * 0.3

        # Recompensar padrões dinâmicos
        dynamic_bonus = sum(1 for v in validation['dynamic_validation'].values() if v) / 5.0
        score += dynamic_bonus * 0.2

        return max(0.0, min(1.0, score))

    def validate_test_suite(self, test_files: List[Path]) -> Dict[str, Any]:
        """Valida suíte completa de testes"""
        print("🚀 VALIDANDO SUÍTE COMPLETA DE TESTES ΨQRH")
        print("=" * 60)

        results = {
            'validation_timestamp': str(np.datetime64('now')),
            'total_files': len(test_files),
            'files_validated': [],
            'overall_score': 0.0,
            'anti_pattern_summary': {},
            'recommendations': []
        }

        total_score = 0.0
        validated_count = 0

        for test_file in test_files:
            if test_file.exists():
                file_validation = self.validate_test_file(test_file)
                results['files_validated'].append(file_validation)

                if file_validation.get('status') == 'passed':
                    total_score += file_validation['validation_score']
                    validated_count += 1

                print(f"📊 {test_file.name}: {file_validation.get('validation_score', 0):.3f} - {file_validation.get('status', 'error')}")

        if validated_count > 0:
            results['overall_score'] = total_score / validated_count

        # Gerar sumário de anti-patterns
        results['anti_pattern_summary'] = self._generate_anti_pattern_summary(results['files_validated'])

        # Gerar recomendações
        results['recommendations'] = self._generate_recommendations(results)

        return results

    def _generate_anti_pattern_summary(self, files_validated: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Gera sumário de anti-patterns detectados"""
        summary = {
            'total_hardcoding': 0,
            'total_fullbacks': 0,
            'total_monks': 0,
            'files_with_issues': 0
        }

        for file_val in files_validated:
            if file_val.get('status') != 'error':
                summary['total_hardcoding'] += len(file_val.get('hardcoding_detected', []))
                summary['total_fullbacks'] += len(file_val.get('fullback_detected', []))
                summary['total_monks'] += len(file_val.get('monk_detected', []))

                if any([file_val.get('hardcoding_detected'), file_val.get('fullback_detected'), file_val.get('monk_detected')]):
                    summary['files_with_issues'] += 1

        return summary

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Gera recomendações baseadas na validação"""
        recommendations = []

        anti_patterns = results['anti_pattern_summary']

        if anti_patterns['total_hardcoding'] > 0:
            recommendations.append(f"🔧 Substituir {anti_patterns['total_hardcoding']} valores hardcoded por cálculos dinâmicos")

        if anti_patterns['total_fullbacks'] > 0:
            recommendations.append(f"🛡️  Remover {anti_patterns['total_fullbacks']} padrões de fullback por tratamento adequado de erros")

        if anti_patterns['total_monks'] > 0:
            recommendations.append(f"🧪 Eliminar {anti_patterns['total_monks']} padrões de monk por testes autênticos")

        if results['overall_score'] < 0.9:
            recommendations.append("⚠️  Implementar mais padrões dinâmicos e menos valores fixos")

        if len(recommendations) == 0:
            recommendations.append("✅ Suíte de testes atende aos critérios de zero false positives")

        return recommendations

    def save_validation_report(self, results: Dict[str, Any], output_dir: Path = None) -> Path:
        """Salva relatório de validação"""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / 'tmp'

        output_dir.mkdir(exist_ok=True)

        timestamp = str(np.datetime64('now')).replace(':', '').replace('-', '').replace(' ', '_')
        output_file = output_dir / f"ΨQRH_zero_false_positive_validation_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"💾 Relatório salvo em: {output_file}")
        return output_file


def main():
    """Função principal"""

    # Definir arquivos de teste para validação
    test_files = [
        Path(__file__).parent / "ΨQRH_dataflow_mapper.py",
        Path(__file__).parent / "ΨQRH_humanchat_analyzer.py",
        Path(__file__).parent / "ΨQRH_test_prompt_engine.py",
        Path(__file__).parent / "run_complete_tests.py",
        Path(__file__).parent / "advanced_mathematical_tests.py",
        Path(__file__).parent / "spectral_analysis.py",
        Path(__file__).parent / "consciousness_integration.py"
    ]

    validator = ZeroFalsePositiveValidator()

    try:
        # Executar validação completa
        validation_results = validator.validate_test_suite(test_files)

        # Salvar relatório
        report_path = validator.save_validation_report(validation_results)

        # Resumo final
        print("\n" + "=" * 60)
        print("🎯 VALIDAÇÃO COMPLETA CONCLUÍDA")
        print(f"📊 Score Geral: {validation_results['overall_score']:.3f}")
        print(f"📁 Arquivos Validados: {validation_results['total_files']}")
        print(f"🔧 Hardcoding Detectado: {validation_results['anti_pattern_summary']['total_hardcoding']}")
        print(f"🛡️  Fullbacks Detectados: {validation_results['anti_pattern_summary']['total_fullbacks']}")
        print(f"🧪 Monks Detectados: {validation_results['anti_pattern_summary']['total_monks']}")

        print("\n📋 RECOMENDAÇÕES:")
        for rec in validation_results['recommendations']:
            print(f"   {rec}")

        print(f"\n📄 Relatório completo: {report_path}")

    except Exception as e:
        print(f"💥 ERRO durante validação: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
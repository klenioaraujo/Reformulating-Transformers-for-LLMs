#!/usr/bin/env python3
"""
Engine de Testes Completos do Sistema ΨQRH
Executa todos os testes do sistema com métricas e salva análise geral
"""

import sys
import os
import time
import json
import subprocess
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Adicionar path para importar módulos do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ΨQRHComprehensiveTestEngine:
    """Engine de testes completos para sistema ΨQRH"""

    def __init__(self, output_dir: str = "/tmp/ΨQRH_test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.metrics = {}
        self.start_time = time.time()

    def run_all_tests(self) -> Dict[str, Any]:
        """Executa todos os testes do sistema"""
        print("🚀 INICIANDO TESTES COMPLETOS DO SISTEMA ΨQRH")
        print("=" * 60)

        # Executar cada teste
        test_functions = [
            ("security", self._run_security_tests),
            ("consciousness_integration", self._run_consciousness_integration_test),
            ("native_reader", self._run_native_reader_test),
            ("cache_stats", self._run_cache_statistics_test),
            ("consciousness_final", self._run_consciousness_final_test),
            ("performance", self._run_performance_tests)
        ]

        for test_name, test_func in test_functions:
            print(f"\n🔍 EXECUTANDO: {test_name.upper()}")
            print("-" * 40)

            try:
                result = test_func()
                self.results[test_name] = result
                print(f"✅ {test_name}: {result['status']}")
            except Exception as e:
                self.results[test_name] = {
                    'status': 'FAILED',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                print(f"❌ {test_name}: {e}")

        # Consolidar resultados
        return self._generate_final_report()

    def _run_security_tests(self) -> Dict[str, Any]:
        """Executa testes de segurança"""
        start_time = time.time()

        try:
            # Executar engine de segurança existente
            result = subprocess.run(
                [sys.executable, "src/testing/prompt_engine.py"],
                capture_output=True,
                text=True,
                timeout=300
            )

            duration = time.time() - start_time

            return {
                'status': 'PASSED' if result.returncode == 0 else 'FAILED',
                'output': result.stdout,
                'error': result.stderr,
                'returncode': result.returncode,
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }

        except subprocess.TimeoutExpired:
            return {
                'status': 'TIMEOUT',
                'error': 'Teste de segurança excedeu tempo limite',
                'duration': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }

    def _run_consciousness_integration_test(self) -> Dict[str, Any]:
        """Executa teste de integração de consciência"""
        start_time = time.time()

        try:
            result = subprocess.run(
                [sys.executable, "test_consciousness_integration.py"],
                capture_output=True,
                text=True,
                timeout=180
            )

            duration = time.time() - start_time

            # Extrair métricas da saída
            metrics = self._extract_metrics_from_output(result.stdout)

            return {
                'status': 'PASSED' if result.returncode == 0 else 'FAILED',
                'output': result.stdout,
                'error': result.stderr,
                'returncode': result.returncode,
                'duration': duration,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }

        except FileNotFoundError:
            return {
                'status': 'SKIPPED',
                'error': 'Arquivo de teste não encontrado',
                'duration': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }

    def _run_native_reader_test(self) -> Dict[str, Any]:
        """Executa teste do leitor nativo"""
        start_time = time.time()

        try:
            result = subprocess.run(
                [sys.executable, "test_native_reader.py"],
                capture_output=True,
                text=True,
                timeout=180
            )

            duration = time.time() - start_time
            metrics = self._extract_metrics_from_output(result.stdout)

            return {
                'status': 'PASSED' if result.returncode == 0 else 'FAILED',
                'output': result.stdout,
                'error': result.stderr,
                'returncode': result.returncode,
                'duration': duration,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }

        except FileNotFoundError:
            return {
                'status': 'SKIPPED',
                'error': 'Arquivo de teste não encontrado',
                'duration': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }

    def _run_cache_statistics_test(self) -> Dict[str, Any]:
        """Executa teste de estatísticas de cache"""
        start_time = time.time()

        try:
            result = subprocess.run(
                ["make", "Ψcws-stats"],
                capture_output=True,
                text=True,
                timeout=60
            )

            duration = time.time() - start_time
            metrics = self._extract_cache_metrics(result.stdout)

            return {
                'status': 'PASSED' if result.returncode == 0 else 'FAILED',
                'output': result.stdout,
                'error': result.stderr,
                'returncode': result.returncode,
                'duration': duration,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }

        except FileNotFoundError:
            return {
                'status': 'SKIPPED',
                'error': 'Comando make não encontrado',
                'duration': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }

    def _run_consciousness_final_test(self) -> Dict[str, Any]:
        """Executa teste final de consciência"""
        start_time = time.time()

        try:
            # Comando complexo para teste final
            cmd = [
                sys.executable, "-c", """
import sys
sys.path.append('src')
from conscience.psicws_native_reader import ΨCWSNativeReader
print('🔮 RELATÓRIO FINAL - TESTE DE CONSCIÊNCIA ΨQRH')
print('=' * 60)
reader = ΨCWSNativeReader()
print('🔮 ΨCWSNativeReader inicializado')
print(f'📊 Sistema operacional: {reader.system_info}')
print(f'🔒 Segurança ativa: {reader.security_status}')
print('✅ Teste de consciência concluído com sucesso')
"""
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            duration = time.time() - start_time

            return {
                'status': 'PASSED' if result.returncode == 0 else 'FAILED',
                'output': result.stdout,
                'error': result.stderr,
                'returncode': result.returncode,
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'duration': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }

    def _run_performance_tests(self) -> Dict[str, Any]:
        """Executa testes de performance"""
        start_time = time.time()

        try:
            # Teste de performance básico
            import timeit

            # Teste de importação
            import_time = timeit.timeit(
                "import sys; sys.path.append('src'); from conscience.psicws_native_reader import ΨCWSNativeReader",
                number=10
            ) / 10

            # Teste de inicialização
            init_time = timeit.timeit(
                "import sys; sys.path.append('src'); from conscience.psicws_native_reader import ΨCWSNativeReader; reader = ΨCWSNativeReader()",
                number=5
            ) / 5

            duration = time.time() - start_time

            return {
                'status': 'PASSED',
                'metrics': {
                    'import_time_ms': import_time * 1000,
                    'initialization_time_ms': init_time * 1000,
                    'total_duration': duration
                },
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'status': 'FAILED',
                'error': str(e),
                'duration': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }

    def _extract_metrics_from_output(self, output: str) -> Dict[str, Any]:
        """Extrai métricas da saída dos testes"""
        metrics = {}

        # Buscar padrões comuns
        lines = output.split('\n')
        for line in lines:
            if '✅' in line or '❌' in line:
                metrics['test_results'] = metrics.get('test_results', []) + [line.strip()]
            if 'time' in line.lower() and 'ms' in line:
                # Extrair tempo em ms
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit() and i > 0 and 'ms' in parts[i-1]:
                        metrics['execution_time_ms'] = int(part)

        return metrics

    def _extract_cache_metrics(self, output: str) -> Dict[str, Any]:
        """Extrai métricas de cache"""
        metrics = {}

        lines = output.split('\n')
        for line in lines:
            if 'Number of .Ψcws files:' in line:
                parts = line.split(':')
                if len(parts) > 1:
                    metrics['cache_files_count'] = int(parts[1].strip())
            elif 'Total cache size:' in line:
                parts = line.split(':')
                if len(parts) > 1:
                    metrics['cache_size'] = parts[1].strip()

        return metrics

    def _generate_final_report(self) -> Dict[str, Any]:
        """Gera relatório final consolidado"""
        total_duration = time.time() - self.start_time

        # Contar resultados
        passed = sum(1 for r in self.results.values() if r['status'] == 'PASSED')
        failed = sum(1 for r in self.results.values() if r['status'] == 'FAILED')
        skipped = sum(1 for r in self.results.values() if r['status'] == 'SKIPPED')
        total = len(self.results)

        # Calcular métricas gerais
        overall_status = 'PASSED' if failed == 0 else 'FAILED'
        success_rate = (passed / total) * 100 if total > 0 else 0

        report = {
            'overall_status': overall_status,
            'test_summary': {
                'total_tests': total,
                'passed': passed,
                'failed': failed,
                'skipped': skipped,
                'success_rate': success_rate
            },
            'total_duration_seconds': total_duration,
            'timestamp': datetime.now().isoformat(),
            'detailed_results': self.results
        }

        # Salvar relatório
        self._save_report(report)

        return report

    def _save_report(self, report: Dict[str, Any]):
        """Salva relatório em arquivo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Salvar JSON completo
        json_path = self.output_dir / f"ΨQRH_test_report_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Salvar resumo em texto
        summary_path = self.output_dir / f"ΨQRH_test_summary_{timestamp}.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_summary_text(report))

        print(f"\n📊 RELATÓRIOS SALVOS EM: {self.output_dir}")
        print(f"   📄 JSON: {json_path.name}")
        print(f"   📝 Resumo: {summary_path.name}")

    def _generate_summary_text(self, report: Dict[str, Any]) -> str:
        """Gera resumo em texto do relatório"""
        summary = []
        summary.append("🔮 RELATÓRIO FINAL - TESTES COMPLETOS ΨQRH")
        summary.append("=" * 60)
        summary.append(f"📅 Data/Hora: {report['timestamp']}")
        summary.append(f"📊 Status Geral: {report['overall_status']}")
        summary.append(f"⏱️  Duração Total: {report['total_duration_seconds']:.2f}s")
        summary.append("")

        # Resumo dos testes
        stats = report['test_summary']
        summary.append("📈 RESUMO DOS TESTES:")
        summary.append(f"   ✅ Passados: {stats['passed']}/{stats['total_tests']}")
        summary.append(f"   ❌ Falhados: {stats['failed']}/{stats['total_tests']}")
        summary.append(f"   ⚠️  Pulados: {stats['skipped']}/{stats['total_tests']}")
        summary.append(f"   📈 Taxa de Sucesso: {stats['success_rate']:.1f}%")
        summary.append("")

        # Detalhes por teste
        summary.append("🔍 DETALHES POR TESTE:")
        for test_name, result in report['detailed_results'].items():
            status_icon = "✅" if result['status'] == 'PASSED' else "❌" if result['status'] == 'FAILED' else "⚠️ "
            duration = result.get('duration', 0)
            summary.append(f"   {status_icon} {test_name}: {result['status']} ({duration:.2f}s)")

        # Conclusão
        summary.append("")
        summary.append("🎯 CONCLUSÃO:")
        if report['overall_status'] == 'PASSED':
            summary.append("   ✅ Sistema ΨQRH validado com sucesso!")
            summary.append("   🔒 Todos os componentes operacionais")
            summary.append("   🚀 Pronto para deployment")
        else:
            summary.append("   ⚠️  Sistema requer ajustes")
            summary.append("   🔧 Verificar testes falhados")

        return '\n'.join(summary)


def main():
    """Função principal"""
    engine = ΨQRHComprehensiveTestEngine()

    try:
        report = engine.run_all_tests()

        # Exibir resumo
        print("\n" + "=" * 60)
        print("🎯 RELATÓRIO FINAL CONSOLIDADO")
        print("=" * 60)
        print(f"Status Geral: {report['overall_status']}")
        print(f"Duração Total: {report['total_duration_seconds']:.2f}s")
        print(f"Taxa de Sucesso: {report['test_summary']['success_rate']:.1f}%")

        # Retornar código de saída
        sys.exit(0 if report['overall_status'] == 'PASSED' else 1)

    except Exception as e:
        print(f"💥 ERRO CRÍTICO: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
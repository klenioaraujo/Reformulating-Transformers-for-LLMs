"""
Gerador de Relatórios de Teste para ΨQRH - Implementação Real

Gera relatórios abrangentes baseados em dados reais de testes:
- Relatórios detalhados com dados brutos
- Análises estatísticas rigorosas
- Visualizações dinâmicas
"""

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class TestReport:
    """Estrutura de relatório de teste"""
    report_id: str
    report_type: str
    timestamp: str
    overall_score: float
    test_results: Dict[str, Any]
    statistical_analysis: Dict[str, Any]
    recommendations: List[str]
    raw_data_references: List[str]

class TestReporter:
    """Gerador de relatórios de teste"""

    def __init__(self, output_dir: str = "tmp"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Subdiretórios
        self.reports_dir = self.output_dir / "reports"
        self.data_dir = self.output_dir / "raw_data"
        self.plots_dir = self.output_dir / "plots"

        for dir_path in [self.reports_dir, self.data_dir, self.plots_dir]:
            dir_path.mkdir(exist_ok=True)

    def generate_comprehensive_test_report(self, mathematical_results: Dict[str, Any],
                                         spectral_results: Dict[str, Any]) -> str:
        """Gera relatório completo de testes"""
        logger.info("Gerando relatório completo de testes...")

        start_time = time.time()

        # Consolidar resultados
        consolidated_results = self._consolidate_results(mathematical_results, spectral_results)

        # Análise estatística
        statistical_analysis = self._perform_statistical_analysis(consolidated_results)

        # Gerar recomendações
        recommendations = self._generate_recommendations(consolidated_results, statistical_analysis)

        # Criar relatório
        report = TestReport(
            report_id=f"ΨQRH_TEST_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type="COMPREHENSIVE_TEST_ANALYSIS",
            timestamp=datetime.now().isoformat(),
            overall_score=statistical_analysis['overall_score'],
            test_results=consolidated_results,
            statistical_analysis=statistical_analysis,
            recommendations=recommendations,
            raw_data_references=self._save_raw_data(mathematical_results, spectral_results)
        )

        # Salvar relatório
        report_path = self._save_report(report)

        # Gerar visualizações
        self._generate_visualizations(consolidated_results, statistical_analysis)

        # Gerar sumário executivo
        self._generate_executive_summary(report)

        logger.info(f"Relatório completo gerado: {report_path}")
        return report_path

    def _consolidate_results(self, math_results: Dict[str, Any], spectral_results: Dict[str, Any]) -> Dict[str, Any]:
        """Consolida resultados de diferentes análises"""
        consolidated = {
            'validation_type': 'DYNAMIC_COMPREHENSIVE',
            'timestamp': datetime.now().isoformat(),
            'mathematical_validation': {},
            'spectral_analysis': {},
            'composite_metrics': {}
        }

        # Resultados matemáticos
        if 'detailed_results' in math_results:
            math_scores = []
            for test_id, result in math_results['detailed_results'].items():
                consolidated['mathematical_validation'][test_id] = {
                    'test_name': result['test_name'],
                    'status': result['status'],
                    'score': result['score'],
                    'execution_time': result.get('execution_time', 0.0)
                }
                if result['status'] == 'passed':
                    math_scores.append(result['score'])

            if math_scores:
                consolidated['composite_metrics']['mathematical_score'] = np.mean(math_scores)

        # Resultados espectrais
        if 'overall_metrics' in spectral_results:
            consolidated['spectral_analysis'] = spectral_results['overall_metrics']
            consolidated['composite_metrics']['spectral_score'] = spectral_results['overall_metrics'].get('composite_score', 0.0)

        # Score geral
        math_score = consolidated['composite_metrics'].get('mathematical_score', 0.0)
        spectral_score = consolidated['composite_metrics'].get('spectral_score', 0.0)

        if math_score > 0 and spectral_score > 0:
            consolidated['composite_metrics']['overall_score'] = (math_score + spectral_score) / 2
        else:
            consolidated['composite_metrics']['overall_score'] = max(math_score, spectral_score)

        return consolidated

    def _perform_statistical_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza análise estatística dos resultados"""
        analysis = {
            'basic_statistics': {},
            'correlation_analysis': {},
            'performance_metrics': {},
            'quality_assessment': {}
        }

        try:
            # Coletar todos os scores
            all_scores = []

            # Scores matemáticos
            math_scores = []
            for test_id, test_data in results['mathematical_validation'].items():
                if test_data['status'] == 'passed':
                    math_scores.append(test_data['score'])
                    all_scores.append(test_data['score'])

            # Scores espectrais
            spectral_metrics = results['spectral_analysis']
            for metric_name, metric_value in spectral_metrics.items():
                if isinstance(metric_value, (int, float)) and 0 <= metric_value <= 1:
                    all_scores.append(metric_value)

            # Estatísticas básicas
            if all_scores:
                analysis['basic_statistics'] = {
                    'mean_score': float(np.mean(all_scores)),
                    'std_score': float(np.std(all_scores)),
                    'min_score': float(min(all_scores)),
                    'max_score': float(max(all_scores)),
                    'num_metrics': len(all_scores)
                }

                # Análise de performance
                analysis['performance_metrics'] = {
                    'excellent_metrics': len([s for s in all_scores if s >= 0.95]),
                    'good_metrics': len([s for s in all_scores if 0.85 <= s < 0.95]),
                    'acceptable_metrics': len([s for s in all_scores if 0.75 <= s < 0.85]),
                    'poor_metrics': len([s for s in all_scores if s < 0.75])
                }

                # Avaliação de qualidade
                excellent_pct = analysis['performance_metrics']['excellent_metrics'] / len(all_scores)
                good_pct = analysis['performance_metrics']['good_metrics'] / len(all_scores)

                if excellent_pct >= 0.8:
                    quality = 'EXCELLENT'
                elif excellent_pct + good_pct >= 0.8:
                    quality = 'GOOD'
                elif excellent_pct + good_pct >= 0.6:
                    quality = 'ACCEPTABLE'
                else:
                    quality = 'NEEDS_IMPROVEMENT'

                analysis['quality_assessment'] = {
                    'overall_quality': quality,
                    'excellent_percentage': excellent_pct,
                    'good_percentage': good_pct
                }

            analysis['overall_score'] = results['composite_metrics'].get('overall_score', 0.0)

        except Exception as e:
            logger.error(f"Erro na análise estatística: {e}")
            analysis['error'] = str(e)

        return analysis

    def _generate_recommendations(self, results: Dict[str, Any], stats: Dict[str, Any]) -> List[str]:
        """Gera recomendações baseadas nos resultados"""
        recommendations = []

        # Análise de qualidade geral
        quality = stats.get('quality_assessment', {}).get('overall_quality', 'UNKNOWN')

        if quality == 'EXCELLENT':
            recommendations.append("✅ O framework ΨQRH demonstra excelente estabilidade matemática e espectral")
            recommendations.append("✅ Pronto para implantação em produção e benchmarks públicos")
        elif quality == 'GOOD':
            recommendations.append("⚠️  O framework ΨQRH apresenta boa estabilidade, com pequenas áreas para otimização")
            recommendations.append("✅ Adequado para a maioria das aplicações, considere otimizações específicas")
        elif quality == 'ACCEPTABLE':
            recommendations.append("⚠️  ΨQRH requer melhorias em algumas áreas antes da implantação em produção")
            recommendations.append("🔧 Foque na otimização das métricas com scores mais baixos")
        else:
            recommendations.append("❌ ΨQRH necessita de melhorias significativas antes de qualquer implantação")
            recommendations.append("🔧 Realize uma revisão completa das implementações matemáticas")

        # Recomendações específicas baseadas nos resultados
        math_results = results['mathematical_validation']
        for test_id, test_data in math_results.items():
            if test_data['score'] < 0.85:
                recommendations.append(f"🔧 Otimizar {test_data['test_name']} (score: {test_data['score']:.3f})")

        spectral_metrics = results['spectral_analysis']
        low_spectral_metrics = [k for k, v in spectral_metrics.items()
                              if isinstance(v, (int, float)) and v < 0.8]
        if low_spectral_metrics:
            recommendations.append(f"🔧 Melhorar métricas espectrais: {', '.join(low_spectral_metrics)}")

        # Recomendações gerais
        recommendations.append("📊 Continuar monitoramento das propriedades matemáticas durante o desenvolvimento")
        recommendations.append("🔬 Realizar validação adicional com datasets do mundo real")
        recommendations.append("⚡ Considerar otimizações de performance para aplicações em tempo real")

        return recommendations

    def _save_raw_data(self, math_results: Dict[str, Any], spectral_results: Dict[str, Any]) -> List[str]:
        """Salva dados brutos para referência"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []

        try:
            # Salvar resultados matemáticos
            math_file = self.data_dir / f"mathematical_results_{timestamp}.json"
            with open(math_file, 'w') as f:
                json.dump(math_results, f, indent=2)
            saved_files.append(str(math_file))

            # Salvar resultados espectrais
            spectral_file = self.data_dir / f"spectral_results_{timestamp}.json"
            with open(spectral_file, 'w') as f:
                json.dump(spectral_results, f, indent=2)
            saved_files.append(str(spectral_file))

            # Salvar dados consolidados em CSV para análise
            if 'detailed_results' in math_results:
                csv_data = []
                for test_id, test_data in math_results['detailed_results'].items():
                    csv_data.append({
                        'test_id': test_id,
                        'test_name': test_data['test_name'],
                        'status': test_data['status'],
                        'score': test_data['score'],
                        'timestamp': test_data.get('timestamp', '')
                    })

                df = pd.DataFrame(csv_data)
                csv_file = self.data_dir / f"test_results_{timestamp}.csv"
                df.to_csv(csv_file, index=False)
                saved_files.append(str(csv_file))

        except Exception as e:
            logger.error(f"Erro ao salvar dados brutos: {e}")

        return saved_files

    def _save_report(self, report: TestReport) -> str:
        """Salva relatório completo"""
        report_path = self.reports_dir / f"{report.report_id}.json"

        # Converter para dict
        report_dict = asdict(report)

        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=self._json_serializer)

        # Também salvar como YAML para legibilidade
        yaml_path = self.reports_dir / f"{report.report_id}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(report_dict, f, default_flow_style=False)

        return str(report_path)

    def _generate_visualizations(self, results: Dict[str, Any], stats: Dict[str, Any]):
        """Gera visualizações dos resultados"""
        try:
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")

            # Gráfico 1: Scores dos testes matemáticos
            math_scores = []
            test_names = []

            for test_id, test_data in results['mathematical_validation'].items():
                math_scores.append(test_data['score'])
                test_names.append(test_data['test_name'])

            if math_scores:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Barras dos scores
                bars = ax1.bar(range(len(math_scores)), math_scores, color='skyblue', alpha=0.7)
                ax1.set_xlabel('Testes')
                ax1.set_ylabel('Score')
                ax1.set_title('Scores dos Testes Matemáticos')
                ax1.set_xticks(range(len(math_scores)))
                ax1.set_xticklabels([f"Teste {i+1}" for i in range(len(math_scores))], rotation=45)
                ax1.grid(True, alpha=0.3)

                # Adicionar valores nas barras
                for bar, score in zip(bars, math_scores):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom')

                # Gráfico 2: Distribuição de performance
                performance = stats['performance_metrics']
                categories = ['Excelente', 'Bom', 'Aceitável', 'Precisa Melhoria']
                values = [
                    performance['excellent_metrics'],
                    performance['good_metrics'],
                    performance['acceptable_metrics'],
                    performance['poor_metrics']
                ]

                ax2.pie(values, labels=categories, autopct='%1.1f%%', startangle=90)
                ax2.set_title('Distribuição de Performance')

                plt.tight_layout()
                plot_path = self.plots_dir / "test_results_visualization.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()

                logger.info(f"Visualizações geradas: {plot_path}")

        except Exception as e:
            logger.error(f"Erro na geração de visualizações: {e}")

    def _generate_executive_summary(self, report: TestReport):
        """Gera sumário executivo em markdown"""
        summary_path = self.reports_dir / f"{report.report_id}_summary.md"

        with open(summary_path, 'w') as f:
            f.write("# Sumário Executivo - Testes ΨQRH\n\n")
            f.write(f"**Relatório**: {report.report_id}\n")
            f.write(f"**Data**: {report.timestamp}\n\n")

            f.write("## Resultados Principais\n\n")
            f.write(f"- **Score Geral**: {report.overall_score:.3f}\\n")
            f.write(f"- **Qualidade**: {report.statistical_analysis.get('quality_assessment', {}).get('overall_quality', 'N/A')}\\n")
            f.write(f"- **Testes Realizados**: {len(report.test_results.get('mathematical_validation', {}))}\\n")
            f.write(f"- **Análises Espectrais**: {len(report.test_results.get('spectral_analysis', {}))}\n\n")

            f.write("## Recomendações\n\n")
            for rec in report.recommendations:
                f.write(f"- {rec}\\n")

            f.write("\n## Próximos Passos\n\n")
            f.write("1. Implementar as recomendações específicas\n")
            f.write("2. Realizar validação adicional com dados reais\n")
            f.write("3. Preparar para benchmarks públicos\n")
            f.write("4. Documentar resultados para publicação\n")

    def _json_serializer(self, obj):
        """Serializador customizado para JSON"""
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)

    def generate_test_summary(self, report_path: str) -> str:
        """Gera sumário de teste baseado em relatório existente"""
        try:
            with open(report_path, 'r') as f:
                report_data = json.load(f)

            summary_path = Path(report_path).with_suffix('.summary.md')

            with open(summary_path, 'w') as f:
                f.write("# ΨQRH Test Summary\n\n")
                f.write("## Quick Assessment\n\n")

                overall_score = report_data['overall_score']
                if overall_score >= 0.95:
                    f.write("✅ **EXCELLENT** - Ready for production and public benchmarking\n")
                elif overall_score >= 0.85:
                    f.write("⚠️  **GOOD** - Suitable for most applications, minor optimizations needed\n")
                elif overall_score >= 0.75:
                    f.write("🔧 **ACCEPTABLE** - Requires improvements before production deployment\n")
                else:
                    f.write("❌ **NEEDS WORK** - Significant improvements required\n")

                f.write(f"\n**Overall Score**: {overall_score:.3f}\n")
                f.write(f"**Test Date**: {report_data['timestamp']}\n")

            return str(summary_path)

        except Exception as e:
            logger.error(f"Erro ao gerar sumário: {e}")
            return ""

def run_comprehensive_testing():
    """Função para executar testes completos e gerar relatórios"""
    # Esta função seria chamada após a execução dos testes
    # Por enquanto é um placeholder para demonstração

    logger.info("Sistema de relatórios de teste ΨQRH implementado")
    logger.info("Execute os testes primeiro, depois chame generate_comprehensive_test_report()")

if __name__ == "__main__":
    run_comprehensive_testing()
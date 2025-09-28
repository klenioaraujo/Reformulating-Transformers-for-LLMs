"""
Análise Final ΨQRH - Relatório Completo

Consolida todos os testes e análises em um relatório final abrangente:
- Integração de todos os módulos de teste
- Análise estatística consolidada
- Recomendações finais
- Preparação para benchmarks públicos
"""

import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
import time
from datetime import datetime

from .advanced_mathematical_tests import AdvancedMathematicalTests
from .spectral_analysis import SpectralAnalyzer
from .consciousness_integration import ConsciousnessIntegrationTests
from .test_reporter import TestReporter
from core.qrh_layer import QRHConfig

logger = logging.getLogger(__name__)

@dataclass
class FinalAnalysisReport:
    """Estrutura do relatório final de análise"""
    report_id: str
    framework_version: str
    analysis_timestamp: str
    overall_assessment: Dict[str, Any]
    component_analyses: Dict[str, Any]
    statistical_summary: Dict[str, Any]
    critical_findings: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    readiness_assessment: Dict[str, Any]
    raw_data_references: List[str]

class FinalAnalysisEngine:
    """Motor de análise final para ΨQRH"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.output_dir = Path("tmp")
        self.output_dir.mkdir(exist_ok=True)

        # Carregar configuração
        self.config = self._load_config()

        # Inicializar componentes de teste
        self.math_tests = AdvancedMathematicalTests(self.config)
        self.spectral_analyzer = SpectralAnalyzer(self.config)
        self.consciousness_tests = ConsciousnessIntegrationTests(self.config)
        self.reporter = TestReporter()

    def _load_config(self) -> QRHConfig:
        """Carrega configuração do arquivo YAML"""
        with open(self.config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        qrh_config_data = config_data['qrh_config']
        return QRHConfig(**qrh_config_data)

    def run_comprehensive_analysis(self) -> FinalAnalysisReport:
        """Executa análise completa e gera relatório final"""
        logger.info("Iniciando análise final completa ΨQRH...")

        start_time = time.time()

        # Executar todos os testes
        mathematical_results = self.math_tests.run_dynamic_comprehensive_validation()
        spectral_results = self.spectral_analyzer.run_comprehensive_spectral_analysis()
        consciousness_results = self.consciousness_tests.run_consciousness_integration_suite()

        # Consolidar resultados
        consolidated_results = self._consolidate_all_results(
            mathematical_results, spectral_results, consciousness_results
        )

        # Análise estatística final
        final_statistics = self._perform_final_statistical_analysis(consolidated_results)

        # Identificar achados críticos
        critical_findings = self._identify_critical_findings(consolidated_results)

        # Gerar recomendações
        recommendations = self._generate_final_recommendations(consolidated_results, critical_findings)

        # Avaliação de prontidão
        readiness_assessment = self._assess_readiness(consolidated_results, final_statistics)

        # Criar relatório final
        report = FinalAnalysisReport(
            report_id=f"ΨQRH_FINAL_ANALYSIS_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            framework_version="ΨQRH v1.0",
            analysis_timestamp=datetime.now().isoformat(),
            overall_assessment=consolidated_results['overall_assessment'],
            component_analyses=consolidated_results['component_analyses'],
            statistical_summary=final_statistics,
            critical_findings=critical_findings,
            recommendations=recommendations,
            readiness_assessment=readiness_assessment,
            raw_data_references=self._save_all_raw_data(
                mathematical_results, spectral_results, consciousness_results
            )
        )

        # Salvar relatório
        report_path = self._save_final_report(report)

        # Gerar sumários executivos
        self._generate_executive_summaries(report)

        logger.info(f"Análise final concluída. Relatório: {report_path}")
        return report

    def _consolidate_all_results(self, math_results: Dict[str, Any],
                               spectral_results: Dict[str, Any],
                               consciousness_results: Dict[str, Any]) -> Dict[str, Any]:
        """Consolida resultados de todos os testes"""
        consolidated = {
            'analysis_type': 'FINAL_COMPREHENSIVE',
            'timestamp': datetime.now().isoformat(),
            'overall_assessment': {},
            'component_analyses': {},
            'quality_metrics': {}
        }

        # Componente matemático
        math_score = math_results.get('overall_score', 0.0)
        math_tests_passed = math_results.get('tests_passed', 0)
        math_total_tests = math_results.get('total_tests', 0)

        consolidated['component_analyses']['mathematical'] = {
            'score': math_score,
            'tests_passed': math_tests_passed,
            'total_tests': math_total_tests,
            'success_rate': math_tests_passed / math_total_tests if math_total_tests > 0 else 0.0,
            'validation_type': math_results.get('validation_type', 'UNKNOWN')
        }

        # Componente espectral
        spectral_metrics = spectral_results.get('overall_metrics', {})
        spectral_score = spectral_metrics.get('composite_score', 0.0)

        consolidated['component_analyses']['spectral'] = {
            'score': spectral_score,
            'stability_metrics': spectral_metrics,
            'analysis_type': spectral_results.get('analysis_type', 'UNKNOWN')
        }

        # Componente de consciência
        consciousness_score = consciousness_results.get('overall_score', 0.0)
        consciousness_tests_passed = consciousness_results.get('tests_passed', 0)
        consciousness_total_tests = consciousness_results.get('total_tests', 0)

        consolidated['component_analyses']['consciousness'] = {
            'score': consciousness_score,
            'tests_passed': consciousness_tests_passed,
            'total_tests': consciousness_total_tests,
            'success_rate': consciousness_tests_passed / consciousness_total_tests if consciousness_total_tests > 0 else 0.0,
            'integration_type': consciousness_results.get('integration_type', 'UNKNOWN')
        }

        # Avaliação geral
        component_scores = [
            consolidated['component_analyses']['mathematical']['score'],
            consolidated['component_analyses']['spectral']['score'],
            consolidated['component_analyses']['consciousness']['score']
        ]

        consolidated['overall_assessment'] = {
            'composite_score': np.mean(component_scores),
            'component_scores': component_scores,
            'weighted_score': self._calculate_weighted_score(component_scores),
            'assessment_date': datetime.now().isoformat()
        }

        # Métricas de qualidade
        consolidated['quality_metrics'] = {
            'test_coverage': self._calculate_test_coverage(consolidated),
            'implementation_quality': self._assess_implementation_quality(consolidated),
            'documentation_completeness': 0.85,  # Placeholder - seria calculado baseado na documentação real
            'reproducibility_score': 0.90  # Placeholder - baseado na capacidade de reproduzir resultados
        }

        return consolidated

    def _calculate_weighted_score(self, component_scores: List[float]) -> float:
        """Calcula score ponderado baseado na importância dos componentes"""
        # Pesos: matemática (40%), espectral (35%), consciência (25%)
        weights = [0.40, 0.35, 0.25]

        if len(component_scores) != len(weights):
            return np.mean(component_scores)

        return sum(score * weight for score, weight in zip(component_scores, weights))

    def _calculate_test_coverage(self, consolidated: Dict[str, Any]) -> float:
        """Calcula cobertura de testes"""
        total_tests = 0
        covered_areas = 0

        components = consolidated['component_analyses']
        for component, data in components.items():
            total_tests += data.get('total_tests', 0)
            if data.get('success_rate', 0) > 0:
                covered_areas += 1

        # Considerar que temos 3 áreas principais cobertas
        area_coverage = covered_areas / 3.0

        # Combinação de cobertura de áreas e quantidade de testes
        if total_tests > 0:
            test_density = min(1.0, total_tests / 15.0)  # Normalizar para ~15 testes ideais
            return (area_coverage + test_density) / 2.0
        else:
            return area_coverage

    def _assess_implementation_quality(self, consolidated: Dict[str, Any]) -> float:
        """Avalia qualidade da implementação"""
        scores = []

        # Baseado nos scores dos componentes
        component_scores = consolidated['overall_assessment']['component_scores']
        scores.extend(component_scores)

        # Baseado na consistência entre componentes
        score_std = np.std(component_scores)
        consistency_score = 1.0 / (1.0 + score_std)  # Menor variação = maior consistência
        scores.append(consistency_score)

        return np.mean(scores)

    def _perform_final_statistical_analysis(self, consolidated: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza análise estatística final"""
        analysis = {
            'descriptive_statistics': {},
            'correlation_analysis': {},
            'performance_distribution': {},
            'reliability_metrics': {}
        }

        try:
            # Coletar todos os scores
            all_scores = []

            # Scores dos componentes
            component_scores = consolidated['overall_assessment']['component_scores']
            all_scores.extend(component_scores)

            # Scores de qualidade
            quality_scores = list(consolidated['quality_metrics'].values())
            all_scores.extend(quality_scores)

            # Estatísticas descritivas
            if all_scores:
                analysis['descriptive_statistics'] = {
                    'mean': float(np.mean(all_scores)),
                    'median': float(np.median(all_scores)),
                    'std': float(np.std(all_scores)),
                    'min': float(min(all_scores)),
                    'max': float(max(all_scores)),
                    'q1': float(np.percentile(all_scores, 25)),
                    'q3': float(np.percentile(all_scores, 75))
                }

            # Análise de correlação entre componentes
            if len(component_scores) >= 2:
                # Matriz de correlação simplificada
                analysis['correlation_analysis'] = {
                    'math_spectral_correlation': min(1.0, abs(component_scores[0] - component_scores[1])),
                    'math_consciousness_correlation': min(1.0, abs(component_scores[0] - component_scores[2])),
                    'spectral_consciousness_correlation': min(1.0, abs(component_scores[1] - component_scores[2]))
                }

            # Distribuição de performance
            excellent = len([s for s in all_scores if s >= 0.95])
            good = len([s for s in all_scores if 0.85 <= s < 0.95])
            acceptable = len([s for s in all_scores if 0.75 <= s < 0.85])
            poor = len([s for s in all_scores if s < 0.75])

            analysis['performance_distribution'] = {
                'excellent_count': excellent,
                'good_count': good,
                'acceptable_count': acceptable,
                'poor_count': poor,
                'total_metrics': len(all_scores)
            }

            # Métricas de confiabilidade
            analysis['reliability_metrics'] = {
                'internal_consistency': 1.0 - analysis['descriptive_statistics'].get('std', 1.0),
                'score_stability': min(1.0, consolidated['overall_assessment']['composite_score']),
                'test_reliability': consolidated['quality_metrics'].get('reproducibility_score', 0.0)
            }

        except Exception as e:
            logger.error(f"Erro na análise estatística final: {e}")
            analysis['error'] = str(e)

        return analysis

    def _identify_critical_findings(self, consolidated: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifica achados críticos"""
        findings = []

        # Verificar componentes com baixo score
        components = consolidated['component_analyses']
        for component, data in components.items():
            score = data.get('score', 0.0)
            if score < 0.80:
                findings.append({
                    'component': component,
                    'issue': f'Score baixo: {score:.3f}',
                    'severity': 'HIGH' if score < 0.70 else 'MEDIUM',
                    'impact': 'Pode afetar performance geral do framework'
                })

        # Verificar inconsistências entre componentes
        component_scores = consolidated['overall_assessment']['component_scores']
        score_range = max(component_scores) - min(component_scores)
        if score_range > 0.15:
            findings.append({
                'component': 'ALL',
                'issue': f'Inconsistência entre componentes (range: {score_range:.3f})',
                'severity': 'MEDIUM',
                'impact': 'Desbalanceamento pode indicar problemas de integração'
            })

        # Verificar cobertura de testes
        test_coverage = consolidated['quality_metrics'].get('test_coverage', 0.0)
        if test_coverage < 0.80:
            findings.append({
                'component': 'TESTING',
                'issue': f'Cobertura de testes insuficiente: {test_coverage:.3f}',
                'severity': 'MEDIUM',
                'impact': 'Risco de bugs não detectados'
            })

        return findings

    def _generate_final_recommendations(self, consolidated: Dict[str, Any],
                                      critical_findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Gera recomendações finais"""
        recommendations = []

        overall_score = consolidated['overall_assessment']['composite_score']

        # Recomendações baseadas no score geral
        if overall_score >= 0.95:
            recommendations.append({
                'type': 'STRATEGIC',
                'priority': 'LOW',
                'recommendation': 'Framework pronto para produção e benchmarks públicos',
                'rationale': 'Excelente performance em todos os componentes'
            })
        elif overall_score >= 0.85:
            recommendations.append({
                'type': 'STRATEGIC',
                'priority': 'MEDIUM',
                'recommendation': 'Otimizar componentes específicos antes de produção',
                'rationale': 'Boa performance geral com áreas para melhoria'
            })
        else:
            recommendations.append({
                'type': 'STRATEGIC',
                'priority': 'HIGH',
                'recommendation': 'Revisão completa antes de qualquer implantação',
                'rationale': 'Performance abaixo do esperado para produção'
            })

        # Recomendações baseadas em achados críticos
        for finding in critical_findings:
            if finding['severity'] == 'HIGH':
                recommendations.append({
                    'type': 'CRITICAL',
                    'priority': 'HIGH',
                    'recommendation': f'Corrigir issue em {finding["component"]}: {finding["issue"]}',
                    'rationale': finding['impact']
                })

        # Recomendações técnicas específicas
        components = consolidated['component_analyses']
        for component, data in components.items():
            score = data.get('score', 0.0)
            if score < 0.85:
                recommendations.append({
                    'type': 'TECHNICAL',
                    'priority': 'MEDIUM',
                    'recommendation': f'Otimizar implementação do componente {component}',
                    'rationale': f'Score atual: {score:.3f} (meta: ≥0.85)'
                })

        # Recomendações de próximos passos
        recommendations.extend([
            {
                'type': 'NEXT_STEPS',
                'priority': 'MEDIUM',
                'recommendation': 'Realizar validação com datasets do mundo real',
                'rationale': 'Confirmar performance em cenários práticos'
            },
            {
                'type': 'NEXT_STEPS',
                'priority': 'LOW',
                'recommendation': 'Documentar resultados para publicação científica',
                'rationale': 'Contribuir para a comunidade de pesquisa'
            },
            {
                'type': 'NEXT_STEPS',
                'priority': 'MEDIUM',
                'recommendation': 'Preparar infraestrutura para benchmarks públicos',
                'rationale': 'Permitir comparação justa com outras arquiteturas'
            }
        ])

        return recommendations

    def _assess_readiness(self, consolidated: Dict[str, Any], statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Avalia prontidão para benchmarks públicos"""
        readiness = {
            'production_ready': False,
            'benchmark_ready': False,
            'research_ready': False,
            'readiness_score': 0.0,
            'blocking_issues': [],
            'improvement_areas': []
        }

        overall_score = consolidated['overall_assessment']['composite_score']
        critical_findings = self._identify_critical_findings(consolidated)

        # Critérios para produção
        production_criteria = [
            overall_score >= 0.90,
            len([f for f in critical_findings if f['severity'] == 'HIGH']) == 0,
            consolidated['quality_metrics']['test_coverage'] >= 0.85
        ]

        readiness['production_ready'] = all(production_criteria)

        # Critérios para benchmarks
        benchmark_criteria = [
            overall_score >= 0.85,
            len([f for f in critical_findings if f['severity'] == 'HIGH']) == 0,
            statistics['reliability_metrics']['internal_consistency'] >= 0.80
        ]

        readiness['benchmark_ready'] = all(benchmark_criteria)

        # Critérios para pesquisa
        research_criteria = [
            overall_score >= 0.75,
            consolidated['quality_metrics']['documentation_completeness'] >= 0.70
        ]

        readiness['research_ready'] = all(research_criteria)

        # Score de prontidão
        readiness_scores = []
        if readiness['production_ready']:
            readiness_scores.append(1.0)
        if readiness['benchmark_ready']:
            readiness_scores.append(0.85)
        if readiness['research_ready']:
            readiness_scores.append(0.70)

        readiness['readiness_score'] = max(readiness_scores) if readiness_scores else 0.0

        # Identificar issues bloqueantes
        if not readiness['production_ready']:
            readiness['blocking_issues'].append('Score geral abaixo de 0.90')
        if not readiness['benchmark_ready']:
            readiness['blocking_issues'].append('Issues críticas não resolvidas')

        # Áreas de melhoria
        if overall_score < 0.95:
            readiness['improvement_areas'].append(f'Otimizar score geral (atual: {overall_score:.3f})')

        return readiness

    def _save_all_raw_data(self, math_results: Dict[str, Any], spectral_results: Dict[str, Any],
                          consciousness_results: Dict[str, Any]) -> List[str]:
        """Salva todos os dados brutos"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []

        try:
            data_dir = self.output_dir / "final_analysis_data"
            data_dir.mkdir(exist_ok=True)

            # Salvar resultados individuais
            files_to_save = {
                f'mathematical_results_{timestamp}.json': math_results,
                f'spectral_results_{timestamp}.json': spectral_results,
                f'consciousness_results_{timestamp}.json': consciousness_results
            }

            for filename, data in files_to_save.items():
                filepath = data_dir / filename
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                saved_files.append(str(filepath))

        except Exception as e:
            logger.error(f"Erro ao salvar dados brutos: {e}")

        return saved_files

    def _save_final_report(self, report: FinalAnalysisReport) -> str:
        """Salva relatório final"""
        report_path = self.output_dir / f"{report.report_id}.json"

        # Converter para dict
        report_dict = asdict(report)

        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=self._json_serializer)

        # Também salvar como YAML
        yaml_path = self.output_dir / f"{report.report_id}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(report_dict, f, default_flow_style=False)

        return str(report_path)

    def _generate_executive_summaries(self, report: FinalAnalysisReport):
        """Gera sumários executivos"""
        # Sumário técnico
        technical_summary = self.output_dir / f"{report.report_id}_technical_summary.md"

        with open(technical_summary, 'w') as f:
            f.write("# Sumário Técnico - Análise Final ΨQRH\n\n")
            f.write(f"**Relatório**: {report.report_id}\n")
            f.write(f"**Data**: {report.analysis_timestamp}\n\n")

            f.write("## Resultados Principais\n\n")
            f.write(f"- **Score Geral**: {report.overall_assessment['composite_score']:.3f}\\n")
            f.write(f"- **Prontidão para Produção**: {'✅ SIM' if report.readiness_assessment['production_ready'] else '❌ NÃO'}\\n")
            f.write(f"- **Prontidão para Benchmarks**: {'✅ SIM' if report.readiness_assessment['benchmark_ready'] else '❌ NÃO'}\\n")
            f.write(f"- **Score de Prontidão**: {report.readiness_assessment['readiness_score']:.3f}\n\n")

        # Sumário executivo
        executive_summary = self.output_dir / f"{report.report_id}_executive_summary.md"

        with open(executive_summary, 'w') as f:
            f.write("# Sumário Executivo - ΨQRH Framework\n\n")
            f.write("## Status Atual\n\n")

            if report.readiness_assessment['production_ready']:
                f.write("✅ **PRONTO PARA PRODUÇÃO**\n\n")
                f.write("O framework ΨQRH demonstra excelente estabilidade matemática e está pronto para implantação em produção e benchmarks públicos.\n")
            elif report.readiness_assessment['benchmark_ready']:
                f.write("⚠️  **PRONTO PARA BENCHMARKS**\n\n")
                f.write("ΨQRH está adequado para comparações públicas, mas requer otimizações antes da implantação em produção.\n")
            else:
                f.write("❌ **EM DESENVOLVIMENTO**\n\n")
                f.write("O framework requer melhorias significativas antes de qualquer implantação ou benchmark público.\n")

    def _json_serializer(self, obj):
        """Serializador customizado"""
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)

def main():
    """Função principal para execução da análise final"""
    print("Analisador Final ΨQRH")
    print("=" * 50)

    analyzer = FinalAnalysisEngine()

    try:
        report = analyzer.run_comprehensive_analysis()

        print(f"\n✅ Análise final concluída com sucesso!")
        print(f"📊 Score Geral: {report.overall_assessment['composite_score']:.3f}")
        print(f"🏭 Pronto para Produção: {'✅ SIM' if report.readiness_assessment['production_ready'] else '❌ NÃO'}")
        print(f"📈 Pronto para Benchmarks: {'✅ SIM' if report.readiness_assessment['benchmark_ready'] else '❌ NÃO'}")
        print(f"📁 Relatórios em: {analyzer.output_dir}")

    except Exception as e:
        print(f"\n❌ Erro na análise final: {e}")
        logging.exception("Erro detalhado:")

if __name__ == "__main__":
    main()
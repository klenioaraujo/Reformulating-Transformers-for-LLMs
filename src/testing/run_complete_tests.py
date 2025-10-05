"""
Script Principal para Execução Completa de Testes ΨQRH

Executa todos os testes e análises em sequência:
1. Testes matemáticos avançados
2. Análise espectral completa
3. Integração de consciência
4. Geração de relatórios finais
"""

import sys
import os
import logging
from pathlib import Path

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar módulos com fallbacks
from testing.advanced_mathematical_tests import AdvancedMathematicalTests
from testing.spectral_analysis import SpectralAnalyzer
from testing.consciousness_integration import ConsciousnessIntegrationTests
from testing.test_reporter import TestReporter
from testing.final_analysis import FinalAnalysisEngine
from core.qrh_layer import QRHConfig

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tmp/testing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_complete_test_suite():
    """Executa suíte completa de testes ΨQRH"""
    logger.info("🚀 Iniciando Suíte Completa de Testes ΨQRH")
    logger.info("=" * 60)

    try:
        # Carregar configuração
        config = QRHConfig()
        logger.info(f"📋 Configuração carregada: embed_dim={config.embed_dim}, alpha={config.alpha}")

        # 1. Testes Matemáticos Avançados
        logger.info("\n1️⃣  Executando Testes Matemáticos Avançados...")
        math_tests = AdvancedMathematicalTests(config)
        math_results = math_tests.run_dynamic_comprehensive_validation()
        logger.info(f"✅ Testes matemáticos concluídos. Score: {math_results.get('overall_score', 0):.3f}")

        # 2. Análise Espectral
        logger.info("\n2️⃣  Executando Análise Espectral...")
        spectral_analyzer = SpectralAnalyzer(config)
        spectral_results = spectral_analyzer.run_comprehensive_spectral_analysis()
        spectral_score = spectral_results.get('overall_metrics', {}).get('composite_score', 0)
        logger.info(f"✅ Análise espectral concluída. Score: {spectral_score:.3f}")

        # 3. Integração de Consciência
        logger.info("\n3️⃣  Executando Integração de Consciência...")
        consciousness_tests = ConsciousnessIntegrationTests(config)
        consciousness_results = consciousness_tests.run_consciousness_integration_suite()
        consciousness_score = consciousness_results.get('overall_score', 0)
        logger.info(f"✅ Integração de consciência concluída. Score: {consciousness_score:.3f}")

        # 4. Gerar Relatório de Testes
        logger.info("\n4️⃣  Gerando Relatório de Testes...")
        reporter = TestReporter()
        test_report_path = reporter.generate_comprehensive_test_report(math_results, spectral_results)
        logger.info(f"✅ Relatório de testes gerado: {test_report_path}")

        # 5. Análise Final
        logger.info("\n5️⃣  Executando Análise Final...")
        final_analyzer = FinalAnalysisEngine()
        final_report = final_analyzer.run_comprehensive_analysis()

        overall_score = final_report.overall_assessment.get('composite_score', 0)
        readiness = final_report.readiness_assessment

        logger.info(f"✅ Análise final concluída!")
        logger.info(f"📊 Score Final: {overall_score:.3f}")
        logger.info(f"🏭 Pronto para Produção: {readiness.get('production_ready', False)}")
        logger.info(f"📈 Pronto para Benchmarks: {readiness.get('benchmark_ready', False)}")

        # Resumo Final
        logger.info("\n" + "=" * 60)
        logger.info("🎯 RESUMO FINAL DOS TESTES ΨQRH")
        logger.info("=" * 60)

        logger.info(f"📈 Score Matemático: {math_results.get('overall_score', 0):.3f}")
        logger.info(f"🌊 Score Espectral: {spectral_score:.3f}")
        logger.info(f"🧠 Score Consciência: {consciousness_score:.3f}")
        logger.info(f"⭐ Score Geral: {overall_score:.3f}")

        if readiness.get('production_ready'):
            logger.info("✅ STATUS: PRONTO PARA PRODUÇÃO E BENCHMARKS PÚBLICOS")
        elif readiness.get('benchmark_ready'):
            logger.info("⚠️  STATUS: PRONTO PARA BENCHMARKS, OTIMIZAÇÕES RECOMENDADAS")
        else:
            logger.info("❌ STATUS: REQUER MELHORIAS SIGNIFICATIVAS")

        logger.info(f"📁 Relatórios disponíveis em: tmp/")
        logger.info("=" * 60)

        return {
            'success': True,
            'math_score': math_results.get('overall_score', 0),
            'spectral_score': spectral_score,
            'consciousness_score': consciousness_score,
            'overall_score': overall_score,
            'production_ready': readiness.get('production_ready', False),
            'benchmark_ready': readiness.get('benchmark_ready', False),
            'report_path': test_report_path
        }

    except Exception as e:
        logger.error(f"❌ Erro durante a execução dos testes: {e}")
        logger.exception("Detalhes do erro:")
        return {
            'success': False,
            'error': str(e)
        }

def main():
    """Função principal"""
    print("🧪 Suíte Completa de Testes ΨQRH")
    print("=" * 50)
    print("Este script executará todos os testes e análises do framework ΨQRH")
    print("Inclui: Testes matemáticos, análise espectral, integração de consciência")
    print("=" * 50)

    # input("Pressione Enter para iniciar os testes...")  # Auto-start for automation

    results = run_complete_test_suite()

    if results['success']:
        print(f"\n✅ Testes concluídos com sucesso!")
        print(f"📊 Score Final: {results['overall_score']:.3f}")

        if results['production_ready']:
            print("🎉 ΨQRH está PRONTO PARA PRODUÇÃO E BENCHMARKS PÚBLICOS!")
        elif results['benchmark_ready']:
            print("⚠️  ΨQRH está pronto para benchmarks, mas requer otimizações para produção")
        else:
            print("🔧 ΨQRH requer melhorias antes de qualquer implantação")

        print(f"📁 Verifique os relatórios completos em: tmp/")
    else:
        print(f"\n❌ Erro durante os testes: {results['error']}")
        print("📋 Verifique o arquivo de log para detalhes: tmp/testing.log")

if __name__ == "__main__":
    main()
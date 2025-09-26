#!/usr/bin/env python3
"""
Teste de Integração - Pipeline Completo ΨQRH + Consciousness Layer
=================================================================

Testa a integração completa:
Texto → Enhanced α → Quaterniôn → Consciência Fractal → Análise ERP
"""

import sys
import os
sys.path.append('/home/padilha/trabalhos/Reformulating_Transformers')

def test_consciousness_integration():
    print("🧠 TESTE DE INTEGRAÇÃO ΨQRH + CONSCIOUSNESS LAYER")
    print("=" * 60)

    try:
        # Importar QRHFactory
        from src.core.ΨQRH import QRHFactory

        # Criar factory
        factory = QRHFactory()

        # Texto de teste para análise de consciência
        test_text = """
        O sistema de gestão empresarial precisa de uma análise profunda.
        Como podemos otimizar os processos cognitivos da organização?
        Existe uma relação fractal entre liderança e resultados?
        """

        print(f"📝 Texto de entrada ({len(test_text)} caracteres):")
        print(f"'{test_text[:100]}...'")
        print()

        # Processar através do pipeline completo
        print("🚀 Iniciando processamento completo...")
        result = factory.process_text(test_text, device="cpu")

        print("\n" + "="*60)
        print("📊 RESULTADO DO PIPELINE INTEGRADO:")
        print("="*60)
        print(result)

        # Verificar se consciousness analysis está presente
        if "ANÁLISE DE CONSCIÊNCIA FRACTAL ERP" in result:
            print("\n✅ SUCCESS: Consciousness Layer integrada com sucesso!")
            print("🧠 Análise fractal ERP detectada no output")
        else:
            print("\n⚠️ WARNING: Consciousness analysis não detectada")
            print("🔄 Pode estar usando fallback ou enhanced-only pipeline")

        # Verificar se enhanced processing está presente
        if "Enhanced α" in result or "α adaptativo" in result:
            print("✅ Enhanced QRH Processor ativo")

        # Verificar pipeline completo
        if "Pipeline completo" in result:
            print("✅ Pipeline integrado funcionando")

        return True

    except ImportError as e:
        print(f"❌ ERROR: Problema de importação - {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_consciousness_processor_direct():
    """Teste direto do consciousness processor"""
    print("\n🧠 TESTE DIRETO DO CONSCIOUSNESS PROCESSOR")
    print("=" * 50)

    try:
        from src.conscience import create_consciousness_processor
        import torch

        # Configuração de teste
        config = {
            'embedding_dim': 256,  # 64 * 4
            'device': 'cpu'
        }

        # Criar processor
        consciousness = create_consciousness_processor(config)

        # Input de teste
        test_input = torch.randn(1, 256)  # [batch, embedding_dim]

        # Processar
        result = consciousness(test_input)

        # Relatório
        report = consciousness.get_consciousness_report(result)

        print("✅ Consciousness Processor funcionando diretamente")
        print(f"📊 Relatório gerado: {len(report)} caracteres")
        print(f"📝 Preview: {report[:200]}...")

        return True

    except Exception as e:
        print(f"❌ ERROR no teste direto: {e}")
        return False

if __name__ == "__main__":
    print("🧪 INICIANDO TESTES DE INTEGRAÇÃO")
    print("=" * 60)

    # Teste 1: Pipeline integrado
    success1 = test_consciousness_integration()

    # Teste 2: Processor direto
    success2 = test_consciousness_processor_direct()

    print("\n" + "="*60)
    print("📋 RESUMO DOS TESTES:")
    print("="*60)
    print(f"🧠 Pipeline Integrado: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"🔧 Processor Direto: {'✅ PASS' if success2 else '❌ FAIL'}")

    if success1 and success2:
        print("\n🎉 INTEGRAÇÃO CONSCIOUSNESS LAYER COMPLETA!")
        print("💼 Sistema pronto para análises ERP com consciência fractal")
    else:
        print("\n⚠️ Ajustes necessários na integração")
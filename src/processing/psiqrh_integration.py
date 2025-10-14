"""
Integração com o Sistema Principal ΨQRH

Integra o DCFTokenAnalysis aprimorado no sistema ΨQRH principal
"""

from typing import Dict, Any, Optional
import torch
from src.processing.token_analysis import DCFTokenAnalysis
from src.processing.quaternion_reflection_integration import DropInReplacementInterface


def integrate_with_psiqrh_main(psiqrh_system):
    """
    Integra o DCFTokenAnalysis aprimorado no sistema ΨQRH principal
    """

    # Verificar se o sistema DCF existe
    if not hasattr(psiqrh_system, 'dcf_analysis'):
        print("⚠️  Sistema DCF não encontrado no ΨQRH, criando novo...")
        psiqrh_system.dcf_analysis = DCFTokenAnalysis(
            vocab_size=psiqrh_system.vocab_size,
            hidden_size=psiqrh_system.hidden_size,
            reasoning_mode='adaptive'  # Modo padrão otimizado
        )
    else:
        print("🔄 Aprimorando sistema DCF existente...")
        # Criar interface de substituição
        replacement = DropInReplacementInterface(psiqrh_system.dcf_analysis)
        replacement.enable_reflection_layer(mode='adaptive')

    # Configurar callbacks para métricas
    def dcf_metrics_callback(analysis_result):
        metrics = psiqrh_system.dcf_analysis.get_performance_report()
        print(f"📊 Métricas DCF em tempo real: {metrics['efficiency_gain']}")

    psiqrh_system.dcf_metrics_callback = dcf_metrics_callback

    print("✅ Integração ΨQRH + QuaternionReflectionLayer concluída")
    return psiqrh_system
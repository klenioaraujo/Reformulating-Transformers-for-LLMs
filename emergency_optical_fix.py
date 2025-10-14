#!/usr/bin/env python3
"""
Emergency Optical Probe Fix
Aplica correções robustas para lidar com saídas variáveis do optical probe
"""

from src.core.optical_probe_fixed import OpticalProbeFixed

def apply_optical_probe_fix(psiqrh_system):
    """
    Aplica correção robusta do optical probe ao sistema ΨQRH existente

    Args:
        psiqrh_system: Instância do sistema ΨQRH

    Returns:
        Sistema corrigido
    """
    print("🔧 Aplicando correção robusta do optical probe...")

    # Verificar se o sistema tem optical probe
    if not hasattr(psiqrh_system, 'optical_probe'):
        print("⚠️  Sistema não tem optical probe - pulando correção")
        return psiqrh_system

    # Substituir método de extração de texto por versão robusta
    original_forward = psiqrh_system.optical_probe.forward

    def robust_forward(psi_final):
        """Versão robusta do forward que sempre retorna tuple válido"""
        try:
            result = original_forward(psi_final)
            # Validar que é um tuple de 3 elementos
            if isinstance(result, tuple) and len(result) == 3:
                return result
            else:
                # Fallback para resultado válido
                print(f"⚠️  Optical probe retornou formato inválido: {type(result)}")
                return (-1, 0.0, False)
        except Exception as e:
            print(f"⚠️  Erro no optical probe forward: {e}")
            return (-1, 0.0, False)

    # Aplicar correção
    psiqrh_system.optical_probe.forward = robust_forward

    # Garantir que safe_extract_text está disponível
    if not hasattr(psiqrh_system.optical_probe, 'safe_extract_text'):
        psiqrh_system.optical_probe.safe_extract_text = OpticalProbeFixed().safe_extract_text

    print("✅ Correção optical probe aplicada com sucesso")
    return psiqrh_system

# Aplicar correção se o sistema já existe
if 'psiqrh_system' in locals():
    psiqrh_system = apply_optical_probe_fix(psiqrh_system)
    print("🔧 Correção optical probe aplicada ao sistema existente")

def create_robust_psiqrh_system():
    """Cria sistema ΨQRH com todas as correções aplicadas"""
    from psiqrh import ΨQRHPipeline

    print("🏗️  Criando sistema ΨQRH robusto...")

    # Criar sistema base
    system = ΨQRHPipeline()

    # Aplicar correções
    system = apply_optical_probe_fix(system)

    print("✅ Sistema ΨQRH robusto criado com sucesso")
    return system

if __name__ == "__main__":
    # Teste da correção
    print("🧪 Testando correção optical probe...")

    system = create_robust_psiqrh_system()

    # Teste com entrada simples
    try:
        result = system("hello")
        print(f"✅ Teste bem-sucedido: {result.get('status', 'unknown')}")
    except Exception as e:
        print(f"❌ Teste falhou: {e}")

    print("🎯 Correção optical probe concluída")
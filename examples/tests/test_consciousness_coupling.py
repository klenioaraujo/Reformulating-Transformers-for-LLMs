#!/usr/bin/env python3
"""
Script de Teste: Validação do Acoplamento Consciência-Espectro
===============================================================

Testa se o módulo de consciência está ACOPLADO ao espectro quaterniônico real,
eliminando completamente o fallback sintético.

Esperado:
- consciousness_distribution varia conforme input (NÃO uniforme)
- FCI varia conforme input (NÃO fixo)
- Coerência > 0.0 (campo reflete sinal real)
- D adapta baseado em FCI e energia espectral
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.core.ΨQRH import QRHFactory
import yaml

def test_consciousness_coupling():
    """Testa acoplamento entre consciência e espectro quaterniônico."""

    print("="*80)
    print("TESTE DE ACOPLAMENTO: Consciência ↔ Espectro Quaterniônico")
    print("="*80)

    # Inicializar factory
    factory = QRHFactory()

    # Textos de teste com características DIFERENTES
    test_cases = [
        {
            'text': 'Hello',
            'expected': 'Texto curto, baixa complexidade → FCI baixo'
        },
        {
            'text': 'The quick brown fox jumps over the lazy dog',
            'expected': 'Texto médio, diversidade média → FCI médio'
        },
        {
            'text': 'In the realm of quantum mechanics, the superposition principle states that particles exist in all possible states simultaneously until observed, collapsing the wave function.',
            'expected': 'Texto longo, alta complexidade → FCI alto'
        }
    ]

    print("\n📊 Testando 3 textos com características diferentes...\n")

    results = []

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'─'*80}")
        print(f"TESTE {i}/3: {case['text'][:50]}...")
        print(f"Expectativa: {case['expected']}")
        print(f"{'─'*80}\n")

        try:
            # Processar texto
            result = factory.process_text(case['text'], device='cpu')

            # Extrair métricas de consciência
            if isinstance(result, dict) and 'consciousness_results' in result:
                consciousness = result['consciousness_results']

                # Extrair métricas chave
                psi_dist = consciousness.get('consciousness_distribution')
                fci_evo = consciousness.get('fci_evolution')
                fractal_field = consciousness.get('fractal_field')
                diffusion = consciousness.get('diffusion_coefficient')

                # Calcular estatísticas
                import torch

                psi_mean = psi_dist.mean().item()
                psi_std = psi_dist.std().item()
                psi_max = psi_dist.max().item()
                psi_min = psi_dist.min().item()

                fci_final = fci_evo[-1].item()
                fci_mean = fci_evo.mean().item()
                fci_std = fci_evo.std().item()

                field_magnitude = torch.norm(fractal_field, dim=-1).mean().item()

                # Calcular coerência (correlação espacial)
                field_flat = fractal_field.flatten()
                field_shifted = torch.roll(field_flat, 1)
                covariance = torch.mean((field_flat - field_flat.mean()) * (field_shifted - field_shifted.mean()))
                field_var = field_flat.var()
                coherence = (covariance / (field_var + 1e-10)).item()
                coherence = max(0.0, min(1.0, abs(coherence)))

                d_mean = diffusion.mean().item()
                d_std = diffusion.std().item()

                results.append({
                    'text': case['text'][:30],
                    'psi_mean': psi_mean,
                    'psi_std': psi_std,
                    'fci_final': fci_final,
                    'fci_mean': fci_mean,
                    'coherence': coherence,
                    'field_magnitude': field_magnitude,
                    'd_mean': d_mean
                })

                print(f"✅ RESULTADOS:")
                print(f"   Ψ Distribution: mean={psi_mean:.6f}, std={psi_std:.6f}, max={psi_max:.6f}, min={psi_min:.6f}")
                print(f"   FCI: final={fci_final:.4f}, mean={fci_mean:.4f}, std={fci_std:.4f}")
                print(f"   Campo Fractal: magnitude={field_magnitude:.4f}, coerência={coherence:.4f}")
                print(f"   Difusão: mean={d_mean:.4f}, std={d_std:.4f}")

                # Validações
                validations = []

                # 1. Distribuição NÃO uniforme (std > 0)
                if psi_std > 1e-5:
                    validations.append("✅ Ψ NÃO uniforme (std > 0)")
                else:
                    validations.append("❌ Ψ uniforme (PROBLEMA: desacoplado)")

                # 2. FCI variando (std > 0)
                if fci_std > 1e-5 or abs(fci_final - fci_mean) > 1e-5:
                    validations.append("✅ FCI variando ao longo do tempo")
                else:
                    validations.append("❌ FCI fixo (PROBLEMA: não adaptativo)")

                # 3. Coerência > 0
                if coherence > 0.01:
                    validations.append(f"✅ Coerência > 0 (campo real)")
                else:
                    validations.append(f"❌ Coerência ≈ 0 (PROBLEMA: campo sintético)")

                # 4. Difusão variando espacialmente
                if d_std > 1e-5:
                    validations.append("✅ D varia espacialmente (adaptativo)")
                else:
                    validations.append("❌ D constante (PROBLEMA: não adaptativo)")

                print(f"\n📋 VALIDAÇÕES:")
                for validation in validations:
                    print(f"   {validation}")

            else:
                print(f"❌ Resultado não contém consciousness_results")
                results.append({'text': case['text'][:30], 'error': 'No consciousness data'})

        except Exception as e:
            print(f"❌ ERRO: {e}")
            import traceback
            traceback.print_exc()
            results.append({'text': case['text'][:30], 'error': str(e)})

    # Análise comparativa
    print(f"\n\n{'='*80}")
    print("ANÁLISE COMPARATIVA: Variação entre textos")
    print(f"{'='*80}\n")

    if len(results) == 3 and all('error' not in r for r in results):
        print("📊 Tabela comparativa:\n")
        print(f"{'Texto':<32} {'Ψ_mean':<10} {'Ψ_std':<10} {'FCI':<10} {'Coerência':<10} {'D_mean':<10}")
        print(f"{'-'*92}")
        for r in results:
            print(f"{r['text']:<32} {r['psi_mean']:<10.6f} {r['psi_std']:<10.6f} {r['fci_final']:<10.4f} {r['coherence']:<10.4f} {r['d_mean']:<10.4f}")

        # Verificar variação ENTRE textos
        print(f"\n📈 VALIDAÇÃO FINAL: Acoplamento Real")

        # CORREÇÃO: Verificar Ψ_std ao invés de Ψ_mean (mean sempre será 1/embed_dim após normalização L1)
        psi_stds = [r['psi_std'] for r in results]
        fci_finals = [r['fci_final'] for r in results]
        coherences = [r['coherence'] for r in results]

        import numpy as np
        psi_std_variation = np.std(psi_stds)  # Variação do DESVIO PADRÃO de Ψ
        fci_variation = np.std(fci_finals)
        coherence_variation = np.std(coherences)

        print(f"   Variação Ψ_std entre textos: {psi_std_variation:.6f} {'✅' if psi_std_variation > 1e-4 else '❌'}")
        print(f"   Variação FCI entre textos: {fci_variation:.4f} {'✅' if fci_variation > 0.01 else '❌'}")
        print(f"   Variação Coerência entre textos: {coherence_variation:.4f} {'✅' if coherence_variation > 0.01 else '❌'}")

        if psi_std_variation > 1e-4 and fci_variation > 0.01:
            print(f"\n✅✅✅ ACOPLAMENTO CONFIRMADO: Métricas variam conforme input real")
        else:
            print(f"\n❌❌❌ ACOPLAMENTO FALHOU: Métricas insuficientemente variáveis")
    else:
        print("❌ Não foi possível completar análise comparativa (erros nos testes)")

    print(f"\n{'='*80}")
    print("FIM DO TESTE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    test_consciousness_coupling()

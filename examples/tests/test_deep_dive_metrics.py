#!/usr/bin/env python3
"""
Teste do endpoint deep_dive para verificar métricas dinâmicas reais
"""
import requests
import json

def test_deep_dive(text, label):
    """Testa o endpoint com um texto específico"""
    print(f"\n{'='*60}")
    print(f"TESTE: {label}")
    print(f"Texto: '{text}'")
    print('='*60)

    try:
        response = requests.post(
            'http://localhost:5000/api/v1/analyze/deep_dive',
            json={'text': text},
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()

            # Extrair métricas relevantes
            power_law = data['fractal_consciousness_analysis']['power_law_fit']
            fci_comp = data['fractal_consciousness_analysis']['fci_components']
            fractal_dim = data['fractal_consciousness_analysis']['fractal_dimension']

            print("\n📊 MÉTRICAS EXTRAÍDAS:")
            print(f"  Power Law Fit:")
            print(f"    β (expoente): {power_law['beta_exponent']:.6f}")
            print(f"    R²: {power_law['r_squared']:.6f}")
            print(f"    Pontos usados: {power_law['points_used']}")

            print(f"\n  Dimensão Fractal:")
            print(f"    Raw: {fractal_dim['raw_value']:.6f}")
            print(f"    Final: {fractal_dim['final_value']:.6f}")

            print(f"\n  Componentes FCI:")
            print(f"    D_EEG raw: {fci_comp['d_eeg']['raw']:.6f}")
            print(f"    D_EEG norm: {fci_comp['d_eeg']['normalized']:.6f}")
            print(f"    H_fMRI raw: {fci_comp['h_fmri']['raw']:.6f}")
            print(f"    H_fMRI norm: {fci_comp['h_fmri']['normalized']:.6f}")
            print(f"    CLZ raw: {fci_comp['clz']['raw']:.6f}")
            print(f"    CLZ norm: {fci_comp['clz']['normalized']:.6f}")

            return data
        else:
            print(f"❌ Erro HTTP {response.status_code}")
            print(response.text)
            return None

    except Exception as e:
        print(f"❌ Erro: {e}")
        return None

if __name__ == '__main__':
    # Testar com textos diferentes
    result1 = test_deep_dive('ola', 'Texto Curto')
    result2 = test_deep_dive('ola mundo', 'Texto Médio')
    result3 = test_deep_dive('ola mundo como vai voce hoje', 'Texto Longo')

    # Comparação
    if result1 and result2 and result3:
        print("\n" + "="*60)
        print("COMPARAÇÃO DE RESULTADOS")
        print("="*60)

        for name, result in [('Curto', result1), ('Médio', result2), ('Longo', result3)]:
            beta = result['fractal_consciousness_analysis']['power_law_fit']['beta_exponent']
            d_eeg = result['fractal_consciousness_analysis']['fci_components']['d_eeg']['raw']
            print(f"{name:8s}: β={beta:7.4f}, D_EEG={d_eeg:7.4f}")

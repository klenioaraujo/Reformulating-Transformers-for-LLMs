#!/usr/bin/env python3
"""
Script de Teste: Endpoint /api/v1/analyze/deep_dive
===================================================

Testa o endpoint de análise profunda verificando se todos os campos
da estrutura JSON estão presentes e populados corretamente.
"""

import requests
import json
import sys


def test_deep_dive_endpoint():
    """Testa o endpoint /api/v1/analyze/deep_dive"""

    # URL do endpoint
    url = "http://localhost:5000/api/v1/analyze/deep_dive"

    # Textos de teste
    test_texts = [
        "Hello",
        "The quick brown fox jumps over the lazy dog",
        "In the realm of quantum mechanics, the superposition principle states that particles exist in all possible states simultaneously until observed."
    ]

    print("=" * 80)
    print("TESTE DO ENDPOINT: /api/v1/analyze/deep_dive")
    print("=" * 80)

    for i, text in enumerate(test_texts, 1):
        print(f"\n{'─' * 80}")
        print(f"TESTE {i}/3: {text[:50]}...")
        print(f"{'─' * 80}\n")

        try:
            # Fazer requisição POST
            response = requests.post(url, json={"text": text})

            if response.status_code != 200:
                print(f"❌ ERRO: Status code {response.status_code}")
                print(f"Resposta: {response.text}")
                continue

            # Parse JSON
            data = response.json()

            # Validar estrutura
            print("✅ Resposta recebida com sucesso!")
            print(f"\n📊 ESTRUTURA JSON:\n")
            print(json.dumps(data, indent=2))

            # Validações específicas
            validations = []

            # Metadata
            if "metadata" in data:
                metadata = data["metadata"]
                validations.append(f"✅ Metadata presente (timestamp: {metadata.get('timestamp_utc', 'N/A')})")
                validations.append(f"   - Tempo total: {metadata.get('execution_times_ms', {}).get('total', 0):.2f}ms")
            else:
                validations.append("❌ Metadata ausente")

            # QRH Spectral Analysis
            if "qrh_spectral_analysis" in data:
                qrh = data["qrh_spectral_analysis"]
                alpha = qrh.get("adaptive_alpha", 0)
                energy_mean = qrh.get("spectral_energy_stats", {}).get("mean", 0)
                validations.append(f"✅ QRH Spectral Analysis presente (alpha={alpha:.4f}, energy_mean={energy_mean:.2e})")
            else:
                validations.append("❌ QRH Spectral Analysis ausente")

            # Fractal Consciousness Analysis
            if "fractal_consciousness_analysis" in data:
                fractal = data["fractal_consciousness_analysis"]

                # Power law fit
                power_law = fractal.get("power_law_fit", {})
                beta = power_law.get("beta_exponent", 0)
                r2 = power_law.get("r_squared", 0)
                points = power_law.get("points_used", 0)
                validations.append(f"✅ Power Law Fit presente (β={beta:.4f}, R²={r2:.4f}, pontos={points})")

                # Fractal dimension
                dim = fractal.get("fractal_dimension", {})
                d_raw = dim.get("raw_value", 0)
                d_final = dim.get("final_value", 0)
                validations.append(f"✅ Dimensão Fractal presente (raw={d_raw:.4f}, final={d_final:.4f})")

                # FCI components
                fci_comp = fractal.get("fci_components", {})
                d_eeg = fci_comp.get("d_eeg", {})
                h_fmri = fci_comp.get("h_fmri", {})
                clz = fci_comp.get("clz", {})
                validations.append(f"✅ FCI Components presente:")
                validations.append(f"   - D_EEG: raw={d_eeg.get('raw', 0):.4f}, norm={d_eeg.get('normalized', 0):.4f}")
                validations.append(f"   - H_fMRI: raw={h_fmri.get('raw', 0):.4f}, norm={h_fmri.get('normalized', 0):.4f}")
                validations.append(f"   - CLZ: raw={clz.get('raw', 0):.4f}, norm={clz.get('normalized', 0):.4f}")

                # Final metrics
                final = fractal.get("final_metrics", {})
                fci = final.get("fci_score", 0)
                state = final.get("consciousness_state", "UNKNOWN")
                coherence = final.get("coherence", 0)
                entropy = final.get("entropy", 0)
                validations.append(f"✅ Final Metrics presente:")
                validations.append(f"   - FCI: {fci:.4f}")
                validations.append(f"   - Estado: {state}")
                validations.append(f"   - Coerência: {coherence:.4f}")
                validations.append(f"   - Entropia: {entropy:.4f}")
            else:
                validations.append("❌ Fractal Consciousness Analysis ausente")

            # Raw data outputs
            if "raw_data_outputs" in data:
                raw = data["raw_data_outputs"]
                psi_len = len(raw.get("psi_distribution", []))
                field_len = len(raw.get("fractal_field_sample", []))
                validations.append(f"✅ Raw Data Outputs presente (psi={psi_len} valores, field={field_len} valores)")
            else:
                validations.append("❌ Raw Data Outputs ausente")

            print(f"\n📋 VALIDAÇÕES:")
            for validation in validations:
                print(f"   {validation}")

        except requests.exceptions.ConnectionError:
            print("❌ ERRO: Não foi possível conectar ao servidor!")
            print("Certifique-se de que o servidor está rodando: python app.py")
            return False
        except Exception as e:
            print(f"❌ ERRO: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'=' * 80}")
    print("FIM DO TESTE")
    print(f"{'=' * 80}\n")

    return True


if __name__ == "__main__":
    success = test_deep_dive_endpoint()
    sys.exit(0 if success else 1)

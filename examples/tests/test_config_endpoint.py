#!/usr/bin/env python3
"""
Test /api/config endpoint
"""
import requests
import json

def test_config_endpoint():
    """Testa o endpoint /api/config"""
    url = "http://localhost:5000/api/config"

    print("🧪 Testing /api/config endpoint...")
    print(f"📍 URL: {url}\n")

    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()

            print("✅ Status: SUCCESS")
            print(f"📊 Response keys: {list(data.keys())}\n")

            # Config files
            if 'config_files' in data:
                print("📁 Config Files:")
                print(f"  - qrh_config: {len(data['config_files'].get('qrh_config', {}))} sections")
                print(f"  - consciousness_metrics: {len(data['config_files'].get('consciousness_metrics', {}))} sections\n")

            # Runtime values
            if 'runtime_values' in data:
                print("⚙️  Runtime Values:")
                rv = data['runtime_values']

                if 'qrh_layer' in rv:
                    qrh = rv['qrh_layer']
                    print(f"  QRH Layer:")
                    print(f"    - embed_dim: {qrh.get('embed_dim')}")
                    print(f"    - alpha: {qrh.get('alpha')}")
                    print(f"    - device: {qrh.get('device')}")

                if 'consciousness_metrics' in rv:
                    cm = rv['consciousness_metrics']
                    print(f"\n  Consciousness Metrics:")

                    if 'fractal_dimension' in cm:
                        fd = cm['fractal_dimension']
                        print(f"    Fractal Dimension:")
                        print(f"      - min: {fd.get('min')}")
                        print(f"      - max: {fd.get('max')}")
                        print(f"      - normalizer: {fd.get('normalizer')}")

                    if 'component_max_values' in cm:
                        cmv = cm['component_max_values']
                        print(f"    Component Max Values:")
                        print(f"      - d_eeg_max: {cmv.get('d_eeg_max')}")
                        print(f"      - h_fmri_max: {cmv.get('h_fmri_max')}")
                        print(f"      - clz_max: {cmv.get('clz_max')}")

                    if 'state_thresholds' in cm:
                        st = cm['state_thresholds']
                        print(f"    State Thresholds:")
                        print(f"      - emergence: {st.get('emergence')}")
                        print(f"      - meditation: {st.get('meditation')}")
                        print(f"      - analysis: {st.get('analysis')}")

                    if 'fci_weights' in cm:
                        fw = cm['fci_weights']
                        print(f"    FCI Weights:")
                        print(f"      - d_eeg: {fw.get('d_eeg')}")
                        print(f"      - h_fmri: {fw.get('h_fmri')}")
                        print(f"      - clz: {fw.get('clz')}")

                    if 'correlation_method' in cm:
                        print(f"    Correlation Method: {cm.get('correlation_method')}")

            # Config paths
            if 'config_paths' in data:
                print(f"\n📂 Config Paths:")
                for key, path in data['config_paths'].items():
                    print(f"  - {key}: {path}")

            print("\n" + "="*60)
            print("📄 Full JSON response:")
            print(json.dumps(data, indent=2))

        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Server não está rodando")
        print("💡 Execute: python3 app.py ou make restart-dev")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == '__main__':
    test_config_endpoint()
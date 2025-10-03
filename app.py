#!/usr/bin/env python3
"""
ΨQRH Flask API - Backend para Sistema de Chat com Consciência Fractal
====================================================================

API REST que expõe o framework ΨQRH para chat e análise de consciência,
retornando dados estruturados para visualização GLS no frontend.
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
import sys
import os
from typing import Dict, Any

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.core.ΨQRH import QRHFactory

app = Flask(__name__)
CORS(app)  # Habilitar CORS para frontend

# Inicializar ΨQRH Factory
print("🚀 Inicializando ΨQRH API...")
qrh_factory = None

try:
    qrh_factory = QRHFactory()
    print("✅ ΨQRH Factory inicializada com sucesso")
except Exception as e:
    print(f"❌ Erro ao inicializar ΨQRH Factory: {e}")


@app.route('/')
def index():
    """Página inicial da API"""
    return jsonify({
        'name': 'ΨQRH API',
        'version': '1.0.0',
        'description': 'API REST para Sistema de Chat com Consciência Fractal',
        'endpoints': {
            '/chat': 'POST - Processar mensagem de chat',
            '/health': 'GET - Status do sistema',
            '/metrics': 'GET - Métricas de consciência atuais'
        }
    })


@app.route('/health')
def health():
    """Endpoint de saúde do sistema"""
    status = 'healthy' if qrh_factory is not None else 'unhealthy'

    return jsonify({
        'status': status,
        'system': 'ΨQRH API',
        'components': {
            'qrh_factory': 'loaded' if qrh_factory is not None else 'failed',
            'consciousness_processor': 'loaded' if hasattr(qrh_factory, 'consciousness_processor') and qrh_factory.consciousness_processor else 'unavailable'
        }
    })


@app.route('/chat', methods=['POST'])
def chat():
    """
    Endpoint principal de chat

    Recebe: {'message': 'texto do usuário'}
    Retorna: {
        'response': 'resposta do ΨQRH',
        'consciousness_metrics': {métricas de consciência}
    }
    """
    if qrh_factory is None:
        return jsonify({
            'error': 'ΨQRH system not initialized',
            'status': 'error'
        }), 500

    try:
        data = request.get_json()

        if not data or 'message' not in data:
            return jsonify({
                'error': 'Missing message field',
                'status': 'error'
            }), 400

        user_message = data['message']

        # Processar mensagem através do ΨQRH
        result = qrh_factory.process_text(user_message, device='cpu')

        # Extrair dados estruturados para resposta
        response_data = {
            'status': 'success',
            'user_message': user_message,
            'timestamp': torch.rand(1).item()  # Timestamp sintético
        }

        # Se o resultado é um dicionário (com análise de consciência)
        if isinstance(result, dict) and 'text_analysis' in result:
            response_data['response'] = result['text_analysis']

            # Extrair métricas de consciência se disponíveis
            if 'consciousness_results' in result:
                consciousness_data = extract_consciousness_metrics(result['consciousness_results'])
                response_data['consciousness_metrics'] = consciousness_data
        else:
            # Resultado simples (string)
            response_data['response'] = result
            response_data['consciousness_metrics'] = None

        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            'error': f'Processing error: {str(e)}',
            'status': 'error'
        }), 500


@app.route('/metrics')
def get_metrics():
    """Endpoint para obter métricas de consciência atuais"""
    if qrh_factory is None or not hasattr(qrh_factory, 'consciousness_results'):
        return jsonify({
            'error': 'Consciousness metrics not available',
            'status': 'error'
        }), 404

    try:
        if qrh_factory.consciousness_results:
            metrics = extract_consciousness_metrics(qrh_factory.consciousness_results)
            return jsonify({
                'status': 'success',
                'metrics': metrics
            })
        else:
            return jsonify({
                'error': 'No consciousness data available',
                'status': 'error'
            }), 404

    except Exception as e:
        return jsonify({
            'error': f'Error extracting metrics: {str(e)}',
            'status': 'error'
        }), 500


def extract_consciousness_metrics(consciousness_results: Dict[str, Any]) -> Dict[str, Any]:
    """Extrai métricas estruturadas dos resultados de consciência"""
    try:
        metrics = {}

        # FCI (Fractal Consciousness Index)
        fci_evolution = consciousness_results.get('fci_evolution', [0.0])
        if isinstance(fci_evolution, torch.Tensor):
            metrics['fci'] = fci_evolution[-1].item()
        elif isinstance(fci_evolution, list) and len(fci_evolution) > 0:
            metrics['fci'] = fci_evolution[-1]
        else:
            metrics['fci'] = 0.0

        # Estado de consciência
        state = consciousness_results.get('final_consciousness_state', None)
        if state:
            metrics['state'] = getattr(state, 'name', 'UNKNOWN')
            metrics['fractal_dimension'] = getattr(state, 'fractal_dimension', 1.0)
        else:
            metrics['state'] = 'UNKNOWN'
            metrics['fractal_dimension'] = 1.0

        # Outras métricas
        metrics['processing_steps'] = consciousness_results.get('processing_steps', 0)
        metrics['convergence_achieved'] = consciousness_results.get('convergence_achieved', False)

        # Campo fractal
        fractal_field = consciousness_results.get('fractal_field', None)
        if fractal_field is not None:
            metrics['field_magnitude'] = torch.norm(fractal_field).item() if isinstance(fractal_field, torch.Tensor) else 0.0
        else:
            metrics['field_magnitude'] = 0.0

        # Distribuição de consciência
        psi_dist = consciousness_results.get('consciousness_distribution', None)
        if psi_dist is not None:
            # Calcular entropia da distribuição
            epsilon = 1e-10
            psi_safe = torch.clamp(psi_dist, min=epsilon)
            log_psi = torch.log(psi_safe)
            entropy_raw = -torch.sum(psi_dist * log_psi, dim=-1).mean()
            metrics['entropy'] = entropy_raw.item() if not torch.isnan(entropy_raw) else 0.0

            # Pico e dispersão
            metrics['peak_distribution'] = psi_dist.max().item() if isinstance(psi_dist, torch.Tensor) else 0.0
            metrics['distribution_spread'] = psi_dist.std().item() if isinstance(psi_dist, torch.Tensor) else 0.0
        else:
            metrics['entropy'] = 0.0
            metrics['peak_distribution'] = 0.0
            metrics['distribution_spread'] = 0.0

        return metrics

    except Exception as e:
        print(f"Error extracting consciousness metrics: {e}")
        return {
            'fci': 0.0,
            'state': 'ERROR',
            'fractal_dimension': 1.0,
            'processing_steps': 0,
            'convergence_achieved': False,
            'field_magnitude': 0.0,
            'entropy': 0.0,
            'peak_distribution': 0.0,
            'distribution_spread': 0.0
        }


# GLS configuration should be handled in the frontend only


if __name__ == '__main__':
    print("\n🌐 ΨQRH API Starting...")
    print("=" * 50)
    print("Endpoints:")
    print("  GET  /          - API information")
    print("  GET  /health    - System health")
    print("  POST /chat      - Process chat message")
    print("  GET  /metrics   - Current consciousness metrics")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5000, debug=True)
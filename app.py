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

from psiqrh import ΨQRHPipeline
from src.conscience.gls_output_generator import GLSOutputGenerator

app = Flask(__name__)
CORS(app)  # Habilitar CORS para frontend

# Inicializar ΨQRH Pipeline e GLS Generator
print("🚀 Inicializando ΨQRH API...")
qrh_pipeline = None
gls_generator = None

try:
    qrh_pipeline = ΨQRHPipeline(task="text-generation", device="cpu")
    print("✅ ΨQRH Pipeline inicializada com sucesso")
except Exception as e:
    print(f"❌ Erro ao inicializar ΨQRH Pipeline: {e}")

try:
    gls_generator = GLSOutputGenerator()
    print("✅ GLS Output Generator inicializado com sucesso")
except Exception as e:
    print(f"❌ Erro ao inicializar GLS Output Generator: {e}")


@app.route('/')
def index():
    """Página inicial - Frontend web"""
    return render_template('index.html')


@app.route('/chat.html')
def chat_page():
    """Interface de Chat"""
    return render_template('chat.html')


@app.route('/harmonic_gls_demo.html')
def harmonic_gls():
    """Visualização Harmônica GLS"""
    return render_template('harmonic_gls_demo.html')


@app.route('/deep_dive_demo.html')
def deep_dive_demo():
    """Demo de Análise Profunda (Deep Dive)"""
    return render_template('deep_dive_demo.html')


@app.route('/api')
def api_info():
    """Informações da API"""
    return jsonify({
        'name': 'ΨQRH API',
        'version': '1.0.0',
        'description': 'API REST para Sistema de Chat com Consciência Fractal',
        'endpoints': {
            '/api/chat': 'POST - Processar mensagem de chat',
            '/api/v1/analyze/deep_dive': 'POST - Análise profunda detalhada (JSON completo)',
            '/api/health': 'GET - Status do sistema',
            '/api/metrics': 'GET - Métricas de consciência atuais',
            '/api/config': 'GET - Configurações do sistema'
        }
    })


@app.route('/api/health')
@app.route('/health')
def health():
    """Endpoint de saúde do sistema"""
    status = 'healthy' if qrh_pipeline is not None else 'unhealthy'

    return jsonify({
        'status': status,
        'system': 'ΨQRH API',
        'components': {
            'qrh_pipeline': 'loaded' if qrh_pipeline is not None else 'failed',
            'consciousness_processor': 'loaded' if hasattr(qrh_pipeline, 'consciousness_processor') and qrh_pipeline.consciousness_processor else 'unavailable',
            'gls_generator': 'loaded' if gls_generator is not None else 'failed'
        }
    })


@app.route('/api/chat', methods=['POST'])
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
    if qrh_pipeline is None:
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

        # Processar mensagem através do ΨQRH Pipeline
        result = qrh_pipeline(user_message)

        # Extrair dados estruturados para resposta
        response_data = {
            'status': result.get('status', 'success'),
            'user_message': user_message,
            'timestamp': torch.rand(1).item(),  # Timestamp sintético
            'processing_parameters': {
                'pipeline_config': {
                    'task': qrh_pipeline.task,
                    'device': qrh_pipeline.device,
                    'embed_dim': qrh_pipeline.config['embed_dim'],
                    'alpha': qrh_pipeline.config['alpha'],
                    'beta': qrh_pipeline.config['beta']
                }
            }
        }

        # Extrair resposta do pipeline
        if result.get('status') == 'success':
            response_data['response'] = result.get('response', '')

            # Extrair métricas físicas do pipeline
            if 'physical_metrics' in result:
                physical_metrics = result['physical_metrics']
                response_data['physical_metrics'] = physical_metrics

                # Mapear para formato de consciência para compatibilidade
                response_data['consciousness_metrics'] = {
                    'fci': physical_metrics.get('FCI', 0.0),
                    'state': physical_metrics.get('consciousness_state', 'UNKNOWN'),
                    'fractal_dimension': physical_metrics.get('D_fractal', 1.0)
                }

            # Adicionar métricas de validação matemática
            if 'mathematical_validation' in result:
                response_data['validation'] = result['mathematical_validation']

            # Gerar dados GLS se disponível e houver métricas de consciência
            if gls_generator is not None and 'consciousness_metrics' in response_data:
                try:
                    # Criar estrutura de consciência compatível com GLS
                    consciousness_results = {
                        'fci_evolution': [response_data['consciousness_metrics']['fci']],
                        'final_consciousness_state': type('State', (), {
                            'name': response_data['consciousness_metrics']['state'],
                            'fractal_dimension': response_data['consciousness_metrics']['fractal_dimension']
                        })()
                    }
                    gls_data = generate_gls_output(consciousness_results)
                    response_data['gls_data'] = gls_data
                except Exception as e:
                    print(f"⚠️  GLS generation error: {e}")
                    response_data['gls_data'] = None
        else:
            # Erro no processamento
            response_data['response'] = f"Erro no processamento: {result.get('error', 'Desconhecido')}"
            response_data['consciousness_metrics'] = None
            response_data['gls_data'] = None

        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            'error': f'Processing error: {str(e)}',
            'status': 'error'
        }), 500


@app.route('/api/config')
@app.route('/config')
def get_config():
    """
    Endpoint para obter todas as configurações do sistema
    Retorna valores de configs/*.yaml usados pelo sistema
    """
    if qrh_pipeline is None:
        return jsonify({
            'error': 'ΨQRH system not initialized',
            'status': 'error'
        }), 500

    try:
        import yaml

        # Carregar configurações dos arquivos YAML
        config_dir = os.path.join(BASE_DIR, 'configs')

        # qrh_config.yaml
        qrh_config_path = os.path.join(config_dir, 'qrh_config.yaml')
        qrh_config = {}
        if os.path.exists(qrh_config_path):
            with open(qrh_config_path, 'r') as f:
                qrh_config = yaml.safe_load(f)

        # consciousness_metrics.yaml
        consciousness_metrics_path = os.path.join(config_dir, 'consciousness_metrics.yaml')
        consciousness_metrics_config = {}
        if os.path.exists(consciousness_metrics_path):
            with open(consciousness_metrics_path, 'r') as f:
                consciousness_metrics_config = yaml.safe_load(f)

        # Valores em uso pelo sistema (runtime)
        runtime_values = {
            'pipeline_config': {
                'task': qrh_pipeline.task,
                'device': qrh_pipeline.device,
                'embed_dim': qrh_pipeline.config['embed_dim'],
                'alpha': qrh_pipeline.config['alpha'],
                'beta': qrh_pipeline.config['beta'],
                'I0': qrh_pipeline.config['I0'],
                'omega': qrh_pipeline.config['omega'],
                'k': qrh_pipeline.config['k']
            }
        }

        # Adicionar valores de consciência se disponível
        if hasattr(qrh_pipeline, 'consciousness_processor') and qrh_pipeline.consciousness_processor:
            cp = qrh_pipeline.consciousness_processor
            # Para o novo pipeline, as métricas podem ser diferentes
            runtime_values['consciousness_metrics'] = {
                'available': True,
                'processor_type': str(type(cp))
            }

        return jsonify({
            'status': 'success',
            'config_files': {
                'qrh_config': qrh_config,
                'consciousness_metrics': consciousness_metrics_config
            },
            'runtime_values': runtime_values,
            'config_paths': {
                'qrh_config': qrh_config_path,
                'consciousness_metrics': consciousness_metrics_path
            }
        })

    except Exception as e:
        return jsonify({
            'error': f'Error loading configuration: {str(e)}',
            'status': 'error'
        }), 500


@app.route('/api/v1/analyze/deep_dive', methods=['POST'])
def deep_dive_analysis():
    """
    Endpoint de análise profunda com todos os detalhes do processamento ΨQRH

    Retorna estrutura JSON completa com:
    - Metadata (timestamp, versão, tempos de execução)
    - QRH Spectral Analysis (alpha, energia, fase)
    - Fractal Consciousness Analysis (lei de potência, dimensão fractal, FCI)
    - Raw Data Outputs (tensores para análise externa)
    """
    if qrh_pipeline is None:
        return jsonify({
            'error': 'ΨQRH system not initialized',
            'status': 'error'
        }), 500

    try:
        import time
        from datetime import datetime
        import numpy as np

        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing text field',
                'status': 'error'
            }), 400

        input_text = data['text']

        # Timestamps para medir tempo de execução
        t_start = time.time()

        # Processar texto através do ΨQRH Pipeline
        result = qrh_pipeline(input_text)

        t_total = (time.time() - t_start) * 1000  # ms

        # Extrair dados de cada etapa
        response = {
            "metadata": {
                "input_text": input_text,
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "framework_version": "ΨQRH v2.0.0 (Pipeline)",
                "execution_times_ms": {
                    "total": round(t_total, 2),
                    "pipeline_processing": round(t_total, 2)
                }
            },
            "qrh_spectral_analysis": {},
            "fractal_consciousness_analysis": {},
            "raw_data_outputs": {}
        }

        # Extrair dados espectrais do pipeline
        if result.get('status') == 'success' and 'physical_metrics' in result:
            physical_metrics = result['physical_metrics']

            response["qrh_spectral_analysis"] = {
                "adaptive_alpha": physical_metrics.get('alpha_calibrated', 0.0),
                "adaptive_beta": physical_metrics.get('beta_calibrated', 0.0),
                "spectral_energy_stats": {
                    "fractal_dimension": physical_metrics.get('D_fractal', 1.0),
                    "consciousness_index": physical_metrics.get('FCI', 0.0)
                }
            }

        # Extrair análise de consciência fractal do pipeline
        if result.get('status') == 'success':
            physical_metrics = result.get('physical_metrics', {})

            response["fractal_consciousness_analysis"] = {
                "power_law_fit": {
                    "beta_exponent": 0.0,  # Não disponível no pipeline simplificado
                    "r_squared": 0.0,
                    "points_used": 0
                },
                "fractal_dimension": {
                    "raw_value": physical_metrics.get('D_fractal', 1.0),
                    "final_value": physical_metrics.get('D_fractal', 1.0)
                },
                "fci_components": {
                    "d_eeg": {"raw": 0.0, "normalized": 0.0},
                    "h_fmri": {"raw": 0.0, "normalized": 0.0},
                    "clz": {"raw": 0.0, "normalized": 0.0}
                },
                "final_metrics": {
                    "fci_score": physical_metrics.get('FCI', 0.0),
                    "consciousness_state": physical_metrics.get('consciousness_state', 'UNKNOWN'),
                    "coherence": 0.0,
                    "diffusion_coefficient": 0.0,
                    "entropy": 0.0
                }
            }

        # Adicionar análise quântica do pipeline se disponível
        if 'quantum_interpretation' in result:
            quantum_data = result['quantum_interpretation']
            response["quantum_interpretation"] = quantum_data

        # Raw data outputs - limitado para o novo pipeline
        response["raw_data_outputs"] = {
            "pipeline_response": result.get('response', ''),
            "processing_time": result.get('processing_time', 0.0)
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f'Deep dive analysis error: {str(e)}',
            'status': 'error'
        }), 500


@app.route('/api/metrics')
@app.route('/metrics')
def get_metrics():
    """Endpoint para obter métricas de consciência atuais"""
    if qrh_pipeline is None:
        return jsonify({
            'error': 'ΨQRH Pipeline not available',
            'status': 'error'
        }), 404

    try:
        # Para o novo pipeline, as métricas são baseadas na configuração atual
        metrics = {
            'pipeline_status': 'active',
            'task': qrh_pipeline.task,
            'device': qrh_pipeline.device,
            'embed_dim': qrh_pipeline.config['embed_dim'],
            'auto_calibration': qrh_pipeline.enable_auto_calibration,
            'noncommutative_geometry': qrh_pipeline.enable_noncommutative
        }

        return jsonify({
            'status': 'success',
            'metrics': metrics
        })

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


def generate_gls_output(consciousness_results: Dict[str, Any]) -> Dict[str, Any]:
    """Gera dados GLS para visualização no frontend"""
    try:
        if gls_generator is None:
            return {
                'status': 'error',
                'error': 'GLS generator not available'
            }

        # Gerar código Processing e p5.js
        processing_code = gls_generator.generate_processing_code(consciousness_results)
        p5js_code = gls_generator.generate_p5js_code(consciousness_results)

        # Extrair parâmetros visuais para renderização direta
        fci_evolution = consciousness_results.get('fci_evolution', [0.0])
        if isinstance(fci_evolution, torch.Tensor):
            fci = fci_evolution[-1].item()
        elif isinstance(fci_evolution, list) and len(fci_evolution) > 0:
            fci = fci_evolution[-1]
        else:
            fci = 0.0

        state = consciousness_results.get('final_consciousness_state', None)
        fractal_dim = getattr(state, 'fractal_dimension', 1.0) if state else 1.0

        # Mapear para parâmetros visuais
        visual_params = gls_generator._map_consciousness_to_visual(fci, state, fractal_dim)

        return {
            'status': 'success',
            'processing_code': processing_code,
            'p5js_code': p5js_code,
            'visual_params': {
                'complexity': visual_params['complexity'],
                'colors': visual_params['colors'],
                'rotation_speed': visual_params['rotation_speed'],
                'fci': visual_params['fci'],
                'state': visual_params['state'],
                'fractal_dim': visual_params['fractal_dim']
            }
        }

    except Exception as e:
        print(f"Error generating GLS output: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }


if __name__ == '__main__':
    print("\n🌐 ΨQRH API Starting...")
    print("=" * 70)
    print("Endpoints:")
    print("  GET  /                           - Frontend page")
    print("  GET  /api/health                 - System health")
    print("  POST /api/chat                   - Process chat message")
    print("  POST /api/v1/analyze/deep_dive   - Deep dive analysis (full JSON)")
    print("  GET  /api/metrics                - Current consciousness metrics")
    print("  GET  /api/config                 - System configuration")
    print("=" * 70)

    app.run(host='0.0.0.0', port=5000, debug=True)
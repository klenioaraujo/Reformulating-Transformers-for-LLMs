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
from src.conscience.gls_output_generator import GLSOutputGenerator

app = Flask(__name__)
CORS(app)  # Habilitar CORS para frontend

# Inicializar ΨQRH Factory e GLS Generator
print("🚀 Inicializando ΨQRH API...")
qrh_factory = None
gls_generator = None

try:
    qrh_factory = QRHFactory()
    print("✅ ΨQRH Factory inicializada com sucesso")
except Exception as e:
    print(f"❌ Erro ao inicializar ΨQRH Factory: {e}")

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
    status = 'healthy' if qrh_factory is not None else 'unhealthy'

    return jsonify({
        'status': status,
        'system': 'ΨQRH API',
        'components': {
            'qrh_factory': 'loaded' if qrh_factory is not None else 'failed',
            'consciousness_processor': 'loaded' if hasattr(qrh_factory, 'consciousness_processor') and qrh_factory.consciousness_processor else 'unavailable',
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
            'timestamp': torch.rand(1).item(),  # Timestamp sintético
            'processing_parameters': {
                'qrh_config': {
                    'embed_dim': qrh_factory.config.embed_dim,
                    'alpha': qrh_factory.config.alpha,
                    'use_learned_rotation': qrh_factory.config.use_learned_rotation,
                    'device': 'cpu'
                },
                'consciousness_config': qrh_factory.consciousness_config,
                'psicws_config': qrh_factory.psicws_config
            }
        }

        # Se o resultado é um dicionário (com análise de consciência)
        if isinstance(result, dict) and 'text_analysis' in result:
            response_data['response'] = result['text_analysis']

            # Extrair métricas de consciência se disponíveis
            if 'consciousness_results' in result:
                consciousness_data = extract_consciousness_metrics(result['consciousness_results'])
                response_data['consciousness_metrics'] = consciousness_data

                # Adicionar parâmetros específicos do processamento
                if 'layer1_fractal' in result:
                    layer1_data = result['layer1_fractal']
                    response_data['processing_parameters']['layer1_fractal'] = {
                        'alpha_adaptive': layer1_data.get('alpha', 0.0),
                        'shape': layer1_data.get('shape', []),
                        'statistics': layer1_data.get('statistics', {}),
                        'values_count': len(layer1_data.get('values', {}).get('magnitude', []))
                    }

                # Gerar dados GLS se disponível
                if gls_generator is not None:
                    try:
                        gls_data = generate_gls_output(result['consciousness_results'])
                        response_data['gls_data'] = gls_data
                    except Exception as e:
                        print(f"⚠️  GLS generation error: {e}")
                        response_data['gls_data'] = None
        else:
            # Resultado simples (string)
            response_data['response'] = result
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
    if qrh_factory is None:
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
        with open(qrh_config_path, 'r') as f:
            qrh_config = yaml.safe_load(f)

        # consciousness_metrics.yaml
        consciousness_metrics_path = os.path.join(config_dir, 'consciousness_metrics.yaml')
        with open(consciousness_metrics_path, 'r') as f:
            consciousness_metrics_config = yaml.safe_load(f)

        # Valores em uso pelo sistema (runtime)
        runtime_values = {
            'qrh_layer': {
                'embed_dim': qrh_factory.config.embed_dim,
                'alpha': qrh_factory.config.alpha,
                'use_learned_rotation': qrh_factory.config.use_learned_rotation,
                'device': qrh_factory.config.device,
            }
        }

        # Adicionar valores de consciência se disponível
        if hasattr(qrh_factory, 'consciousness_processor') and qrh_factory.consciousness_processor:
            cp = qrh_factory.consciousness_processor
            if hasattr(cp, 'metrics'):
                metrics = cp.metrics
                runtime_values['consciousness_metrics'] = {
                    'fractal_dimension': {
                        'min': metrics.fractal_dimension_min,
                        'max': metrics.fractal_dimension_max,
                        'normalizer': metrics.fractal_dimension_normalizer
                    },
                    'component_max_values': {
                        'd_eeg_max': metrics.d_eeg_max,
                        'h_fmri_max': metrics.h_fmri_max,
                        'clz_max': metrics.clz_max
                    },
                    'state_thresholds': {
                        'emergence': metrics.threshold_emergence,
                        'meditation': metrics.threshold_meditation,
                        'analysis': metrics.threshold_analysis
                    },
                    'fci_weights': {
                        'd_eeg': metrics.fci_weights[0].item(),
                        'h_fmri': metrics.fci_weights[1].item(),
                        'clz': metrics.fci_weights[2].item()
                    },
                    'correlation_method': metrics.correlation_method
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
    if qrh_factory is None:
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

        # Processar texto através do ΨQRH
        result = qrh_factory.process_text(input_text, device='cpu')

        t_total = (time.time() - t_start) * 1000  # ms

        # Extrair dados de cada etapa
        response = {
            "metadata": {
                "input_text": input_text,
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "framework_version": "ΨQRH v1.0.0",
                "execution_times_ms": {
                    "total": round(t_total, 2),
                    "qrh_spectral_processing": 0.0,  # Será preenchido
                    "fractal_consciousness_processing": 0.0  # Será preenchido
                }
            },
            "qrh_spectral_analysis": {},
            "fractal_consciousness_analysis": {},
            "raw_data_outputs": {}
        }

        # Extrair dados espectrais (Layer1 Fractal)
        if isinstance(result, dict) and 'layer1_fractal' in result:
            layer1 = result['layer1_fractal']
            stats = layer1.get('statistics', {})
            values = layer1.get('values', {})

            response["qrh_spectral_analysis"] = {
                "adaptive_alpha": layer1.get('alpha', 0.0),
                "spectral_energy_stats": {
                    "mean": stats.get('magnitude_mean', 0.0),
                    "std": stats.get('magnitude_std', 0.0),
                    "min": stats.get('magnitude_min', 0.0),
                    "max": stats.get('magnitude_max', 0.0),
                    "total_energy_scale": np.log1p(stats.get('energy_total', 0.0))
                },
                "quaternion_phase_stats": {
                    "mean": stats.get('phase_mean', 0.0),
                    "std": 0.0  # Calcular se disponível
                }
            }

        # Extrair análise de consciência fractal
        if isinstance(result, dict) and 'consciousness_results' in result:
            consciousness = result['consciousness_results']

            # Extrair métricas detalhadas do consciousness_processor
            if hasattr(qrh_factory, 'consciousness_processor') and qrh_factory.consciousness_processor:
                cp = qrh_factory.consciousness_processor
                metrics_obj = cp.metrics

                # Power law fit (se disponível nos resultados)
                beta_exponent = 0.0
                r_squared = 0.0
                points_used = 0

                # Tentar extrair do histórico de métricas se disponível
                if hasattr(metrics_obj, 'last_fractal_dimension_raw'):
                    # Valores armazenados durante o processamento
                    pass

                # Dimensão fractal
                state = consciousness.get('final_consciousness_state')
                if state:
                    fractal_dim_final = state.fractal_dimension
                else:
                    fractal_dim_final = 1.0

                # FCI components
                fci_evolution = consciousness.get('fci_evolution', torch.tensor([0.0]))
                if isinstance(fci_evolution, torch.Tensor):
                    fci_final = fci_evolution[-1].item()
                else:
                    fci_final = 0.0

                # Extrair componentes FCI do histórico (valores REAIS calculados)
                d_eeg_raw = 0.025  # Default
                h_fmri_raw = 2.0  # Default
                clz_raw = 0.75  # Default
                d_eeg_norm = 0.0
                h_fmri_norm = 0.0
                clz_norm = 0.0
                beta_exponent = 0.0
                r_squared = 0.0
                points_used = 0

                # Tentar extrair do último FCI calculado
                if hasattr(metrics_obj, 'fci_history') and len(metrics_obj.fci_history) > 0:
                    last_fci = metrics_obj.fci_history[-1]
                    components = last_fci.components

                    d_eeg_raw = float(components.get('D_EEG', d_eeg_raw))
                    h_fmri_raw = float(components.get('H_fMRI', h_fmri_raw))
                    clz_raw = float(components.get('CLZ', clz_raw))
                    d_eeg_norm = float(components.get('D_EEG_normalized', d_eeg_norm))
                    h_fmri_norm = float(components.get('H_fMRI_normalized', h_fmri_norm))
                    clz_norm = float(components.get('CLZ_normalized', clz_norm))

                # Tentar extrair dados da lei de potência (se armazenados)
                if hasattr(metrics_obj, 'last_beta_exponent'):
                    beta_exponent = float(metrics_obj.last_beta_exponent)
                if hasattr(metrics_obj, 'last_r_squared'):
                    r_squared = float(metrics_obj.last_r_squared)
                if hasattr(metrics_obj, 'last_points_used'):
                    points_used = int(metrics_obj.last_points_used)

                # Campo fractal e outras métricas
                fractal_field = consciousness.get('fractal_field')
                if fractal_field is not None and isinstance(fractal_field, torch.Tensor):
                    field_flat = fractal_field.flatten()
                    field_shifted = torch.roll(field_flat, 1)
                    covariance = torch.mean((field_flat - field_flat.mean()) * (field_shifted - field_shifted.mean()))
                    field_var = field_flat.var()
                    coherence = (covariance / (field_var + 1e-10)).item()
                    coherence = max(0.0, min(1.0, abs(coherence)))
                else:
                    coherence = 0.0

                diffusion = consciousness.get('diffusion_coefficient', torch.tensor(0.0))
                if isinstance(diffusion, torch.Tensor):
                    diffusion_mean = diffusion.mean().item()
                else:
                    diffusion_mean = 0.0

                # Entropia
                psi_dist = consciousness.get('consciousness_distribution')
                if psi_dist is not None and isinstance(psi_dist, torch.Tensor):
                    epsilon = 1e-10
                    psi_safe = torch.clamp(psi_dist, min=epsilon)
                    log_psi = torch.log(psi_safe)
                    entropy_raw = -torch.sum(psi_dist * log_psi, dim=-1).mean()
                    entropy = entropy_raw.item() if not torch.isnan(entropy_raw) else 0.0
                else:
                    entropy = 0.0

                # Dimensão fractal raw (antes do clamp)
                fractal_dim_raw = fractal_dim_final
                if hasattr(metrics_obj, 'last_fractal_dimension_raw'):
                    fractal_dim_raw = metrics_obj.last_fractal_dimension_raw

                response["fractal_consciousness_analysis"] = {
                    "power_law_fit": {
                        "beta_exponent": float(beta_exponent),
                        "r_squared": float(r_squared),
                        "points_used": int(points_used)
                    },
                    "fractal_dimension": {
                        "raw_value": float(fractal_dim_raw),  # Antes do clamp
                        "final_value": float(fractal_dim_final)  # Após clamp
                    },
                    "fci_components": {
                        "d_eeg": {
                            "raw": d_eeg_raw,
                            "normalized": d_eeg_norm
                        },
                        "h_fmri": {
                            "raw": h_fmri_raw,
                            "normalized": h_fmri_norm
                        },
                        "clz": {
                            "raw": clz_raw,
                            "normalized": clz_norm
                        }
                    },
                    "final_metrics": {
                        "fci_score": fci_final,
                        "consciousness_state": state.name if state else "UNKNOWN",
                        "coherence": coherence,
                        "diffusion_coefficient": diffusion_mean,
                        "entropy": entropy
                    }
                }

                # Raw data outputs
                if psi_dist is not None and isinstance(psi_dist, torch.Tensor):
                    psi_list = psi_dist.flatten().tolist()[:64]  # Primeiros 64 valores
                else:
                    psi_list = []

                if fractal_field is not None and isinstance(fractal_field, torch.Tensor):
                    field_list = fractal_field.flatten().tolist()[:64]  # Amostra
                else:
                    field_list = []

                response["raw_data_outputs"] = {
                    "psi_distribution": psi_list,
                    "fractal_field_sample": field_list
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
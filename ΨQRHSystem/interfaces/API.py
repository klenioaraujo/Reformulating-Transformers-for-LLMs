#!/usr/bin/env python3
"""
ΨQRH API Interface - Interface REST API

Serviço REST para o sistema ΨQRH modular.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from typing import Dict, Any, Optional
import os
import sys
from datetime import datetime

# Adicionar diretório base ao path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.dirname(BASE_DIR))

from ΨQRHSystem.configs.SystemConfig import SystemConfig
from ..core.PipelineManager import PipelineManager


class ΨQRHAPI:
    """
    API REST para o sistema ΨQRH
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Inicializa API

        Args:
            config: Configuração do sistema (opcional)
        """
        self.config = config or SystemConfig()
        self.pipeline = None
        self.app = None

    def initialize_pipeline(self):
        """
        Inicializa pipeline ΨQRH
        """
        if self.pipeline is None:
            print("🚀 Inicializando pipeline ΨQRH para API...")
            self.pipeline = PipelineManager(self.config)
            print("✅ Pipeline ΨQRH pronto para API!")

    def create_app(self) -> Flask:
        """
        Cria aplicação Flask

        Returns:
            Aplicação Flask configurada
        """
        self.app = Flask(__name__)
        CORS(self.app)  # Habilitar CORS

        # Inicializar pipeline
        self.initialize_pipeline()

        # Registrar rotas
        self._register_routes()

        return self.app

    def _register_routes(self):
        """
        Registra rotas da API
        """
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Verificação de saúde da API"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'service': 'ΨQRH API'
            })

        @self.app.route('/process', methods=['POST'])
        def process_text():
            """Processa texto através do pipeline ΨQRH"""
            try:
                data = request.get_json()

                if not data or 'text' not in data:
                    return jsonify({
                        'error': 'Campo "text" obrigatório',
                        'status': 'error'
                    }), 400

                text = data['text']
                options = data.get('options', {})

                print(f"🧠 API processando: '{text[:50]}...'")

                # Processar texto
                result = self.pipeline.process(text)

                # Adicionar metadados da API
                result['api_metadata'] = {
                    'processed_at': datetime.now().isoformat(),
                    'request_id': request.headers.get('X-Request-ID', 'unknown'),
                    'client_ip': request.remote_addr
                }

                return jsonify(result)

            except Exception as e:
                print(f"❌ Erro na API: {e}")
                return jsonify({
                    'error': str(e),
                    'status': 'error',
                    'timestamp': datetime.now().isoformat()
                }), 500

        @self.app.route('/info', methods=['GET'])
        def get_info():
            """Retorna informações do sistema"""
            try:
                info = self.pipeline.get_pipeline_status()
                info['api_info'] = {
                    'version': '1.0.0',
                    'service': 'ΨQRH Modular API',
                    'endpoints': [
                        '/health',
                        '/process',
                        '/info',
                        '/batch',
                        '/config'
                    ]
                }
                return jsonify(info)
            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500

        @self.app.route('/batch', methods=['POST'])
        def process_batch():
            """Processa lote de textos"""
            try:
                data = request.get_json()

                if not data or 'texts' not in data:
                    return jsonify({
                        'error': 'Campo "texts" obrigatório (lista de textos)',
                        'status': 'error'
                    }), 400

                texts = data['texts']
                if not isinstance(texts, list):
                    return jsonify({
                        'error': 'Campo "texts" deve ser uma lista',
                        'status': 'error'
                    }), 400

                print(f"📁 Processando lote via API: {len(texts)} textos")

                results = []
                for i, text in enumerate(texts):
                    try:
                        result = self.pipeline.process(text)
                        results.append({
                            'index': i,
                            'input': text,
                            'result': result,
                            'status': 'success'
                        })
                    except Exception as e:
                        results.append({
                            'index': i,
                            'input': text,
                            'error': str(e),
                            'status': 'error'
                        })

                return jsonify({
                    'status': 'completed',
                    'total_processed': len(results),
                    'results': results,
                    'timestamp': datetime.now().isoformat()
                })

            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500

        @self.app.route('/config', methods=['GET'])
        def get_config():
            """Retorna configuração atual (informações seguras)"""
            try:
                config_info = {
                    'model': {
                        'embed_dim': self.config.model.embed_dim,
                        'max_history': self.config.model.max_history,
                        'vocab_size': self.config.model.vocab_size
                    },
                    'physics': {
                        'I0': self.config.physics.I0,
                        'alpha': self.config.physics.alpha,
                        'beta': self.config.physics.beta,
                        'omega': self.config.physics.omega
                    },
                    'system': {
                        'device': self.config.device,
                        'enable_components': self.config.enable_components
                    }
                }
                return jsonify(config_info)
            except Exception as e:
                return jsonify({
                    'error': str(e),
                    'status': 'error'
                }), 500

    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """
        Executa servidor API

        Args:
            host: Host para bind
            port: Porta para bind
            debug: Modo debug
        """
        if self.app is None:
            self.create_app()

        print(f"🌐 Iniciando ΨQRH API em {host}:{port}")
        print(f"📋 Endpoints disponíveis:")
        print(f"   GET  /health - Verificação de saúde")
        print(f"   POST /process - Processar texto único")
        print(f"   GET  /info - Informações do sistema")
        print(f"   POST /batch - Processar lote de textos")
        print(f"   GET  /config - Configuração do sistema")
        print()

        self.app.run(host=host, port=port, debug=debug)


def create_app(config_path: Optional[str] = None) -> Flask:
    """
    Factory function para criar aplicação Flask

    Args:
        config_path: Caminho para configuração (opcional)

    Returns:
        Aplicação Flask
    """
    # Carregar configuração se fornecida
    config = None
    if config_path and os.path.exists(config_path):
        config = SystemConfig.from_yaml(config_path)

    # Criar API
    api = ΨQRHAPI(config)
    return api.create_app()


def main():
    """
    Função principal para executar API standalone
    """
    import argparse

    parser = argparse.ArgumentParser(description="ΨQRH API Server")
    parser.add_argument('--host', default='0.0.0.0', help='Host para bind (padrão: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Porta para bind (padrão: 5000)')
    parser.add_argument('--config', type=str, help='Arquivo de configuração YAML')
    parser.add_argument('--debug', action='store_true', help='Modo debug')

    args = parser.parse_args()

    try:
        # Criar e executar API
        api = ΨQRHAPI()

        if args.config:
            api.config = SystemConfig.from_yaml(args.config)

        api.run(host=args.host, port=args.port, debug=args.debug)

    except Exception as e:
        print(f"❌ Erro ao iniciar API: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
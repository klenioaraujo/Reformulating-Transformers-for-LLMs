#!/usr/bin/env python3
"""
Multi-Model Manager - Gerenciamento Completo de Múltiplos Modelos ΨQRH

Sistema central para gerenciamento de múltiplos modelos simultaneamente,
permitindo troca dinâmica, cache inteligente e otimização de recursos.
"""

import torch
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import threading
import yaml

from .ModelRegistry import ModelRegistry
from .DynamicModelLoader import DynamicModelLoader
from .AutoCalibration import AutoCalibration

class MultiModelManager:
    """
    Gerenciador multi-modelo para sistema ΨQRH

    Permite carregar, gerenciar e trocar entre múltiplos modelos
    simultaneamente com cache inteligente e otimização de memória.
    """

    def __init__(self, config_path: str = "../config/multi_model_config.yaml"):
        """
        Inicializa o gerenciador multi-modelo

        Args:
            config_path: Caminho para configuração multi-modelo
        """
        self.config = self._load_config(config_path)
        self.registry = ModelRegistry(self.config["model_management"]["model_registry_path"])
        self.loader = DynamicModelLoader(config_path)

        # ZERO CACHE: Removido sistema de cache
        self.active_model = None
        self.model_lock = threading.Lock()

        # Auto-calibração (desabilitada temporariamente para evitar erro)
        # self.auto_calibration = AutoCalibration(self.config)
        self.auto_calibration = None

        print("🚀 Multi-Model Manager inicializado")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carrega configuração multi-modelo"""
        config_file = Path("configs/multi_model_config.yaml")
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Configuração padrão"""
        return {
            "model_management": {
                "default_model": "gpt2",
                "max_loaded_models": 3,
                "memory_limit_gb": 8.0
            }
        }

    def load_model(self, model_name: str, set_active: bool = True) -> bool:
        """
        Carrega um modelo específico

        Args:
            model_name: Nome do modelo a carregar
            set_active: Se deve definir como modelo ativo

        Returns:
            True se carregado com sucesso, False caso contrário
        """
        with self.model_lock:
            try:
                print(f"🔄 Carregando modelo: {model_name}")

                # ZERO CACHE: Sempre carregar modelo do zero

                # Verificar limite de modelos carregados
                if len(self.loaded_models) >= self.config["model_management"]["max_loaded_models"]:
                    self._unload_least_recently_used()

                # Carregar modelo (ZERO CACHE: sempre do zero)
                model, info = self.loader.load_model(model_name)

                if model is not None:
                    # ZERO FALLBACK POLICY: Verificar se é modelo real
                    if info.get('status') == 'gpt2_simulated' or info.get('type') == 'gpt2_simulated':
                        raise RuntimeError(f"ZERO FALLBACK POLICY: Modelo simulado '{model_name}' rejeitado. Sistema deve usar apenas modelos reais.")

                    # ZERO CACHE: Não armazenar em cache
                    if set_active:
                        self.active_model = model_name

                    print(f"✅ Modelo {model_name} carregado com sucesso")
                    return True
                else:
                    print(f"❌ Falha ao carregar modelo {model_name}")
                    return False

            except Exception as e:
                print(f"❌ Erro ao carregar modelo {model_name}: {e}")
                return False

    def unload_model(self, model_name: str) -> bool:
        """
        Descarrega um modelo

        Args:
            model_name: Nome do modelo a descarregar

        Returns:
            True se descarregado com sucesso
        """
        with self.model_lock:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]

                if self.active_model == model_name:
                    self.active_model = None

                print(f"🗑️  Modelo {model_name} descarregado")
                return True

            return False

    def switch_to_model(self, model_name: str) -> bool:
        """
        Troca para um modelo específico

        Args:
            model_name: Nome do modelo

        Returns:
            True se troca realizada com sucesso
        """
        with self.model_lock:
            # ZERO CACHE: Sempre tentar carregar modelo do zero
            if self.load_model(model_name, set_active=True):
                return True

            # ZERO FALLBACK POLICY: Não tentar modelos simulados - falhar claramente
            raise RuntimeError(f"ZERO FALLBACK POLICY: Modelo '{model_name}' não encontrado. Sistema deve falhar claramente.")

    def get_active_model(self) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """
        Obtém o modelo ativo atual

        Returns:
            Tupla (modelo, informações) ou None se nenhum ativo
        """
        # ZERO CACHE: Sempre recarregar modelo ativo
        if self.active_model:
            try:
                model, info = self.loader.load_model(self.active_model)
                return model, info
            except:
                self.active_model = None

        return None

    def get_model_info(self, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Obtém informações de um modelo

        Args:
            model_name: Nome do modelo (usa ativo se None)

        Returns:
            Informações do modelo
        """
        target_model = model_name or self.active_model

        # ZERO CACHE: Sempre obter informações atualizadas
        if target_model:
            try:
                _, info = self.loader.load_model(target_model)
                return info
            except:
                pass

        return None

    def list_loaded_models(self) -> List[str]:
        """
        Lista modelos carregados

        Returns:
            Lista de nomes de modelos carregados (ZERO CACHE: sempre vazia)
        """
        # ZERO CACHE: Não há modelos em cache
        return []

    def list_available_models(self) -> List[str]:
        """
        Lista todos os modelos disponíveis

        Returns:
            Lista de nomes de modelos disponíveis
        """
        return self.registry.list_available_models()

    def get_system_status(self) -> Dict[str, Any]:
        """
        Obtém status completo do sistema multi-modelo

        Returns:
            Status detalhado do sistema
        """
        status = {
            "active_model": self.active_model,
            "loaded_models": {},  # ZERO CACHE: sempre vazio
            "available_models": self.list_available_models(),
            "registry_summary": self.registry.get_model_summary(),
            "memory_usage": {"total_parameters": 0, "estimated_size_mb": 0, "loaded_models_count": 0}  # ZERO CACHE
        }

        return status

    # ZERO CACHE: Função _unload_least_recently_used removida

    # ZERO CACHE: Função _estimate_memory_usage removida

    def scan_and_register_models(self):
        """Escaneia e registra modelos disponíveis"""
        print("🔍 Escaneando modelos disponíveis...")
        self.loader.scan_and_register_models()
        print("✅ Escaneamento concluído")

    def set_default_model(self, model_name: str):
        """
        Define o modelo padrão

        Args:
            model_name: Nome do modelo padrão
        """
        self.config["model_management"]["default_model"] = model_name
        self.loader.set_default_model(model_name)
        print(f"✅ Modelo padrão definido como: {model_name}")

    def preload_models(self, model_names: List[str]):
        """
        Pré-carrega múltiplos modelos

        Args:
            model_names: Lista de nomes de modelos
        """
        print(f"🔄 Pré-carregando {len(model_names)} modelos...")

        for name in model_names:
            self.load_model(name, set_active=False)

        print("✅ Pré-carregamento concluído")

    # ZERO CACHE: Função optimize_memory removida

    def process_with_model(self, model_name: str, input_text: str) -> Optional[Dict[str, Any]]:
        """
        Processa texto com um modelo específico

        Args:
            model_name: Nome do modelo
            input_text: Texto de entrada

        Returns:
            Resultado do processamento ou None se erro
        """
        try:
            # Garantir que o modelo está carregado e ativo
            if not self.switch_to_model(model_name):
                return None

            # Obter modelo ativo
            model_data = self.get_active_model()
            if not model_data:
                return None

            model, info = model_data

            # Simulação de processamento (adaptar conforme necessário)
            result = {
                "model_used": model_name,
                "input_text": input_text,
                "processed_at": datetime.now().isoformat(),
                "model_info": info
            }

            return result

        except Exception as e:
            print(f"❌ Erro ao processar com modelo {model_name}: {e}")
            return None

    def get_model_statistics(self) -> Dict[str, Any]:
        """
        Obtém estatísticas dos modelos

        Returns:
            Estatísticas detalhadas (ZERO CACHE: sempre vazias)
        """
        stats = {
            "total_loaded": 0,  # ZERO CACHE
            "active_model": self.active_model,
            "model_usage": {}  # ZERO CACHE
        }

        return stats
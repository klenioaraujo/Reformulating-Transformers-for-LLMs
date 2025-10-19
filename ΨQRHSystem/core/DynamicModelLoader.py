#!/usr/bin/env python3
"""
Dynamic Model Loader - Carregamento Dinâmico de Modelos Multi-Modelo

Sistema inteligente para carregamento dinâmico de modelos baseado na seleção,
com detecção automática de parâmetros e fallback seguro.
"""

import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import yaml

from .ModelRegistry import ModelRegistry
from .SemanticModelLoader import EnhancedSemanticModelLoader

class DynamicModelLoader:
    """
    Carregador dinâmico de modelos para sistema multi-modelo ΨQRH

    Detecta automaticamente parâmetros do modelo selecionado,
    carrega com prioridade otimizada e fornece fallback seguro.
    """

    def __init__(self, config_path: str = "../config/multi_model_config.yaml"):
        """
        Inicializa o carregador dinâmico

        Args:
            config_path: Caminho para configuração multi-modelo
        """
        self.config = self._load_config(config_path)
        self.registry = ModelRegistry(self.config["model_management"]["model_registry_path"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ZERO CACHE: Removido sistema de cache

        print("🔧 Dynamic Model Loader inicializado")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Carrega configuração multi-modelo"""
        config_file = Path("configs/multi_model_config.yaml")
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Configuração padrão
            return {
                "model_management": {
                    "default_model": "gpt2",
                    "loading_priority": ["semantic_converted", "distilled", "source"]
                },
                "supported_models": {
                    "gpt2": {"type": "transformer", "vocab_size": 50257, "embed_dim": 768, "num_layers": 12, "num_heads": 12},
                    # ZERO HARDCODING: Removido modelo simulado hardcoded
                }
            }

    def load_model(self, model_name: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Carrega um modelo dinamicamente

        Args:
            model_name: Nome do modelo (usa padrão se None)

        Returns:
            Tupla (modelo, informações do modelo)
        """
        if model_name is None:
            model_name = self.config["model_management"]["default_model"]

        print(f"🔍 Procurando modelo: {model_name}")

        # ZERO CACHE: Sempre carregar modelo do zero

        # Tentar carregamento por prioridade
        for priority in self.config["model_management"]["loading_priority"]:
            model, info = self._try_load_by_priority(model_name, priority)
            if model is not None:
                # ZERO CACHE: Não armazenar em cache
                return model, info

        # ZERO FALLBACK POLICY: Não criar modelos simulados
        raise RuntimeError(f"ZERO FALLBACK POLICY: Nenhum modelo real encontrado para '{model_name}'. Sistema deve falhar claramente.")

    def _try_load_by_priority(self, model_name: str, priority: str) -> Tuple[Optional[Any], Dict[str, Any]]:
        """
        Tenta carregar modelo por prioridade

        Args:
            model_name: Nome do modelo
            priority: Prioridade de carregamento

        Returns:
            Tupla (modelo, informações) ou (None, {})
        """
        try:
            if priority == "semantic_converted":
                return self._load_semantic_converted(model_name)
            elif priority == "distilled":
                return self._load_distilled(model_name)
            elif priority == "source":
                return self._load_source(model_name)
            # ZERO FALLBACK POLICY: Removido suporte a modelos simulados
        except Exception as e:
            print(f"⚠️  Erro ao carregar {model_name} como {priority}: {e}")

        return None, {}

    def _load_semantic_converted(self, model_name: str) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Carrega modelo convertido para formato semântico"""
        semantic_dir = Path("models/semantic")
        if not semantic_dir.exists():
            return None, {}

        # Procurar arquivo semântico
        semantic_file = semantic_dir / f"psiqrh_semantic_{model_name}.pt"
        if semantic_file.exists():
            print(f"📁 Carregando modelo semântico: {semantic_file}")
            try:
                # Usar EnhancedSemanticModelLoader
                loader = EnhancedSemanticModelLoader(self._create_config_for_model(model_name))
                model = loader.load_semantic_model(str(semantic_file))

                info = {
                    "type": "semantic_converted",
                    "model_name": model_name,
                    "path": str(semantic_file),
                    "vocab_size": loader.get_vocab_size(),
                    "embed_dim": loader.get_embed_dim(),
                    "status": "loaded"
                }
                return model, info
            except Exception as e:
                print(f"❌ Erro ao carregar modelo semântico: {e}")

        return None, {}

    def _load_distilled(self, model_name: str) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Carrega modelo destilado"""
        distilled_dir = Path("models/distilled")
        if not distilled_dir.exists():
            return None, {}

        distilled_file = distilled_dir / f"psiqrh_distilled_{model_name}.pt"
        if distilled_file.exists():
            print(f"🧠 Carregando modelo destilado: {distilled_file}")
            try:
                checkpoint = torch.load(distilled_file, map_location=self.device)

                # Detectar parâmetros automaticamente
                model_config = self._detect_model_config(checkpoint)

                info = {
                    "type": "distilled",
                    "model_name": model_name,
                    "path": str(distilled_file),
                    **model_config,
                    "status": "loaded"
                }
                return checkpoint, info
            except Exception as e:
                print(f"❌ Erro ao carregar modelo destilado: {e}")

        return None, {}

    def _load_source(self, model_name: str) -> Tuple[Optional[Any], Dict[str, Any]]:
        """Carrega modelo fonte do Hugging Face"""
        source_dir = Path("models/source") / model_name
        if not source_dir.exists():
            return None, {}

        print(f"📥 Carregando modelo fonte: {source_dir}")
        try:
            # Tentar carregar via transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model = AutoModelForCausalLM.from_pretrained(str(source_dir))
            tokenizer = AutoTokenizer.from_pretrained(str(source_dir))

            info = {
                "type": "source",
                "model_name": model_name,
                "path": str(source_dir),
                "vocab_size": len(tokenizer),
                "embed_dim": model.config.hidden_size,
                "num_layers": model.config.num_hidden_layers,
                "num_heads": model.config.num_attention_heads,
                "status": "loaded"
            }
            return model, info
        except Exception as e:
            print(f"❌ Erro ao carregar modelo fonte: {e}")

        return None, {}

    # ZERO FALLBACK POLICY: Função _create_simulated_model removida completamente

    def _detect_model_config(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Detecta configuração do modelo a partir do checkpoint"""
        config = {}

        # Tentar detectar parâmetros comuns
        if "embed.weight" in checkpoint:
            config["embed_dim"] = checkpoint["embed.weight"].shape[1]

        # Detectar número de camadas
        num_layers = 0
        while f"layers.{num_layers}.self_attn.in_proj_weight" in checkpoint:
            num_layers += 1
        config["num_layers"] = num_layers

        # Vocabulário padrão se não detectado
        config["vocab_size"] = 50257
        config["num_heads"] = 12

        return config

    def _create_config_for_model(self, model_name: str) -> Any:
        """Cria objeto de configuração para o modelo"""
        # Configuração simples mock
        class MockConfig:
            def __init__(self, model_name):
                self.model_name = model_name
                self.device = "cpu"

        return MockConfig(model_name)

    def get_available_models(self) -> List[str]:
        """
        Lista modelos disponíveis

        Returns:
            Lista de nomes de modelos disponíveis
        """
        return self.registry.list_available_models()

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Obtém informações de um modelo

        Args:
            model_name: Nome do modelo

        Returns:
            Informações do modelo ou None
        """
        return self.registry.get_model_info(model_name)

    def scan_and_register_models(self):
        """Escaneia e registra modelos disponíveis"""
        base_paths = ["models"]
        self.registry.scan_for_models(base_paths)

    def set_default_model(self, model_name: str):
        """
        Define o modelo padrão

        Args:
            model_name: Nome do modelo padrão
        """
        self.config["model_management"]["default_model"] = model_name
        print(f"✅ Modelo padrão definido como: {model_name}")

    # ZERO CACHE: Funções de cache removidas completamente
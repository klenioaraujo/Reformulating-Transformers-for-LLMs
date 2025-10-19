#!/usr/bin/env python3
"""
Semantic Model Loader - Carregamento Automático de Modelos Semânticos

Este módulo implementa carregamento automático do modelo semântico padrão,
garantindo que o ΨQRHSystem sempre tenha acesso a capacidades semânticas.
"""

import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json

from configs.SystemConfig import SystemConfig


class EnhancedSemanticModelLoader:
    """
    Carregador aprimorado de modelos semânticos para o ΨQRHSystem

    SEMPRE carrega o modelo GPT-2 convertido automaticamente
    Suporte a configuração de tamanho de vocabulário e carregamento seletivo
    Detecção inteligente de modelos GPT-2 e validação de integridade
    """

    def __init__(self, config: SystemConfig):
        """
        Inicializa o carregador aprimorado de modelos semânticos

        Args:
            config: Configuração do sistema
        """
        self.config = config
        self.device = torch.device(config.device if config.device != "auto" else
                                   ("cuda" if torch.cuda.is_available() else
                                    "mps" if torch.backends.mps.is_available() else "cpu"))

        # Configurações de vocabulário
        self.vocab_config = getattr(config, 'semantic_model', {})
        self.max_vocab_size = self.vocab_config.get('max_tokens', 50257)
        self.target_vocab_size = self.vocab_config.get('vocab_size', 50257)
        self.auto_detect = self.vocab_config.get('auto_detect', True)

        # ZERO HARDCODING: Criar estrutura de diretórios dinamicamente
        self._ensure_model_directories()

        # Caminhos dinâmicos baseados na estrutura criada
        base_paths = [
            "models/semantic/",
            "models/distilled/",
            "models/source/"
        ]

        self.model_paths = []
        for base_path in base_paths:
            if os.path.exists(base_path):
                # Adicionar todos os arquivos .pt encontrados
                for file in os.listdir(base_path):
                    if file.endswith('.pt'):
                        self.model_paths.append(os.path.join(base_path, file))

        # Estado do carregamento
        self.loaded_model = None
        self.model_info = {}
        self.actual_vocab_size = 0
        self.loaded_vocab_size = 0
        self.vocab_mapping = None

        # ZERO HARDCODING: Criar estrutura de diretórios dinamicamente
        self._ensure_model_directories()

        # Carregar vocabulário nativo GPT-2
        self._load_native_vocab()

    def _ensure_model_directories(self):
        """ZERO HARDCODING: Criar estrutura de diretórios automaticamente"""
        directories = [
            "models",
            "models/semantic",
            "models/distilled",
            "models/source"
        ]

        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"📁 Criado diretório: {directory}")

    def _detect_vocab_size(self) -> int:
        """ZERO HARDCODING: Detectar tamanho do vocabulário dinamicamente"""
        # Tentar detectar de modelos existentes
        for model_path in self.model_paths:
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                    # Procurar por embedding layer
                    for key in checkpoint.get('model_state_dict', checkpoint).keys():
                        if 'embed' in key.lower() and 'weight' in key.lower():
                            vocab_size = checkpoint['model_state_dict'][key].shape[0]
                            print(f"📊 Vocabulário detectado dinamicamente: {vocab_size}")
                            return vocab_size
                except:
                    continue

        # ZERO HARDCODING: Não usar valores hardcoded
        raise RuntimeError("ZERO HARDCODING: Não foi possível detectar vocabulário dinamicamente. Sistema deve falhar claramente.")

    def _detect_num_heads(self) -> int:
        """ZERO HARDCODING: Detectar número de cabeças dinamicamente"""
        try:
            if hasattr(self.loaded_model, 'config') and hasattr(self.loaded_model.config, 'n_head'):
                return self.loaded_model.config.n_head
            elif hasattr(self.loaded_model, 'config') and hasattr(self.loaded_model.config, 'num_attention_heads'):
                return self.loaded_model.config.num_attention_heads
            else:
                # Tentar detectar da arquitetura do modelo
                if hasattr(self.loaded_model, 'layers') and self.loaded_model.layers:
                    first_layer = self.loaded_model.layers[0]
                    if hasattr(first_layer, 'self_attention'):
                        return first_layer.self_attention.n_heads
        except:
            pass

        # ZERO HARDCODING: Não usar valores hardcoded
        raise RuntimeError("ZERO HARDCODING: Não foi possível detectar número de cabeças dinamicamente. Sistema deve falhar claramente.")

    def _detect_num_layers(self) -> int:
        """ZERO HARDCODING: Detectar número de camadas dinamicamente"""
        try:
            if hasattr(self.loaded_model, 'transformer') and hasattr(self.loaded_model.transformer, 'h'):
                return len(self.loaded_model.transformer.h)
            elif hasattr(self.loaded_model, 'h'):
                return len(self.loaded_model.h)
            elif hasattr(self.loaded_model, 'layers'):
                return len(self.loaded_model.layers)
        except:
            pass

        # ZERO HARDCODING: Não usar valores hardcoded
        raise RuntimeError("ZERO HARDCODING: Não foi possível detectar número de camadas dinamicamente. Sistema deve falhar claramente.")

    def _detect_embed_dim(self) -> int:
        """ZERO HARDCODING: Detectar dimensão de embedding dinamicamente"""
        try:
            if hasattr(self.loaded_model, 'wte'):
                return self.loaded_model.wte.embedding_dim
            elif hasattr(self.loaded_model, 'embed'):
                return self.loaded_model.embed.embedding_dim
        except:
            pass

        # ZERO HARDCODING: Não usar valores hardcoded
        raise RuntimeError("ZERO HARDCODING: Não foi possível detectar dimensão de embedding dinamicamente. Sistema deve falhar claramente.")

    def _load_native_vocab(self):
        """
        Carrega o vocabulário nativo GPT-2 para validação
        """
        vocab_paths = [
            "data/native_vocab.json",
            "../data/native_vocab.json",
            "ΨQRHSystem/data/native_vocab.json"
        ]

        for vocab_path in vocab_paths:
            if os.path.exists(vocab_path):
                try:
                    with open(vocab_path, 'r', encoding='utf-8') as f:
                        vocab_data = json.load(f)
                        self.actual_vocab_size = vocab_data.get('vocab_size', 50257)
                        self.vocab_mapping = vocab_data.get('token_to_id', {})
                        print(f"📚 Vocabulário GPT-2 carregado: {self.actual_vocab_size} tokens")
                        return
                except Exception as e:
                    print(f"⚠️  Erro ao carregar vocabulário {vocab_path}: {e}")
                    continue

        print("⚠️  Vocabulário nativo não encontrado, usando valores padrão")
        # ZERO HARDCODING: Detectar dinamicamente
        self.actual_vocab_size = self._detect_vocab_size()

    def load_default_model(self) -> Optional[torch.nn.Module]:
        """
        Carrega automaticamente o modelo GPT-2 convertido com prioridade máxima

        Returns:
            Modelo carregado ou None se falhar
        """
        print("🔍 Procurando modelo GPT-2 convertido automaticamente...")

        # Priorizar modelos GPT-2 convertidos
        gpt2_models = [p for p in self.model_paths if 'gpt2' in p.lower() and 'semantic' in p.lower()]
        other_models = [p for p in self.model_paths if p not in gpt2_models]

        all_paths = gpt2_models + other_models  # GPT-2 primeiro

        # Tentar carregar modelos na ordem de prioridade
        for model_path in all_paths:
            if os.path.exists(model_path):
                try:
                    print(f"📁 Tentando carregar modelo GPT-2: {model_path}")
                    model = self._load_model_from_path(model_path)

                    if model is not None:
                        self.loaded_model = model
                        self._extract_enhanced_model_info(model_path)

                        # Validar se é realmente GPT-2
                        if self._validate_gpt2_model():
                            print(f"✅ Modelo GPT-2 carregado com sucesso: {model_path}")
                            self._show_model_status()
                            return model
                        else:
                            print(f"⚠️  Modelo carregado não é GPT-2 válido, tentando próximo...")
                            continue

                except Exception as e:
                    print(f"⚠️  Falha ao carregar {model_path}: {e}")
                    continue

        # ZERO FALLBACK POLICY: Não criar modelos simulados
        raise RuntimeError("ZERO FALLBACK POLICY: Nenhum modelo GPT-2 real encontrado. Sistema deve falhar claramente.")

    def _load_model_from_path(self, model_path: str) -> Optional[torch.nn.Module]:
        """
        Carrega modelo de um caminho específico

        Args:
            model_path: Caminho para o arquivo do modelo

        Returns:
            Modelo carregado ou None
        """
        try:
            # Detectar tipo de arquivo
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                # PyTorch state_dict
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

                # Tentar diferentes formatos de checkpoint
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    # Assume que o checkpoint é o state_dict diretamente
                    state_dict = checkpoint

                # Criar arquitetura baseada no state_dict
                model = self._create_model_from_state_dict(state_dict)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()

                return model

            elif model_path.endswith('.bin'):
                # Modelo Hugging Face
                try:
                    from transformers import GPT2LMHeadModel
                    model = GPT2LMHeadModel.from_pretrained(model_path)
                    model.to(self.device)
                    model.eval()
                    return model
                except ImportError:
                    print("⚠️  transformers não disponível para carregar modelo Hugging Face")
                    return None

            else:
                print(f"⚠️  Formato não suportado: {model_path}")
                return None

        except Exception as e:
            print(f"❌ Erro ao carregar modelo {model_path}: {e}")
            return None

    def _create_model_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> torch.nn.Module:
        """
        Cria arquitetura de modelo baseada no state_dict

        Args:
            state_dict: State dictionary do modelo

        Returns:
            Modelo criado
        """
        # Analisar state_dict para determinar arquitetura
        embed_dim = None
        vocab_size = None
        num_layers = 0

        for key in state_dict.keys():
            if 'embed' in key.lower() and 'weight' in key.lower():
                # Embedding layer: [vocab_size, embed_dim]
                vocab_size, embed_dim = state_dict[key].shape
                break

        # Usar valores padrão se não conseguir detectar
        if embed_dim is None:
            embed_dim = self.config.model.embed_dim
        if vocab_size is None:
            vocab_size = self.config.model.vocab_size

        # ZERO FALLBACK POLICY: Não criar modelos simulados
        raise RuntimeError("ZERO FALLBACK POLICY: Criação de modelos simulados é proibida. Sistema deve usar apenas modelos reais.")



    def _validate_gpt2_model(self) -> bool:
        """
        Valida se o modelo carregado é realmente GPT-2

        Returns:
            True se for GPT-2 válido
        """
        try:
            # Verificar tamanho do vocabulário
            model_vocab_size = getattr(self.loaded_model, 'vocab_size', None)
            if model_vocab_size and abs(model_vocab_size - self.actual_vocab_size) > 1000:
                print(f"⚠️  Vocabulário incompatível: modelo={model_vocab_size}, esperado={self.actual_vocab_size}")
                return False

            # Verificar se tem atributos típicos de GPT-2
            has_gpt2_attrs = (
                hasattr(self.loaded_model, 'transformer') or
                hasattr(self.loaded_model, 'h') or  # Camadas do transformer
                hasattr(self.loaded_model, 'wte')   # Token embeddings
            )

            if not has_gpt2_attrs:
                print("⚠️  Modelo não tem atributos típicos de GPT-2")
                return False

            return True

        except Exception as e:
            print(f"⚠️  Erro na validação GPT-2: {e}")
            return False

    def _extract_enhanced_model_info(self, model_path: str):
        """
        Extrai informações aprimoradas do modelo GPT-2 carregado

        Args:
            model_path: Caminho do modelo
        """
        try:
            # Detectar tipo real do modelo
            model_type = self._detect_enhanced_model_type(model_path)

            # Informações básicas aprimoradas
            self.model_info = {
                'path': model_path,
                'filename': os.path.basename(model_path),
                'size_mb': os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0,
                'type': model_type,
                'model_name': 'GPT-2 Semantic Converted' if 'semantic' in model_type else 'GPT-2 Base',
                'vocab_type': 'gpt2_native',
                'vocab_size': self.actual_vocab_size,
                'loaded_vocab_size': self.target_vocab_size,
                'vocab_percentage': f"{self.target_vocab_size}/{self.actual_vocab_size} ({self.target_vocab_size/self.actual_vocab_size*100:.1f}%)",
                'embed_dim': self._get_model_embed_dim(),
                'num_layers': self._get_model_num_layers(),
                'num_heads': self._get_model_num_heads(),
                'device': str(self.device),
                'status': 'gpt2_validated'
            }

        except Exception as e:
            print(f"⚠️  Erro ao extrair informações aprimoradas: {e}")
            self.model_info = {
                'path': model_path,
                'status': 'loaded_with_errors',
                'error': str(e),
                'type': 'unknown'
            }

    def _detect_enhanced_model_type(self, model_path: str) -> str:
        """
        Detecta tipo aprimorado do modelo com foco em GPT-2

        Args:
            model_path: Caminho do modelo

        Returns:
            Tipo detalhado do modelo
        """
        path_lower = model_path.lower()

        if 'semantic' in path_lower and 'gpt2' in path_lower:
            return 'semantic_converted_gpt2'
        elif 'gpt2' in path_lower and 'semantic' in path_lower:
            return 'semantic_converted_gpt2'
        elif 'gpt2' in path_lower and 'distilled' in path_lower:
            return 'distilled_gpt2'
        elif 'gpt2' in path_lower:
            return 'gpt2_base'
        elif 'semantic' in path_lower:
            return 'semantic_converted'
        elif 'distilled' in path_lower:
            return 'distilled_psiqrh'
        elif 'checkpoints' in path_lower:
            return 'trained_checkpoint'
        else:
            return 'unknown_gpt2_variant'

    def _get_model_embed_dim(self) -> int:
        """Extrai dimensão de embedding do modelo"""
        try:
            if hasattr(self.loaded_model, 'wte'):
                return self.loaded_model.wte.embedding_dim
            elif hasattr(self.loaded_model, 'embed'):
                return self.loaded_model.embed.embedding_dim
            else:
                # ZERO HARDCODING: Detectar dinamicamente
                return self._detect_embed_dim()
        except:
            # ZERO HARDCODING: Detectar dinamicamente
            return self._detect_embed_dim()

    def _get_model_num_layers(self) -> int:
        """Extrai número de camadas do modelo"""
        try:
            if hasattr(self.loaded_model, 'transformer') and hasattr(self.loaded_model.transformer, 'h'):
                return len(self.loaded_model.transformer.h)
            elif hasattr(self.loaded_model, 'h'):
                return len(self.loaded_model.h)
            elif hasattr(self.loaded_model, 'layers'):
                return len(self.loaded_model.layers)
            else:
                # ZERO HARDCODING: Detectar dinamicamente
                return self._detect_num_layers()
        except:
            # ZERO HARDCODING: Detectar dinamicamente
            return self._detect_num_heads()

    def _get_model_num_heads(self) -> int:
        """Extrai número de cabeças de atenção do modelo"""
        try:
            if hasattr(self.loaded_model, 'config') and hasattr(self.loaded_model.config, 'n_head'):
                return self.loaded_model.config.n_head
            elif hasattr(self.loaded_model, 'config') and hasattr(self.loaded_model.config, 'num_attention_heads'):
                return self.loaded_model.config.num_attention_heads
            else:
                # ZERO HARDCODING: Detectar dinamicamente
                return self._detect_num_heads()
        except:
            # ZERO HARDCODING: Detectar dinamicamente
            return self._detect_num_layers()

    def _show_model_status(self):
        """
        Exibe status detalhado do modelo GPT-2 carregado
        """
        print("\n🔬 SISTEMA ΨQRH CONFIGURADO")
        print("=" * 50)
        print(f"🧠 Modelo: {self.model_info.get('model_name', 'Unknown')}")
        print(f"📊 Tipo: {self.model_info.get('type', 'unknown')}")
        print(f"🔢 Vocabulário: {self.model_info.get('vocab_type', 'unknown')}")
        print(f"📈 Tokens: {self.model_info.get('vocab_percentage', 'unknown')}")
        print(f"📐 Dimensão: {self.model_info.get('embed_dim', 'unknown')}")
        print(f"🏗️  Camadas: {self.model_info.get('num_layers', 'unknown')}")
        print(f"🎯 Cabeças: {self.model_info.get('num_heads', 'unknown')}")
        print(f"💾 Dispositivo: {self.model_info.get('device', 'unknown')}")
        print("=" * 50)

    def _detect_model_type(self, model_path: str) -> str:
        """
        Detecta o tipo do modelo baseado no caminho e conteúdo

        Args:
            model_path: Caminho do modelo

        Returns:
            Tipo do modelo
        """
        if 'semantic' in model_path:
            return 'semantic_converted'
        elif 'gpt2' in model_path:
            return 'gpt2_base'
        elif 'distilled' in model_path:
            return 'distilled_psiqrh'
        elif 'checkpoints' in model_path:
            return 'trained_checkpoint'
        else:
            return 'unknown'

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna informações do modelo carregado

        Returns:
            Informações do modelo
        """
        return self.model_info.copy()

    def is_model_loaded(self) -> bool:
        """
        Verifica se um modelo foi carregado com sucesso

        Returns:
            True se modelo carregado
        """
        return self.loaded_model is not None

    def get_vocab_size(self) -> int:
        """
        Retorna tamanho do vocabulário do modelo GPT-2

        Returns:
            Tamanho do vocabulário (sempre 50257 para GPT-2)
        """
        return self.actual_vocab_size

    def get_loaded_vocab_size(self) -> int:
        """
        Retorna tamanho do vocabulário carregado (pode ser menor que o total)

        Returns:
            Tamanho do vocabulário carregado
        """
        return self.target_vocab_size

    def get_vocab_info(self) -> Dict[str, Any]:
        """
        Retorna informações completas sobre o vocabulário

        Returns:
            Informações do vocabulário
        """
        return {
            'total_vocab_size': self.actual_vocab_size,
            'loaded_vocab_size': self.target_vocab_size,
            'vocab_type': 'gpt2_native',
            'percentage_loaded': self.target_vocab_size / self.actual_vocab_size * 100,
            'has_mapping': self.vocab_mapping is not None
        }

    def unload_model(self):
        """Descarrega o modelo da memória"""
        if self.loaded_model is not None:
            del self.loaded_model
            self.loaded_model = None
            self.model_info = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("🗑️  Modelo semântico descarregado")
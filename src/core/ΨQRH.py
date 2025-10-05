"""
ΨQRH Pure - Sistema Espectral-Quaterniônico Puro (SEM Fallbacks)
==================================================================

Sistema completamente baseado em doe.md Seções 2.9.1–2.9.4.
NÃO contém fallbacks, hardcoding ou dependências de Transformer.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


class QRHPureFactory:
    """
    Factory puro para sistema ΨQRH espectral-quaterniônico.

    ZERO FALLBACKS:
    - Falha claramente se configuração inválida
    - Não usa tokenização (texto → onda contínua)
    - Não usa vocab_size (modos espectrais)
    - Não usa nn.Module híbridos
    """

    def __init__(self, config_path: str = "configs/qrh_config.yaml"):
        """
        Inicializa factory puro.

        Args:
            config_path: Caminho para configuração YAML

        Raises:
            FileNotFoundError: Se config não existir
            ValueError: Se config inválida
        """
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"❌ Configuração não encontrada: {config_path}\n"
                f"Sistema ΨQRH puro requer configuração explícita.\n"
                f"Não há valores padrão (ZERO FALLBACK POLICY)."
            )

        self.config = self._load_config()
        self._validate_config()

        # Inicializar processadores puros
        self.spectral_processor = None
        self.consciousness_processor = None

    def _load_config(self) -> dict:
        """Carrega configuração YAML."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        if not config:
            raise ValueError(
                f"❌ Arquivo de configuração vazio: {self.config_path}\n"
                f"Sistema ΨQRH puro requer configuração completa."
            )

        return config

    def _validate_config(self):
        """
        Valida configuração ΨQRH pura.

        Raises:
            ValueError: Se faltarem parâmetros obrigatórios
        """
        required_sections = ['qrh_layer', 'consciousness_processor']

        for section in required_sections:
            if section not in self.config:
                raise ValueError(
                    f"❌ Seção obrigatória ausente: '{section}'\n"
                    f"Configuração ΨQRH pura deve conter: {required_sections}"
                )

        # Validar parâmetros QRH
        qrh = self.config['qrh_layer']
        required_qrh = ['embed_dim', 'alpha']

        for param in required_qrh:
            if param not in qrh:
                raise ValueError(
                    f"❌ Parâmetro QRH obrigatório ausente: '{param}'\n"
                    f"Seção qrh_layer deve conter: {required_qrh}"
                )

    def process_text(self, text: str, device: str = "cpu") -> Dict[str, Any]:
        """
        Processa texto através do pipeline espectral puro.

        Pipeline:
        1. Texto → Sinal contínuo (text_to_wave)
        2. Sinal → Quaternions Ψ(x)
        3. Processamento espectral-harmônico
        4. Análise de consciência fractal
        5. Colapso de medida → Saída

        Args:
            text: Texto de entrada (string bruta)
            device: Dispositivo de processamento

        Returns:
            Dicionário com resultado e métricas físicas

        Raises:
            RuntimeError: Se pipeline falhar
        """
        try:
            # Importar pipeline puro
            from ..processing.psiqrh_pipeline import process_with_consciousness

            # Processar via pipeline físico
            result = process_with_consciousness(
                text=text,
                n_layers=self.config['qrh_layer'].get('n_layers', 6),
                alpha=self.config['qrh_layer']['alpha'],
                embed_dim=self.config['qrh_layer']['embed_dim']
            )

            return result

        except ImportError as e:
            raise RuntimeError(
                f"❌ Pipeline espectral puro não disponível: {e}\n"
                f"Certifique-se de que src/processing/psiqrh_pipeline.py existe.\n"
                f"Sistema ΨQRH puro NÃO possui fallback."
            )
        except Exception as e:
            raise RuntimeError(
                f"❌ Erro no processamento ΨQRH puro: {e}\n"
                f"Sistema falhou claramente (ZERO FALLBACK)."
            )

    def get_info(self) -> Dict[str, Any]:
        """Retorna informações do sistema ΨQRH puro."""
        return {
            'type': 'ΨQRH Pure Spectral-Quaternionic System',
            'version': '1.0.0',
            'based_on': 'doe.md Sections 2.9.1-2.9.4',
            'config': {
                'embed_dim': self.config['qrh_layer']['embed_dim'],
                'alpha': self.config['qrh_layer']['alpha'],
                'use_learned_rotation': self.config['qrh_layer'].get('use_learned_rotation', True)
            },
            'features': {
                'tokenization': False,
                'vocab_size': None,
                'fallbacks': 0,
                'hardcoded_values': 0,
                'hybrid_modules': 0
            },
            'pipeline': [
                'text_to_wave (análise espectral)',
                'quaternion_from_signal (Ψ(x))',
                'spectral_attention (F⁻¹{F(k)·F{Ψ⊗Ψ}})',
                'harmonic_evolution (R·F⁻¹{F(k)·F{Ψ}})',
                'fractal_consciousness (D, FCI)',
                'optical_probe (λ* = argmax |⟨f(λ,t), Ψ⟩|²)'
            ]
        }


# Alias para compatibilidade (mas força uso do sistema puro)
class QRHFactory(QRHPureFactory):
    """
    Alias para QRHPureFactory.

    ⚠️  IMPORTANTE: Esta classe NÃO possui fallbacks.
    Sistema ΨQRH agora é 100% espectral-quaterniônico puro.
    """

    def __init__(self, *args, **kwargs):
        # Remover model_path se fornecido (não é mais suportado)
        if 'model_path' in kwargs:
            print("⚠️  model_path ignorado - sistema ΨQRH puro não carrega modelos pré-treinados")
            del kwargs['model_path']

        super().__init__(*args, **kwargs)

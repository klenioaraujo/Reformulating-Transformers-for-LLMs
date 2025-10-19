#!/usr/bin/env python3
"""
ConfigManager - Gerenciador Central de Configurações para o Sistema ΨQRH
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """
    Carrega, valida e fornece acesso a todas as configurações do sistema.
    """

    def __init__(self, base_path: str = ".", default_config_path: str = "configs/config.yaml"):
        """
        Inicializa o gerenciador de configurações.

        Args:
            base_path: O caminho base do projeto.
            default_config_path: Caminho para o arquivo de configuração principal.
        """
        self.base_path = Path(base_path).resolve()
        self.config_path = self.base_path / default_config_path
        
        self.config = self._load_config_file(self.config_path)
        self.pipeline_config = self._load_config_file(self.base_path / "configs/pipeline_config.yaml")
        self.model_config = self._load_config_file(self.base_path / "configs/model_config.yaml") # Exemplo
        
        if not self.config:
            raise FileNotFoundError(f"Arquivo de configuração principal não encontrado em {self.config_path}")

        print("✅ ConfigManager inicializado com sucesso.")

    def _load_config_file(self, path: Path) -> Optional[Dict[str, Any]]:
        """
        Carrega um único arquivo de configuração YAML.
        """
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"⚠️  Erro ao carregar o arquivo de configuração {path}: {e}")
                return None
        return None

    def get_config(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Obtém um valor de uma seção de configuração.

        Args:
            section: A seção principal da configuração (ex: 'qrh_layer').
            key: A chave específica dentro da seção. Se None, retorna a seção inteira.
            default: Valor padrão a ser retornado se a chave não for encontrada.

        Returns:
            O valor da configuração ou o valor padrão.
        """
        section_data = self.config.get(section, {})
        if key:
            return section_data.get(key, default)
        return section_data

    def get_pipeline_config(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Obtém um valor da configuração do pipeline.
        """
        if not self.pipeline_config:
            return default
            
        section_data = self.pipeline_config.get(section, {})
        if key:
            return section_data.get(key, default)
        return section_data

    def get_full_config(self) -> Dict[str, Any]:
        """
        Retorna o dicionário de configuração completo.
        """
        return self.config

    def get_full_pipeline_config(self) -> Optional[Dict[str, Any]]:
        """
        Retorna o dicionário de configuração do pipeline completo.
        """
        return self.pipeline_config

    def get_vocab_path(self) -> str:
        """
        Retorna o caminho para o arquivo de vocabulário usando a lógica do sistema legado.
        Implementa a seleção multi-tier com fallback para native_vocab.json
        """
        import os
        from pathlib import Path

        # Multi-tier fallback approach from legacy system
        base_dir = Path(self.base_path).resolve()

        # Lista de caminhos de vocabulário em ordem de preferência
        vocab_paths = [
            base_dir / "models" / "gpt2_full_spectral_embeddings" / "vocab.json",
            base_dir / "models" / "source" / "gpt2" / "vocab.json",
            base_dir / "data" / "native_vocab.json",
            base_dir / "dynamic_quantum_vocabulary.json",
        ]

        # Adicionar caminhos relativos ao diretório atual
        current_dir = Path.cwd()
        vocab_paths.extend([
            current_dir / "models" / "gpt2_full_spectral_embeddings" / "vocab.json",
            current_dir / "models" / "source" / "gpt2" / "vocab.json",
            current_dir / "data" / "native_vocab.json",
            current_dir / "dynamic_quantum_vocabulary.json",
        ])

        # Buscar o primeiro arquivo que existe
        for vocab_path in vocab_paths:
            if vocab_path.exists():
                print(f"✅ Vocabulário selecionado: {vocab_path}")
                return str(vocab_path)

        # Fallback final - usar native_vocab.json se não existir
        fallback_path = base_dir / "data" / "native_vocab.json"
        print(f"⚠️  Nenhum arquivo de vocabulário encontrado, usando fallback: {fallback_path}")
        return str(fallback_path)

# Exemplo de uso (para teste)
if __name__ == '__main__':
    try:
        # Supondo que você execute este script do diretório raiz do projeto
        config_manager = ConfigManager(base_path=".")
        
        embed_dim = config_manager.get_pipeline_config("quantum_matrix", "embed_dim", 64)
        print(f"Dimensão do Embedding (do pipeline_config.yaml): {embed_dim}")

        model_name = config_manager.get_config("model", "name", "default_model")
        print(f"Nome do Modelo (do config.yaml): {model_name}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Ocorreu um erro: {e}")

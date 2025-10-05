#!/usr/bin/env python3
"""
Configuration loader for ΨQRH Transformer examples
Loads configurations directly from example_configs.yaml
"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Carrega configuração de arquivo YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_example_config(example_name: str, **override_params) -> Dict[str, Any]:
    """
    Obtém configuração para exemplo específico usando apenas example_configs.yaml

    Args:
        example_name: Nome do script de exemplo (ex: "basic_usage.py")
        **override_params: Parâmetros para sobrescrever na configuração

    Returns:
        Configuração completa para o exemplo
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_data = load_config(os.path.join(base_dir, "configs", "example_configs.yaml"))

    # Buscar configuração específica do exemplo
    example_configs = config_data.get("example_configs", {})

    # Extrair nome base do arquivo (sem .py)
    example_key = example_name.replace(".py", "")

    if example_key not in example_configs:
        raise ValueError(f"Nenhuma configuração encontrada para exemplo: {example_name}")

    # Obter configuração do exemplo
    example_config = example_configs[example_key]

    # Construir configuração completa
    config = {}

    # Adicionar parâmetros do modelo
    if "model" in example_config:
        config.update(example_config["model"])

    # Adicionar parâmetros de validação
    if "validation" in example_config:
        config.update(example_config["validation"])

    # Adicionar parâmetros de teste
    if "testing" in example_config:
        config.update(example_config["testing"])

    # Adicionar parâmetros de cenários
    if "scenarios" in example_config:
        config.update(example_config["scenarios"])

    # Aplicar sobrescritas
    config.update(override_params)

    return config


def get_scientific_test_config(test_id: str, **override_params) -> Dict[str, Any]:
    """
    Obtém configuração para teste científico usando apenas example_configs.yaml

    Args:
        test_id: ID do teste científico (ex: "SCI_001")
        **override_params: Parâmetros para sobrescrever na configuração

    Returns:
        Configuração completa para o teste científico
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_data = load_config(os.path.join(base_dir, "configs", "example_configs.yaml"))

    # Buscar configuração específica do teste científico
    scientific_configs = config_data.get("scientific_test_configs", {})

    if test_id not in scientific_configs:
        raise ValueError(f"Nenhuma configuração encontrada para teste: {test_id}")

    # Obter configuração do teste
    test_config = scientific_configs[test_id]

    # Construir configuração completa
    config = {}

    # Adicionar parâmetros do modelo
    if "model" in test_config:
        config.update(test_config["model"])

    # Adicionar parâmetros de validação
    if "validation" in test_config:
        config.update(test_config["validation"])

    # Adicionar parâmetros de teste
    if "testing" in test_config:
        config.update(test_config["testing"])

    # Adicionar parâmetros de quaternion
    if "quaternion" in test_config:
        config.update(test_config["quaternion"])

    # Aplicar sobrescritas
    config.update(override_params)

    return config
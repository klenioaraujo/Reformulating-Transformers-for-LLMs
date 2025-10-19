#!/usr/bin/env python3
"""
Model Registry - Sistema de Registro de Modelos Multi-Modelo

Gerencia o registro central de todos os modelos disponíveis no sistema ΨQRH,
incluindo modelos fonte, destilados, semânticos e simulados.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import yaml

class ModelRegistry:
    """
    Sistema de registro central para modelos multi-modelo ΨQRH

    Mantém registro de todos os modelos disponíveis e seus metadados,
    permitindo detecção automática e seleção dinâmica.
    """

    def __init__(self, registry_path: str = "models/model_registry.json"):
        """
        Inicializa o registro de modelos

        Args:
            registry_path: Caminho para o arquivo de registro
        """
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()
        self._ensure_registry_structure()

    def _load_registry(self) -> Dict[str, Any]:
        """Carrega o registro de modelos do disco"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Erro ao carregar registro: {e}")
                return self._create_empty_registry()
        else:
            return self._create_empty_registry()

    def _create_empty_registry(self) -> Dict[str, Any]:
        """Cria um registro vazio"""
        return {
            "format_version": "1.0",
            "last_updated": datetime.now().isoformat(),
            "models": {}
        }

    def _ensure_registry_structure(self):
        """Garante que a estrutura do registro está correta"""
        if "format_version" not in self.registry:
            self.registry["format_version"] = "1.0"
        if "last_updated" not in self.registry:
            self.registry["last_updated"] = datetime.now().isoformat()
        if "models" not in self.registry:
            self.registry["models"] = {}

    def _save_registry(self):
        """Salva o registro no disco"""
        try:
            self.registry["last_updated"] = datetime.now().isoformat()
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(self.registry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Erro ao salvar registro: {e}")

    def register_model(self, model_name: str, model_info: Dict[str, Any]):
        """
        Registra um modelo no sistema

        Args:
            model_name: Nome do modelo
            model_info: Informações do modelo
        """
        self.registry["models"][model_name] = {
            "name": model_name,
            "registered_at": datetime.now().isoformat(),
            "status": "available",
            **model_info
        }
        self._save_registry()

    def unregister_model(self, model_name: str):
        """
        Remove um modelo do registro

        Args:
            model_name: Nome do modelo a remover
        """
        if model_name in self.registry["models"]:
            del self.registry["models"][model_name]
            self._save_registry()

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Obtém informações de um modelo

        Args:
            model_name: Nome do modelo

        Returns:
            Informações do modelo ou None se não encontrado
        """
        return self.registry["models"].get(model_name)

    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lista todos os modelos registrados

        Args:
            model_type: Tipo de modelo para filtrar (opcional)

        Returns:
            Lista de modelos
        """
        models = list(self.registry["models"].values())

        if model_type:
            models = [m for m in models if m.get("type") == model_type]

        return models

    def list_available_models(self) -> List[str]:
        """
        Lista nomes de todos os modelos disponíveis

        Returns:
            Lista de nomes de modelos
        """
        return list(self.registry["models"].keys())

    def is_model_registered(self, model_name: str) -> bool:
        """
        Verifica se um modelo está registrado

        Args:
            model_name: Nome do modelo

        Returns:
            True se registrado, False caso contrário
        """
        return model_name in self.registry["models"]

    def update_model_status(self, model_name: str, status: str):
        """
        Atualiza o status de um modelo

        Args:
            model_name: Nome do modelo
            status: Novo status
        """
        if model_name in self.registry["models"]:
            self.registry["models"][model_name]["status"] = status
            self.registry["models"][model_name]["last_updated"] = datetime.now().isoformat()
            self._save_registry()

    def scan_for_models(self, base_paths: List[str]):
        """
        Escaneia diretórios em busca de modelos

        Args:
            base_paths: Lista de caminhos base para escanear
        """
        print("🔍 Escaneando por modelos disponíveis...")

        for base_path in base_paths:
            base_dir = Path(base_path)
            if not base_dir.exists():
                continue

            # Escanear modelos fonte
            source_dir = base_dir / "source"
            if source_dir.exists():
                self._scan_source_models(source_dir)

            # Escanear modelos destilados
            distilled_dir = base_dir / "distilled"
            if distilled_dir.exists():
                self._scan_distilled_models(distilled_dir)

            # Escanear modelos semânticos
            semantic_dir = base_dir / "semantic"
            if semantic_dir.exists():
                self._scan_semantic_models(semantic_dir)

        self._save_registry()
        print(f"✅ Escaneamento concluído. {len(self.registry['models'])} modelos registrados.")

    def _scan_source_models(self, source_dir: Path):
        """Escaneia modelos fonte"""
        for model_dir in source_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "metadata.json").exists():
                try:
                    with open(model_dir / "metadata.json", 'r') as f:
                        metadata = json.load(f)

                    model_name = f"source_{model_dir.name}"
                    self.register_model(model_name, {
                        "type": "source",
                        "path": str(model_dir),
                        "model_name": model_dir.name,
                        "metadata": metadata
                    })
                except Exception as e:
                    print(f"⚠️  Erro ao registrar modelo fonte {model_dir.name}: {e}")

    def _scan_distilled_models(self, distilled_dir: Path):
        """Escaneia modelos destilados"""
        for model_file in distilled_dir.glob("*.pt"):
            model_name = f"distilled_{model_file.stem}"
            self.register_model(model_name, {
                "type": "distilled",
                "path": str(model_file),
                "source_model": model_file.stem.replace("psiqrh_distilled_", ""),
                "file_size": model_file.stat().st_size
            })

    def _scan_semantic_models(self, semantic_dir: Path):
        """Escaneia modelos semânticos"""
        for model_file in semantic_dir.glob("*.pt"):
            model_name = f"semantic_{model_file.stem}"
            self.register_model(model_name, {
                "type": "semantic",
                "path": str(model_file),
                "source_model": model_file.stem.replace("psiqrh_semantic_", ""),
                "file_size": model_file.stat().st_size
            })

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Obtém resumo dos modelos registrados

        Returns:
            Resumo estatístico dos modelos
        """
        models = self.registry["models"]
        summary = {
            "total_models": len(models),
            "by_type": {},
            "by_status": {}
        }

        for model in models.values():
            model_type = model.get("type", "unknown")
            status = model.get("status", "unknown")

            summary["by_type"][model_type] = summary["by_type"].get(model_type, 0) + 1
            summary["by_status"][status] = summary["by_status"].get(status, 0) + 1

        return summary

    def export_registry(self, output_path: str):
        """
        Exporta o registro para um arquivo

        Args:
            output_path: Caminho de saída
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)

    def import_registry(self, input_path: str):
        """
        Importa registro de um arquivo

        Args:
            input_path: Caminho do arquivo de entrada
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            imported = json.load(f)

        # Mesclar com registro atual
        self.registry["models"].update(imported.get("models", {}))
        self._save_registry()
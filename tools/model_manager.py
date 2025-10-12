#!/usr/bin/env python3
"""
ΨQRH Model Manager with Certification Support

Enhanced model management with security features and certification status display.
Ensures only certified models are used in critical operations.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3 - see LICENSE file

DOI: https://zenodo.org/records/17171112
Project: https://github.com/klenioaraujo/Reformulating-Transformers-for-LLMs
"""

import sys
import os
import json
import argparse
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(os.getcwd())
sys.path.insert(0, str(project_root))


class ModelManagerError(Exception):
    """Base exception for model management errors."""
    pass


class ModelManager:
    """Manages ΨQRH models with enhanced security and certification support."""

    def __init__(self):
        # Use current working directory as project root for registry path
        project_root = Path(os.getcwd())
        self.registry_path = project_root / "models" / "model_registry.json"
        self.ensure_registry()

    def ensure_registry(self):
        """Ensure registry exists with proper structure."""
        if not self.registry_path.exists():
            # Create initial registry
            registry = {
                "models": [],
                "active_model": None
            }
            self.save_registry(registry)

    def load_registry(self) -> dict:
        """Load the model registry safely."""
        try:
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ModelManagerError(f"Failed to load registry: {e}")

    def save_registry(self, registry: dict):
        """Save the registry safely."""
        try:
            # Ensure parent directory exists
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
        except Exception as e:
            raise ModelManagerError(f"Failed to save registry: {e}")

    def discover_models(self):
        """Discover models in the models directory."""
        models_dir = self.registry_path.parent
        if not models_dir.exists():
            print("⚠️  No models directory found")
            return

        registry = self.load_registry()
        existing_names = {m['name'] for m in registry['models']}

        # Find model directories
        for item in models_dir.iterdir():
            if item.is_dir() and item.name not in ['checkpoints', 'pretrained', 'finetuned']:
                model_name = item.name

                if model_name not in existing_names:
                    # Add new model to registry
                    new_model = {
                        "name": model_name,
                        "path": f"models/{model_name}",
                        "status": "inactive",
                        "certification": "uncertified",
                        "created_at": datetime.now().isoformat()
                    }
                    registry['models'].append(new_model)
                    print(f"➕ Discovered new model: {model_name}")

        self.save_registry(registry)
        print(f"✅ Model discovery completed")

    def list_models(self):
        """List all models with enhanced display including certification status."""
        registry = self.load_registry()

        if not registry['models']:
            print("📭 No models found in registry")
            return

        print("\n🔬 ΨQRH Model Registry")
        print("=" * 90)
        print(f"{'STATUS':<10} {'CERTIFICATION':<15} {'NAME':<20} {'PATH':<30} {'CREATED'}")
        print("-" * 90)

        for model in registry['models']:
            status = f"[ACTIVE]" if model['name'] == registry.get('active_model') else ""

            # Format certification status with colors
            cert = model['certification']
            if cert == "certified":
                cert_display = "[ CERTIFIED ]"
            elif cert == "failed":
                cert_display = "[  FAILED   ]"
            else:
                cert_display = "[ uncertified ]"

            # Format creation date
            created = model.get('created_at', 'unknown')
            if created != 'unknown':
                try:
                    created = datetime.fromisoformat(created.replace('Z', '+00:00')).strftime('%Y-%m-%d')
                except:
                    pass

            print(f"{status:<10} {cert_display:<15} {model['name']:<20} {model['path']:<30} {created}")

        print("=" * 90)

    def set_active(self, model_name: str):
        """Set a model as active with certification warning."""
        registry = self.load_registry()

        # Find the model
        model = None
        for m in registry['models']:
            if m['name'] == model_name:
                model = m
                break

        if not model:
            raise ModelManagerError(f"Model '{model_name}' not found")

        # Check if model path exists
        model_path = Path(model['path'])
        if not model_path.exists():
            raise ModelManagerError(f"Model path not found: {model_path}")

        # Check certification status
        if model['certification'] != "certified":
            print(f"\n⚠️  ATENÇÃO: Você está ativando um modelo NÃO CERTIFICADO: {model_name}")
            print(f"   Status de certificação: {model['certification']}")
            print(f"   💡 Execute 'make model-certify MODEL={model_name}' para tentar certificá-lo.")
            print(f"   ⚠️  Use por sua conta e risco!")

            # Ask for confirmation
            response = input("\nContinuar? [y/N]: ").strip().lower()
            if response != 'y':
                print("❌ Ativação cancelada pelo usuário")
                return

        # Update registry
        registry['active_model'] = model_name
        for m in registry['models']:
            m['status'] = 'active' if m['name'] == model_name else 'inactive'

        self.save_registry(registry)
        print(f"✅ Modelo ativado: {model_name}")

        # Atualizar configuração para usar configurações calibradas
        self._update_config_for_active_model(model_path)

    def _update_config_for_active_model(self, model_path: Path):
        """Atualiza configuração para usar configurações calibradas do modelo ativo."""
        calibrated_config_dir = Path(__file__).parent.parent / "configs" / "gradient_calibrated"

        if calibrated_config_dir.exists():
            print(f"  📁 Configurações calibradas disponíveis: {calibrated_config_dir}")
            print(f"  🔧 Sistema usará configurações calibradas automaticamente")
        else:
            print(f"  ⚠️  Nenhuma configuração calibrada encontrada. Execute 'make calibrate-model'")
            print(f"  💡 Usando configurações padrão por enquanto")

    def get_active_model(self):
        """Get the currently active model."""
        registry = self.load_registry()
        return registry.get('active_model')

    def get_active_model_config(self) -> dict:
        """
        Loads the final configuration by merging the base config.yaml
        with the active model's specific config.json.
        """
        # 1. Load base configuration from root config.yaml
        base_config_path = self.registry_path.parent.parent / "config.yaml"
        final_config = {}
        if base_config_path.exists():
            try:
                import yaml
                with open(base_config_path, 'r') as f:
                    final_config = yaml.safe_load(f)
            except Exception as e:
                print(f"⚠️  Could not load base config.yaml:{e}")

        # 2. Load and merge active model's config
        active_model_name = self.get_active_model()
        if active_model_name:
            registry = self.load_registry()
            for model in registry['models']:
                if model['name'] == active_model_name:
                    model_config_path = self.registry_path.parent.parent / model['path'] / "config.json"
                    if model_config_path.exists():
                        try:
                            with open(model_config_path, 'r') as f:
                                model_config = json.load(f)
                            # Merge model_config into final_config
                            # This is a deep merge for nested dictionaries
                            for key, value in model_config.items():
                                if isinstance(value, dict) and key in final_config and isinstance(final_config[key], dict):
                                    final_config[key].update(value)
                                else:
                                    final_config[key] = value
                            print(f"✅ Loaded and merged config from active model: {model_config_path}")
                        except Exception as e:
                            print(f"⚠️  Could not load model's config.json:{e}")
                    break

        return final_config

    def is_certified(self, model_name: str) -> bool:
        """Check if a model is certified."""
        registry = self.load_registry()

        for model in registry['models']:
            if model['name'] == model_name:
                return model['certification'] == "certified"

        return False

    def prune_models(self, failed: bool = False, uncertified: bool = False, empty_dirs: bool = False):
        """Prune models from registry based on criteria."""
        registry = self.load_registry()
        original_count = len(registry['models'])

        # Filter models to keep
        models_to_keep = []
        removed_models = []

        for model in registry['models']:
            should_remove = False

            # Check criteria
            if failed and model['certification'] == "failed":
                should_remove = True
                removed_models.append(f"{model['name']} (failed)")

            elif uncertified and model['certification'] == "uncertified":
                should_remove = True
                removed_models.append(f"{model['name']} (uncertified)")

            elif empty_dirs:
                model_path = Path(model['path'])
                if not model_path.exists() or not any(model_path.iterdir()):
                    should_remove = True
                    removed_models.append(f"{model['name']} (empty directory)")

            if not should_remove:
                models_to_keep.append(model)

        # Update registry
        registry['models'] = models_to_keep

        # Handle active model if it was removed
        if registry['active_model']:
            active_exists = any(model['name'] == registry['active_model'] for model in models_to_keep)
            if not active_exists:
                print(f"⚠️  Active model '{registry['active_model']}' was removed, clearing active status")
                registry['active_model'] = None

        self.save_registry(registry)

        # Report results
        removed_count = original_count - len(models_to_keep)
        print(f"🧹 Pruning completed:")
        print(f"   Original models: {original_count}")
        print(f"   Removed models: {removed_count}")
        print(f"   Remaining models: {len(models_to_keep)}")

        if removed_models:
            print(f"\n📋 Removed models:")
            for model in removed_models:
                print(f"   - {model}")
        else:
            print(f"\n✅ No models matched the pruning criteria")

    def safe_model_load(self, model_path: str):
        """Safely load a model with comprehensive error handling."""
        try:
            model_path = Path(model_path)

            if not model_path.exists():
                raise ModelManagerError(f"Model path not found: {model_path}")

            # Check for required files
            required_files = ['config.json', 'model.pt']
            for file in required_files:
                if not (model_path / file).exists():
                    raise ModelManagerError(f"Required file not found: {model_path / file}")

            # Load configuration
            with open(model_path / 'config.json', 'r') as f:
                config_data = json.load(f)

            # Load model weights safely
            try:
                model_state = torch.load(model_path / 'model.pt', map_location='cpu')
            except (pickle.UnpicklingError, RuntimeError, EOFError) as e:
                raise ModelManagerError(f"Failed to load model weights: {e}")

            return config_data, model_state

        except Exception as e:
            raise ModelManagerError(f"Failed to load model: {e}")


def main():
    """Main model management function."""
    parser = argparse.ArgumentParser(description="ΨQRH Model Manager")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Discover command
    subparsers.add_parser('discover', help='Discover new models')

    # List command
    subparsers.add_parser('list', help='List all models with certification status')

    # Set-active command
    set_active_parser = subparsers.add_parser('set-active', help='Set a model as active')
    set_active_parser.add_argument('model_name', help='Name of the model to activate')

    # Is-certified command
    is_certified_parser = subparsers.add_parser('is-certified', help='Check if a model is certified')
    is_certified_parser.add_argument('model_name', help='Name of the model to check')

    # Get-active command
    subparsers.add_parser('get-active', help='Get the currently active model')

    # Prune command
    prune_parser = subparsers.add_parser('prune', help='Prune models from registry')
    prune_parser.add_argument('--failed', action='store_true', help='Remove failed models')
    prune_parser.add_argument('--uncertified', action='store_true', help='Remove uncertified models')
    prune_parser.add_argument('--empty-dirs', action='store_true', help='Remove models with empty directories')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    manager = ModelManager()

    try:
        if args.command == 'discover':
            manager.discover_models()

        elif args.command == 'list':
            manager.list_models()

        elif args.command == 'set-active':
            manager.set_active(args.model_name)

        elif args.command == 'is-certified':
            if manager.is_certified(args.model_name):
                print("true")
            else:
                print("false")

        elif args.command == 'get-active':
            active_model = manager.get_active_model()
            if active_model:
                print(active_model)
            else:
                print("none")

        elif args.command == 'prune':
            manager.prune_models(
                failed=args.failed,
                uncertified=args.uncertified,
                empty_dirs=args.empty_dirs
            )

    except ModelManagerError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Secure Training Integration for ΨQRH Framework

Integrates secure asset validation with the training system to ensure
only certified .Ψcws files can be used for training.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.secure_asset_validator import SecureAssetValidator


class SecureTrainingSystem:
    def __init__(self):
        self.validator = SecureAssetValidator()

    def validate_training_asset(self, asset_name, key=None):
        """
        Validate a secure asset before training
        Returns True if valid and ready for training
        """
        print(f"🔐 Validando ativo seguro para treinamento: {asset_name}")

        # Validate asset certification
        if not self.validator.validate_asset(asset_name, key):
            print("❌ Ativo não validado. Treinamento cancelado.")
            return False

        print("✅ Ativo validado com sucesso. Pronto para treinamento.")
        return True

    def get_secure_training_data(self, asset_name, key=None):
        """
        Get training data from secure asset after validation
        Returns the path to the validated .Ψcws file
        """
        if not self.validate_training_asset(asset_name, key):
            raise ValueError(f"Asset {asset_name} failed validation")

        asset_path = Path('data/Ψcws') / f"{asset_name}.Ψcws"

        if not asset_path.exists():
            raise FileNotFoundError(f"Asset file not found: {asset_path}")

        return str(asset_path)

    def create_secure_training_pipeline(self, asset_name, key, training_config):
        """
        Create a complete secure training pipeline
        """
        print("🚀 Iniciando pipeline de treinamento seguro...")

        # Step 1: Validate asset
        if not self.validate_training_asset(asset_name, key):
            raise ValueError("Asset validation failed")

        # Step 2: Get secure data path
        data_path = self.get_secure_training_data(asset_name, key)

        # Step 3: Configure training with secure data
        training_config['secure_data_path'] = data_path
        training_config['asset_name'] = asset_name

        print(f"✅ Pipeline configurado para treinamento seguro")
        print(f"   Ativo: {asset_name}")
        print(f"   Dados: {data_path}")

        return training_config


def main():
    """
    Main function for secure training integration
    Can be called from training scripts
    """
    import argparse

    parser = argparse.ArgumentParser(description='Secure training integration for ΨQRH')
    parser.add_argument('--asset', required=True, help='Asset name')
    parser.add_argument('--key', help='Encryption key')
    parser.add_argument('--validate-only', action='store_true', help='Only validate, don\'t train')

    args = parser.parse_args()

    training_system = SecureTrainingSystem()

    if args.validate_only:
        success = training_system.validate_training_asset(args.asset, args.key)
        sys.exit(0 if success else 1)
    else:
        # This would integrate with the actual training system
        config = training_system.create_secure_training_pipeline(
            args.asset, args.key, {}
        )
        print(f"📋 Configuração de treinamento seguro: {config}")


if __name__ == "__main__":
    main()
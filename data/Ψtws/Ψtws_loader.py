#!/usr/bin/env python3
"""
ΨTWS File Loader
===============

Utility script for loading and processing ΨTWS training data files.
"""

import os
import yaml
from typing import Dict, Any, List


class ΨTWSLoader:
    """Loader for ΨTWS training data files."""

    def __init__(self, data_dir: str = "data/Ψtws"):
        self.data_dir = data_dir
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load ΨTWS configuration file."""
        config_path = os.path.join(self.data_dir, "Ψtws_config.yaml")
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Configuration file not found at {config_path}")
            return {}

    def load_training_files(self) -> List[Dict[str, Any]]:
        """Load all training data files."""
        training_files = []

        for filename in self.config.get('file_structure', {}).get('training_files', []):
            file_path = os.path.join(self.data_dir, filename)
            if os.path.exists(file_path):
                training_files.append(self._parse_Ψtws_file(file_path))

        return training_files

    def load_validation_files(self) -> List[Dict[str, Any]]:
        """Load validation data files."""
        validation_files = []

        for filename in self.config.get('file_structure', {}).get('validation_files', []):
            file_path = os.path.join(self.data_dir, filename)
            if os.path.exists(file_path):
                validation_files.append(self._parse_Ψtws_file(file_path))

        return validation_files

    def load_test_files(self) -> List[Dict[str, Any]]:
        """Load test data files."""
        test_files = []

        for filename in self.config.get('file_structure', {}).get('test_files', []):
            file_path = os.path.join(self.data_dir, filename)
            if os.path.exists(file_path):
                test_files.append(self._parse_Ψtws_file(file_path))

        return test_files

    def _parse_Ψtws_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a ΨTWS file and extract structured data."""
        data = {
            'file_path': file_path,
            'metadata': {},
            'spectral_data': {},
            'encryption_info': {},
            'scientific_mask': {},
            'validation_info': {}
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            current_section = None
            for line in lines:
                line = line.strip()

                if not line or line.startswith('#'):
                    continue

                if line.startswith('MAGIC_HEADER:'):
                    data['metadata']['magic_header'] = line.split(':', 1)[1].strip()
                elif line.startswith('VERSION:'):
                    data['metadata']['version'] = line.split(':', 1)[1].strip()
                elif line.startswith('CREATION_DATE:'):
                    data['metadata']['creation_date'] = line.split(':', 1)[1].strip()
                elif line.startswith('ENCRYPTION_LAYERS:'):
                    data['metadata']['encryption_layers'] = int(line.split(':', 1)[1].strip())
                elif line.startswith('SPECTRAL_DIMENSION:'):
                    data['metadata']['spectral_dimension'] = int(line.split(':', 1)[1].strip())
                elif line.startswith('INPUT_TEXT:'):
                    data['metadata']['input_text'] = line.split(':', 1)[1].strip().strip('"')
                elif line == 'SPECTRAL_DATA:':
                    current_section = 'spectral_data'
                elif line == 'ENCRYPTION_INFO:':
                    current_section = 'encryption_info'
                elif line == 'SCIENTIFIC_MASK:':
                    current_section = 'scientific_mask'
                elif line == 'VALIDATION_HASH:' or line == 'TEST_HASH:' or line == 'END_OF_FILE':
                    current_section = None
                elif current_section and line.startswith('  - '):
                    # Parse key-value pairs
                    key_value = line[4:].split(':', 1)
                    if len(key_value) == 2:
                        key = key_value[0].strip()
                        value = key_value[1].strip()

                        # Try to convert to appropriate type
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif value.replace('.', '').isdigit():
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)

                        data[current_section][key] = value

        except Exception as e:
            print(f"Error parsing ΨTWS file {file_path}: {e}")

        return data

    def get_file_statistics(self) -> Dict[str, Any]:
        """Get statistics about available ΨTWS files."""
        stats = {
            'total_files': 0,
            'training_files': 0,
            'validation_files': 0,
            'test_files': 0,
            'file_sizes': {},
            'spectral_dimensions': []
        }

        all_files = []
        all_files.extend(self.load_training_files())
        all_files.extend(self.load_validation_files())
        all_files.extend(self.load_test_files())

        stats['total_files'] = len(all_files)
        stats['training_files'] = len(self.load_training_files())
        stats['validation_files'] = len(self.load_validation_files())
        stats['test_files'] = len(self.load_test_files())

        for file_data in all_files:
            file_path = file_data['file_path']
            stats['file_sizes'][file_path] = os.path.getsize(file_path)

            if 'spectral_dimension' in file_data['metadata']:
                stats['spectral_dimensions'].append(file_data['metadata']['spectral_dimension'])

        return stats


if __name__ == "__main__":
    # Example usage
    loader = ΨTWSLoader()

    print("🔍 ΨTWS File Loader Test")
    print("=" * 50)

    # Load files
    training_files = loader.load_training_files()
    validation_files = loader.load_validation_files()
    test_files = loader.load_test_files()

    print(f"📚 Training files: {len(training_files)}")
    print(f"📊 Validation files: {len(validation_files)}")
    print(f"🧪 Test files: {len(test_files)}")

    # Show statistics
    stats = loader.get_file_statistics()
    print(f"\n📈 Statistics:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  File sizes: {stats['file_sizes']}")
    print(f"  Spectral dimensions: {stats['spectral_dimensions']}")

    # Show sample data from first training file
    if training_files:
        print(f"\n📖 Sample from first training file:")
        sample = training_files[0]
        print(f"  Input text: {sample['metadata'].get('input_text', 'N/A')[:100]}...")
        print(f"  Spectral dim: {sample['metadata'].get('spectral_dimension', 'N/A')}")
        print(f"  Encryption layers: {sample['metadata'].get('encryption_layers', 'N/A')}")
#!/usr/bin/env python3
"""
CWSDataManager - Unified Manager for .Ψcws Format

This class centralizes all operations related to the .Ψcws spectral format,
providing a unified interface for loading, saving, listing, and converting
data sources to the .Ψcws format.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import torch
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.conscience.psicws_native_reader import ΨCWSNativeReader


class CWSDataManager:
    """
    Unified manager for .Ψcws spectral format operations.

    This class provides a centralized interface for all .Ψcws-related
    operations including loading, saving, listing, and converting data sources.
    """

    def __init__(self, cache_dir: str = "data/Ψcws"):
        """
        Initialize the CWSDataManager.

        Args:
            cache_dir: Directory where .Ψcws files are stored
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize native reader
        self.reader = ΨCWSNativeReader(cache_dir)

    def load(self, path: Union[str, Path]) -> torch.Tensor:
        """
        Load a .Ψcws file into a PyTorch tensor.

        Args:
            path: Path to the .Ψcws file

        Returns:
            PyTorch tensor containing the spectral data
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix != '.Ψcws':
            raise ValueError(f"File must have .Ψcws extension: {path}")

        try:
            # Load using native reader
            cws_file = self.reader.load_by_name(path.stem)
            if cws_file is None:
                raise ValueError(f"Failed to load .Ψcws file: {path}")

            # Convert spectral data to tensor
            spectral_data = cws_file.spectral_data
            tensor_data = torch.tensor(spectral_data.data, dtype=torch.float32)

            return tensor_data

        except Exception as e:
            raise RuntimeError(f"Error loading .Ψcws file {path}: {e}")

    def save(self, tensor: torch.Tensor, path: Union[str, Path],
             metadata: Optional[Dict[str, Any]] = None):
        """
        Save a PyTorch tensor in .Ψcws format.

        Args:
            tensor: PyTorch tensor to save
            path: Output path for the .Ψcws file
            metadata: Optional metadata to include in the file
        """
        path = Path(path)

        if path.suffix != '.Ψcws':
            path = path.with_suffix('.Ψcws')

        try:
            # Convert tensor to numpy array
            data_array = tensor.detach().cpu().numpy()

            # Create basic metadata if not provided
            if metadata is None:
                metadata = {
                    'shape': tensor.shape,
                    'dtype': str(tensor.dtype),
                    'device': str(tensor.device)
                }

            # For now, save as numpy array with metadata
            # In a full implementation, this would use the proper ΨCWSFile format
            np.savez(path, data=data_array, metadata=metadata)

            print(f"✅ Saved .Ψcws file: {path}")

        except Exception as e:
            raise RuntimeError(f"Error saving .Ψcws file {path}: {e}")

    def list(self, pattern: str = "**/*.Ψcws") -> List[Dict[str, Any]]:
        """
        List available .Ψcws files with their metadata.

        Args:
            pattern: Glob pattern for file discovery

        Returns:
            List of dictionaries with file information
        """
        try:
            files = list(self.cache_dir.glob(pattern))
            file_info_list = []

            for file_path in files:
                try:
                    stat = file_path.stat()

                    file_info = {
                        'name': file_path.name,
                        'path': str(file_path),
                        'size_bytes': stat.st_size,
                        'size_mb': round(stat.st_size / (1024 * 1024), 2),
                        'modified_time': stat.st_mtime,
                        'stem': file_path.stem
                    }

                    file_info_list.append(file_info)

                except Exception as e:
                    print(f"⚠️ Error processing {file_path}: {e}")

            return file_info_list

        except Exception as e:
            print(f"❌ Error listing .Ψcws files: {e}")
            return []

    def convert(self, source_type: str, source_path: str,
                output_path: Optional[str] = None, **kwargs) -> str:
        """
        Convert data source to .Ψcws format.

        Args:
            source_type: Type of source ('pdf', 'wiki', 'text')
            source_path: Path to source data
            output_path: Optional output path for .Ψcws file
            **kwargs: Additional conversion parameters

        Returns:
            Path to the created .Ψcws file
        """
        if source_type == 'pdf':
            return self._convert_pdf(source_path, output_path, **kwargs)
        elif source_type == 'wiki':
            return self._convert_wiki(source_path, output_path, **kwargs)
        elif source_type == 'text':
            return self._convert_text(source_path, output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    def _convert_pdf(self, pdf_path: str, output_path: Optional[str] = None, **kwargs) -> str:
        """
        Convert PDF to .Ψcws format.

        Args:
            pdf_path: Path to PDF file
            output_path: Optional output path
            **kwargs: Additional parameters

        Returns:
            Path to created .Ψcws file
        """
        try:
            # Import PDF converter if available
            try:
                from src.utils.pdf_converter import PDFToCWSConverter
                converter = PDFToCWSConverter()
                cws_path = converter.convert(pdf_path, output_path)
                return cws_path

            except ImportError:
                # Fallback implementation
                print(f"⚠️ PDF converter not available, using fallback for {pdf_path}")

                if output_path is None:
                    output_path = self.cache_dir / f"{Path(pdf_path).stem}.Ψcws"

                # Create dummy spectral data
                dummy_data = torch.randn(100, 256)  # Example: 100 tokens, 256 dimensions
                self.save(dummy_data, output_path, {
                    'source': 'pdf',
                    'original_file': pdf_path,
                    'conversion_method': 'fallback'
                })

                return str(output_path)

        except Exception as e:
            raise RuntimeError(f"Error converting PDF {pdf_path}: {e}")

    def _convert_wiki(self, topic: str, output_path: Optional[str] = None, **kwargs) -> str:
        """
        Convert Wikipedia topic to .Ψcws format.

        Args:
            topic: Wikipedia topic name
            output_path: Optional output path
            **kwargs: Additional parameters

        Returns:
            Path to created .Ψcws file
        """
        try:
            # Import Wikipedia converter if available
            try:
                from wiki_to_psicws_converter import WikipediaToCWSConverter
                converter = WikipediaToCWSConverter()
                cws_path = converter.convert_topic(topic, output_path)
                return cws_path

            except ImportError:
                # Fallback implementation
                print(f"⚠️ Wikipedia converter not available, using fallback for topic '{topic}'")

                if output_path is None:
                    output_path = self.cache_dir / f"wikipedia_{topic.replace(' ', '_')}.Ψcws"

                # Create dummy spectral data
                dummy_data = torch.randn(200, 512)  # Example: 200 tokens, 512 dimensions
                self.save(dummy_data, output_path, {
                    'source': 'wikipedia',
                    'topic': topic,
                    'conversion_method': 'fallback'
                })

                return str(output_path)

        except Exception as e:
            raise RuntimeError(f"Error converting Wikipedia topic '{topic}': {e}")

    def _convert_text(self, text: str, output_path: Optional[str] = None, **kwargs) -> str:
        """
        Convert text to .Ψcws format.

        Args:
            text: Input text
            output_path: Optional output path
            **kwargs: Additional parameters

        Returns:
            Path to created .Ψcws file
        """
        try:
            if output_path is None:
                # Create hash-based filename
                import hashlib
                text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
                output_path = self.cache_dir / f"text_{text_hash}.Ψcws"

            # Simple text-to-spectral conversion
            # In a real implementation, this would use proper spectral analysis
            tokens = text.split()
            num_tokens = len(tokens)
            embed_dim = kwargs.get('embed_dim', 256)

            # Create spectral-like representation
            spectral_data = torch.randn(num_tokens, embed_dim)

            self.save(spectral_data, output_path, {
                'source': 'text',
                'text_length': num_tokens,
                'embed_dim': embed_dim,
                'conversion_method': 'direct'
            })

            return str(output_path)

        except Exception as e:
            raise RuntimeError(f"Error converting text to .Ψcws: {e}")

    def get_dataset(self, pattern: str = "**/*.Ψcws") -> List[torch.Tensor]:
        """
        Load multiple .Ψcws files as a dataset.

        Args:
            pattern: Glob pattern for file discovery

        Returns:
            List of tensors from .Ψcws files
        """
        file_info_list = self.list(pattern)
        dataset = []

        for file_info in file_info_list:
            try:
                tensor = self.load(file_info['path'])
                dataset.append(tensor)
                print(f"✅ Loaded {file_info['name']}: shape {tensor.shape}")

            except Exception as e:
                print(f"❌ Error loading {file_info['name']}: {e}")

        return dataset

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the .Ψcws data management system.

        Returns:
            Health status report
        """
        try:
            files = self.list()
            total_files = len(files)
            total_size = sum(f['size_bytes'] for f in files)

            # Test loading a few files
            test_files = files[:min(3, len(files))]
            loadable_files = 0

            for file_info in test_files:
                try:
                    self.load(file_info['path'])
                    loadable_files += 1
                except:
                    pass

            return {
                'status': 'healthy' if loadable_files == len(test_files) else 'warning',
                'total_files': total_files,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'test_files_loaded': f"{loadable_files}/{len(test_files)}",
                'cache_directory': str(self.cache_dir)
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'cache_directory': str(self.cache_dir)
            }


# Global instance for convenience
_global_cws_manager = None

def get_cws_manager() -> CWSDataManager:
    """
    Get global instance of CWSDataManager.

    Returns:
        Global CWSDataManager instance
    """
    global _global_cws_manager
    if _global_cws_manager is None:
        _global_cws_manager = CWSDataManager()
    return _global_cws_manager


if __name__ == "__main__":
    # Test the CWSDataManager
    print("🧪 Testing CWSDataManager...")

    manager = CWSDataManager()

    # List available files
    files = manager.list()
    print(f"📁 Found {len(files)} .Ψcws files")

    # Health check
    health = manager.health_check()
    print(f"🚦 Health status: {health['status']}")

    # Test conversion
    if len(files) == 0:
        print("🧪 Testing text conversion...")
        test_text = "This is a test text for .Ψcws conversion"
        cws_path = manager.convert('text', test_text)
        print(f"✅ Created: {cws_path}")
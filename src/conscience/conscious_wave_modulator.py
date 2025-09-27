#!/usr/bin/env python3
"""
Conscious Wave Modulator - Processador de Arquivos para .Ψcws
===========================================================

Converte múltiplos tipos de arquivo (PDF, TXT, SQL, CSV, JSON) para
formato .Ψcws (Psi Conscious Wave Spectrum) com embedding de ondas caóticas
e análise de consciência fractal.

Formato .cwm:
- Header: Metadados e parâmetros de onda
- Spectral Data: Embeddings e trajetórias caóticas
- Content Metadata: Conteúdo original e clusters semânticos
- QRH Compatibility: Tensors compatíveis com QRHLayer

Pipeline: Arquivo → Extração → Encoding Consciente → .Ψcws → QRH Processing
"""

import torch
import torch.nn as nn
import numpy as np
import hashlib
import json
import gzip
import struct
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, asdict
import time

# Importar sistema de segurança
from .secure_Ψcws_protector import create_secure_Ψcws_protector

# File processors imports (with fallbacks)
try:
    import PyMuPDF as fitz  # pymupdf
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    try:
        import PyPDF2
        HAS_PYPDF2 = True
    except ImportError:
        HAS_PYPDF2 = False

import csv
import io


@dataclass
class ΨCWSHeader:
    """Header structure for .Ψcws files."""
    magic_number: str = "ΨCWS1"
    version: str = "1.0"
    file_type: str = ""
    content_hash: str = ""
    timestamp: str = ""
    wave_parameters: Dict[str, float] = None

    def __post_init__(self):
        if self.wave_parameters is None:
            self.wave_parameters = {
                "amplitude_base": 1.0,
                "frequency_range": [0.5, 5.0],
                "phase_offsets": [0.0, 0.7854, 1.5708, 2.3562],  # 0, π/4, π/2, 3π/4
                "chaotic_seed": 12345
            }


@dataclass
class ΨCWSSpectralData:
    """Spectral data structure for .Ψcws files."""
    wave_embeddings: torch.Tensor = None
    chaotic_trajectories: torch.Tensor = None
    fourier_spectra: torch.Tensor = None
    consciousness_metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.consciousness_metrics is None:
            self.consciousness_metrics = {
                "complexity": 0.0,
                "coherence": 0.0,
                "adaptability": 0.0,
                "integration": 0.0
            }


@dataclass
class ΨCWSContentMetadata:
    """Content metadata structure for .Ψcws files."""
    original_source: str = ""
    extracted_text: str = ""
    key_concepts: List[str] = None
    semantic_clusters: List[List[float]] = None

    def __post_init__(self):
        if self.key_concepts is None:
            self.key_concepts = []
        if self.semantic_clusters is None:
            self.semantic_clusters = []


@dataclass
class ΨCWSFile:
    """Complete .Ψcws file structure."""
    header: ΨCWSHeader
    spectral_data: ΨCWSSpectralData
    content_metadata: ΨCWSContentMetadata
    qrh_tensor: torch.Tensor = None  # Pre-computed tensor for QRH

    def save(self, file_path: Union[str, Path]):
        """Save .Ψcws file to disk."""
        file_path = Path(file_path)

        # Prepare data for serialization with proper numpy conversion
        def safe_numpy_convert(tensor):
            if tensor is None:
                return None
            if hasattr(tensor, 'numpy'):
                return tensor.numpy().tolist()  # Convert to Python list for JSON
            elif hasattr(tensor, 'tolist'):
                return tensor.tolist()
            else:
                return tensor

        data = {
            'header': asdict(self.header),
            'spectral_data': {
                'wave_embeddings': safe_numpy_convert(self.spectral_data.wave_embeddings),
                'chaotic_trajectories': safe_numpy_convert(self.spectral_data.chaotic_trajectories),
                'fourier_spectra': safe_numpy_convert(self.spectral_data.fourier_spectra),
                'consciousness_metrics': self.spectral_data.consciousness_metrics
            },
            'content_metadata': asdict(self.content_metadata),
            'qrh_tensor': safe_numpy_convert(self.qrh_tensor)
        }

        # Compress and save
        json_data = json.dumps(data, indent=2, ensure_ascii=False)
        compressed_data = gzip.compress(json_data.encode('utf-8'))

        with open(file_path, 'wb') as f:
            f.write(compressed_data)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> 'CWMFile':
        """Load .Ψcws file from disk."""
        file_path = Path(file_path)

        with open(file_path, 'rb') as f:
            compressed_data = f.read()

        json_data = gzip.decompress(compressed_data).decode('utf-8')
        data = json.loads(json_data)

        # Safe tensor reconstruction
        def safe_tensor_convert(data_array):
            if data_array is None:
                return None
            return torch.tensor(data_array, dtype=torch.float32)

        # Reconstruct tensors
        header = ΨCWSHeader(**data['header'])

        spectral_data = ΨCWSSpectralData()
        spectral_data.wave_embeddings = safe_tensor_convert(data['spectral_data']['wave_embeddings'])
        spectral_data.chaotic_trajectories = safe_tensor_convert(data['spectral_data']['chaotic_trajectories'])
        spectral_data.fourier_spectra = safe_tensor_convert(data['spectral_data']['fourier_spectra'])
        spectral_data.consciousness_metrics = data['spectral_data']['consciousness_metrics']

        content_metadata = ΨCWSContentMetadata(**data['content_metadata'])

        qrh_tensor = safe_tensor_convert(data['qrh_tensor'])

        return cls(header, spectral_data, content_metadata, qrh_tensor)


class ConsciousWaveModulator:
    """
    Modulator de ondas conscientes que converte arquivos múltiplos
    para formato .Ψcws com embedding caótico e análise fractal.
    """

    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = self._default_config()

        self.config = config
        self.device = config.get('device', 'cpu')

        # Wave parameters
        self.base_amplitude = config.get('base_amplitude', 1.0)
        self.frequency_range = config.get('frequency_range', [0.5, 5.0])
        self.phase_consciousness = config.get('phase_consciousness', 0.7854)  # π/4
        self.chaotic_r = config.get('chaotic_r', 3.9)
        self.embedding_dim = config.get('embedding_dim', 256)
        self.sequence_length = config.get('sequence_length', 64)

        # File processors
        self.processors = {
            'pdf': self._process_pdf,
            'txt': self._process_txt,
            'sql': self._process_sql,
            'json': self._process_json,
            'csv': self._process_csv
        }

        # Cache for processed files
        self.cache_dir = Path(config.get('cache_dir', 'data/cwm_cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"🌊 ConsciousWaveModulator inicializado")
        print(f"   Embedding dim: {self.embedding_dim}")
        print(f"   Sequence length: {self.sequence_length}")
        print(f"   Cache dir: {self.cache_dir}")

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for the modulator."""
        return {
            'device': 'cpu',
            'base_amplitude': 1.0,
            'frequency_range': [0.5, 5.0],
            'phase_consciousness': 0.7854,
            'chaotic_r': 3.9,
            'embedding_dim': 256,
            'sequence_length': 64,
            'cache_dir': 'data/Ψcws_cache',
            'compression': True,
            'auto_cleanup': True
        }

    def process_file(self, file_path: Union[str, Path]) -> ΨCWSFile:
        """
        Process any supported file to .Ψcws format.

        Args:
            file_path: Path to the input file

        Returns:
            ΨCWSFile object
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check cache first
        cache_path = self._get_cache_path(file_path)
        if cache_path.exists():
            print(f"📋 Loading from cache: {cache_path.name}")
            return CWMFile.load(cache_path)

        # Detect file type and process
        file_type = file_path.suffix.lower().lstrip('.')

        if file_type not in self.processors:
            raise ValueError(f"Unsupported file type: {file_type}")

        print(f"🔄 Processing {file_type.upper()} file: {file_path.name}")

        # Extract content
        extracted_text = self.processors[file_type](file_path)

        # Create .Ψcws file
        Ψcws_file = self._create_Ψcws_file(file_path, extracted_text, file_type)

        # Cache the result
        if self.config.get('cache_enabled', True):
            Ψcws_file.save(cache_path)
            print(f"💾 Cached to: {cache_path.name}")

        return Ψcws_file

    def _get_cache_path(self, file_path: Path) -> Path:
        """Generate cache path for a file."""
        # Create hash of file path and modification time
        file_stat = file_path.stat()
        hash_input = f"{file_path.absolute()}_{file_stat.st_mtime}"
        file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:16]

        cache_name = f"{file_hash}_{file_path.stem}.Ψcws"
        return self.cache_dir / cache_name

    def _create_Ψcws_file(
        self,
        file_path: Path,
        extracted_text: str,
        file_type: str
    ) -> ΨCWSFile:
        """Create .Ψcws file from extracted text."""

        # Create header
        content_hash = hashlib.sha256(extracted_text.encode()).hexdigest()
        header = ΨCWSHeader(
            file_type=file_type,
            content_hash=content_hash,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            wave_parameters={
                "amplitude_base": self.base_amplitude,
                "frequency_range": self.frequency_range,
                "phase_offsets": [0.0, self.phase_consciousness, 2*self.phase_consciousness, 3*self.phase_consciousness],
                "chaotic_seed": hash(extracted_text) % (2**32)
            }
        )

        # Generate conscious wave embeddings
        wave_embeddings = self._generate_wave_embeddings(extracted_text)

        # Generate chaotic trajectories
        chaotic_trajectories = self._generate_chaotic_trajectories(extracted_text)

        # Compute Fourier spectra
        fourier_spectra = self._compute_fourier_spectra(wave_embeddings)

        # Compute consciousness metrics
        consciousness_metrics = self._compute_consciousness_metrics(
            wave_embeddings,
            chaotic_trajectories,
            fourier_spectra
        )

        # Create spectral data
        spectral_data = ΨCWSSpectralData(
            wave_embeddings=wave_embeddings,
            chaotic_trajectories=chaotic_trajectories,
            fourier_spectra=fourier_spectra,
            consciousness_metrics=consciousness_metrics
        )

        # Extract semantic information
        key_concepts = self._extract_key_concepts(extracted_text)
        semantic_clusters = self._generate_semantic_clusters(extracted_text)

        # Create content metadata
        content_metadata = ΨCWSContentMetadata(
            original_source=str(file_path),
            extracted_text=extracted_text[:10000],  # Limit size
            key_concepts=key_concepts,
            semantic_clusters=semantic_clusters
        )

        # Generate QRH-compatible tensor
        qrh_tensor = self._generate_qrh_tensor(wave_embeddings, chaotic_trajectories)

        return ΨCWSFile(header, spectral_data, content_metadata, qrh_tensor)

    def _generate_wave_embeddings(self, text: str) -> torch.Tensor:
        """Generate conscious wave embeddings from text."""
        # Convert text to numerical sequence
        char_sequence = [ord(char) / 127.0 - 1.0 for char in text]  # Normalize to [-1, 1]

        # Pad or truncate to sequence length
        if len(char_sequence) > self.sequence_length:
            char_sequence = char_sequence[:self.sequence_length]
        else:
            char_sequence.extend([0.0] * (self.sequence_length - len(char_sequence)))

        # Create base wave embedding
        base_embedding = torch.tensor(char_sequence, dtype=torch.float32)

        # Generate multi-dimensional embedding using wave functions
        embeddings = torch.zeros(self.sequence_length, self.embedding_dim)

        for i in range(self.sequence_length):
            for j in range(self.embedding_dim):
                # Base frequency based on character and position
                freq = self.frequency_range[0] + (self.frequency_range[1] - self.frequency_range[0]) * (j / self.embedding_dim)

                # Phase modulation based on character value
                phase = self.phase_consciousness * base_embedding[i]

                # Generate conscious wave
                wave_value = self.base_amplitude * np.sin(2 * np.pi * freq * i + phase)

                # Add chaotic modulation
                chaotic_mod = self._logistic_map(base_embedding[i].item(), self.chaotic_r)

                embeddings[i, j] = wave_value * (1 + 0.3 * chaotic_mod)

        return embeddings

    def _generate_chaotic_trajectories(self, text: str) -> torch.Tensor:
        """Generate chaotic trajectories using logistic map."""
        # Initialize with text-based seed
        x = abs(hash(text) % 1000) / 1000.0 * 0.5 + 0.25  # Range [0.25, 0.75]

        trajectory_points = 256  # Number of trajectory points
        trajectory = torch.zeros(trajectory_points)

        for i in range(trajectory_points):
            x = self.chaotic_r * x * (1 - x)
            trajectory[i] = x

        return trajectory

    def _logistic_map(self, x: float, r: float) -> float:
        """Single iteration of logistic map."""
        x = abs(x) % 1.0  # Ensure x in [0, 1]
        return r * x * (1 - x)

    def _compute_fourier_spectra(self, wave_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute Fourier spectra of wave embeddings."""
        # Take FFT along sequence dimension
        fft_result = torch.fft.fft(wave_embeddings, dim=0)

        # Return magnitude spectrum
        magnitude_spectrum = torch.abs(fft_result)

        return magnitude_spectrum

    def _compute_consciousness_metrics(
        self,
        wave_embeddings: torch.Tensor,
        chaotic_trajectories: torch.Tensor,
        fourier_spectra: torch.Tensor
    ) -> Dict[str, float]:
        """Compute consciousness metrics from wave data."""

        # Complexity: Based on entropy of wave embeddings
        wave_flat = wave_embeddings.flatten()
        hist = torch.histc(wave_flat, bins=50)
        prob = hist / hist.sum()
        prob = prob[prob > 0]
        complexity = -torch.sum(prob * torch.log2(prob)).item()

        # Coherence: Based on autocorrelation of trajectories
        autocorr = torch.corrcoef(torch.stack([
            chaotic_trajectories[:-1],
            chaotic_trajectories[1:]
        ]))[0, 1]
        coherence = abs(autocorr.item()) if not torch.isnan(autocorr) else 0.0

        # Adaptability: Based on spectral diversity
        spectral_std = torch.std(fourier_spectra).item()
        spectral_mean = torch.mean(fourier_spectra).item()
        adaptability = spectral_std / (spectral_mean + 1e-10)

        # Integration: Based on cross-correlation between different dimensions
        if wave_embeddings.shape[1] > 1:
            correlations = []
            for i in range(min(10, wave_embeddings.shape[1] - 1)):
                corr = torch.corrcoef(torch.stack([
                    wave_embeddings[:, i],
                    wave_embeddings[:, i + 1]
                ]))[0, 1]
                if not torch.isnan(corr):
                    correlations.append(abs(corr.item()))
            integration = np.mean(correlations) if correlations else 0.0
        else:
            integration = 0.0

        return {
            "complexity": min(complexity / 10.0, 1.0),  # Normalize
            "coherence": coherence,
            "adaptability": min(adaptability, 1.0),
            "integration": integration
        }

    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text (simple implementation)."""
        # Simple keyword extraction based on word frequency
        words = text.lower().split()
        word_counts = {}

        for word in words:
            # Filter out short words and common words
            if len(word) > 3 and word.isalpha():
                word_counts[word] = word_counts.get(word, 0) + 1

        # Get top 10 most frequent words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        key_concepts = [word for word, count in sorted_words[:10]]

        return key_concepts

    def _generate_semantic_clusters(self, text: str) -> List[List[float]]:
        """Generate semantic clusters (simple implementation)."""
        # Split text into sentences and create simple numerical representations
        sentences = text.split('.')[:5]  # Max 5 sentences
        clusters = []

        for sentence in sentences:
            if sentence.strip():
                # Simple numerical representation based on character statistics
                cluster = [
                    len(sentence) / 100.0,  # Length
                    sentence.count(' ') / 20.0,  # Word count approximation
                    sum(ord(c) for c in sentence[:10]) / 1000.0,  # Character sum
                    sentence.count(',') / 5.0,  # Punctuation density
                ]
                clusters.append(cluster)

        return clusters

    def _generate_qrh_tensor(
        self,
        wave_embeddings: torch.Tensor,
        chaotic_trajectories: torch.Tensor
    ) -> torch.Tensor:
        """Generate QRH-compatible tensor."""
        seq_len, embed_dim = wave_embeddings.shape

        # Create quaternion-compatible tensor [1, seq_len, 4*reduced_dim]
        qrh_dim = 64  # Reduced dimension for QRH compatibility

        # Reduce embedding dimension
        if embed_dim > qrh_dim:
            # Average pooling to reduce dimension
            pooled_embeddings = wave_embeddings.view(seq_len, qrh_dim, -1).mean(dim=2)
        else:
            # Pad to required dimension
            padding = torch.zeros(seq_len, qrh_dim - embed_dim)
            pooled_embeddings = torch.cat([wave_embeddings, padding], dim=1)

        # Create 4D quaternion structure
        qrh_tensor = torch.zeros(1, seq_len, 4 * qrh_dim)

        # Fill quaternion components
        qrh_tensor[0, :, :qrh_dim] = pooled_embeddings  # Real component
        qrh_tensor[0, :, qrh_dim:2*qrh_dim] = pooled_embeddings * 0.5  # i component
        qrh_tensor[0, :, 2*qrh_dim:3*qrh_dim] = pooled_embeddings * 0.3  # j component
        qrh_tensor[0, :, 3*qrh_dim:] = pooled_embeddings * 0.2  # k component

        # Modulate with chaotic trajectories
        trajectory_expanded = chaotic_trajectories[:seq_len].unsqueeze(1).expand(-1, 4 * qrh_dim)
        qrh_tensor[0] *= (1 + 0.1 * trajectory_expanded)

        return qrh_tensor

    # File processors
    def _process_pdf(self, file_path: Path) -> str:
        """Process PDF file."""
        if HAS_PYMUPDF:
            return self._process_pdf_pymupdf(file_path)
        elif HAS_PYPDF2:
            return self._process_pdf_pypdf2(file_path)
        else:
            raise ImportError("No PDF processing library available. Install PyMuPDF or PyPDF2.")

    def _process_pdf_pymupdf(self, file_path: Path) -> str:
        """Process PDF using PyMuPDF."""
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text

    def _process_pdf_pypdf2(self, file_path: Path) -> str:
        """Process PDF using PyPDF2."""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text

    def _process_txt(self, file_path: Path) -> str:
        """Process text file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def _process_sql(self, file_path: Path) -> str:
        """Process SQL file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            sql_content = f.read()

        # Extract meaningful information from SQL
        extracted_info = f"SQL Analysis:\n"
        extracted_info += f"Tables mentioned: {sql_content.count('TABLE')}\n"
        extracted_info += f"SELECT statements: {sql_content.count('SELECT')}\n"
        extracted_info += f"INSERT statements: {sql_content.count('INSERT')}\n"
        extracted_info += f"Content:\n{sql_content}"

        return extracted_info

    def _process_json(self, file_path: Path) -> str:
        """Process JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert JSON to readable text
        return json.dumps(data, indent=2, ensure_ascii=False)

    def _process_csv(self, file_path: Path) -> str:
        """Process CSV file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Read and analyze CSV
            sample = f.read(1024)
            f.seek(0)

            # Detect delimiter
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter

            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)

        # Convert to readable text
        text = f"CSV Analysis: {len(rows)} rows\n"
        if rows:
            text += f"Headers: {', '.join(rows[0])}\n"
            text += "Sample data:\n"
            for row in rows[:10]:  # First 10 rows
                text += ', '.join(str(cell) for cell in row) + '\n'

        return text

    def batch_convert(self, input_dir: Union[str, Path], output_dir: Union[str, Path] = None):
        """Batch convert files to ΨCWS format."""
        input_dir = Path(input_dir)

        if output_dir is None:
            output_dir = input_dir / "cwm_output"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all supported files
        supported_extensions = list(self.processors.keys())
        files_to_process = []

        for ext in supported_extensions:
            files_to_process.extend(input_dir.glob(f"*.{ext}"))

        print(f"🔄 Batch converting {len(files_to_process)} files")

        results = []
        for file_path in files_to_process:
            try:
                Ψcws_file = self.process_file(file_path)
                output_path = output_dir / f"{file_path.stem}.Ψcws"
                Ψcws_file.save(output_path)
                results.append({'file': file_path.name, 'status': 'success', 'output': output_path.name})
                print(f"✅ {file_path.name} → {output_path.name}")
            except Exception as e:
                results.append({'file': file_path.name, 'status': 'error', 'error': str(e)})
                print(f"❌ {file_path.name}: {e}")

        return results
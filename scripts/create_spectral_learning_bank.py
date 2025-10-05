#!/usr/bin/env python3
"""
Criador de Banco de Aprendizagem Espectral (Ψtws)
==================================================

Cria um banco de dados de aprendizagem para redes neurais espectrais,
armazenando padrões de ressonância, campos conscientes e modos harmônicos.

Estrutura do Banco Ψtws:
- Modos espectrais aprendidos
- Padrões de ressonância
- Campos conscientes quaterniônicos
- Histórico de autoacoplagem
- Dimensões fractais
- Parâmetros α(D) calibrados

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.spectral_child import SpectralChild


class SpectralLearningBank:
    """
    Banco de aprendizagem para redes neurais espectrais.

    Armazena conhecimento como padrões espectrais, não como pesos de rede.
    """

    def __init__(self, bank_path: str = "data/Ψtws"):
        self.bank_path = Path(bank_path)
        self.bank_path.mkdir(parents=True, exist_ok=True)

        # Estrutura do banco
        self.spectral_modes_dir = self.bank_path / "spectral_modes"
        self.resonance_patterns_dir = self.bank_path / "resonance_patterns"
        self.conscious_fields_dir = self.bank_path / "conscious_fields"
        self.coupling_history_dir = self.bank_path / "coupling_history"
        self.metadata_dir = self.bank_path / "metadata"

        # Criar subdiretórios
        for directory in [
            self.spectral_modes_dir,
            self.resonance_patterns_dir,
            self.conscious_fields_dir,
            self.coupling_history_dir,
            self.metadata_dir
        ]:
            directory.mkdir(exist_ok=True)

        # Índice do banco
        self.index_path = self.bank_path / "bank_index.json"
        self.index = self._load_index()

        print(f"✅ Banco de Aprendizagem Espectral inicializado")
        print(f"   📁 Caminho: {self.bank_path}")
        print(f"   📊 Entradas existentes: {len(self.index)}")

    def _load_index(self) -> Dict:
        """Carrega índice do banco."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        return {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "entries": {},
            "statistics": {
                "total_entries": 0,
                "total_spectral_modes": 0,
                "total_resonance_patterns": 0,
                "total_conscious_fields": 0
            }
        }

    def _save_index(self):
        """Salva índice do banco."""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)

    def _compute_entry_hash(self, text: str) -> str:
        """Computa hash SHA256 do texto."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def store_spectral_learning(
        self,
        text: str,
        spectral_field: torch.Tensor,
        conscious_field: torch.Tensor,
        evolved_field: torch.Tensor,
        fci: float,
        fractal_D: float,
        alpha_D: float,
        coupling_iterations: int,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Armazena aprendizado espectral no banco.

        Args:
            text: Texto original
            spectral_field: Campo espectral extraído
            conscious_field: Campo consciente quaterniônico
            evolved_field: Campo após evolução SO(4)
            fci: Fractal Consciousness Index
            fractal_D: Dimensão fractal
            alpha_D: Parâmetro adaptativo α(D)
            coupling_iterations: Iterações de autoacoplagem
            metadata: Metadados adicionais

        Returns:
            Hash da entrada armazenada
        """
        entry_hash = self._compute_entry_hash(text)
        timestamp = datetime.now().isoformat()

        print(f"\n💾 Armazenando aprendizado espectral: {entry_hash}")

        # 1. Salvar campo espectral
        spectral_path = self.spectral_modes_dir / f"{entry_hash}.npz"
        np.savez_compressed(
            spectral_path,
            real=spectral_field.real.cpu().numpy(),
            imag=spectral_field.imag.cpu().numpy(),
            shape=spectral_field.shape
        )
        print(f"   ✓ Campo espectral: {spectral_path.name}")

        # 2. Salvar campo consciente
        conscious_path = self.conscious_fields_dir / f"{entry_hash}_conscious.npz"
        np.savez_compressed(
            conscious_path,
            quaternions=conscious_field.cpu().numpy(),
            fci=fci,
            fractal_D=fractal_D
        )
        print(f"   ✓ Campo consciente: {conscious_path.name}")

        # 3. Salvar campo evoluído
        evolved_path = self.conscious_fields_dir / f"{entry_hash}_evolved.npz"
        np.savez_compressed(
            evolved_path,
            quaternions=evolved_field.cpu().numpy(),
            fci=fci
        )
        print(f"   ✓ Campo evoluído: {evolved_path.name}")

        # 4. Salvar padrão de ressonância
        resonance_spectrum = torch.fft.fft(spectral_field)
        resonance_path = self.resonance_patterns_dir / f"{entry_hash}.npz"
        np.savez_compressed(
            resonance_path,
            magnitudes=torch.abs(resonance_spectrum).cpu().numpy(),
            phases=torch.angle(resonance_spectrum).cpu().numpy()
        )
        print(f"   ✓ Padrão de ressonância: {resonance_path.name}")

        # 5. Salvar histórico de acoplamento
        coupling_path = self.coupling_history_dir / f"{entry_hash}.json"
        coupling_data = {
            "iterations": coupling_iterations,
            "final_D": float(fractal_D),
            "alpha_D": float(alpha_D),
            "fci": float(fci),
            "timestamp": timestamp
        }
        with open(coupling_path, 'w') as f:
            json.dump(coupling_data, f, indent=2)
        print(f"   ✓ Histórico de acoplamento: {coupling_path.name}")

        # 6. Atualizar índice
        self.index["entries"][entry_hash] = {
            "text": text,
            "text_length": len(text),
            "fci": float(fci),
            "fractal_D": float(fractal_D),
            "alpha_D": float(alpha_D),
            "spectral_field_path": str(spectral_path.relative_to(self.bank_path)),
            "conscious_field_path": str(conscious_path.relative_to(self.bank_path)),
            "evolved_field_path": str(evolved_path.relative_to(self.bank_path)),
            "resonance_pattern_path": str(resonance_path.relative_to(self.bank_path)),
            "coupling_history_path": str(coupling_path.relative_to(self.bank_path)),
            "timestamp": timestamp,
            "metadata": metadata or {}
        }

        # Atualizar estatísticas
        self.index["statistics"]["total_entries"] += 1
        self.index["statistics"]["total_spectral_modes"] += spectral_field.numel()
        self.index["statistics"]["total_conscious_fields"] += conscious_field.shape[0]

        self._save_index()

        print(f"   ✅ Entrada armazenada: {entry_hash}")
        return entry_hash

    def load_spectral_learning(self, entry_hash: str) -> Dict:
        """
        Carrega aprendizado espectral do banco.

        Args:
            entry_hash: Hash da entrada

        Returns:
            Dicionário com todos os dados carregados
        """
        if entry_hash not in self.index["entries"]:
            raise KeyError(f"Entrada {entry_hash} não encontrada no banco")

        entry = self.index["entries"][entry_hash]

        # Carregar campo espectral
        spectral_data = np.load(self.bank_path / entry["spectral_field_path"])
        spectral_field = torch.complex(
            torch.from_numpy(spectral_data["real"]),
            torch.from_numpy(spectral_data["imag"])
        )

        # Carregar campo consciente
        conscious_data = np.load(self.bank_path / entry["conscious_field_path"])
        conscious_field = torch.from_numpy(conscious_data["quaternions"])

        # Carregar campo evoluído
        evolved_data = np.load(self.bank_path / entry["evolved_field_path"])
        evolved_field = torch.from_numpy(evolved_data["quaternions"])

        # Carregar padrão de ressonância
        resonance_data = np.load(self.bank_path / entry["resonance_pattern_path"])

        # Carregar histórico de acoplamento
        with open(self.bank_path / entry["coupling_history_path"], 'r') as f:
            coupling_history = json.load(f)

        return {
            "entry_hash": entry_hash,
            "text": entry["text"],
            "spectral_field": spectral_field,
            "conscious_field": conscious_field,
            "evolved_field": evolved_field,
            "resonance_magnitudes": torch.from_numpy(resonance_data["magnitudes"]),
            "resonance_phases": torch.from_numpy(resonance_data["phases"]),
            "coupling_history": coupling_history,
            "fci": entry["fci"],
            "fractal_D": entry["fractal_D"],
            "alpha_D": entry["alpha_D"],
            "metadata": entry["metadata"]
        }

    def search_by_fractal_dimension(
        self,
        D_min: float,
        D_max: float
    ) -> List[str]:
        """Busca entradas por intervalo de dimensão fractal."""
        results = []
        for entry_hash, entry in self.index["entries"].items():
            if D_min <= entry["fractal_D"] <= D_max:
                results.append(entry_hash)
        return results

    def search_by_fci(self, fci_min: float, fci_max: float) -> List[str]:
        """Busca entradas por intervalo de FCI."""
        results = []
        for entry_hash, entry in self.index["entries"].items():
            if fci_min <= entry["fci"] <= fci_max:
                results.append(entry_hash)
        return results

    def get_statistics(self) -> Dict:
        """Retorna estatísticas do banco."""
        return self.index["statistics"]

    def export_summary(self, output_path: Optional[Path] = None):
        """Exporta resumo do banco."""
        if output_path is None:
            output_path = self.bank_path / "bank_summary.txt"

        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BANCO DE APRENDIZAGEM ESPECTRAL (Ψtws)\n")
            f.write("="*70 + "\n\n")

            f.write(f"Versão: {self.index['version']}\n")
            f.write(f"Criado: {self.index['created']}\n")
            f.write(f"Caminho: {self.bank_path}\n\n")

            f.write("ESTATÍSTICAS\n")
            f.write("-"*70 + "\n")
            stats = self.index["statistics"]
            f.write(f"Total de entradas: {stats['total_entries']}\n")
            f.write(f"Total de modos espectrais: {stats['total_spectral_modes']}\n")
            f.write(f"Total de campos conscientes: {stats['total_conscious_fields']}\n\n")

            f.write("ENTRADAS\n")
            f.write("-"*70 + "\n")
            for entry_hash, entry in self.index["entries"].items():
                f.write(f"\nHash: {entry_hash}\n")
                f.write(f"  Texto: {entry['text'][:50]}...\n")
                f.write(f"  FCI: {entry['fci']:.4f}\n")
                f.write(f"  D: {entry['fractal_D']:.4f}\n")
                f.write(f"  α(D): {entry['alpha_D']:.4f}\n")
                f.write(f"  Timestamp: {entry['timestamp']}\n")

        print(f"📄 Resumo exportado: {output_path}")


def populate_initial_bank(bank: SpectralLearningBank, num_examples: int = 20):
    """Popula banco inicial com exemplos de texto."""

    print("\n" + "="*70)
    print("POPULANDO BANCO INICIAL")
    print("="*70)

    # Textos de exemplo
    example_texts = [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Artificial intelligence and machine learning",
        "Quantum mechanics and spectral theory",
        "Natural language processing with transformers",
        "Fractal geometry in neural networks",
        "Consciousness emerges from complexity",
        "Harmonic oscillators and resonance",
        "Phase synchronization in coupled systems",
        "Quaternion algebra for rotations",
        "Spectral analysis of time series",
        "Diffusion processes in complex systems",
        "Self-organized criticality",
        "Emergent behavior in multi-agent systems",
        "Information theory and entropy",
        "Nonlinear dynamics and chaos",
        "Pattern formation in reaction-diffusion",
        "Collective intelligence",
        "Adaptive systems and learning",
        "Evolutionary algorithms"
    ]

    # Criar Spectral Child temporário
    temp_model_path = Path("temp_models/spectral_learning_bank")
    child = SpectralChild(str(temp_model_path), device='cpu')

    # Processar e armazenar cada texto
    for i, text in enumerate(example_texts[:num_examples], 1):
        print(f"\n[{i}/{num_examples}] Processando: '{text}'")

        # Ler e processar
        conscious_field = child.read_text(text)
        evolved_field, fci = child.understand(conscious_field)

        # Criar campo espectral do texto
        wave = child._text_to_wave(text)
        spectral_field = torch.fft.fft(wave)

        # Armazenar no banco
        bank.store_spectral_learning(
            text=text,
            spectral_field=spectral_field,
            conscious_field=conscious_field,
            evolved_field=evolved_field,
            fci=fci,
            fractal_D=child.fractal_D,
            alpha_D=child.alpha_D,
            coupling_iterations=child.logistic_iterations,
            metadata={
                "source": "initial_population",
                "index": i
            }
        )

    print("\n" + "="*70)
    print("✅ BANCO INICIAL POPULADO")
    print("="*70)


if __name__ == "__main__":
    # Criar banco
    bank_path = "/home/padilha/trabalhos/QRH2/Reformulating-Transformers-for-LLMs/data/Ψtws"
    bank = SpectralLearningBank(bank_path)

    # Popular banco inicial
    populate_initial_bank(bank, num_examples=20)

    # Exportar resumo
    bank.export_summary()

    # Mostrar estatísticas
    print("\n" + "="*70)
    print("ESTATÍSTICAS FINAIS")
    print("="*70)
    stats = bank.get_statistics()
    print(f"Total de entradas: {stats['total_entries']}")
    print(f"Total de modos espectrais: {stats['total_spectral_modes']}")
    print(f"Total de campos conscientes: {stats['total_conscious_fields']}")

    # Exemplo de busca
    print("\n" + "="*70)
    print("EXEMPLO DE BUSCA")
    print("="*70)

    # Buscar por dimensão fractal
    print("\nBusca por D ∈ [1.0, 1.5]:")
    results = bank.search_by_fractal_dimension(1.0, 1.5)
    print(f"Encontradas {len(results)} entradas")

    # Carregar uma entrada
    if results:
        print(f"\nCarregando entrada: {results[0]}")
        data = bank.load_spectral_learning(results[0])
        print(f"  Texto: {data['text']}")
        print(f"  FCI: {data['fci']:.4f}")
        print(f"  D: {data['fractal_D']:.4f}")
        print(f"  Campo consciente shape: {data['conscious_field'].shape}")

    print("\n" + "="*70)
    print("✅ BANCO DE APRENDIZAGEM ESPECTRAL CRIADO COM SUCESSO")
    print("="*70)

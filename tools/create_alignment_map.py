import torch
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from psiqrh import ΨQRHPipeline
from src.core.quantum_linguistic_genesis import QuantumLinguisticGenesis

def create_map():
    print("🔬 Creating alignment map from Quantum Linguistic Genesis...")

    # Initialize Quantum Linguistic Genesis system
    genesis = QuantumLinguisticGenesis(embed_dim=64, device='cpu')
    quantum_vocab, char_to_idx = genesis.get_quantum_vocabulary_tensor()

    print(f"🧬 Using Genesis vocabulary: {len(quantum_vocab)} linguistic primitives")
    print(f"   🔬 Tensor shape: {quantum_vocab.shape}")

    # Create spectral map directly from genesis vocabulary
    # quantum_vocab is already [vocab_size, embed_dim, 4]
    spectral_map = quantum_vocab.clone()

    # Ensure proper shape and normalization
    spectral_map = spectral_map.float()

    # Save the new spectral map
    output_path = "data/spectral_vocab_map.pt"
    torch.save(spectral_map, output_path)

    print(f"✅ New spectral alignment map saved to: {output_path}")
    print(f"   Shape: {spectral_map.shape} (vocab_size={spectral_map.shape[0]}, embed_dim={spectral_map.shape[1]})")
    print(f"   Data type: {spectral_map.dtype}")
    print(f"   Device: {spectral_map.device}")

    # Verify the map was saved correctly
    if os.path.exists(output_path):
        loaded_map = torch.load(output_path)
        if torch.equal(loaded_map, spectral_map):
            print("✅ Map verification successful")
        else:
            print("⚠️  Map verification failed - data mismatch")
    else:
        print("❌ Map file not found after saving")

    # Show sample mappings
    print("\n🔍 Sample character mappings:")
    chars = list(char_to_idx.keys())[:10]  # Show first 10 characters
    for char in chars:
        idx = char_to_idx[char]
        state = spectral_map[idx]
        norm = torch.norm(state).item()
        print(f"   '{char}': idx={idx}, norm={norm:.4f}")

if __name__ == "__main__":
    create_map()
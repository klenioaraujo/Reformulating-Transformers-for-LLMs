#!/usr/bin/env python3
"""
Teste de conversão PDF→CWM usando ΨQRH ConsciousWaveModulator
=============================================================

Converte o PDF específico d41d8cd98f00b204e9800998ecf8427e.pdf para formato .cwm
usando o pipeline de consciência fractal.
"""

import sys
import os
from pathlib import Path
import traceback

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_pdf_to_cwm():
    """Testa conversão do PDF específico para .cwm"""

    pdf_path = Path("/home/padilha/trabalhos/Reformulating_Transformers/src/conceptual/models/d41d8cd98f00b204e9800998ecf8427e.pdf")

    print("🔮 ΨQRH PDF→CWM Conversion Test")
    print("=" * 50)
    print(f"📄 PDF: {pdf_path.name}")
    print(f"📊 Size: {pdf_path.stat().st_size / (1024*1024):.2f} MB")

    try:
        # Import ConsciousWaveModulator
        from src.conscience.conscious_wave_modulator import ConsciousWaveModulator

        print("✅ ConsciousWaveModulator imported successfully")

        # Create cache directory
        cache_dir = Path("data/cwm_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Configure modulator
        config = {
            'cache_dir': str(cache_dir),
            'embedding_dim': 256,
            'sequence_length': 64,
            'device': 'cpu',
            'base_amplitude': 1.0,
            'frequency_range': [0.5, 5.0],
            'phase_consciousness': 0.7854,  # π/4
            'chaotic_r': 3.9,
            'cache_enabled': True,
            'compression': True
        }

        # Initialize modulator
        print("🌊 Initializing ConsciousWaveModulator...")
        modulator = ConsciousWaveModulator(config)

        # Process PDF
        print("🔄 Processing PDF through consciousness pipeline...")
        cwm_file = modulator.process_file(pdf_path)

        # Generate output path with hash format
        import hashlib
        file_stat = pdf_path.stat()
        hash_input = f"{pdf_path.absolute()}_{file_stat.st_mtime}"
        file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:16]

        output_path = cache_dir / f"{file_hash}_{pdf_path.stem}.cwm"

        # Save CWM file
        print(f"💾 Saving to: {output_path}")
        cwm_file.save(output_path)

        print("\n✨ Conversion completed successfully!")
        print("=" * 50)

        # Display results
        print("📊 CWM File Analysis:")
        print(f"   Header magic: {cwm_file.header.magic_number}")
        print(f"   File type: {cwm_file.header.file_type}")
        print(f"   Content hash: {cwm_file.header.content_hash[:16]}...")
        print(f"   Timestamp: {cwm_file.header.timestamp}")

        print("\n🌊 Wave Parameters:")
        wave_params = cwm_file.header.wave_parameters
        print(f"   Amplitude base: {wave_params['amplitude_base']}")
        print(f"   Frequency range: {wave_params['frequency_range']}")
        print(f"   Phase offsets: {[f'{p:.4f}' for p in wave_params['phase_offsets']]}")
        print(f"   Chaotic seed: {wave_params['chaotic_seed']}")

        print("\n🧠 Consciousness Metrics:")
        metrics = cwm_file.spectral_data.consciousness_metrics
        print(f"   Complexity: {metrics['complexity']:.4f}")
        print(f"   Coherence: {metrics['coherence']:.4f}")
        print(f"   Adaptability: {metrics['adaptability']:.4f}")
        print(f"   Integration: {metrics['integration']:.4f}")

        print("\n📈 Spectral Data:")
        if cwm_file.spectral_data.wave_embeddings is not None:
            embeddings = cwm_file.spectral_data.wave_embeddings
            print(f"   Wave embeddings shape: {embeddings.shape}")
            print(f"   Mean amplitude: {embeddings.mean().item():.6f}")
            print(f"   Std amplitude: {embeddings.std().item():.6f}")

        if cwm_file.spectral_data.chaotic_trajectories is not None:
            trajectories = cwm_file.spectral_data.chaotic_trajectories
            print(f"   Chaotic trajectory points: {len(trajectories)}")
            print(f"   Trajectory range: [{trajectories.min().item():.4f}, {trajectories.max().item():.4f}]")

        if cwm_file.qrh_tensor is not None:
            qrh = cwm_file.qrh_tensor
            print(f"   QRH tensor shape: {qrh.shape}")
            print(f"   QRH quaternion dims: {qrh.shape[-1] // 4} per component")

        print("\n📝 Content Metadata:")
        print(f"   Source: {cwm_file.content_metadata.original_source}")
        print(f"   Text length: {len(cwm_file.content_metadata.extracted_text)} chars")
        print(f"   Key concepts: {len(cwm_file.content_metadata.key_concepts)}")
        if cwm_file.content_metadata.key_concepts:
            print(f"   Top concepts: {', '.join(cwm_file.content_metadata.key_concepts[:5])}")

        print(f"\n💾 Output file: {output_path}")
        output_size = output_path.stat().st_size
        print(f"📊 CWM file size: {output_size / (1024*1024):.2f} MB")
        compression_ratio = (pdf_path.stat().st_size - output_size) / pdf_path.stat().st_size * 100
        print(f"📉 Compression: {compression_ratio:.1f}%")

        # Test loading the file
        print("\n🔄 Testing CWM file loading...")
        from src.conscience.conscious_wave_modulator import CWMFile
        loaded_cwm = CWMFile.load(output_path)
        print("✅ CWM file loaded successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install torch numpy PyMuPDF")
        return False

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        print("\n📋 Full traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pdf_to_cwm()
    if success:
        print("\n🎉 Test completed successfully!")
        exit(0)
    else:
        print("\n💥 Test failed!")
        exit(1)
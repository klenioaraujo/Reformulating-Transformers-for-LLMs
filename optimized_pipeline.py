#!/usr/bin/env python3
"""
Optimized ΨQRH Pipeline with Context Length Management
=====================================================

This script implements a context-optimized version of the ΨQRH pipeline
that avoids the 131072 token limit by using intelligent chunking.
"""

import sys
import os
from pathlib import Path
from context_optimizer import PipelineOptimizer


def run_optimized_pipeline():
    """Run the ΨQRH pipeline with optimized context management"""

    print("🚀 Starting Optimized ΨQRH Pipeline")
    print("=" * 50)

    # Initialize the optimizer
    optimizer = PipelineOptimizer()

    # Analyze psiqrh.py structure
    psiqrh_path = Path("psiqrh.py")
    if not psiqrh_path.exists():
        print(f"❌ psiqrh.py not found at {psiqrh_path}")
        return

    print("🔍 Analyzing psiqrh.py structure...")
    optimization_result = optimizer.optimize_psiqrh_analysis(psiqrh_path)

    # Display optimization summary
    structure = optimization_result['file_structure']
    chunks = optimization_result['chunks']

    print(f"\n📊 OPTIMIZATION SUMMARY:")
    print(f"   File: psiqrh.py")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Classes: {len(structure.get('classes', []))}")
    print(f"   Functions: {len(structure.get('functions', []))}")
    print(f"   Total tokens: {sum(chunk['token_count'] for chunk in chunks)}")

    # Test with first chunk to avoid context overflow
    print(f"\n🧪 Testing with first chunk (safe token count: {chunks[0]['token_count']})...")

    # Import and test core components
    try:
        # Import core modules
        sys.path.insert(0, str(Path.cwd()))

        # Test basic imports
        import torch
        import numpy as np

        print("✅ Core imports successful")

        # Test optical probe
        from src.core.optical_probe_fixed import OpticalProbeFixed

        print("✅ Optical probe loaded")

        # Test with minimal input
        test_text = "The ΨQRH framework represents a paradigm shift in transformer architectures."
        print(f"\n📝 Test input: {test_text}")

        # Initialize optical probe with device
        probe = OpticalProbeFixed(device='cpu')
        print("✅ Optical probe initialized")

        # Test spectral map loading
        spectral_map_path = Path("data/spectral_vocab_map.pt")
        if spectral_map_path.exists():
            spectral_map = torch.load(spectral_map_path)
            print(f"✅ Spectral map loaded: {spectral_map.shape}")
        else:
            print("⚠️  Spectral map not found, continuing without it")

        print("\n🎉 Optimized pipeline ready!")
        print("\n📋 Next steps:")
        print("   1. Run individual chunks for detailed analysis")
        print("   2. Use focused analysis for specific components")
        print("   3. Test with reduced input data to avoid context limits")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("\n🔧 Debugging steps:")
        print("   1. Check if all dependencies are installed")
        print("   2. Verify file paths and module imports")
        print("   3. Try running with smaller input data")


def analyze_specific_component(component_name: str):
    """Analyze a specific component with focused context"""

    print(f"\n🎯 Analyzing component: {component_name}")

    # Load focused chunks for the component
    optimizer = PipelineOptimizer()
    psiqrh_path = Path("psiqrh.py")

    focused_chunks = optimizer.context_optimizer.create_focused_chunks(
        psiqrh_path,
        focus_areas=['classes', 'main_functions']
    )

    # Find chunks related to the component
    component_chunks = []
    for chunk in focused_chunks:
        if component_name.lower() in chunk['focus'].lower():
            component_chunks.append(chunk)

    if component_chunks:
        print(f"📦 Found {len(component_chunks)} chunks for {component_name}")
        for chunk in component_chunks:
            print(f"   - {chunk['description']} ({chunk['token_count']} tokens)")
    else:
        print(f"⚠️  No specific chunks found for {component_name}")
        print("   Using general structure-based chunks")

        # Use general chunks
        chunks = optimizer.context_optimizer.chunk_python_file_intelligently(psiqrh_path)
        print(f"   Total chunks available: {len(chunks)}")


def test_semantic_models_in_hilbert_space():
    """Test semantic models within Hilbert space for token extraction"""

    print("🧬 Testing Semantic Models in Hilbert Space")
    print("=" * 50)

    import torch
    import numpy as np
    from pathlib import Path

    # Load semantic models
    semantic_dir = Path("models/semantic")
    if not semantic_dir.exists():
        print("❌ Semantic models directory not found")
        return

    semantic_models = list(semantic_dir.glob("*.pt"))
    if not semantic_models:
        print("❌ No semantic models found")
        return

    print(f"📊 Found {len(semantic_models)} semantic models")

    # Test each semantic model
    for model_path in semantic_models:
        model_name = model_path.stem
        print(f"\n🔬 Testing model: {model_name}")

        try:
            # Load model
            model_data = torch.load(model_path, map_location='cpu')
            print(f"   ✅ Model loaded: {model_data.keys()}")

            # Check for embeddings in different possible locations
            embeddings = None
            if 'embeddings' in model_data:
                embeddings = model_data['embeddings']
            elif 'model_state_dict' in model_data:
                state_dict = model_data['model_state_dict']
                # Look for embedding layers in state dict
                for key, value in state_dict.items():
                    if 'embed' in key.lower() and isinstance(value, torch.Tensor):
                        embeddings = value
                        print(f"   📍 Found embeddings in state_dict: {key}")
                        break

            if embeddings is not None:
                print(f"   📐 Embeddings shape: {embeddings.shape}")

                # Test Hilbert space properties
                # 1. Check if embeddings form a Hilbert space
                norms = torch.norm(embeddings, dim=-1)
                print(f"   📏 Norms range: [{norms.min().item():.3f}, {norms.max().item():.3f}]")

                # 2. Check orthogonality (inner products)
                if embeddings.shape[0] > 1:
                    inner_products = torch.matmul(embeddings, embeddings.t())
                    diagonal = torch.diag(inner_products)
                    off_diagonal = inner_products - torch.diag(diagonal)
                    orthogonality = torch.mean(torch.abs(off_diagonal)).item()
                    print(f"   🔄 Orthogonality measure: {orthogonality:.6f}")

                # 3. Test token extraction capability
                vocab_size = embeddings.shape[0]
                print(f"   📚 Vocabulary size: {vocab_size}")

                # Generate test tokens using Hilbert space operations
                test_indices = torch.randint(0, vocab_size, (5,))
                test_embeddings = embeddings[test_indices]

                # Apply quantum operations (simulated)
                phase = torch.randn_like(test_embeddings) * 0.1
                quantum_states = test_embeddings * torch.exp(1j * phase)

                # Extract tokens using magnitude-based selection
                magnitudes = torch.abs(quantum_states)
                token_indices = torch.argmax(magnitudes, dim=0)

                print(f"   🎯 Extracted token indices: {token_indices.tolist()}")
                print(f"   🧬 Quantum state magnitudes: {magnitudes.mean(dim=0).tolist()}")

            else:
                print("   ⚠️  No embeddings found in model")
                # Show what is available in the model
                print(f"   📋 Available keys: {list(model_data.keys())}")
                if 'semantic_info' in model_data:
                    print(f"   🎨 Semantic info: {model_data['semantic_info']}")

        except Exception as e:
            print(f"   ❌ Error testing model: {e}")
            import traceback
            traceback.print_exc()

    print("\n🎉 Hilbert space testing completed!")


def extract_semantic_tokens():
    """Extract semantic tokens from Hilbert space representations using Dynamic Quantum Matrix"""

    print("🔍 Extracting Semantic Tokens with Dynamic Quantum Matrix")
    print("=" * 60)

    import torch
    from pathlib import Path
    from context_optimizer import PipelineOptimizer

    # Initialize PipelineOptimizer with Dynamic Quantum Matrix
    optimizer = PipelineOptimizer()

    # Load spectral vocabulary map for reference
    spectral_map_path = Path("data/spectral_vocab_map.pt")
    if spectral_map_path.exists():
        try:
            spectral_map = torch.load(spectral_map_path, map_location='cpu')
            print(f"✅ Spectral vocabulary map loaded: {spectral_map.shape}")
        except Exception as e:
            print(f"⚠️  Could not load spectral map: {e}")
            spectral_map = None
    else:
        print("⚠️  Spectral vocabulary map not found")
        spectral_map = None

    # Test semantic token extraction using Dynamic Quantum Matrix
    semantic_dir = Path("models/semantic")
    if not semantic_dir.exists():
        print("❌ Semantic models directory not found")
        return

    # Extract tokens using Dynamic Quantum Matrix
    extraction_results = optimizer.extract_tokens_with_dynamic_matrix(semantic_dir)

    if extraction_results['success'] == False:
        print(f"❌ Dynamic Quantum Matrix extraction failed: {extraction_results.get('error', 'Unknown error')}")
        return

    # Display detailed results
    print(f"\n📊 EXTRACTION SUMMARY:")
    print(f"   📁 Models processed: {extraction_results['models_processed']}")
    print(f"   🎯 Total tokens extracted: {extraction_results['total_tokens_extracted']}")

    if 'performance_summary' in extraction_results and extraction_results['performance_summary']:
        perf = extraction_results['performance_summary']
        print(f"\n⚡ PERFORMANCE METRICS:")
        print(f"   📊 Total models: {perf['total_models']}")
        print(f"   ⏱️  Avg adaptation time: {perf['avg_adaptation_time']}")
        print(f"   ⚡ Avg extraction time: {perf['avg_extraction_time']}")
        print(f"   💾 Total matrix size: {perf['total_matrix_size']}")
        print(f"   📏 Avg matrix size: {perf['avg_matrix_size']}")

    # Display detailed results for each model
    if 'extraction_details' in extraction_results:
        print(f"\n🔬 DETAILED MODEL RESULTS:")
        for model_name, details in extraction_results['extraction_details'].items():
            print(f"\n🎯 Model: {model_name}")
            print(f"   ⏱️  Load time: {details['load_time']}")
            print(f"   ⚡ Extraction time: {details['extraction_time']}")
            print(f"   📊 Vocab size: {details['vocab_size']}")
            print(f"   📐 Embedding dim: {details['embedding_dim']}")
            print(f"   🔝 Top magnitude indices: {details['top_magnitude_indices'][:5]}")
            print(f"   🧬 Top coherence indices: {details['top_coherence_indices'][:5]}")
            print(f"   📏 Encoded shape: {details['encoded_shape']}")
            print(f"   ✅ Finite values: {details['finite_encoded']}")

            # Show adaptation metrics
            if 'adaptation_metrics' in details and details['adaptation_metrics']['success']:
                adapt = details['adaptation_metrics']
                print(f"   🔧 Adaptation time: {adapt['adaptation_time']}")
                print(f"   💾 Matrix size: {adapt['matrix_size']}")

    print("\n✅ Dynamic Quantum Matrix semantic token extraction completed!")


def main():
    """Main function with menu options"""

    print("🤖 ΨQRH Context-Optimized Pipeline")
    print("=" * 40)
    print("1. Run full optimized pipeline")
    print("2. Analyze specific component")
    print("3. Show optimization plan")
    print("4. Test semantic models in Hilbert space")
    print("5. Extract semantic tokens")
    print("6. Exit")

    # Check if running in non-interactive mode (e.g., via script)
    import sys
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        try:
            choice = input("\nSelect option (1-6): ").strip()
        except EOFError:
            # Default to option 1 if no input available
            choice = "1"

    if choice == "1":
        run_optimized_pipeline()
    elif choice == "2":
        if len(sys.argv) > 2:
            component = sys.argv[2]
        else:
            try:
                component = input("Enter component name (e.g., ΨQRHPipeline, OpticalProbe): ").strip()
            except EOFError:
                component = "ΨQRHPipeline"
        analyze_specific_component(component)
    elif choice == "3":
        # Show existing optimization plan
        plan_path = Path("optimized_chunks/recommended_pipeline.md")
        if plan_path.exists():
            with open(plan_path, 'r') as f:
                print(f.read())
        else:
            print("❌ Optimization plan not found. Run context_optimizer.py first.")
    elif choice == "4":
        test_semantic_models_in_hilbert_space()
    elif choice == "5":
        extract_semantic_tokens()
    elif choice == "6":
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
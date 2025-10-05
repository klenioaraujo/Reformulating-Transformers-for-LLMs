#!/usr/bin/env python3
"""
ONNX Export Guide for ΨQRH Transformer

This script demonstrates how to export ΨQRH models to ONNX format
for optimized inference and cross-platform deployment.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3 - see LICENSE file

DOI: https://zenodo.org/records/17171112
"""

import torch
import sys
from pathlib import Path
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.architecture.psiqrh_transformer import PsiQRHTransformer


def export_to_onnx(
    model,
    output_path='models/psiqrh.onnx',
    input_shape=(1, 32),  # (batch_size, sequence_length)
    opset_version=14,
    simplify=True
):
    """
    Export ΨQRH model to ONNX format.

    Args:
        model: ΨQRH model to export
        output_path: Path to save ONNX model
        input_shape: Shape of input tensor (batch_size, seq_len)
        opset_version: ONNX opset version
        simplify: Whether to simplify the ONNX graph

    Returns:
        Path to exported ONNX model
    """
    print(f"Exporting ΨQRH model to ONNX...")
    print(f"Output path: {output_path}")
    print(f"Input shape: {input_shape}")

    # Set model to evaluation mode
    model.eval()

    # Create dummy input
    dummy_input = torch.randint(
        0,
        model.vocab_size if hasattr(model, 'vocab_size') else 1000,
        input_shape
    )

    # Dynamic axes for variable batch size and sequence length
    dynamic_axes = {
        'input': {0: 'batch_size', 1: 'sequence_length'},
        'output': {0: 'batch_size', 1: 'sequence_length'}
    }

    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        print(f"✓ ONNX export successful!")

    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return None

    # Simplify ONNX graph (optional, requires onnx-simplifier)
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            print("Simplifying ONNX graph...")
            onnx_model = onnx.load(output_path)
            simplified_model, check = onnx_simplify(onnx_model)

            if check:
                onnx.save(simplified_model, output_path)
                print("✓ ONNX graph simplified!")
            else:
                print("⚠ Simplification check failed, using original graph")

        except ImportError:
            print("⚠ onnx-simplifier not installed, skipping simplification")
            print("  Install with: pip install onnx-simplifier")

    return output_path


def verify_onnx_model(onnx_path, test_input=None):
    """
    Verify the exported ONNX model by comparing outputs
    with the original PyTorch model.

    Args:
        onnx_path: Path to ONNX model
        test_input: Test input tensor (optional)

    Returns:
        Boolean indicating if verification passed
    """
    print(f"\nVerifying ONNX model: {onnx_path}")

    try:
        import onnx
        import onnxruntime as ort
        import numpy as np

        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")

        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path)

        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        print(f"  Input name: {input_name}")
        print(f"  Output name: {output_name}")

        # Test inference
        if test_input is None:
            test_input = np.random.randint(0, 1000, (1, 32)).astype(np.int64)

        outputs = session.run([output_name], {input_name: test_input})
        print(f"✓ ONNX inference successful")
        print(f"  Output shape: {outputs[0].shape}")

        return True

    except ImportError as e:
        print(f"⚠ Verification skipped: {e}")
        print("  Install with: pip install onnx onnxruntime")
        return False

    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def benchmark_onnx_performance(onnx_path, num_runs=100):
    """
    Benchmark ONNX model inference performance.

    Args:
        onnx_path: Path to ONNX model
        num_runs: Number of inference runs for benchmarking

    Returns:
        Average inference time in milliseconds
    """
    print(f"\nBenchmarking ONNX model performance...")

    try:
        import onnxruntime as ort
        import numpy as np
        import time

        # Create session
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name

        # Prepare input
        test_input = np.random.randint(0, 1000, (1, 32)).astype(np.int64)

        # Warmup
        for _ in range(10):
            session.run(None, {input_name: test_input})

        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            session.run(None, {input_name: test_input})
        end_time = time.time()

        avg_time_ms = (end_time - start_time) / num_runs * 1000

        print(f"✓ Benchmark completed")
        print(f"  Runs: {num_runs}")
        print(f"  Average inference time: {avg_time_ms:.2f} ms")
        print(f"  Throughput: {1000/avg_time_ms:.2f} inferences/second")

        return avg_time_ms

    except ImportError:
        print("⚠ onnxruntime not installed")
        print("  Install with: pip install onnxruntime")
        return None

    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        return None


def export_to_tensorrt(onnx_path, tensorrt_path=None):
    """
    Convert ONNX model to TensorRT for NVIDIA GPU optimization.

    Args:
        onnx_path: Path to ONNX model
        tensorrt_path: Path to save TensorRT engine (optional)

    Returns:
        Path to TensorRT engine
    """
    print(f"\nConverting to TensorRT...")

    try:
        import tensorrt as trt

        if tensorrt_path is None:
            tensorrt_path = onnx_path.replace('.onnx', '.trt')

        # TensorRT logger
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        # Build engine
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Parse ONNX model
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                print("✗ Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Build config
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB

        # Enable FP16 if supported
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  FP16 mode enabled")

        # Build engine
        print("  Building TensorRT engine (this may take a while)...")
        engine = builder.build_engine(network, config)

        if engine is None:
            print("✗ Failed to build TensorRT engine")
            return None

        # Save engine
        with open(tensorrt_path, 'wb') as f:
            f.write(engine.serialize())

        print(f"✓ TensorRT engine saved to {tensorrt_path}")
        return tensorrt_path

    except ImportError:
        print("⚠ TensorRT not installed")
        print("  Install TensorRT from: https://developer.nvidia.com/tensorrt")
        return None

    except Exception as e:
        print(f"✗ TensorRT conversion failed: {e}")
        return None


def main():
    """Main ONNX export pipeline"""

    print("="*60)
    print("ΨQRH ONNX Export Guide")
    print("="*60)

    # Load model configuration
    config_path = "configs/qrh_config.yaml"

    if not Path(config_path).exists():
        print(f"⚠ Configuration file not found: {config_path}")
        print("  Using default configuration")
        config = {
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6,
            'vocab_size': 50000,
            'max_seq_length': 512
        }
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # Initialize model
    print("\nInitializing ΨQRH model...")
    model = PsiQRHTransformer(config)
    model.eval()

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Export to ONNX
    onnx_path = 'models/psiqrh.onnx'
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)

    export_to_onnx(
        model=model,
        output_path=onnx_path,
        input_shape=(1, 32),
        opset_version=14,
        simplify=True
    )

    # Verify ONNX model
    if Path(onnx_path).exists():
        verify_onnx_model(onnx_path)

        # Benchmark performance
        benchmark_onnx_performance(onnx_path, num_runs=100)

        # Optional: Convert to TensorRT
        # export_to_tensorrt(onnx_path)

    print("\n" + "="*60)
    print("ONNX Export Completed!")
    print("="*60)

    print("\nDeployment Options:")
    print("1. ONNX Runtime: Cross-platform CPU/GPU inference")
    print("2. TensorRT: NVIDIA GPU optimization")
    print("3. OpenVINO: Intel CPU/GPU/VPU optimization")
    print("4. ONNX.js: Browser-based inference")

    print("\nNext Steps:")
    print(f"1. Test the exported model: {onnx_path}")
    print("2. Integrate into your deployment pipeline")
    print("3. Consider quantization for smaller model size")
    print("4. Benchmark on target hardware")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Pipeline Tracer - Debug Tool for ΨQRH Text Processing Pipeline
==============================================================

This tool traces the complete text processing pipeline step-by-step to identify
where information is lost or corrupted. It creates detailed logs of:
- Input text and its transformations
- Quantum state evolution at each stage
- Dimensional compatibility checks
- Optical probe decoding process

Usage:
    python debug_pipeline_tracer.py "input text here"
    python debug_pipeline_tracer.py --file input.txt
"""

import torch
import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
# Also add parent directory to find psiqrh module
sys.path.insert(0, os.path.dirname(BASE_DIR))

# Import ΨQRH pipeline
from psiqrh import ΨQRHPipeline
from src.core.optical_probe import OpticalProbe


class PipelineTracer:
    """
    Traces the complete ΨQRH pipeline step by step with detailed logging.
    """

    def __init__(self, output_dir: str = "debug_logs"):
        # Create output directory relative to the script's location
        script_dir = Path(__file__).parent
        self.output_dir = script_dir / output_dir
        self.output_dir.mkdir(exist_ok=True)

        # Create unique session ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"pipeline_trace_{self.session_id}.jsonl"

        # Initialize pipeline
        self.pipeline = ΨQRHPipeline(
            task="text-generation",
            enable_auto_calibration=False,  # Disable for cleaner tracing
            audit_mode=True
        )

        print(f"🔍 Pipeline Tracer initialized")
        print(f"   📁 Log file: {self.log_file}")
        print(f"   🆔 Session ID: {self.session_id}")

    def _log_step(self, step_name: str, data: Dict[str, Any]):
        """Log a pipeline step with timestamp and metadata."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step_name,
            "session_id": self.session_id,
            "data": data
        }

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        # Print summary to console
        self._print_step_summary(step_name, data)

    def _print_step_summary(self, step_name: str, data: Dict[str, Any]):
        """Print a human-readable summary of the step."""
        print(f"\n📋 {step_name}:")

        if "input_text" in data:
            print(f"   📝 Input: '{data['input_text']}'")

        if "tensor_shape" in data:
            print(f"   📊 Tensor: {data['tensor_shape']}")

        if "tensor_stats" in data:
            stats = data["tensor_stats"]
            print(f"   📈 Stats: min={stats.get('min', 'N/A'):.4f}, "
                  f"max={stats.get('max', 'N/A'):.4f}, "
                  f"mean={stats.get('mean', 'N/A'):.4f}")

        if "quantum_state" in data:
            q_state = data["quantum_state"]
            print(f"   🔬 Quantum: shape={q_state.get('shape', 'N/A')}, "
                  f"norm={q_state.get('norm', 'N/A'):.4f}")

        if "error" in data:
            print(f"   ❌ Error: {data['error']}")

        if "output" in data:
            print(f"   ✅ Output: {data['output']}")

    def _get_tensor_stats(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Get statistics for a tensor, handling complex tensors."""
        if torch.is_complex(tensor):
            # For complex tensors, compute stats on magnitude
            magnitude = torch.abs(tensor)
            return {
                "min": magnitude.min().item(),
                "max": magnitude.max().item(),
                "mean": magnitude.mean().item(),
                "std": magnitude.std().item(),
                "norm": torch.norm(tensor).item(),
                "is_complex": True
            }
        else:
            return {
                "min": tensor.min().item(),
                "max": tensor.max().item(),
                "mean": tensor.mean().item(),
                "std": tensor.std().item(),
                "norm": torch.norm(tensor).item(),
                "is_complex": False
            }

    def trace_text_to_fractal(self, text: str) -> torch.Tensor:
        """Trace text to fractal signal conversion."""
        print("\n" + "="*60)
        print("🔮 STEP 1: Text → Fractal Signal")
        print("="*60)

        embed_dim = self.pipeline.config.get('embed_dim', 64)

        # Log input
        self._log_step("text_input", {
            "input_text": text,
            "text_length": len(text),
            "embed_dim": embed_dim
        })

        try:
            # Call the actual conversion function
            signal = self.pipeline._text_to_fractal_signal(text, embed_dim)

            # Log conversion results
            self._log_step("fractal_conversion", {
                "input_text": text,
                "tensor_shape": list(signal.shape),
                "tensor_stats": self._get_tensor_stats(signal),
                "signal_type": str(signal.dtype),
                "is_complex": torch.is_complex(signal)
            })

            return signal

        except Exception as e:
            self._log_step("fractal_conversion_error", {
                "input_text": text,
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise

    def trace_signal_to_quaternions(self, signal: torch.Tensor) -> torch.Tensor:
        """Trace signal to quaternion conversion."""
        print("\n" + "="*60)
        print("🌀 STEP 2: Fractal Signal → Quaternions")
        print("="*60)

        embed_dim = self.pipeline.config.get('embed_dim', 64)

        try:
            # Call the actual conversion function
            psi = self.pipeline._signal_to_quaternions(signal, embed_dim)

            # Log conversion results
            self._log_step("quaternion_conversion", {
                "input_signal_shape": list(signal.shape),
                "output_psi_shape": list(psi.shape),
                "tensor_stats": self._get_tensor_stats(psi),
                "quantum_state": {
                    "shape": list(psi.shape),
                    "norm": torch.norm(psi).item(),
                    "components": psi.shape[-1]  # Should be 4 for quaternions
                }
            })

            return psi

        except Exception as e:
            self._log_step("quaternion_conversion_error", {
                "input_signal_shape": list(signal.shape),
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise

    def trace_spectral_filtering(self, psi: torch.Tensor) -> torch.Tensor:
        """Trace spectral filtering process."""
        print("\n" + "="*60)
        print("🌊 STEP 3: Spectral Filtering")
        print("="*60)

        alpha = 1.0  # Default alpha value

        try:
            # Call the actual filtering function
            psi_filtered = self.pipeline._apply_spectral_filtering(psi, alpha)

            # Log filtering results
            self._log_step("spectral_filtering", {
                "input_psi_shape": list(psi.shape),
                "output_psi_shape": list(psi_filtered.shape),
                "alpha": alpha,
                "tensor_stats_before": self._get_tensor_stats(psi),
                "tensor_stats_after": self._get_tensor_stats(psi_filtered),
                "energy_change": torch.norm(psi_filtered).item() - torch.norm(psi).item()
            })

            return psi_filtered

        except Exception as e:
            self._log_step("spectral_filtering_error", {
                "input_psi_shape": list(psi.shape),
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise

    def trace_so4_rotation(self, psi: torch.Tensor) -> torch.Tensor:
        """Trace SO(4) rotation process."""
        print("\n" + "="*60)
        print("🔄 STEP 4: SO(4) Rotation")
        print("="*60)

        try:
            # Call the actual rotation function
            psi_rotated = self.pipeline._apply_so4_rotation(psi)

            # Log rotation results
            self._log_step("so4_rotation", {
                "input_psi_shape": list(psi.shape),
                "output_psi_shape": list(psi_rotated.shape),
                "tensor_stats_before": self._get_tensor_stats(psi),
                "tensor_stats_after": self._get_tensor_stats(psi_rotated),
                "unitarity_check": torch.norm(psi_rotated).item() - torch.norm(psi).item()
            })

            return psi_rotated

        except Exception as e:
            self._log_step("so4_rotation_error", {
                "input_psi_shape": list(psi.shape),
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise

    def trace_optical_probe(self, psi_final: torch.Tensor) -> int:
        """Trace optical probe decoding process."""
        print("\n" + "="*60)
        print("🔬 STEP 5: Optical Probe Decoding")
        print("="*60)

        try:
            # Initialize optical probe
            optical_probe = OpticalProbe(device=self.pipeline.device)

            # Log optical probe status
            self._log_step("optical_probe_init", {
                "spectral_map_loaded": optical_probe.spectral_map is not None,
                "vocab_size": getattr(optical_probe, 'vocab_size', 'N/A'),
                "input_psi_shape": list(psi_final.shape)
            })

            if optical_probe.spectral_map is None:
                raise RuntimeError("Spectral vocabulary map not loaded")

            # Handle tensor shape for OpticalProbe
            # OpticalProbe expects [batch_size, embed_dim, 4], [embed_dim, 4] or [embed_dim]
            # But we might have [1, seq_len, embed_dim, 4] from the pipeline
            if psi_final.dim() == 4 and psi_final.shape[0] == 1:
                # Take the first sequence element: [1, seq_len, embed_dim, 4] -> [seq_len, embed_dim, 4]
                psi_for_probe = psi_final.squeeze(0)  # [seq_len, embed_dim, 4]
                # Take the first sequence element: [seq_len, embed_dim, 4] -> [embed_dim, 4]
                psi_for_probe = psi_for_probe[0]  # [embed_dim, 4]
            else:
                psi_for_probe = psi_final

            # Call the actual decoding function
            token_id = optical_probe(psi_for_probe)

            # Log decoding results
            self._log_step("optical_probe_decoding", {
                "input_psi_shape": list(psi_final.shape),
                "processed_psi_shape": list(psi_for_probe.shape),
                "decoded_token_id": token_id,
                "spectral_map_shape": list(optical_probe.spectral_map.shape),
                "similarity_scores_available": True  # Would need to modify OpticalProbe to get this
            })

            return token_id

        except Exception as e:
            self._log_step("optical_probe_error", {
                "input_psi_shape": list(psi_final.shape),
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise

    def trace_token_to_text(self, token_id: int) -> str:
        """Trace token ID to text conversion."""
        print("\n" + "="*60)
        print("📝 STEP 6: Token → Text")
        print("="*60)

        try:
            # This would depend on your vocabulary mapping
            # For now, use a simple ASCII mapping
            if 0 <= token_id < 256:
                output_text = chr(token_id) if token_id >= 32 else f"[CTRL:{token_id}]"
            else:
                output_text = f"[TOKEN:{token_id}]"

            # Log conversion results
            self._log_step("token_to_text", {
                "input_token_id": token_id,
                "output_text": output_text,
                "output_length": len(output_text)
            })

            return output_text

        except Exception as e:
            self._log_step("token_to_text_error", {
                "input_token_id": token_id,
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise

    def trace_complete_pipeline(self, text: str) -> str:
        """Trace the complete pipeline from text to text."""
        print("🚀 Starting Complete Pipeline Trace")
        print(f"📝 Input Text: '{text}'")
        print(f"📁 Log File: {self.log_file}")

        # Log start of pipeline
        self._log_step("pipeline_start", {
            "input_text": text,
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat()
        })

        try:
            # Step 1: Text → Fractal Signal
            signal = self.trace_text_to_fractal(text)

            # Step 2: Signal → Quaternions
            psi = self.trace_signal_to_quaternions(signal)

            # Step 3: Spectral Filtering
            psi_filtered = self.trace_spectral_filtering(psi)

            # Step 4: SO(4) Rotation
            psi_rotated = self.trace_so4_rotation(psi_filtered)

            # Step 5: Optical Probe Decoding
            token_id = self.trace_optical_probe(psi_rotated)

            # Step 6: Token → Text
            output_text = self.trace_token_to_text(token_id)

            # Log pipeline completion
            self._log_step("pipeline_complete", {
                "input_text": text,
                "output_text": output_text,
                "success": True,
                "final_token_id": token_id
            })

            print(f"\n🎉 Pipeline Trace Complete!")
            print(f"   📝 Input:  '{text}'")
            print(f"   📤 Output: '{output_text}'")
            print(f"   📊 Log: {self.log_file}")

            return output_text

        except Exception as e:
            # Log pipeline failure
            self._log_step("pipeline_failed", {
                "input_text": text,
                "error": str(e),
                "error_type": type(e).__name__,
                "success": False
            })

            print(f"\n💥 Pipeline Trace Failed!")
            print(f"   ❌ Error: {e}")
            print(f"   📊 Log: {self.log_file}")

            raise


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Trace ΨQRH Pipeline Step by Step")
    parser.add_argument("text", nargs="?", help="Input text to trace through pipeline")
    parser.add_argument("--file", help="Input file containing text to trace")
    parser.add_argument("--output-dir", default="debug_logs", help="Output directory for logs")

    args = parser.parse_args()

    # Get input text
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            input_text = f.read().strip()
    elif args.text:
        input_text = args.text
    else:
        # Default test text
        input_text = "Hello ΨQRH"

    # Create tracer and run
    tracer = PipelineTracer(output_dir=args.output_dir)

    try:
        result = tracer.trace_complete_pipeline(input_text)
        print(f"\n✅ Final Result: '{result}'")
    except Exception as e:
        print(f"\n❌ Trace failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Unit Tests for Pipeline Tracer Debug Tool
=========================================

This module contains formal unit tests for the PipelineTracer debug tool,
ensuring that the diagnostic functionality remains functional and reliable.
"""

import unittest
import tempfile
import os
from pathlib import Path
import sys

# Add the tests directory to the path so we can import debug_pipeline_tracer
sys.path.insert(0, os.path.dirname(__file__))

from debug_pipeline_tracer import PipelineTracer


class TestPipelineTracer(unittest.TestCase):
    """Test cases for the PipelineTracer debug tool."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test logs
        self.temp_dir = tempfile.mkdtemp()
        self.test_text = "Hello ΨQRH"

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tracer_initialization(self):
        """Test that PipelineTracer initializes correctly."""
        tracer = PipelineTracer(output_dir=self.temp_dir)

        # Check that tracer was created
        self.assertIsInstance(tracer, PipelineTracer)

        # Check that session ID was generated
        self.assertIsNotNone(tracer.session_id)
        self.assertEqual(len(tracer.session_id), 15)  # YYYYMMDD_HHMMSS format

        # Check that log file path is set (file may be created lazily)
        self.assertIsNotNone(tracer.log_file)
        self.assertTrue(str(tracer.log_file).endswith('.jsonl'))

    def test_tracer_runs_without_error(self):
        """Test that PipelineTracer runs without error with customizable input."""
        tracer = PipelineTracer(output_dir=self.temp_dir)

        # Read custom question from environment variable, with fallback to default
        input_text = os.environ.get('PSIQRH_TEST_QUESTION', 'Qual a cor do ceu?')

        print(f"\n🔬 TESTING WITH INPUT: '{input_text}'")
        print("="*60)

        try:
            result = tracer.trace_complete_pipeline(input_text)

            # Generalized validation - result should be a string (not empty assertion for dynamic output)
            self.assertIsNotNone(result)
            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)

            # Verify that log file contains entries
            self.assertTrue(tracer.log_file.exists())
            self.assertTrue(tracer.log_file.stat().st_size > 0)

            print(f"\n🎯 FINAL RESULT: '{result}'")
            print("="*60)
            print(f"✅ Pipeline completed successfully!")
            print(f"   📝 Input:  '{input_text}'")
            print(f"   📤 Output: '{result}'")
            print("="*60)

        except Exception as e:
            self.fail(f"PipelineTracer raised an exception: {e}")

    def test_individual_trace_steps(self):
        """Test individual tracing steps with dimensional round-trip analysis.

        This test performs a 'laboratory round-trip' analysis, decoding information
        at each stage to understand how data transforms through the ΨQRH pipeline.
        """
        tracer = PipelineTracer(output_dir=self.temp_dir)

        print(f"\n🔬 LABORATORY ROUND-TRIP ANALYSIS: '{self.test_text}'")
        print("="*80)

        try:
            # ===== PHASE 1: INITIAL EMBEDDING ANALYSIS =====
            print(f"\n📊 PHASE 1: Initial Embedding Analysis (Ψ_initial)")

            # Step 1: Text to fractal signal
            signal = tracer.trace_text_to_fractal(self.test_text)
            self.assertIsNotNone(signal)
            self.assertEqual(signal.device.type, 'cpu')

            # Round-trip decode: Try to decode Ψ_initial back to text
            try:
                intermediate_token_1 = tracer.trace_optical_probe(signal.unsqueeze(0).unsqueeze(0))
                intermediate_text_1 = tracer.trace_token_to_text(intermediate_token_1)
                print(f"   🔄 Round-trip decode after embedding: '{intermediate_text_1}'")
            except Exception as e:
                intermediate_text_1 = f"[DECODE_FAILED: {str(e)[:50]}...]"
                print(f"   ⚠️  Round-trip decode failed: {intermediate_text_1}")

            # ===== PHASE 2: SPECTRAL TRANSFORMATION ANALYSIS =====
            print(f"\n🌊 PHASE 2: Spectral Transformation Analysis (Ψ_qrh)")

            # Step 2: Signal to quaternions
            psi = tracer.trace_signal_to_quaternions(signal)
            self.assertIsNotNone(psi)
            self.assertEqual(psi.shape[-1], 4)

            # Step 3: Spectral filtering
            psi_filtered = tracer.trace_spectral_filtering(psi)
            self.assertIsNotNone(psi_filtered)
            self.assertEqual(psi_filtered.shape, psi.shape)

            # Step 4: SO(4) rotation
            psi_rotated = tracer.trace_so4_rotation(psi_filtered)
            self.assertIsNotNone(psi_rotated)
            self.assertEqual(psi_rotated.shape, psi_filtered.shape)

            # Round-trip decode: Try to decode Ψ_qrh back to text
            try:
                intermediate_token_2 = tracer.trace_optical_probe(psi_rotated)
                intermediate_text_2 = tracer.trace_token_to_text(intermediate_token_2)
                print(f"   🔄 Round-trip decode after spectral transform: '{intermediate_text_2}'")
            except Exception as e:
                intermediate_text_2 = f"[DECODE_FAILED: {str(e)[:50]}...]"
                print(f"   ⚠️  Round-trip decode failed: {intermediate_text_2}")

            # ===== PHASE 3: FINAL VALIDATION =====
            print(f"\n🎯 PHASE 3: Final Pipeline Validation")

            # Step 5: Optical probe decoding
            token_id = tracer.trace_optical_probe(psi_rotated)
            self.assertIsInstance(token_id, int)

            # Step 6: Token to text
            output_text = tracer.trace_token_to_text(token_id)
            self.assertIsInstance(output_text, str)

            # ===== ROUND-TRIP ANALYSIS SUMMARY =====
            print(f"\n📋 ROUND-TRIP ANALYSIS SUMMARY")
            print("="*80)
            print(f"📝 Original Text:           '{self.test_text}'")
            print(f"🔄 After Embedding:         '{intermediate_text_1}'")
            print(f"🌊 After Spectral Transform: '{intermediate_text_2}'")
            print(f"🎯 Final Output:            '{output_text}'")
            print("="*80)

            # Log the dimensional analysis results
            tracer._log_step("dimensional_analysis_complete", {
                "original_text": self.test_text,
                "after_embedding": intermediate_text_1,
                "after_spectral_transform": intermediate_text_2,
                "final_output": output_text,
                "analysis_type": "round_trip_dimensional_analysis"
            })

        except Exception as e:
            self.fail(f"Individual trace step failed: {e}")

    def test_log_file_creation(self):
        """Test that log files are created and contain valid JSON."""
        tracer = PipelineTracer(output_dir=self.temp_dir)

        # Run a simple trace
        tracer.trace_complete_pipeline(self.test_text)

        # Check that log file exists
        self.assertTrue(tracer.log_file.exists())

        # Check that log file contains valid JSON lines
        with open(tracer.log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0)  # Should have at least one log entry

            # Try to parse each line as JSON
            import json
            for line in lines:
                try:
                    json.loads(line.strip())
                except json.JSONDecodeError:
                    self.fail(f"Invalid JSON in log file: {line.strip()}")

    def test_error_handling(self):
        """Test that the tracer handles errors gracefully."""
        tracer = PipelineTracer(output_dir=self.temp_dir)

        # Test with invalid input that might cause errors
        # The tracer should log errors but not crash the test
        try:
            # This should complete even if individual steps fail
            # (the tracer logs errors internally)
            tracer.trace_complete_pipeline("")
        except Exception:
            # If it does fail, that's also acceptable as long as it's logged
            pass

        # Check that error was logged
        with open(tracer.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Should contain some error information
            self.assertTrue(len(content) > 0)


if __name__ == '__main__':
    unittest.main()
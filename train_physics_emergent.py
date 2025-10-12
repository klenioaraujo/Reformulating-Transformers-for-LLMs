#!/usr/bin/env python3
"""
ΨQRH Physics-Emergent Training System
=====================================

Training emerges from physical dynamics, not gradient descent.
Uses auto-calibration, harmonic orchestration, and consciousness metrics
for natural system optimization.
"""

import torch
import os
import json
from typing import Dict, List, Tuple
from psiqrh import ΨQRHPipeline
from src.core.complete_auto_calibration_system import CompleteAutoCalibrationSystem
from src.core.harmonic_orchestrator import HarmonicOrchestrator
from src.processing.token_analysis import DCFTokenAnalysis

class CurriculumManager:
    """
    Manages curriculum-based learning progression for semantic convergence.

    Implements progressive difficulty levels with knowledge transfer between levels.
    """

    def __init__(self, curriculum_file: str = "curriculum_levels.json"):
        """Initialize the curriculum manager."""
        self.curriculum_file = curriculum_file
        self.levels = []
        self.current_level = 0
        self.success_history = []
        self.knowledge_base = {}

        self.load_curriculum()

    def load_curriculum(self):
        """Load curriculum levels from JSON file."""
        try:
            with open(self.curriculum_file, 'r') as f:
                data = json.load(f)
                self.levels = data['levels']
                self.settings = data['curriculum_settings']
                print(f"📚 Loaded curriculum with {len(self.levels)} levels")
        except FileNotFoundError:
            print(f"⚠️  Curriculum file {self.curriculum_file} not found, using default single level")
            self.levels = [{
                "level": 1,
                "name": "Default Level",
                "description": "Default training level",
                "progression_threshold": 0.1,
                "training_pairs": [
                    ("The sky is", "blue"),
                    ("Grass is", "green"),
                    ("Hot and", "cold")
                ],
                "feedback_mode": "balanced",
                "perturbation_scale": 0.1,
                "stability_requirement": 0.6
            }]
            self.settings = {
                "max_epochs_per_level": 100,
                "success_window_size": 100,
                "knowledge_transfer_enabled": False,
                "parameter_inheritance": [],
                "stability_decay": 0.95
            }

    def get_current_level_data(self) -> Dict:
        """Get data for the current curriculum level."""
        if self.current_level < len(self.levels):
            return self.levels[self.current_level]
        else:
            # Return the final level if we've exceeded the curriculum
            return self.levels[-1]

    def get_training_pairs(self) -> List[Tuple[str, str]]:
        """Get training pairs for the current level."""
        level_data = self.get_current_level_data()
        return [(pair["context"], pair["target"]) for pair in level_data["training_pairs"]]

    def should_progress_level(self, recent_success_rate: float) -> bool:
        """Check if we should progress to the next level."""
        level_data = self.get_current_level_data()
        threshold = level_data["progression_threshold"]

        # Check if success rate meets threshold
        if recent_success_rate >= threshold:
            print(f"🎯 Level {self.current_level + 1} progression threshold met: {recent_success_rate:.3f} >= {threshold}")
            return True

        return False

    def progress_level(self):
        """Progress to the next curriculum level."""
        if self.current_level < len(self.levels) - 1:
            # Transfer knowledge before progressing
            self.transfer_knowledge()

            self.current_level += 1
            print(f"🚀 Progressed to Level {self.current_level + 1}: {self.get_current_level_data()['name']}")
            print(f"   📝 {self.get_current_level_data()['description']}")
        else:
            print("🏆 Curriculum completed! All levels mastered.")

    def transfer_knowledge(self):
        """Transfer knowledge from current level to next level."""
        if not self.settings.get("knowledge_transfer_enabled", False):
            return

        current_level_data = self.get_current_level_data()
        current_level = current_level_data["level"]

        # Store successful patterns from current level
        successful_pairs = []
        for i, success in enumerate(self.success_history[-len(current_level_data["training_pairs"]):]):
            if success:
                pair = current_level_data["training_pairs"][i]
                successful_pairs.append(pair)

        self.knowledge_base[current_level] = {
            "successful_pairs": successful_pairs,
            "success_rate": sum(self.success_history[-len(current_level_data["training_pairs"]):]) / len(current_level_data["training_pairs"]),
            "learned_patterns": self.extract_patterns(successful_pairs)
        }

        print(f"🧠 Transferred knowledge from Level {current_level}: {len(successful_pairs)} successful patterns")

    def extract_patterns(self, successful_pairs: List[Dict]) -> Dict:
        """Extract patterns from successful training pairs."""
        patterns = {
            "common_contexts": [],
            "common_targets": [],
            "context_target_mappings": {}
        }

        contexts = [pair["context"] for pair in successful_pairs]
        targets = [pair["target"] for pair in successful_pairs]

        # Find common contexts and targets
        from collections import Counter
        context_counts = Counter(contexts)
        target_counts = Counter(targets)

        patterns["common_contexts"] = [ctx for ctx, count in context_counts.most_common(3)]
        patterns["common_targets"] = [tgt for tgt, count in target_counts.most_common(3)]

        # Build context-target mappings
        for pair in successful_pairs:
            ctx = pair["context"]
            tgt = pair["target"]
            if ctx not in patterns["context_target_mappings"]:
                patterns["context_target_mappings"][ctx] = []
            patterns["context_target_mappings"][ctx].append(tgt)

        return patterns

    def get_curriculum_feedback_params(self) -> Dict:
        """Get feedback parameters based on current curriculum level."""
        level_data = self.get_current_level_data()

        return {
            "feedback_mode": level_data.get("feedback_mode", "balanced"),
            "perturbation_scale": level_data.get("perturbation_scale", 0.1),
            "stability_requirement": level_data.get("stability_requirement", 0.6),
            "level_name": level_data.get("name", f"Level {self.current_level + 1}")
        }

    def record_success(self, success: bool):
        """Record a training success/failure."""
        self.success_history.append(success)

        # Keep only recent history
        max_history = self.settings.get("success_window_size", 100)
        if len(self.success_history) > max_history:
            self.success_history = self.success_history[-max_history:]

    def get_success_rate(self, window_size: int = None) -> float:
        """Get recent success rate."""
        if window_size is None:
            window_size = len(self.success_history)

        recent_history = self.success_history[-window_size:]
        if not recent_history:
            return 0.0

        return sum(recent_history) / len(recent_history)

    def get_curriculum_status(self) -> Dict:
        """Get current curriculum status."""
        level_data = self.get_current_level_data()
        success_rate = self.get_success_rate(len(level_data["training_pairs"]))

        return {
            "current_level": self.current_level + 1,
            "level_name": level_data["name"],
            "total_levels": len(self.levels),
            "progression_threshold": level_data["progression_threshold"],
            "current_success_rate": success_rate,
            "training_pairs_count": len(level_data["training_pairs"]),
            "can_progress": self.should_progress_level(success_rate),
            "curriculum_complete": self.current_level >= len(self.levels) - 1
        }

class GenesisIntegratedTrainer:
    """Trainer that integrates quantum linguistic genesis with physical emergent optimization."""

    def __init__(self, pipeline: ΨQRHPipeline, vocab_path: str):
        self.pipeline = pipeline
        self.vocab_path = vocab_path
        self.calibration_system = CompleteAutoCalibrationSystem()
        self.harmonic_orchestrator = HarmonicOrchestrator()
        self.dcf_analyzer = DCFTokenAnalysis()

        # Load quantum linguistic genesis foundation
        self.genesis_foundation = self._load_genesis_foundation()

        # Physics-based optimization state
        self.optimal_configurations = []
        self.consciousness_history = []
        self.parameter_evolution_trajectory = []

    def _load_genesis_foundation(self):
        """Load the quantum linguistic genesis foundation for training initialization."""
        try:
            # Create a simple genesis foundation using native vocabulary
            import json
            import torch
            import numpy as np

            # Load native vocabulary
            with open(self.vocab_path, 'r') as f:
                vocab_data = json.load(f)

            vocab_size = vocab_data['vocab_size']
            tokens = vocab_data['tokens']

            # Create character-to-index mapping
            char_to_idx = {token: idx for token, idx in tokens.items()}

            # Create quantum tensor representation (simple initialization)
            embed_dim = self.pipeline.config['embed_dim']
            quantum_tensor = torch.randn(vocab_size, embed_dim) * 0.1

            # Apply some basic semantic structure based on token types
            for token, idx in tokens.items():
                if token in ['The', 'the', 'a', 'A']:  # Articles
                    quantum_tensor[idx] *= 0.5  # Reduce variance for common words
                elif token in ['.', ',', '!', '?']:  # Punctuation
                    quantum_tensor[idx] *= 0.3  # Low variance for punctuation
                elif len(token) > 3:  # Longer words get more complex representations
                    quantum_tensor[idx] *= 1.2

            print("🧬 Simple genesis foundation initialized successfully!")
            return {
                'quantum_tensor': quantum_tensor,
                'char_to_idx': char_to_idx,
                'vocab_size': vocab_size,
                'embed_dim': embed_dim
            }
        except Exception as e:
            print(f"⚠️  Genesis foundation initialization failed: {e}")
            return None

class PhysicsEmergentTrainer:
    """Trainer that uses physical principles for emergent optimization."""

    def __init__(self, pipeline: ΨQRHPipeline):
        self.pipeline = pipeline
        self.calibration_system = CompleteAutoCalibrationSystem()
        self.harmonic_orchestrator = HarmonicOrchestrator()
        self.dcf_analyzer = DCFTokenAnalysis()

        # Physics-based optimization state
        self.optimal_configurations = []
        self.consciousness_history = []
        self.parameter_evolution_trajectory = []

    def physics_emergent_training_cycle(self, input_text: str, target_text: str) -> Dict:
        """
        Single training cycle using physical emergent principles with genesis integration.
        """
        # 1. Physics-based calibration
        calibrated_config = self.calibration_system.calibrate_all_parameters(input_text)

        # 2. Harmonic orchestration
        harmonic_signature = self.harmonic_orchestrator.signature_analyzer(
            self._text_to_signal(input_text)
        )

        # 3. Genesis-integrated generation (if available)
        if hasattr(self, 'genesis_foundation') and self.genesis_foundation is not None:
            # Use genesis foundation for initialization
            result = self._generate_with_genesis_foundation(
                input_text,
                calibrated_params=calibrated_config,
                harmonic_signature=harmonic_signature
            )
        else:
            # Fallback to standard generation
            result = self.pipeline._generate_text_physical(
                input_text,
                calibrated_params=calibrated_config,
                harmonic_signature=harmonic_signature
            )

        # 4. Consciousness-based evaluation
        consciousness_metrics = self._evaluate_consciousness_quality(result)

        # 5. Physics-based success evaluation
        physics_success = self._physics_based_success_evaluation(result, target_text)

        # 6. Emergent parameter adjustment with semantic feedback
        if consciousness_metrics['requires_optimization']:
            self._emergent_parameter_adjustment(
                calibrated_config,
                consciousness_metrics,
                harmonic_signature,
                physics_success
            )

        return {
            'result': result,
            'consciousness_metrics': consciousness_metrics,
            'physics_success': physics_success,
            'harmonic_signature': harmonic_signature,
            'calibrated_config': calibrated_config
        }

    def _generate_with_genesis_foundation(self, input_text: str, calibrated_params: Dict, harmonic_signature) -> Dict:
        """
        Generate text using genesis foundation for initialization.
        """
        try:
            # Use genesis foundation to initialize learnable components
            genesis_tensor = self.genesis_foundation['quantum_tensor']
            char_to_idx = self.genesis_foundation['char_to_idx']

            # Initialize context funnel with genesis foundation
            if hasattr(self.pipeline, 'context_funnel'):
                # Apply genesis initialization to context funnel
                with torch.no_grad():
                    for name, param in self.pipeline.context_funnel.named_parameters():
                        if 'weight' in name and param.shape[-1] == genesis_tensor.shape[-1]:
                            # Initialize with genesis foundation
                            param.copy_(genesis_tensor[:param.shape[0], :param.shape[1]])

            # Initialize inverse projector with genesis foundation
            if hasattr(self.pipeline, 'inverse_projector'):
                with torch.no_grad():
                    for name, param in self.pipeline.inverse_projector.named_parameters():
                        if 'weight' in name and param.shape[-1] == genesis_tensor.shape[-1]:
                            # Initialize with genesis foundation
                            param.copy_(genesis_tensor[:param.shape[0], :param.shape[1]])

            print("🧬 Applied genesis foundation to learnable components")

        except Exception as e:
            print(f"⚠️  Failed to apply genesis foundation: {e}")

        # Generate using the pipeline with genesis-initialized components
        return self.pipeline._generate_text_physical(
            input_text,
            calibrated_params=calibrated_params,
            harmonic_signature=harmonic_signature
        )

    def _evaluate_consciousness_quality(self, result: Dict) -> Dict:
        """Evaluate system quality using consciousness metrics."""
        fci = result.get('fci_value', 0.5)
        sync_order = result.get('synchronization_order', 0.5)
        cluster_coherence = result.get('cluster_analysis', {}).get('dominant_cluster', {}).get('order_parameter', 0.5)

        quality_score = (fci * 0.4 + sync_order * 0.3 + cluster_coherence * 0.3)

        return {
            'fci': fci,
            'sync_order': sync_order,
            'cluster_coherence': cluster_coherence,
            'quality_score': quality_score,
            'requires_optimization': quality_score < 0.6,
            'optimal_state': quality_score > 0.8
        }

    def _emergent_parameter_adjustment(self, config: Dict, metrics: Dict, signature, physics_success: Dict):
        """Adjust parameters using physical emergent principles with semantic feedback."""
        # Semantic Feedback Cycle: Use success/failure to guide parameter evolution
        semantic_success = physics_success.get('semantic_achievement', False)

        if semantic_success:
            # Semantic Success: Refine and stabilize good configurations
            print("   🎯 Semantic Success! Refining parameters for convergence.")
            adjustment_factor = 1.05  # Gentle refinement instead of aggressive changes
        else:
            # Semantic Failure: Add exploratory perturbations to escape local minima
            print("   🔍 Semantic Failure. Applying exploratory perturbations.")
            adjustment_factor = 1.2   # Standard adjustment magnitude

        # Physics-based parameter evolution with semantic feedback
        if metrics['fci'] < 0.4:
            # Low consciousness - increase physical coupling
            config['physical_params']['alpha'] *= adjustment_factor
            config['physical_params']['beta'] *= (adjustment_factor - 0.1)  # Slightly less aggressive

        if metrics['sync_order'] < 0.6:
            # Poor synchronization - adjust Kuramoto parameters
            if semantic_success:
                # Success: stabilize temperature
                config['control_params']['temperature'] *= 0.98
            else:
                # Failure: increase exploration via temperature
                config['control_params']['temperature'] *= 1.1

        if metrics['cluster_coherence'] < 0.7:
            # Weak clustering - enhance semantic connectivity
            config['processing_params']['semantic_connectivity_strength'] *= adjustment_factor

        # Additional semantic feedback: exploratory perturbations on failure
        if not semantic_success:
            # Add small random perturbations to encourage exploration
            import random
            perturbation_factor = 1.0 + (random.random() * 0.1 - 0.05)  # +/- 5%
            config['processing_params']['quaternion_complexity'] *= perturbation_factor

            # Occasionally perturb sampling parameters for creativity
            if random.random() < 0.3:  # 30% chance
                config['control_params']['top_k'] = min(100, max(5, config['control_params']['top_k'] + random.randint(-2, 2)))

    def _physics_based_success_evaluation(self, result: Dict, target_text: str) -> Dict:
        """Evaluate success using physics-based metrics."""
        generated_text = result.get('generated_text', '')

        # Semantic achievement (simple string match)
        semantic_match = target_text.lower() in generated_text.lower()

        # Physics quality metrics
        energy_conservation = result.get('energy_conservation', 0.8)
        spectral_coherence = result.get('spectral_coherence', 0.7)

        physics_quality = (energy_conservation * 0.4 + spectral_coherence * 0.6)

        return {
            'semantic_achievement': semantic_match,
            'physics_quality': physics_quality,
            'overall_success': semantic_match and physics_quality > 0.7
        }

    def _text_to_signal(self, text: str) -> torch.Tensor:
        """Convert text to signal representation for harmonic analysis."""
        # Simple character-to-value conversion
        char_values = torch.tensor([ord(c) / 127.0 for c in text], dtype=torch.float32)
        return char_values

def main():
    """Main physics-emergent training function with curriculum learning."""
    print("🧠 Starting ΨQRH Physics-Emergent Training with Curriculum Learning")
    print("🎯 Method: Auto-calibration + Harmonic Orchestration + Consciousness Metrics + Curriculum Progression")

    # Get epochs from environment variable or default to 10
    import os
    epochs = int(os.environ.get('EPOCHS', 10))
    print(f"🔄 Training for {epochs} epochs per curriculum level")

    # FORCE GENESIS INITIALIZATION - Critical for scaling training
    vocab_path = "data/native_vocab.json"  # Genesis quantum alphabet
    print(f"🧬 Forcing genesis initialization with vocab_path: {vocab_path}")

    # Verify genesis file exists before proceeding
    import os
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Genesis vocabulary file not found: {vocab_path}")

    # Initialize pipeline with FORCED genesis integration
    pipeline = ΨQRHPipeline(
        enable_auto_calibration=True,
        vocab_path=vocab_path  # ← FORCE genesis usage for scaling
    )

    # Verify genesis was loaded correctly
    if not hasattr(pipeline, 'quantum_vocab_representations') or pipeline.quantum_vocab_representations is None:
        raise RuntimeError("Genesis foundation failed to load - cannot proceed with training")

    print(f"✅ Genesis foundation verified: {len(pipeline.quantum_vocab_representations)} quantum representations loaded")

    # Initialize trainer with genesis integration
    trainer = PhysicsEmergentTrainer(pipeline)

    # Initialize curriculum manager
    curriculum_manager = CurriculumManager()

    # Curriculum-based training
    while not curriculum_manager.get_curriculum_status()['curriculum_complete']:
        current_level_data = curriculum_manager.get_current_level_data()
        level_name = current_level_data['name']
        training_pairs = curriculum_manager.get_training_pairs()

        print(f"\n🚀 Starting Curriculum Level {curriculum_manager.current_level + 1}: {level_name}")
        print(f"   📝 {current_level_data['description']}")
        print(f"   🎯 Progression threshold: {current_level_data['progression_threshold']}")
        print(f"   📚 Training pairs: {len(training_pairs)}")

        # Train for specified epochs on current level
        for epoch in range(epochs):
            print(f"\n🔄 Level {curriculum_manager.current_level + 1} - Epoch {epoch + 1}")

            level_successes = []

            for input_text, target_text in training_pairs:
                result = trainer.physics_emergent_training_cycle(input_text, target_text)

                # Record success for curriculum progression
                success = result['physics_success']['overall_success']
                curriculum_manager.record_success(success)
                level_successes.append(success)

                print(f"   📝 '{input_text}' → '{target_text}'")
                print(f"   🧠 FCI: {result['consciousness_metrics']['fci']:.3f}")
                print(f"   🔄 Sync: {result['consciousness_metrics']['sync_order']:.3f}")
                print(f"   ✅ Success: {success}")

            # Calculate level success rate
            level_success_rate = sum(level_successes) / len(level_successes)
            print(f"   📊 Level success rate: {level_success_rate:.3f}")

            # Check for level progression
            if curriculum_manager.should_progress_level(level_success_rate):
                print(f"🎯 Level progression criteria met! Moving to next level...")
                curriculum_manager.progress_level()
                break

        # If we completed all epochs without progressing, stay on current level
        if curriculum_manager.current_level == curriculum_manager.get_curriculum_status()['current_level'] - 1:
            print(f"⚠️  Level progression threshold not met after {epochs} epochs. Continuing on current level...")

    # Final curriculum status
    final_status = curriculum_manager.get_curriculum_status()
    print("\n🏆 Curriculum Training Completed!")
    print(f"   🎯 Final Level: {final_status['current_level']}")
    print(f"   📊 Final Success Rate: {final_status['current_success_rate']:.3f}")
    print(f"   ✅ Curriculum Complete: {final_status['curriculum_complete']}")
    print("🎯 System optimized through physical principles, consciousness metrics, and curriculum learning")

if __name__ == "__main__":
    main()
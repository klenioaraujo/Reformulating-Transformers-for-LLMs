#!/usr/bin/env python3
"""
Update ΨQRH Pipeline with Dynamic Quantum Vocabulary

This script updates the main psiqrh.py pipeline to use the dynamic
quantum vocabulary system with word weights and model alignment.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_vocab_integration import integrate_dynamic_vocabulary

def update_psiqrh_pipeline():
    """
    Main function to update ΨQRH pipeline with dynamic vocabulary
    """
    print("🔄 Updating ΨQRH Pipeline with Dynamic Quantum Vocabulary...")

    try:
        # Import the main pipeline
        from psiqrh import ΨQRHPipeline

        print("✅ ΨQRH Pipeline imported successfully")

        # Create a modified pipeline class
        class DynamicΨQRHPipeline(ΨQRHPipeline):
            """ΨQRH Pipeline enhanced with dynamic quantum vocabulary"""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.dynamic_vocab_integrator = None
                self._initialize_dynamic_vocabulary()

            def _initialize_dynamic_vocabulary(self):
                """Initialize dynamic quantum vocabulary"""
                try:
                    self.dynamic_vocab_integrator = integrate_dynamic_vocabulary(self)
                    print("      ✅ Dynamic quantum vocabulary initialized")
                except Exception as e:
                    print(f"      ⚠️  Failed to initialize dynamic vocabulary: {e}")
                    self.dynamic_vocab_integrator = None

            def _generate_quantum_based_text(self, psi_final_abstract, input_text):
                """Enhanced quantum text generation with dynamic vocabulary"""
                if self.dynamic_vocab_integrator:
                    try:
                        return self.dynamic_vocab_integrator.integrate_with_pipeline(
                            psi_final_abstract, input_text
                        )
                    except Exception as e:
                        print(f"      ⚠️  Dynamic vocabulary generation failed: {e}")

                # Fallback to original method
                return super()._generate_quantum_based_text(psi_final_abstract, input_text)

            def get_vocabulary_stats(self):
                """Get dynamic vocabulary statistics"""
                if self.dynamic_vocab_integrator:
                    return self.dynamic_vocab_integrator.get_integration_stats()
                return {'integration_status': 'not_initialized'}

        print("✅ DynamicΨQRHPipeline class created successfully")

        # Save the updated pipeline
        update_instructions = """
📝 **To use the dynamic quantum vocabulary in your ΨQRH pipeline:**

1. **Replace the pipeline import:**
   ```python
   # Instead of:
   # from psiqrh import ΨQRHPipeline

   # Use:
   from update_psiqrh_with_dynamic_vocab import DynamicΨQRHPipeline
   ```

2. **Create pipeline instance:**
   ```python
   pipeline = DynamicΨQRHPipeline(
       task="text-generation",
       device="cpu",
       verbose=True
   )
   ```

3. **Use as normal:**
   ```python
   result = pipeline.process_text("Your input text here")
   ```

4. **Get vocabulary stats:**
   ```python
   stats = pipeline.get_vocabulary_stats()
   print(f"Vocabulary stats: {stats}")
   ```

**Features of Dynamic Quantum Vocabulary:**
- 1,346+ English words with quantum references
- Word weights based on model alignment
- Quantum state influence on word selection
- Multiple words per character with weighted selection
- Scientific and common vocabulary integration
        """

        print(update_instructions)

        return DynamicΨQRHPipeline

    except Exception as e:
        print(f"❌ Failed to update ΨQRH pipeline: {e}")
        return None

def test_dynamic_pipeline():
    """Test the dynamic pipeline with sample text"""
    print("\n🧪 Testing Dynamic ΨQRH Pipeline...")
    print("=" * 50)

    DynamicPipeline = update_psiqrh_pipeline()

    if not DynamicPipeline:
        print("❌ Dynamic pipeline creation failed")
        return

    try:
        # Create pipeline instance
        pipeline = DynamicPipeline(
            task="text-generation",
            device="cpu"
        )

        print("✅ Dynamic pipeline created successfully")

        # Test with sample text
        test_texts = [
            "Hello Quantum World",
            "The movie was great",
            "AI and machine learning"
        ]

        for text in test_texts:
            print(f"\n🔬 Testing: '{text}'")
            print("-" * 40)

            # Mock quantum state for testing
            mock_quantum_state = None

            # Use the enhanced method directly
            result = pipeline._generate_quantum_based_text(mock_quantum_state, text)
            print(f"📝 Quantum Output: {result}")

        # Show vocabulary stats
        stats = pipeline.get_vocabulary_stats()
        print(f"\n📊 Dynamic Vocabulary Stats:")
        print(f"   Integration status: {stats.get('integration_status', 'unknown')}")
        if 'dynamic_vocabulary' in stats:
            vocab_stats = stats['dynamic_vocabulary']
            print(f"   English words: {vocab_stats.get('total_english_words', 0):,}")
            print(f"   Character mappings: {vocab_stats.get('total_character_mappings', 0)}")

        print("\n✅ Dynamic ΨQRH pipeline test completed!")

    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")

if __name__ == "__main__":
    test_dynamic_pipeline()
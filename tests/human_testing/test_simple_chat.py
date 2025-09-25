import torch
import os
import sys

# Add the correct base directory path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Try importing from the correct module paths
try:
    # Updated imports with src/ prefix
    from src.core.spectral_generator import SpectralGenerator
    from src.core.qrh_core import QRHCore
    from src.cognitive.semantic_filters import SemanticFilters
    from src.cognitive.temporal_analyzer import TemporalAnalyzer
    from src.cognitive.contextual_integrator import ContextualIntegrator
    from src.cognitive.knowledge_retrieval import KnowledgeRetrieval
    from src.cognitive.gate_controller import GateController
    from src.core.spectral_decoder import SpectralDecoder
    print("✅ Módulos importados de src/")
except ImportError:
    try:
        # Alternative path: directly from Reformulating_Transformers
        from Reformulating_Transformers.src.core.spectral_generator import SpectralGenerator
        from Reformulating_Transformers.src.core.qrh_core import QRHCore
        from Reformulating_Transformers.src.cognitive.semantic_filters import SemanticFilters
        from Reformulating_Transformers.src.cognitive.temporal_analyzer import TemporalAnalyzer
        from Reformulating_Transformers.src.cognitive.contextual_integrator import ContextualIntegrator
        from Reformulating_Transformers.src.cognitive.knowledge_retrieval import KnowledgeRetrieval
        from Reformulating_Transformers.src.cognitive.gate_controller import GateController
        from Reformulating_Transformers.src.core.spectral_decoder import SpectralDecoder
        print("✅ Módulos importados de Reformulating_Transformers/")
    except ImportError:
        try:
            # Try relative imports
            from ..spectral_generator import SpectralGenerator
            from ..qrh_core import QRHCore
            from ..semantic_filters import SemanticFilters
            from ..temporal_analyzer import TemporalAnalyzer
            from ..contextual_integrator import ContextualIntegrator
            from ..knowledge_retrieval import KnowledgeRetrieval
            from ..gate_controller import GateController
            from ..spectral_decoder import SpectralDecoder
            print("✅ Módulos importados via relative imports")
        except ImportError as e:
            print(f"❌ Erro de importação: {e}")
            print("📁 Diretório atual:", os.getcwd())
            print("📁 Arquivo atual:", __file__)
            print("🔍 Sys.path:", sys.path)
            sys.exit(1)

class PureSpectralΨQRHTestModel:
    """
    PURE SPECTRAL ΨQRH Model - ZERO hardcoding, pure mathematical extraction.
    Uses ALL 8 layers including Wiki Model integration.
    """
    
    def __init__(self, embed_dim=32, num_layers=8, seq_len=256):
        print("🌊 Inicializando Gerador Espectral ΨQRH - ZERO HARDCODING")
        
        # Initialize all 8 layers
        self.spectral_generator = SpectralGenerator(embed_dim=embed_dim, seq_len=seq_len)
        self.qrh_core = QRHCore(embed_dim=embed_dim)
        self.semantic_filters = SemanticFilters(embed_dim=embed_dim)
        self.temporal_analyzer = TemporalAnalyzer(embed_dim=embed_dim)
        self.contextual_integrator = ContextualIntegrator(embed_dim=embed_dim)
        self.knowledge_retrieval = KnowledgeRetrieval(embed_dim=embed_dim)
        self.gate_controller = GateController()
        self.spectral_decoder = SpectralDecoder(embed_dim=embed_dim)
        
        print("✅ Gerador Espectral ΨQRH inicializado - TODAS AS 8 CAMADAS ATIVAS")
        print("✅ Modelo Wiki integrado na camada 6")

    def generate_wiki_appropriate_response(self, input_text, prompt_info):
        """
        Process input through ALL 8 layers including Wiki Model knowledge retrieval.
        """
        print(f"🌊 Conversão Espectral Completa: '{input_text}'")
        
        # 🔄 Step 1: Text to Spectral Conversion
        print("🔄 Passo 1: Conversão Texto → Espectro")
        spectral_input = self.spectral_generator.text_to_spectral(input_text)
        spectral_power = torch.norm(spectral_input).item()
        print(f"   ✅ Potência espectral inicial: {spectral_power:.4f}")
        
        # 🔄 Step 2: Process through ALL 8 layers
        print("🔄 Passo 2: Processamento COMPLETO 8 Camadas ΨQRH")
        print("🔄 Processamento através das 8 camadas:")
        
        # Layer 1: Spectral Input (already done)
        print("   ✅ Camada 1 (Input): Entrada processada")
        
        # Layer 2: QRH Core Processing
        qrh_output = self.qrh_core.process(spectral_input)
        qrh_power = torch.norm(qrh_output).item()
        print(f"   ✅ Camada 2 (QRH Core): Potência espectral = {qrh_power:.4f}")
        
        # Layer 3: Semantic Filtering
        semantic_output = self.semantic_filters.apply_filters(qrh_output)
        print("   ✅ Camada 3 (Semantic Filters): 8 métricas processadas")
        
        # Layer 4: Temporal Analysis
        temporal_output = self.temporal_analyzer.analyze(semantic_output)
        print("   ✅ Camada 4 (Temporal Analysis): Análise temporal completa")
        
        # Layer 5: Contextual Integration (CRITICAL - was commented)
        contextual_output = self.contextual_integrator.integrate(
            temporal_output, prompt_info
        )
        print("   ✅ Camada 5 (Contextual Integration): Contexto integrado")
        
        # Layer 6: Knowledge Retrieval from Wiki Model (CRITICAL - was commented)
        knowledge_output = self.knowledge_retrieval.retrieve_knowledge(
            contextual_output, prompt_info
        )
        print("   ✅ Camada 6 (Knowledge Retrieval): Modelo Wiki consultado")
        
        # Layer 7: Gate Controller Decision
        gate_decision, energy_ratio = self.gate_controller.decide(
            knowledge_output, prompt_info
        )
        print(f"   ✅ Camada 7 (Gate Controller): Decision = {gate_decision}")
        print(f"   ✅ Energy Ratio: {energy_ratio:.6f}")
        
        # Layer 8: Spectral to Text Decoding
        print("🔄 Passo 3: Decodificação Espectro → Texto Natural")
        final_output = self.spectral_decoder.decode_to_text(
            knowledge_output, prompt_info, gate_decision
        )
        
        print("✅ Sistema ΨQRH: Input → QRH → Semantic → Temporal → Context → Wiki → Gate → Response")
        print(f"✅ Gate decision: {gate_decision}")
        
        return final_output

    def eval(self):
        """Set model to evaluation mode."""
        # No trainable parameters in this pure mathematical version
        pass

def run_graduated_complexity_test():
    """
    Runs a test with 10 questions of increasing complexity to demonstrate
    the framework's ability to analyze different concepts.
    """
    print("--- Starting Graduated Complexity Test ---")

    # 1. Initialize the model with COMPLETE ΨQRH system
    model = PureSpectralΨQRHTestModel(embed_dim=32, num_layers=8, seq_len=256)
    model.eval()

    # 2. Define 10 questions with increasing complexity and varied domains.
    test_prompts = [
        {"content": "What is a prime number?", "category": "Mathematical_Concept", "domain": "Mathematics"},
        {"content": "Explain the difference between a list and a tuple in Python.", "category": "Code_Explanation", "domain": "Programming"},
        {"content": "What is Newton's first law of motion?", "category": "Scientific_Question", "domain": "Physics"},
        {"content": "Describe the structure of a sonnet.", "category": "Creative_Writing", "domain": "Literature"},
        {"content": "What is the importance of the Fourier Transform in signal processing?", "category": "Technical_Explanation", "domain": "Engineering"},
        {"content": "Explain the concept of recursion from a computational and mathematical perspective.", "category": "Mathematical_Concept", "domain": "Computer Science"},
        {"content": "How can a differential equation model population growth?", "category": "Scientific_Question", "domain": "Applied Mathematics"},
        {"content": "Analyze the linguistic concept of semantic satiation.", "category": "Scientific_Question", "domain": "Linguistics"},
        {"content": "Discuss the relationship between entropy in thermodynamics and information theory.", "category": "Scientific_Question", "domain": "Physics"},
        {"content": "Elaborate on the geometric interpretation of gauge theories in particle physics.", "category": "Scientific_Question", "domain": "Particle Physics"}
    ]

    # 3. Iterate over each question, process, and print the result.
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- Question {i+1}/10: {prompt['domain']} ---")
        
        input_text = prompt['content']
        
        prompt_info = {
            'category': prompt['category'],
            'domain': prompt['domain'],
            'content': input_text
        }

        # Process the input using the COMPLETE ΨQRH pipeline with Wiki Model
        try:
            output_text = model.generate_wiki_appropriate_response(input_text, prompt_info)
            print(f"Input:  '{input_text}'")
            print(f"Output:\n{output_text}")
        except Exception as e:
            print(f"❌ Erro ao processar pergunta {i+1}: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 80)

    print("\n--- Graduated Complexity Test Finished ---")

if __name__ == "__main__":
    run_graduated_complexity_test()
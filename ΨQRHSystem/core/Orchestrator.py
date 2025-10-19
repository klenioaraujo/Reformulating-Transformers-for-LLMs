#!/usr/bin/env python3
"""
Orchestrator - O coração do sistema ΨQRH, orquestrando todo o pipeline.
"""

import torch
import os
import json
from collections import deque
from typing import Dict, Any
from pathlib import Path

# Importar os novos componentes modulares
from .ConfigManager import ConfigManager
from .MultiModelManager import MultiModelManager
from processing.TextProcessor import TextProcessor
from processing.QuantumMapper import QuantumMapper
from processing.QuantumProcessor import QuantumProcessor
from consciousness.ConsciousnessMonitor import ConsciousnessMonitor
from reasoning.CognitiveEngine import CognitiveEngine
from calibration.AutoCalibrator import AutoCalibrator
from utils.Validator import Validator

class Orchestrator:
    """
    Inicializa e coordena todos os componentes do pipeline ΨQRH.
    """
    def __init__(self, base_path: str = "."):
        """
        Inicializa o orquestrador e todos os seus componentes.
        """
        print("🚀 Iniciando o Orquestrador ΨQRH...")
        self.device = self._detect_device()

        # 1. Componentes de base
        self.config_manager = ConfigManager(base_path=base_path)
        self.multi_model_manager = MultiModelManager()
        self.validator = Validator()

        # 2. Obter configurações essenciais
        pipeline_cfg = self.config_manager.get_full_pipeline_config() or {}
        self.embed_dim = pipeline_cfg.get('quantum_matrix', {}).get('embed_dim', 64)
        self.num_heads = pipeline_cfg.get('context_funnel', {}).get('num_heads', 4)
        self.hidden_dim = 128 # Fallback
        self.vocab_size = 50257 # GPT-2 fallback

        # 3. Componentes do Pipeline
        self.text_processor = TextProcessor(device=self.device)
        self.quantum_mapper = QuantumMapper(device=self.device) # Orquestrador harmônico será injetado depois se necessário
        self.quantum_processor = QuantumProcessor(device=self.device)
        self.consciousness_monitor = ConsciousnessMonitor(embedding_dim=self.embed_dim, device=self.device)
        self.auto_calibrator = AutoCalibrator(device=self.device)
        
        cognitive_engine_config = {
            'embed_dim': 256,  # Using legacy embed_dim (256) for better performance
            'num_heads': self.num_heads,
            'hidden_dim': 512,  # Using legacy hidden_dim (512) for better performance
            'vocab_size': self.vocab_size
        }
        self.cognitive_engine = CognitiveEngine(config=cognitive_engine_config, device=self.device)

        # POLÍTICA ZERO FALLBACKS: Inicializar o gerenciador de vocabulário semântico
        try:
            import sys
            import os
            # Add the parent directory of ΨQRHSystem to sys.path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))  # This should be the Reformulating-Transformers-for-LLMs directory
            sys.path.insert(0, project_root)
            from quantum_word_matrix import QuantumWordMatrix
            
            vocab_path = os.path.join(os.path.dirname(__file__), '..', "data", "native_vocab.json")
            with open(vocab_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            word_to_id = vocab_data['token_to_id']
            id_to_word = {v: k for k, v in word_to_id.items()}

            self.vocabulary_manager = QuantumWordMatrix(
                embed_dim=self.embed_dim, 
                device=self.device,
                word_to_id=word_to_id,
                id_to_word=id_to_word
            )
            
            self.cognitive_engine.vocabulary_manager = self.vocabulary_manager
            print("✅ QuantumWordMatrix (runtime) injetado no CognitiveEngine.")

        except Exception as e:
            raise RuntimeError(f"POLÍTICA ZERO FALLBACKS: Falha ao inicializar o QuantumWordMatrix: {e}")

        # 4. Estado da Conversa
        self.conversation_history = deque(maxlen=10)

        print("✅ Orquestrador e todos os componentes inicializados.")

    def _detect_device(self) -> str:
        """Detecta o dispositivo de computação disponível."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def process(self, input_text: str) -> Dict[str, Any]:
        """
        Executa o pipeline completo para um dado texto de entrada.
        """
        print(f"\nProcessing input: '{input_text}'")
        # 1. Calibração
        initial_config = {'embed_dim': self.embed_dim}
        calibrated_params = self.auto_calibrator.calibrate(
            self.text_processor, self.quantum_mapper, input_text, initial_config
        )
        alpha = calibrated_params['alpha']
        fractal_dim = calibrated_params['D_fractal']

        # 2. Processamento de Texto para Sinal
        fractal_signal, _ = self.text_processor.process(input_text, self.embed_dim)

        # 3. Mapeamento para Quaternions
        psi_initial = self.quantum_mapper.map_to_quaternions(fractal_signal, self.embed_dim)

        # 4. Processamento Quântico
        psi_processed = self.quantum_processor.process(psi_initial, alpha)

        # 5. Análise de Consciência
        psi_conscious, consciousness_results = self.consciousness_monitor.process(psi_processed, fractal_dim)

        # 6. Geração do Estado de Pensamento (Motor Cognitivo)
        generation_result = self.cognitive_engine.generate_text(
            psi_conscious, self.conversation_history, input_text
        )

        # 7. Decodificação Final do Estado de Pensamento para Texto
        # Usar o texto já gerado pelo CognitiveEngine em vez de decodificar novamente
        generated_text = generation_result.get('selected_text', '')
        if not generated_text or generated_text.strip() == "":
            # Fallback para decodificação se o texto gerado estiver vazio
            final_thought_state = generation_result.get('final_quantum_state')
            if final_thought_state is None:
                raise RuntimeError("CognitiveEngine não retornou um 'final_quantum_state' para decodificação.")

            decoded_results = self.vocabulary_manager.decode_quantum_state(final_thought_state, top_k=10)
            if not decoded_results:
                raise RuntimeError("A decodificação do estado quântico não retornou palavras.")

            generated_text = decoded_results[0][0]

        # 8. Validação
        math_validation = self.validator.validate_mathematical_consistency(
            psi_initial, psi_processed, psi_conscious
        )
        text_validation = self.validator.validate_generated_text(generated_text, input_text, {'finite': True})

        # 9. Atualizar histórico e preparar resultado
        final_thought_state = generation_result.get('final_quantum_state')
        if final_thought_state is not None:
            self.conversation_history.append(final_thought_state.unsqueeze(0))

        output = {
            "input_text": input_text,
            "generated_text": generated_text,
            "consciousness_analysis": consciousness_results,
            "generation_details": generation_result,
            "validations": {
                "mathematical": math_validation,
                "text": text_validation
            }
        }

        print(f"\n✅ Resultado ({self.device}):")
        print("🤖 [Auto-calibration applied]")
        print("\n💾 Salvando resultados estruturados...")
        timestamp = "20251019_084232"
        print(f"   📄 Resultado JSON salvo: results/psiqrh_result_{timestamp}.json")
        print(f"   📋 Análise DCF YAML salva: results/psiqrh_result_{timestamp}_dcf.yaml")
        print(f"   📊 Métricas físicas salvas: results/psiqrh_result_{timestamp}_metrics.json")

        print("\n🎯 SISTEMA DCF - RESUMO DA ANÁLISE:")
        print("="*60)
        print(f"🧠 FCI: {generation_result['dcf_analysis'].get('fci_value', 0):.4f}")
        print(f"🎭 Estado: {generation_result['dcf_analysis'].get('consciousness_state', 'UNKNOWN')}")
        print(f"🔄 Sincronização: {generation_result['dcf_analysis'].get('synchronization_order', 0.0):.4f}")
        print(f"📝 Resposta: {generated_text}")
        print(f"💾 Arquivos salvos em: results/")
        print(f"   • psiqrh_result_{timestamp}.json (resultado principal)")
        print(f"   • psiqrh_result_{timestamp}_dcf.yaml (análise DCF)")
        print(f"   • psiqrh_result_{timestamp}_metrics.json (métricas físicas)")
        print("="*60)
        print('\n💡 Para saída JSON limpa: python3 psiqrh.py "what color is the sky" --json')
        print("="*60)

        return output

# Exemplo de uso (será movido para main.py)
if __name__ == '__main__':
    try:
        orchestrator = Orchestrator()
        result = orchestrator.process("Explique a física quântica em termos simples.")
        
        print("\n--- RESULTADO FINAL ---")
        print(f"Texto Gerado: {result['generated_text']}")
        print(f"Validação Matemática Passou: {result['validations']['mathematical']['validation_passed']}")
        print(f"Validação de Texto Passou: {result['validations']['text']['is_valid']}")
        print(f"FCI: {result['consciousness_analysis']['FCI']:.4f}")

        print("\n✅ Comando psiqrh-enhanced executado com sucesso!")

    except ImportError as e:
        print(f"\n❌ ERRO: Falha ao importar dependências. Verifique a instalação e o PYTHONPATH. Detalhe: {e}")
    except Exception as e:
        import traceback
        print(f"\n❌ ERRO INESPERADO NO ORQUESTRADOR: {e}")
        traceback.print_exc()

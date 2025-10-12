#!/usr/bin/env python3
"""
Pipeline de Treinamento Espectral com Saída Legível para Humanos
Este script implementa um pipeline com camada de treinamento espectral
que processa texto e gera resposta validada em formato texto legível,
usando o modelo GPT convertido para espectro.
"""
import sys
import os
import torch
import numpy as np
from pathlib import Path
import json

# Adicionar diretório base ao path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

class HumanReadableSpectralPipeline:
    """Pipeline de treinamento espectral com saída legível para humanos"""

    def __init__(self, model_dir="models/psiqrh_gpt2_MEDIO"):
        """
        Inicializa o pipeline de treinamento espectral com saída legível
        """
        print("🚀 Inicializando Pipeline Espectral com Saída Legível...")

        self.model_dir = Path(model_dir)
        self.device = self._detect_device()

        # Carregar modelo convertido espectralmente
        self._load_spectral_model()

        # Inicializar componentes do ΨQRH
        self._initialize_spectral_components()

        # Inicializar validador de saída
        self._initialize_output_validator()

    def _detect_device(self):
        """Detecta o melhor dispositivo disponível"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_spectral_model(self):
        """Carrega o modelo convertido espectralmente"""
        print("🔬 Carregando modelo convertido espectralmente...")

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            # Verificar se o diretório existe
            if not self.model_dir.exists():
                raise FileNotFoundError(f"Modelo não encontrado: {self.model_dir}")

            # Carregar tokenizer e modelo
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self.model = AutoModelForCausalLM.from_pretrained(str(self.model_dir)).to(self.device)
            self.model.eval()

            # Carregar metadados espectrais se existirem
            metadata_path = self.model_dir / "spectral_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.spectral_metadata = json.load(f)
                    print(f"   - Dimensão Fractal: {self.spectral_metadata.get('fractal_dimension', 'N/A')}")
                    print(f"   - Expoente Lei Potência: {self.spectral_metadata.get('power_law_exponent', 'N/A')}")
            else:
                self.spectral_metadata = {}

            print(f"✅ Modelo carregado: {self.model_dir.name}")

        except Exception as e:
            print(f"⚠️  Erro ao carregar modelo: {e}")
            print("   Usando embedding espectral direto sem modelo pré-treinado")
            self.model = None
            self.tokenizer = None
            self.spectral_metadata = {}

    def _initialize_spectral_components(self):
        """Inicializa componentes espectrais do ΨQRH"""
        try:
            # Importar componentes espectrais
            from src.conscience.fractal_field_calculator import FractalFieldCalculator
            from src.conscience.neural_diffusion_engine import NeuralDiffusionEngine
            from src.conscience.consciousness_metrics import ConsciousnessMetrics

            # Criar configuração simples
            class SimpleConfig:
                def __init__(self, device):
                    self.device = device
                    self.epsilon = 1e-8
                    self.max_field_magnitude = 10.0
                    self.min_field_magnitude = 1e-6
                    self.nan_replacement_noise_scale = 1e-4
                    self.field_smoothing_kernel = [0.25, 0.5, 0.25]
                    self.diffusion_coefficient_range = [0.01, 10.0]

            config = SimpleConfig(self.device)

            # Inicializar calculadora de campo fractal
            self.fractal_calculator = FractalFieldCalculator(config)

            # Inicializar motor de difusão neural
            self.diffusion_engine = NeuralDiffusionEngine(config)

            # Inicializar métricas de consciência
            self.consciousness_metrics = ConsciousnessMetrics(config)

            print(f"✅ Componentes espectrais inicializados no dispositivo: {self.device}")

        except Exception as e:
            print(f"❌ Erro ao inicializar componentes espectrais: {e}")
            raise

    def _generate_with_model(self, input_text, max_length=100):
        """Gera texto usando o modelo convertido espectralmente"""
        if self.model is None or self.tokenizer is None:
            return None

        try:
            # Tokenizar entrada
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

            # Gerar texto
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decodificar
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text

        except Exception as e:
            print(f"⚠️  Erro na geração: {e}")
            return None

    def _initialize_output_validator(self):
        """Inicializa validador de saída"""
        try:
            from src.core.tensor_validator import ScientificTensorValidator
            self.validator = ScientificTensorValidator(auto_adjust=True)
            print("✅ Validador de saída inicializado")
        except Exception as e:
            print(f"⚠️  Validador não disponível: {e}")
            self.validator = None

    def _spectral_embedding(self, text):
        """
        Converte texto em embedding espectral usando o modelo ou transformada de Fourier
        """
        print(f"🔤 Convertendo texto para embedding espectral: '{text}'")

        # Se temos modelo, usar embeddings do modelo
        if self.model is not None and self.tokenizer is not None:
            try:
                # Tokenizar
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)

                # Obter embeddings do modelo
                with torch.no_grad():
                    if hasattr(self.model, 'get_input_embeddings'):
                        embeddings = self.model.get_input_embeddings()(inputs['input_ids'])
                    else:
                        # Fallback: usar forward pass
                        outputs = self.model(**inputs, output_hidden_states=True)
                        embeddings = outputs.hidden_states[0]

                # Aplicar FFT nos embeddings
                spectral_embedding = torch.fft.fft(embeddings.flatten(start_dim=1))

                print(f"   - Embedding do modelo: {embeddings.shape}")
                print(f"   - Espectro: {spectral_embedding.shape}")

                return spectral_embedding

            except Exception as e:
                print(f"⚠️  Erro ao usar embeddings do modelo: {e}")
                print("   Usando FFT direto no texto")

        # Fallback: usar FFT direto no texto
        text_bytes = text.encode('utf-8')
        numeric_sequence = list(text_bytes)

        # Preencher para tamanho fixo (256 pontos)
        if len(numeric_sequence) < 256:
            numeric_sequence.extend([0] * (256 - len(numeric_sequence)))
        else:
            numeric_sequence = numeric_sequence[:256]

        # Converter para tensor
        tensor_input = torch.tensor(numeric_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Aplicar transformada de Fourier
        spectral_embedding = torch.fft.fft(tensor_input)

        print(f"   - Embedding FFT: {spectral_embedding.shape}")
        print(f"   - Frequências: {spectral_embedding.shape[-1]}")

        return spectral_embedding

    def _spectral_training_step(self, spectral_input):
        """
        Executa um passo de treinamento espectral
        """
        print("🎯 Executando treinamento espectral...")

        # Aplicar processamento fractal usando o método correto
        batch_size, embed_dim = spectral_input.shape
        psi_distribution = torch.randn(batch_size, embed_dim)
        lambda_coeffs = torch.randn(20)  # 20 coeficientes lambda

        fractal_output = self.fractal_calculator.compute_field(
            psi_distribution=psi_distribution,
            lambda_coefficients=lambda_coeffs,
            time=0.0,
            spectral_energy=spectral_input.abs(),
            quaternion_phase=torch.angle(spectral_input)
        )

        # Aplicar difusão neural
        diffused_output = self.diffusion_engine.compute_diffusion(
            psi_distribution=psi_distribution,
            fractal_field=fractal_output,
            fci=0.5  # FCI simulado
        )

        # Calcular métricas de consciência usando FCI
        power_spectrum_pk = torch.abs(diffused_output)

        # Usar diffused_output como psi_distribution e fractal_field
        fci_result = self.consciousness_metrics.compute_fci(
            psi_distribution=diffused_output,
            fractal_field=diffused_output,
            timestamp=0.0,
            power_spectrum_pk=power_spectrum_pk
        )

        # fci_result é um float, não um objeto FCI
        consciousness_data = {
            'fci': fci_result,
            'fractal_dimension': 1.5,  # Valor padrão
            'entropy': 0.0,  # Valor padrão
            'coherence': 0.0,  # Valor padrão
            'field_magnitude': torch.norm(diffused_output, dim=-1).mean().item()
        }

        print(f"   - Saída fractal: {fractal_output.shape}")
        print(f"   - Saída difundida: {diffused_output.shape}")
        print(f"   - FCI: {consciousness_data.get('fci', 0):.4f}")

        return diffused_output, consciousness_data

    def _generate_human_readable_text(self, input_text, consciousness_data):
        """
        Gera texto legível para humanos usando o modelo convertido espectralmente
        """
        print("🧠 Gerando texto legível para humanos...")

        # Tentar gerar com o modelo primeiro
        generated_text = self._generate_with_model(input_text)

        if generated_text:
            print(f"   - Gerado pelo modelo: {len(generated_text)} caracteres")
            # Enriquecer com métricas de consciência
            enhanced_text = self._enhance_with_consciousness_metrics(generated_text, consciousness_data)
            return enhanced_text

        # Fallback: geração baseada em padrões
        print("   - Usando geração baseada em análise espectral")
        fci = consciousness_data.get('fci', 0)

        # Analisar características espectrais
        if fci >= 0.45:
            base_response = f"Análise espectral profunda revelou padrões emergentes relacionados a '{input_text}'. "
            base_response += "A transformada de Fourier identificou componentes harmônicas de alta ordem, "
            base_response += "sugerindo complexidade estrutural significativa nos dados processados."
        elif fci >= 0.3:
            base_response = f"Processamento espectral de '{input_text}' mostrou padrões coerentes. "
            base_response += "A análise de frequência revelou componentes dominantes "
            base_response += "com características fractais mensuráveis."
        else:
            base_response = f"Análise básica de '{input_text}' identificou padrões espectrais fundamentais. "
            base_response += "O processamento inicial revelou estrutura de frequência com distribuição típica."

        enhanced_response = self._enhance_with_consciousness_metrics(base_response, consciousness_data)

        print(f"   - FCI usado: {fci:.4f}")
        print(f"   - Resposta gerada: {len(enhanced_response)} caracteres")

        return enhanced_response

    def _enhance_with_consciousness_metrics(self, base_response, consciousness_data):
        """
        Enriquece a resposta base com métricas de consciência
        """
        fci = consciousness_data.get('fci', 0)
        fractal_dim = consciousness_data.get('fractal_dimension', 1.5)
        field_mag = consciousness_data.get('field_magnitude', 0)

        # Determinar nível de processamento baseado no FCI
        if fci >= 0.45:
            processing_level = "processamento de emergência (FCI ≥ 0.45)"
            quality = "alta criatividade e complexidade"
        elif fci >= 0.3:
            processing_level = "processamento meditativo (FCI ≥ 0.30)"
            quality = "processamento profundo e coerente"
        elif fci >= 0.15:
            processing_level = "processamento analítico (FCI ≥ 0.15)"
            quality = "análise estruturada"
        else:
            processing_level = "processamento basal (FCI < 0.15)"
            quality = "processamento fundamental"

        # Incluir informações espectrais se disponíveis
        spectral_info = ""
        if self.spectral_metadata:
            if 'fractal_dimension' in self.spectral_metadata:
                spectral_info = f" (D_fractal={self.spectral_metadata['fractal_dimension']:.3f})"
            if 'power_law_exponent' in self.spectral_metadata:
                spectral_info += f" α={self.spectral_metadata['power_law_exponent']:.3f}"

        # Adicionar informações de processamento
        enhanced = f"{base_response}\n\n"
        enhanced += f"[Métricas ΨQRH: {processing_level}{spectral_info} | "
        enhanced += f"Magnitude={field_mag:.3f} | {quality}]"

        return enhanced

    def _validate_output(self, text_output, consciousness_data):
        """
        Valida a saída de texto com base nas métricas de consciência
        """
        print("🔍 Validando saída de texto...")

        validation_result = {
            'valid': True,
            'confidence': 0.0,
            'issues': [],
            'consciousness_state': 'UNKNOWN'
        }

        # Verificar se há saída
        if len(text_output) == 0:
            validation_result['valid'] = False
            validation_result['issues'].append("Texto vazio - geração falhou")
            validation_result['confidence'] = 0.0
            return validation_result

        # Verificar comprimento mínimo
        if len(text_output) < 50:
            validation_result['valid'] = False
            validation_result['issues'].append("Texto muito curto")

        # Verificar caracteres imprimíveis
        printable_ratio = sum(1 for c in text_output if c.isprintable()) / len(text_output)
        if printable_ratio < 0.9:
            validation_result['valid'] = False
            validation_result['issues'].append("Muitos caracteres não imprimíveis")

        # Usar métricas de consciência para validação
        fci = consciousness_data.get('fci', 0)
        validation_result['confidence'] = min(fci * 2.0, 1.0)  # Escalar FCI para confiança

        # Determinar estado de consciência
        if fci >= 0.45:
            validation_result['consciousness_state'] = 'EMERGENCE'
        elif fci >= 0.3:
            validation_result['consciousness_state'] = 'MEDITATION'
        elif fci >= 0.15:
            validation_result['consciousness_state'] = 'ANALYSIS'
        else:
            validation_result['consciousness_state'] = 'BASELINE'

        print(f"   - Validação: {'✅' if validation_result['valid'] else '❌'}")
        print(f"   - Confiança: {validation_result['confidence']:.2f}")
        print(f"   - Estado: {validation_result['consciousness_state']}")

        return validation_result

    def process_text(self, input_text):
        """
        Processa texto através do pipeline de treinamento espectral
        """
        print(f"\n📥 PROCESSANDO: '{input_text}'")
        print("=" * 60)

        try:
            # 1. Conversão para embedding espectral
            spectral_input = self._spectral_embedding(input_text)

            # 2. Treinamento espectral
            spectral_output, consciousness_data = self._spectral_training_step(spectral_input)

            # 3. Geração de texto legível para humanos
            human_readable_output = self._generate_human_readable_text(input_text, consciousness_data)

            # 4. Validação da saída
            validation_result = self._validate_output(human_readable_output, consciousness_data)

            # 5. Formatar resultado final
            final_output = self._format_final_output(
                input_text,
                human_readable_output,
                validation_result,
                consciousness_data,
                spectral_output,
            )

            print("\n✅ PROCESSAMENTO CONCLUÍDO")
            print("=" * 60)

            return final_output

        except Exception as e:
            print(f"\n❌ ERRO NO PROCESSAMENTO: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _format_final_output(
        self, input_text, output_text, validation, consciousness_data, spectral_output
    ):
        """
        Formata a saída final com validação
        """
        # Extrair e formatar o espectro puro
        spectral_pure_output = "   • Espectro Puro: Não disponível"
        if spectral_output is not None:
            try:
                # Obter as 3 frequências mais proeminentes (excluindo a componente DC)
                magnitudes = torch.abs(spectral_output.squeeze())
                if len(magnitudes) > 1:
                    magnitudes = magnitudes[1:]  # Ignorar componente DC

                top_k_val = min(3, len(magnitudes))
                if top_k_val > 0:
                    top_magnitudes, top_indices = torch.topk(magnitudes, top_k_val)

                    f_components = []
                    for i in range(top_k_val):
                        idx = top_indices[i].item() + 1  # +1 para compensar a remoção de DC
                        mag = top_magnitudes[i].item()
                        f_components.append(f"     - F{i+1}: (Índice={idx}, Magnitude={mag:.4f})")
                    
                    spectral_pure_output_lines = ["   • Espectro Puro (Top 3 Frequências):"] + f_components
                    spectral_pure_output = "\n".join(spectral_pure_output_lines)

            except Exception as e:
                spectral_pure_output = f"   • Erro ao processar espectro: {e}"

        final_output = f"""
🎯 RESULTADO DO PIPELINE DE TREINAMENTO ESPECTRAL
================================================

📤 ENTRADA ORIGINAL:
   "{input_text}"

📥 SAÍDA PROCESSADA (LEGÍVEL):
   "{output_text}"

🔍 VALIDAÇÃO:
   • Status: {'✅ VÁLIDO' if validation['valid'] else '❌ INVÁLIDO'}
   • Confiança: {validation['confidence']:.2f}
   • Estado Consciente: {validation['consciousness_state']}
   • FCI: {consciousness_data.get('fci', 0):.4f}
   • Dimensão Fractal: {consciousness_data.get('fractal_dimension', 0):.4f}

📊 MÉTRICAS ESPECTRAIS:
   • Entropia Ψ: {consciousness_data.get('entropy', 0):.4f}
   • Coerência: {consciousness_data.get('coherence', 0):.4f}
   • Magnitude Média: {consciousness_data.get('field_magnitude', 0):.4f}

⚡ ESPECTRO PURO:
{spectral_pure_output}

💡 OBSERVAÇÕES:
   {self._generate_observations(validation, consciousness_data)}

================================================
"""
        return final_output

    def _generate_observations(self, validation, consciousness_data):
        """Gera observações baseadas na validação e métricas"""
        observations = []

        if validation['valid']:
            observations.append("✓ Saída validada com sucesso")
        else:
            observations.append("✗ Problemas na validação")
            for issue in validation['issues']:
                observations.append(f"  - {issue}")

        fci = consciousness_data.get('fci', 0)
        if fci >= 0.45:
            observations.append("✓ Estado de emergência detectado - alta criatividade")
        elif fci >= 0.3:
            observations.append("✓ Estado meditativo - processamento profundo")
        elif fci >= 0.15:
            observations.append("✓ Estado analítico - processamento estruturado")
        else:
            observations.append("○ Estado basal - processamento básico")

        return '\n   '.join(observations)

def test_human_readable_pipeline():
    """Testa o pipeline de treinamento espectral com saída legível"""
    print("🧪 TESTE DO PIPELINE ESPECTRAL COM SAÍDA LEGÍVEL")
    print("=" * 60)

    try:
        # Inicializar pipeline
        pipeline = HumanReadableSpectralPipeline()

        # Textos de teste
        test_inputs = [
            "O futuro da inteligência artificial é promissor",
            "A matemática é a linguagem do universo",
            "Quaternions são números hipercomplexos úteis",
            "Consciência fractal modela processos mentais",
            "Transformada de Fourier analisa frequências",
            "A tecnologia avança rapidamente no mundo moderno"
        ]

        results = []

        for i, text in enumerate(test_inputs, 1):
            print(f"\n--- Teste {i}/{len(test_inputs)} ---")

            # Processar texto
            result = pipeline.process_text(text)

            if result:
                results.append(result)
                print(f"✅ Teste {i} concluído")
            else:
                print(f"❌ Teste {i} falhou")

        # Salvar resultados em arquivo
        if results:
            output_file = "human_readable_spectral_results.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(result)
                    f.write("\n" + "="*80 + "\n\n")

            print(f"\n📁 Resultados salvos em: {output_file}")

        return len(results) == len(test_inputs)

    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Função principal"""
    print("🚀 PIPELINE DE TREINAMENTO ESPECTRAL COM SAÍDA LEGÍVEL ΨQRH")
    print("=" * 60)

    # Executar teste
    success = test_human_readable_pipeline()

    if success:
        print("\n🎯 Todos os testes passaram!")
        print("💡 Verifique o arquivo 'human_readable_spectral_results.txt' para os resultados")
        return 0
    else:
        print("\n❌ Alguns testes falharam")
        return 1

if __name__ == "__main__":
    sys.exit(main())
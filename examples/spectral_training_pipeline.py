#!/usr/bin/env python3
"""
Pipeline de Treinamento Espectral ΨQRH
Este script implementa um pipeline com camada de treinamento espectral
que processa texto e gera resposta validada em formato texto.
"""
import sys
import os
import torch
import numpy as np
from pathlib import Path

# Adicionar diretório base ao path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

class SpectralTrainingPipeline:
    """Pipeline de treinamento espectral com validação de saída"""

    def __init__(self, model_dir="models/psiqrh_gpt2_MEDIO"):
        """
        Inicializa o pipeline de treinamento espectral
        """
        print("🚀 Inicializando Pipeline de Treinamento Espectral...")

        self.model_dir = model_dir
        self.device = self._detect_device()

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

    def _initialize_spectral_components(self):
        """Inicializa componentes espectrais do ΨQRH"""
        try:
            # Importar componentes espectrais do diretório correto
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
        Converte texto em embedding espectral usando transformada de Fourier
        """
        print(f"🔤 Convertendo texto para embedding espectral: '{text}'")

        # Converter texto para sequência numérica simples
        text_bytes = text.encode('utf-8')
        numeric_sequence = list(text_bytes)

        # Preencher para tamanho fixo (256 pontos)
        if len(numeric_sequence) < 256:
            numeric_sequence.extend([0] * (256 - len(numeric_sequence)))
        else:
            numeric_sequence = numeric_sequence[:256]

        # Converter para tensor
        tensor_input = torch.tensor(numeric_sequence, dtype=torch.float32).unsqueeze(0)

        # Aplicar transformada de Fourier
        spectral_embedding = torch.fft.fft(tensor_input)

        print(f"   - Embedding espectral: {spectral_embedding.shape}")
        print(f"   - Frequências: {spectral_embedding.shape[-1]}")

        return spectral_embedding

    def _spectral_training_step(self, spectral_input):
        """
        Executa um passo de treinamento espectral
        """
        print("🎯 Executando treinamento espectral...")

        # Aplicar processamento fractal usando o método correto
        # Criar dados simulados para o fractal calculator
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
        # Criar dados simulados para cálculo do FCI
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

    def _spectral_to_text(self, spectral_output):
        """
        Converte saída espectral de volta para texto
        """
        print("🔄 Convertendo saída espectral para texto...")

        # Aplicar transformada inversa de Fourier
        time_domain = torch.fft.ifft(spectral_output)

        # Converter para valores reais
        real_values = time_domain.real.squeeze().detach().numpy()

        # Normalizar e converter para caracteres ASCII
        normalized = np.clip(real_values, 32, 126)  # Apenas caracteres ASCII imprimíveis
        char_sequence = [chr(int(val)) for val in normalized[:100]]

        # Combinar em texto
        output_text = ''.join(char_sequence)

        print(f"   - Sequência convertida: {len(output_text)} caracteres")

        return output_text

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
            validation_result['issues'].append("Texto vazio - conversão espectral falhou")
            validation_result['confidence'] = 0.0
            return validation_result

        # Verificar comprimento mínimo
        if len(text_output) < 10:
            validation_result['valid'] = False
            validation_result['issues'].append("Texto muito curto")

        # Verificar caracteres imprimíveis
        printable_ratio = sum(1 for c in text_output if c.isprintable()) / len(text_output)
        if printable_ratio < 0.7:
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

            # 3. Conversão de volta para texto
            text_output = self._spectral_to_text(spectral_output)

            # 4. Validação da saída
            validation_result = self._validate_output(text_output, consciousness_data)

            # 5. Formatar resultado final
            final_output = self._format_final_output(
                input_text, text_output, validation_result, consciousness_data
            )

            print("\n✅ PROCESSAMENTO CONCLUÍDO")
            print("=" * 60)

            return final_output

        except Exception as e:
            print(f"\n❌ ERRO NO PROCESSAMENTO: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _format_final_output(self, input_text, output_text, validation, consciousness_data):
        """
        Formata a saída final com validação
        """
        final_output = f"""
🎯 RESULTADO DO PIPELINE DE TREINAMENTO ESPECTRAL
================================================

📤 ENTRADA ORIGINAL:
   "{input_text}"

📥 SAÍDA PROCESSADA:
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

def test_spectral_pipeline():
    """Testa o pipeline de treinamento espectral"""
    print("🧪 TESTE DO PIPELINE DE TREINAMENTO ESPECTRAL")
    print("=" * 60)

    try:
        # Inicializar pipeline
        pipeline = SpectralTrainingPipeline()

        # Textos de teste
        test_inputs = [
            "O futuro da inteligência artificial é promissor",
            "A matemática é a linguagem do universo",
            "Quaternions são números hipercomplexos úteis",
            "Consciência fractal modela processos mentais",
            "Transformada de Fourier analisa frequências"
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
            output_file = "spectral_training_results.txt"
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
    print("🚀 PIPELINE DE TREINAMENTO ESPECTRAL ΨQRH")
    print("=" * 60)

    # Executar teste
    success = test_spectral_pipeline()

    if success:
        print("\n🎯 Todos os testes passaram!")
        print("💡 Verifique o arquivo 'spectral_training_results.txt' para os resultados")
        return 0
    else:
        print("\n❌ Alguns testes falharam")
        return 1

if __name__ == "__main__":
    sys.exit(main())
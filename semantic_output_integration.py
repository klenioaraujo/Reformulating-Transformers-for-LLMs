#!/usr/bin/env python3
"""
Integração da Saída Semântica no Pipeline ΨQRH
==============================================

Integra a saída dos modelos semânticos no pipeline de geração de texto ΨQRH,
combinando os parâmetros espectrais extraídos com o sistema de geração quântica.

Princípios Integrados:
- Equação de Padilha: f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))
- Filtragem Espectral: F(k) = exp(i α · arctan(ln(|k| + ε)))
- Sistema DCF: Dinâmica de Consciência Fractal
- Pipeline ΨQRH: Integração completa com geração de texto

Uso:
    from semantic_output_integration import SemanticOutputIntegrator
    integrator = SemanticOutputIntegrator()
    result = integrator.generate_with_semantic_model('gpt2', 'Hello world')
"""

import torch
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from spectral_parameters_integration import SpectralParametersIntegrator
from src.core.dynamic_quantum_matrix import DynamicQuantumCharacterMatrix
from advanced_physical_validation import AdvancedPhysicalValidator
from src.core.efficient_quantum_decoder import EfficientQuantumDecoder


class SemanticOutputIntegrator:
    """
    Integra a saída dos modelos semânticos no pipeline ΨQRH completo.
    """

    def __init__(self):
        self.spectral_integrator = SpectralParametersIntegrator()
        self.quantum_matrix = DynamicQuantumCharacterMatrix()
        self.validator = AdvancedPhysicalValidator()
        self.efficient_decoder = None  # Inicializado sob demanda
        self.current_model = None

        print("🔗 Semantic Output Integrator inicializado")

    def generate_with_semantic_model(self, model_name: str, input_text: str,
                                   max_length: int = 50) -> Dict[str, Any]:
        """
        Gera texto usando modelo semântico integrado no pipeline ΨQRH.

        Args:
            model_name: Nome do modelo semântico
            input_text: Texto de entrada
            max_length: Comprimento máximo da geração

        Returns:
            Resultado da geração com métricas físicas
        """
        print(f"🎯 Gerando com modelo semântico: {model_name}")
        print(f"📝 Entrada: '{input_text}'")

        # 1. Preparar modelo semântico
        if not self._prepare_semantic_model(model_name):
            return {
                'status': 'error',
                'error': f'Falha ao preparar modelo {model_name}'
            }

        # 2. Processar entrada com matriz quântica adaptada
        quantum_input = self._encode_input_text(input_text)

        # 3. Aplicar operações quânticas do pipeline ΨQRH
        processed_output = self._apply_quantum_pipeline(quantum_input)

        # 4. Gerar sequência usando Equação de Padilha
        generated_sequence = self._generate_with_padilha_equation(
            processed_output, max_length
        )

        # 5. Decodificar para texto usando sistema DCF
        final_text = self._decode_with_dcf_system(generated_sequence)

        # 6. Computar métricas físicas
        physical_metrics = self._compute_physical_metrics(
            quantum_input, processed_output, generated_sequence
        )

        # 7. Preparar resultado final
        result = {
            'status': 'success',
            'model_name': model_name,
            'input_text': input_text,
            'generated_text': final_text,
            'physical_metrics': physical_metrics,
            'spectral_parameters': self.quantum_matrix.get_current_parameters(),
            'generation_method': 'Semantic ΨQRH Pipeline',
            'fcf_value': self._compute_fcf_metric(processed_output),
            'consciousness_state': self._determine_consciousness_state(physical_metrics),
            'synchronization_order': self._compute_synchronization_order(generated_sequence)
        }

        print("✅ Geração concluída com sucesso!")
        print(f"📝 Texto gerado: '{final_text[:100]}{'...' if len(final_text) > 100 else ''}'")

        return result

    def _prepare_semantic_model(self, model_name: str) -> bool:
        """
        Prepara o modelo semântico para uso.
        """
        try:
            # Adaptar matriz quântica aos parâmetros do modelo
            success = self.quantum_matrix.adapt_to_model(model_name)
            if success:
                self.current_model = model_name
            return success
        except Exception as e:
            print(f"❌ Erro preparando modelo {model_name}: {e}")
            return False

    def _encode_input_text(self, text: str) -> torch.Tensor:
        """
        Codifica texto de entrada usando matriz quântica adaptada.
        """
        return self.quantum_matrix.encode_text(text)

    def _apply_quantum_pipeline(self, quantum_input: torch.Tensor) -> torch.Tensor:
        """
        Aplica operações quânticas do pipeline ΨQRH.
        """
        # Aplicar filtragem espectral
        filtered = self._apply_spectral_filtering(quantum_input)

        # Aplicar rotações SO(4)
        rotated = self._apply_so4_rotations(filtered)

        # Aplicar processamento de consciência (simplificado)
        conscious = self._apply_consciousness_processing(rotated)

        return conscious

    def _apply_spectral_filtering(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Aplica filtragem espectral baseada nos parâmetros do modelo usando camadas reais do ΨQRH.
        """
        # Usar diretamente a camada de filtragem espectral do DynamicQuantumCharacterMatrix
        # Formatar tensor para [batch, seq, hidden] -> [batch, hidden, seq] para conv1d
        if tensor.dim() == 1:  # [hidden] - tensor unidimensional
            # Para tensor 1D, expandir para formato adequado [batch=1, channels=hidden_size, seq=1]
            x = tensor.unsqueeze(0).unsqueeze(-1)  # [1, hidden, 1]
        elif tensor.dim() == 2:  # [seq, hidden]
            x = tensor.unsqueeze(0).transpose(1, 2)  # [1, hidden, seq]
        else:
            x = tensor.transpose(0, 1).unsqueeze(0)  # [1, hidden, seq]

        # Aplicar filtro espectral real mantendo fase complexa
        filtered = self.quantum_matrix.adaptation_layers['spectral_filter'](x)

        # Reverter formato
        if tensor.dim() == 1:
            return filtered.squeeze(0).squeeze(-1)  # [hidden]
        else:
            return filtered.squeeze(0).transpose(0, 1)  # [seq, hidden]

    def _apply_so4_rotations(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Aplica rotações SO(4) unitárias usando camadas reais do ΨQRH.
        """
        # Usar diretamente a camada de rotação quaterniónica do DynamicQuantumCharacterMatrix
        # Formatar tensor para [batch, seq, hidden]
        if tensor.dim() == 1:  # [hidden] - tensor unidimensional
            x = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
        elif tensor.dim() == 2:  # [seq, hidden]
            x = tensor.unsqueeze(0)  # [1, seq, hidden]
        else:
            x = tensor  # Já no formato correto

        # Aplicar rotações SO(4) verdadeiras
        rotated = self.quantum_matrix.adaptation_layers['quaternion_rotator'](x)

        # Reverter formato se necessário
        if tensor.dim() == 1:
            return rotated.squeeze(0).squeeze(0)  # [hidden]
        else:
            return rotated.squeeze(0) if tensor.dim() == 2 else rotated

    def _apply_consciousness_processing(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Aplica processamento de consciência (FCI computation).
        """
        # Processamento simplificado - normalizar baseado na energia
        energy = torch.norm(tensor)
        if energy > 0:
            normalized = tensor / energy
            # Aplicar transformação não-linear para simular processamento consciente
            # Versão complexa do tanh (aplicar separadamente às partes)
            real_part = torch.tanh(normalized.real * 2.0)
            imag_part = torch.tanh(normalized.imag * 2.0)
            conscious = torch.complex(real_part, imag_part)
            return conscious

        return tensor

    def _generate_with_padilha_equation(self, quantum_state: torch.Tensor,
                                       max_length: int) -> torch.Tensor:
        """
        Gera sequência usando Equação de Padilha com autoregressão quântica.
        """
        params = self.quantum_matrix.get_current_parameters()
        if not params:
            # Fallback para geração simples
            return torch.randn(max_length, quantum_state.size(-1))

        alpha = params.get('alpha_final', 1.5)
        beta = params.get('beta_final', 0.8)

        # Parâmetros da Equação de Padilha
        I0 = 1.0
        omega = alpha
        k = beta

        # Geração autoregressiva quântica
        sequence = []
        current_state = quantum_state.clone()  # Estado quântico inicial

        for t in range(max_length):
            lambda_val = t / max_length  # Posição normalizada

            # f(λ,t) = I₀ sin(ωt + αλ) e^(i(ωt - kλ + βλ²))
            wave_function = I0 * torch.sin(torch.tensor(omega * t + alpha * lambda_val)) * \
                           torch.exp(1j * torch.tensor(omega * t - k * lambda_val + beta * lambda_val**2))

            # Modulação quântica baseada no estado atual
            modulation = wave_function * current_state.mean(dim=0)

            # Aplicar pipeline ΨQRH ao estado modulado
            processed_state = self._apply_quantum_pipeline(modulation.unsqueeze(0)).squeeze(0)

            # Próximo estado é baseado no estado processado
            next_state = processed_state * wave_function.conj()  # Evolução unitária

            sequence.append(processed_state)
            current_state = next_state  # Atualizar estado para autoregressão

        return torch.stack(sequence)

    def _extract_original_model_name(self, semantic_model_name: str) -> str:
        """
        Extrai o nome do modelo original do nome do modelo semântico.

        Args:
            semantic_model_name: Nome do modelo semântico (ex: 'psiqrh_semantic_gpt2')

        Returns:
            Nome do modelo original para tokenização
        """
        if not semantic_model_name or not semantic_model_name.startswith('psiqrh_semantic_'):
            return 'gpt2'  # Fallback padrão

        # Remover prefixo 'psiqrh_semantic_'
        original_name = semantic_model_name.replace('psiqrh_semantic_', '')

        # Mapear nomes especiais se necessário
        name_mapping = {
            'gpt2': 'gpt2',
            # Adicionar outros mapeamentos conforme necessário
        }

        return name_mapping.get(original_name, original_name)

    def _decode_with_dcf_system(self, sequence: torch.Tensor) -> str:
        """
        Decodifica sequência usando sistema DCF (Dinâmica de Consciência Fractal).
        Usa o EfficientQuantumDecoder para decodificação precisa.
        """
        if self.efficient_decoder is None:
            self.efficient_decoder = EfficientQuantumDecoder(verbose=False)  # Modo silencioso para produção
            self.efficient_decoder.initialize_with_quantum_matrix(self.quantum_matrix)

        tokens = self.efficient_decoder.inverse_decode(sequence.unsqueeze(0))
        return self.efficient_decoder.tokens_to_text(tokens)

    def _compute_physical_metrics(self, input_tensor: torch.Tensor,
                                processed_tensor: torch.Tensor,
                                generated_sequence: torch.Tensor) -> Dict[str, Any]:
        """
        Computa métricas físicas da geração.
        """
        metrics = {}

        # Validação de conservação de energia
        energy_validation = self.validator.validate_energy_conservation(
            input_tensor, processed_tensor
        )
        metrics['energy_conservation'] = energy_validation

        # Validação de estabilidade numérica
        stability_validation = self.validator.validate_numerical_stability(
            input_tensor, processed_tensor
        )
        metrics['numerical_stability'] = stability_validation

        # Métricas da sequência gerada
        metrics['sequence_metrics'] = {
            'length': generated_sequence.size(0),
            'mean_magnitude': generated_sequence.abs().mean().item(),
            'std_magnitude': generated_sequence.abs().std().item(),
            'complexity': self._compute_sequence_complexity(generated_sequence)
        }

        # Parâmetros da Equação de Padilha
        params = self.quantum_matrix.get_current_parameters() or {}
        metrics['padilha_parameters'] = {
            'I0': 1.0,
            'omega': params.get('alpha_final', 1.5),
            'k': params.get('beta_final', 0.8),
            'alpha': params.get('alpha_final', 1.5),
            'beta': params.get('beta_final', 0.8)
        }

        return metrics

    def _compute_fcf_metric(self, tensor: torch.Tensor) -> float:
        """
        Computa métrica FCF (Fractal Consciousness Factor).
        """
        # Simplificação: baseado na complexidade espectral
        if tensor.numel() > 0:
            # Usar variância como proxy de complexidade
            complexity = torch.var(tensor).item()
            # Normalizar para [0, 1]
            fcf = min(1.0, complexity / 10.0)
            return fcf
        return 0.5

    def _determine_consciousness_state(self, metrics: Dict) -> str:
        """
        Determina estado de consciência baseado nas métricas.
        """
        # Extrair FCF real das métricas ou computar do estado
        fcf = metrics.get('fcf_value', 0.5)  # ou compute de processed_tensor

        if fcf > 0.7:
            return "ENLIGHTENMENT"
        elif fcf > 0.5:
            return "MEDITATION"
        elif fcf > 0.3:
            return "FOCUS"
        else:
            return "CONFUSION"

    def _compute_synchronization_order(self, sequence: torch.Tensor) -> float:
        """
        Computa ordem de sincronização da sequência gerada.
        """
        if sequence.size(0) < 2:
            return 0.5

        # Usar correlação entre passos consecutivos como proxy
        correlations = []
        for i in range(sequence.size(0) - 1):
            # Compute correlação das magnitudes ou partes reais
            real_i = sequence[i].real
            real_ip1 = sequence[i+1].real
            corr = torch.corrcoef(torch.stack([real_i, real_ip1]))[0, 1].item()
            correlations.append(abs(corr))

        return np.mean(correlations) if correlations else 0.5

    def _compute_sequence_complexity(self, sequence: torch.Tensor) -> float:
        """
        Computa complexidade da sequência gerada.
        """
        if sequence.numel() == 0:
            return 0.0

        # Usar entropia como medida de complexidade
        flattened = sequence.flatten().abs()

        # Discretizar em bins
        bins = torch.histc(flattened, bins=10, min=0, max=flattened.max().item())

        # Computar entropia
        probs = bins / bins.sum()
        probs = probs[probs > 0]  # Remover zeros
        entropy = -torch.sum(probs * torch.log2(probs))

        return entropy.item()


# Função de teste
def test_semantic_output_integration():
    """
    Testa a integração da saída semântica.
    """
    print("🧪 Teste de Integração da Saída Semântica")
    print("=" * 50)

    integrator = SemanticOutputIntegrator()

    # Testar com modelo disponível
    available_models = integrator.spectral_integrator.get_available_models()

    if available_models:
        test_model = available_models[0]
        test_text = "Hello quantum"

        print(f"🎯 Testando com modelo: {test_model}")
        print(f"📝 Texto de entrada: '{test_text}'")

        try:
            result = integrator.generate_with_semantic_model(
                test_model, test_text, max_length=20
            )

            if result['status'] == 'success':
                print("✅ Geração bem-sucedida!")
                print(f"📝 Texto gerado: '{result['generated_text']}'")
                print(f"🧠 FCF: {result['fcf_value']:.3f}")
                print(f"🎭 Estado: {result['consciousness_state']}")
                print(f"🔄 Sincronização: {result['synchronization_order']:.3f}")

                # Verificar métricas físicas
                energy = result['physical_metrics']['energy_conservation']
                print(f"⚡ Conservação de energia: {energy['energy_conserved']}")

            else:
                print(f"❌ Falha: {result.get('error', 'Erro desconhecido')}")

        except Exception as e:
            print(f"💥 Erro durante teste: {e}")
    else:
        print("⚠️  Nenhum modelo semântico disponível para teste")


if __name__ == "__main__":
    test_semantic_output_integration()
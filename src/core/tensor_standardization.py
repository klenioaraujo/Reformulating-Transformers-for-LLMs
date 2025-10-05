#!/usr/bin/env python3
"""
Sistema Unificado de Tensores ΨQRH
===================================

Padronização completa para resolver incompatibilidades dimensionais
entre componentes quânticos, clássicos e de memória.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import torch.nn as nn
import math
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class QRHTensorSpec:
    """
    Especificação padronizada para todos os tensores no sistema ΨQRH

    Resolve incompatibilidades dimensionais através de especificações claras
    e validadas para cada estágio do pipeline.
    """

    # Tensores principais do pipeline
    SPECTRAL_INPUT = {
        'shape': [1, 64, 64],      # [batch, freq_bins, time_frames]
        'dtype': torch.float32,
        'description': 'Representação espectral do texto de entrada',
        'physics_domain': 'classical_spectral'
    }

    QUATERNION_STATES = {
        'shape': [1, 64, 64, 4],   # [batch, dim1, dim2, quaternion_components]
        'dtype': torch.float32,
        'description': 'Estados quaterniônicos após mapeamento Hamilton',
        'physics_domain': 'quantum_quaternion'
    }

    QUANTUM_EMBEDDINGS = {
        'shape': [1, 512],         # [batch, embedding_dim]
        'dtype': torch.float32,
        'description': 'Embeddings quânticos unificados para processamento',
        'physics_domain': 'quantum_unified'
    }

    LANGUAGE_OUTPUT = {
        'shape': [1, 128],         # [batch, sequence_length]
        'dtype': torch.int64,
        'description': 'Saída linguística tokenizada',
        'physics_domain': 'classical_linguistic'
    }

    # Tensores auxiliares para componentes específicos
    MEMORY_BUFFER = {
        'shape': [10, 512],        # [memory_slots, embedding_dim]
        'dtype': torch.float32,
        'description': 'Buffer de memória quântica temporal',
        'physics_domain': 'quantum_memory'
    }

    CONSCIOUSNESS_STATE = {
        'shape': [1, 64],          # [batch, consciousness_features]
        'dtype': torch.float32,
        'description': 'Estado de consciência fractal',
        'physics_domain': 'quantum_consciousness'
    }

    @classmethod
    def get_spec(cls, spec_name: str) -> Dict[str, Any]:
        """Obtém especificação por nome"""
        return getattr(cls, spec_name.upper(), None)

    @classmethod
    def validate_tensor(cls, tensor: torch.Tensor, spec_name: str) -> bool:
        """Valida se tensor corresponde à especificação"""
        spec = cls.get_spec(spec_name)
        if spec is None:
            return False

        expected_shape = spec['shape']
        expected_dtype = spec['dtype']

        # Verificar forma (permitir batch dimension variável)
        if len(tensor.shape) != len(expected_shape):
            return False

        # Verificar dimensões não-batch
        for i in range(1, len(expected_shape)):
            if tensor.shape[i] != expected_shape[i]:
                return False

        # Verificar tipo de dados
        if tensor.dtype != expected_dtype:
            return False

        return True

    @classmethod
    def get_all_specs(cls) -> Dict[str, Dict[str, Any]]:
        """Retorna todas as especificações disponíveis"""
        specs = {}
        for attr_name in dir(cls):
            if not attr_name.startswith('_') and attr_name.isupper():
                attr_value = getattr(cls, attr_name)
                if isinstance(attr_value, dict) and 'shape' in attr_value:
                    specs[attr_name.lower()] = attr_value
        return specs


class UniversalTensorAdapter:
    """
    Conversor universal entre todos os formatos de tensor ΨQRH

    Implementa grafo de conversões para transformar tensores entre
    domínios quânticos e clássicos preservando informação física.
    """

    def __init__(self):
        self.specs = QRHTensorSpec()
        self.conversion_graph = self._build_conversion_graph()
        self._initialize_conversion_matrices()

    def _initialize_conversion_matrices(self):
        """Inicializa matrizes de projeção para conversões"""
        self.projection_matrices = {}

        # Matriz para conversão quaternion → embedding
        self.projection_matrices['quat_to_embed'] = torch.randn(512, 16384) / math.sqrt(16384)

        # Matriz para conversão embedding → quaternion
        self.projection_matrices['embed_to_quat'] = torch.randn(16384, 512) / math.sqrt(512)

        # Matriz para conversão spectral → quaternion
        self.projection_matrices['spec_to_quat'] = torch.randn(64 * 64 * 4, 64 * 64) / math.sqrt(64 * 64)

        # Matriz para conversão embedding → language
        self.projection_matrices['embed_to_lang'] = torch.randn(128, 512) / math.sqrt(512)

        # Matriz para conversão language → embedding
        self.projection_matrices['lang_to_embed'] = torch.randn(512, 128) / math.sqrt(128)

    def convert(self, tensor: torch.Tensor, target_spec: str) -> torch.Tensor:
        """
        Conversão padronizada entre especificações

        Args:
            tensor: Tensor de entrada
            target_spec: Nome da especificação alvo (ex: 'QUANTUM_EMBEDDINGS')

        Returns:
            Tensor convertido para especificação alvo
        """
        source_spec = self._infer_spec(tensor)

        if source_spec == target_spec:
            return tensor

        conversion_path = self._find_conversion_path(source_spec, target_spec)
        if not conversion_path:
            raise ValueError(f"Não há caminho de conversão de {source_spec} para {target_spec}")

        current_tensor = tensor

        for conversion_step in conversion_path:
            current_tensor = self._apply_conversion(current_tensor, conversion_step)

        return self._validate_output(current_tensor, target_spec)

    def _infer_spec(self, tensor: torch.Tensor) -> str:
        """Infere especificação do tensor baseado na forma"""
        shape = list(tensor.shape)
        dtype = tensor.dtype

        # Verificar todas as especificações
        for spec_name, spec in self.specs.get_all_specs().items():
            expected_shape = spec['shape']
            expected_dtype = spec['dtype']

            # Verificar tipo de dados
            if dtype != expected_dtype:
                continue

            # Verificar número de dimensões
            if len(shape) != len(expected_shape):
                continue

            # Verificar dimensões não-batch (índices 1+)
            match = True
            for i in range(1, len(expected_shape)):
                if shape[i] != expected_shape[i]:
                    match = False
                    break

            if match:
                return spec_name.upper()

        # Fallback para inferência baseada em heurísticas
        if len(shape) == 3 and shape[1] == 64 and shape[2] == 64:
            return "SPECTRAL_INPUT"
        elif len(shape) == 4 and shape[3] == 4:
            return "QUATERNION_STATES"
        elif len(shape) == 2 and shape[1] == 512:
            return "QUANTUM_EMBEDDINGS"
        elif len(shape) == 2 and shape[1] == 128 and dtype == torch.int64:
            return "LANGUAGE_OUTPUT"

        return "UNKNOWN"

    def _find_conversion_path(self, source_spec: str, target_spec: str) -> List[str]:
        """Encontra caminho de conversão no grafo"""
        if (source_spec, target_spec) in self.conversion_graph:
            return [f"{source_spec}_to_{target_spec}"]

        # Para caminhos mais complexos, implementar busca em grafo
        # Por enquanto, apenas conversões diretas
        return []

    def _apply_conversion(self, tensor: torch.Tensor, conversion_step: str) -> torch.Tensor:
        """Aplica uma conversão específica"""
        # Convert to lowercase for case-insensitive matching
        step_lower = conversion_step.lower()

        if 'quaternion' in step_lower and 'quantum_embeddings' in step_lower:
            if 'to_quantum_embeddings' in step_lower:
                return self._quaternion_to_embedding(tensor)
            else:
                return self._embedding_to_quaternion(tensor)
        elif 'spectral' in step_lower and 'quaternion' in step_lower:
            return self._spectral_to_quaternion(tensor)
        elif 'quantum_embeddings' in step_lower and 'language' in step_lower:
            if 'to_language' in step_lower:
                return self._embedding_to_language(tensor)
            else:
                return self._language_to_embedding(tensor)
        elif 'memory_buffer' in step_lower and 'quantum_embeddings' in step_lower:
            return self._memory_buffer_to_quantum_embeddings(tensor)
        else:
            raise ValueError(f"Conversão não suportada: {conversion_step}")

    def _build_conversion_graph(self) -> Dict[Tuple[str, str], str]:
        """Constrói grafo de conversões possíveis"""
        return {
            ('QUATERNION_STATES', 'QUANTUM_EMBEDDINGS'): 'quaternion_to_quantum_embeddings',
            ('QUANTUM_EMBEDDINGS', 'QUATERNION_STATES'): 'quantum_embeddings_to_quaternion',
            ('SPECTRAL_INPUT', 'QUATERNION_STATES'): 'spectral_to_quaternion_states',
            ('QUANTUM_EMBEDDINGS', 'LANGUAGE_OUTPUT'): 'quantum_embeddings_to_language_output',
            ('LANGUAGE_OUTPUT', 'QUANTUM_EMBEDDINGS'): 'language_to_quantum_embeddings',
            ('MEMORY_BUFFER', 'QUANTUM_EMBEDDINGS'): 'memory_buffer_to_quantum_embeddings',
        }

    def _quaternion_to_embedding(self, tensor: torch.Tensor) -> torch.Tensor:
        """Converte [1, 64, 64, 4] → [1, 512] preservando informação quântica"""
        if tensor.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {tensor.dim()}D")

        # 1. Aplicar transformada de Fourier quaterniônica
        quat_fft = self._quaternion_fft(tensor)

        # 2. Extrair magnitudes e fases
        magnitudes = torch.norm(quat_fft, dim=-1)  # [1, 64, 64]
        phases = self._extract_quaternion_phases(quat_fft)  # [1, 64, 64, 3]

        # 3. Combinar em embedding unificado
        mag_flat = magnitudes.view(1, -1)  # [1, 4096]
        phase_flat = phases.view(1, -1)    # [1, 12288]

        # 4. Projeção para dimensão alvo
        combined = torch.cat([mag_flat, phase_flat], dim=-1)  # [1, 16384]
        embedding = torch.nn.functional.linear(combined, self.projection_matrices['quat_to_embed'])

        return embedding  # [1, 512]

    def _embedding_to_quaternion(self, tensor: torch.Tensor) -> torch.Tensor:
        """Converte [1, 512] → [1, 64, 64, 4] (aproximação inversa)"""
        # Projeção para espaço expandido
        expanded = torch.nn.functional.linear(tensor, self.projection_matrices['embed_to_quat'])

        # Separar magnitudes e fases
        mag_size = 64 * 64
        magnitudes = expanded[:, :mag_size].view(1, 64, 64)
        phases = expanded[:, mag_size:].view(1, 64, 64, 3)

        # Reconstruir quatérnions
        quaternion = self._reconstruct_quaternions(magnitudes, phases)

        return quaternion  # [1, 64, 64, 4]

    def _spectral_to_quaternion(self, tensor: torch.Tensor) -> torch.Tensor:
        """Converte [1, 64, 64] → [1, 64, 64, 4]"""
        # Expandir espectro para domínio quaterniônico
        expanded = torch.nn.functional.linear(
            tensor.view(1, -1),
            self.projection_matrices['spec_to_quat']
        ).view(1, 64, 64, 4)

        # Normalizar para unitariedade aproximada
        norms = torch.norm(expanded, dim=-1, keepdim=True)
        normalized = expanded / (norms + 1e-10)

        return normalized

    def _embedding_to_language(self, tensor: torch.Tensor) -> torch.Tensor:
        """Converte [1, 512] → [1, 128] (tokens)"""
        # Projeção para espaço de tokens
        logits = torch.nn.functional.linear(tensor, self.projection_matrices['embed_to_lang'])

        # Amostragem greedy (para determinismo)
        tokens = torch.argmax(logits, dim=-1)

        return tokens  # [1, 128]

    def _language_to_embedding(self, tensor: torch.Tensor) -> torch.Tensor:
        """Converte [1, 128] → [1, 512]"""
        # Embedding simples baseada em tokens
        embedding = torch.nn.functional.linear(
            tensor.float(),
            self.projection_matrices['lang_to_embed']
        )

        return embedding  # [1, 512]

    def _memory_buffer_to_quantum_embeddings(self, tensor: torch.Tensor) -> torch.Tensor:
        """Converte [10, 512] → [1, 512] (buffer de memória para embedding atual)"""
        # O buffer de memória tem shape [memory_slots, embedding_dim]
        # Pegamos apenas o embedding mais recente (último slot)
        if tensor.dim() == 2 and tensor.shape[0] > 1:
            # Se temos múltiplos slots, pegamos o mais recente
            current_embedding = tensor[-1:, :]  # [1, 512]
        else:
            # Se já é [1, 512], retorna como está
            current_embedding = tensor.unsqueeze(0) if tensor.dim() == 1 else tensor

        return current_embedding  # [1, 512]

    def _quaternion_fft(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transformada de Fourier quaterniônica simplificada"""
        # FFT nas dimensões espaciais
        fft_result = torch.fft.fftn(tensor, dim=(1, 2))
        return fft_result

    def _extract_quaternion_phases(self, quat_tensor: torch.Tensor) -> torch.Tensor:
        """Extrai fases dos componentes quaterniônicos"""
        # Para simplificar, usar ângulos dos primeiros 3 componentes
        phases = torch.angle(quat_tensor[..., :3])  # [..., 3]
        return phases

    def _reconstruct_quaternions(self, magnitudes: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        """Reconstrói quatérnions a partir de magnitudes e fases"""
        # Componente real (w)
        w = magnitudes * torch.cos(phases[..., 0])

        # Componentes imaginários (x, y, z)
        x = magnitudes * torch.sin(phases[..., 0]) * torch.cos(phases[..., 1])
        y = magnitudes * torch.sin(phases[..., 0]) * torch.sin(phases[..., 1]) * torch.cos(phases[..., 2])
        z = magnitudes * torch.sin(phases[..., 0]) * torch.sin(phases[..., 1]) * torch.sin(phases[..., 2])

        return torch.stack([w, x, y, z], dim=-1)

    def _validate_output(self, tensor: torch.Tensor, target_spec: str) -> torch.Tensor:
        """Valida tensor de saída"""
        if not QRHTensorSpec.validate_tensor(tensor, target_spec):
            print(f"⚠️  Tensor de saída não corresponde à especificação {target_spec}")
            print(f"   Forma esperada: {QRHTensorSpec.get_spec(target_spec)['shape']}")
            print(f"   Forma obtida: {list(tensor.shape)}")

        return tensor


class QRHComponentInterface:
    """
    Interface padronizada para todos os componentes ΨQRH

    Garante que todos os componentes sigam o mesmo padrão de
    inicialização, processamento e tratamento de erros.
    """

    def __init__(self, component_name: str, input_spec: str, output_spec: str):
        self.component_name = component_name
        self.input_spec = input_spec
        self.output_spec = output_spec
        self.adapter = UniversalTensorAdapter()
        self.initialized = False
        self.device = torch.device('cpu')

    def initialize(self):
        """Inicialização padronizada com validação"""
        try:
            self._setup_component()
            self.initialized = True
            print(f"✅ {self.component_name} inicializado")
        except Exception as e:
            print(f"❌ {self.component_name} falhou: {e}")
            raise

    def process(self, input_tensor: torch.Tensor) -> Any:
        """Processamento padronizado com adaptação automática"""
        if not self.initialized:
            raise RuntimeError(f"Componente {self.component_name} não inicializado")

        # 1. Adaptar entrada para especificação esperada
        adapted_input = self.adapter.convert(input_tensor, self.input_spec)

        # 2. Executar processamento específico do componente
        try:
            raw_output = self._internal_process(adapted_input)
        except Exception as e:
            raise RuntimeError(f"Processamento em {self.component_name} falhou: {e}")

        # 3. Adaptar saída para especificação padrão (se for tensor)
        if isinstance(raw_output, torch.Tensor):
            adapted_output = self.adapter.convert(raw_output, self.output_spec)
            return adapted_output
        else:
            # Para saídas não-tensor (como strings do gerador de linguagem), retornar diretamente
            return raw_output

    def _setup_component(self):
        """Setup específico do componente - deve ser implementado pelas subclasses"""
        raise NotImplementedError("Subclasses devem implementar _setup_component")

    def _internal_process(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Processamento interno específico - deve ser implementado pelas subclasses"""
        raise NotImplementedError("Subclasses devem implementar _internal_process")

    def to_device(self, device: torch.device):
        """Move componente para dispositivo específico"""
        self.device = device
        # Mover matrizes do adaptador para o dispositivo
        for key, matrix in self.adapter.projection_matrices.items():
            self.adapter.projection_matrices[key] = matrix.to(device)


class TensorValidation:
    """Validação de tensores com restrições físicas"""

    @staticmethod
    def validate_unitarity(tensor: torch.Tensor, tolerance: float = 1e-6) -> bool:
        """Garante preservação de unitariedade quaterniônica"""
        if tensor.shape[-1] == 4:  # Tensor quaterniônico
            norms = torch.norm(tensor, dim=-1)
            return torch.allclose(norms, torch.ones_like(norms), atol=tolerance)
        return True

    @staticmethod
    def validate_energy_conservation(input_tensor: torch.Tensor,
                                    output_tensor: torch.Tensor,
                                    tolerance: float = 0.05) -> bool:
        """Verifica conservação de energia espectral"""
        input_energy = torch.sum(input_tensor.abs() ** 2)
        output_energy = torch.sum(output_tensor.abs() ** 2)
        ratio = output_energy / (input_energy + 1e-10)
        return abs(ratio - 1.0) <= tolerance

    @staticmethod
    def validate_physical_constraints(tensor: torch.Tensor, spec_name: str) -> Dict[str, bool]:
        """Valida todas as restrições físicas para um tensor"""
        results = {}

        # Unitaridade (para quatérnions)
        if 'quaternion' in spec_name.lower():
            results['unitarity'] = TensorValidation.validate_unitarity(tensor)

        # Valores finitos
        results['finite_values'] = torch.isfinite(tensor).all().item()

        # Range físico razoável
        results['reasonable_range'] = torch.abs(tensor).max().item() < 1e10

        return results


# Função de compatibilidade
def create_unified_tensor_system() -> Tuple[QRHTensorSpec, UniversalTensorAdapter]:
    """
    Factory function para criar sistema unificado de tensores

    Returns:
        Tupla com especificações e adaptador universal
    """
    specs = QRHTensorSpec()
    adapter = UniversalTensorAdapter()

    return specs, adapter


if __name__ == "__main__":
    # Teste do sistema unificado
    print("🧮 Testando Sistema Unificado de Tensores ΨQRH...")

    # Criar sistema
    specs, adapter = create_unified_tensor_system()

    # Teste de conversões
    print("\n🔄 Testando conversões de tensor:")

    # Tensor quaterniônico de teste
    quat_tensor = torch.randn(1, 64, 64, 4)
    print(f"Tensor quaterniônico: {quat_tensor.shape}")

    # Converter para embedding
    embed_tensor = adapter.convert(quat_tensor, "QUANTUM_EMBEDDINGS")
    print(f"Embedding quântico: {embed_tensor.shape}")

    # Converter de volta
    quat_back = adapter.convert(embed_tensor, "QUATERNION_STATES")
    print(f"Quaterniônico reconstruído: {quat_back.shape}")

    # Validação
    print("\n✅ Validações físicas:")
    unitarity_ok = TensorValidation.validate_unitarity(quat_tensor)
    energy_ok = TensorValidation.validate_energy_conservation(quat_tensor, quat_back)

    print(f"   Unitaridade original: {unitarity_ok}")
    print(f"   Conservação de energia: {energy_ok}")

    print("\n🎯 Sistema unificado de tensores inicializado com sucesso!")
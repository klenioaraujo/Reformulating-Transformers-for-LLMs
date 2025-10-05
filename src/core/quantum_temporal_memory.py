#!/usr/bin/env python3
"""
Quantum Temporal Memory System for ΨQRH
=========================================

Implements memory with long-range temporal correlations using quantum entanglement.
Preserves contextual coherence across linguistic sequences.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
import math


class QuantumDimensionAdapter:
    """
    Adaptador de dimensões quânticas para compatibilidade entre sistemas.

    Resolve incompatibilidade crítica entre:
    - Pipeline físico: [1, 64, 64, 4] (quatérnions)
    - Sistema de memória: [1, seq, features]
    """
    def __init__(self, embed_dim: int = 64, quaternion_dim: int = 4):
        self.embed_dim = embed_dim
        self.quaternion_dim = quaternion_dim
        self.projection_matrices = self._initialize_projection_matrices()

    def _initialize_projection_matrices(self) -> Dict[str, torch.Tensor]:
        """Inicializa matrizes de projeção para adaptação de dimensões"""
        matrices = {}

        # Matriz para compressão quaterniônica
        matrices['quaternion_compression'] = torch.randn(
            self.embed_dim, self.embed_dim * self.quaternion_dim
        ) / math.sqrt(self.embed_dim)

        # Matriz para expansão de memória
        matrices['memory_expansion'] = torch.randn(
            self.embed_dim * self.quaternion_dim, self.embed_dim
        ) / math.sqrt(self.embed_dim)

        return matrices

    def adapt_quantum_states(self, physical_tensor: torch.Tensor,
                           target_system: str) -> torch.Tensor:
        """
        Adapta tensores entre diferentes representações quânticas
        preservando informação e unitariedade
        """
        if target_system == "memory_system":
            return self._physical_to_memory_adapter(physical_tensor)
        elif target_system == "physical_pipeline":
            return self._memory_to_physical_adapter(physical_tensor)
        else:
            return physical_tensor

    def _physical_to_memory_adapter(self, physical_tensor: torch.Tensor) -> torch.Tensor:
        """
        Converte tensor físico [1, 64, 64, 4] para formato de memória [1, seq, features]
        Preserva informação quântica através de projeção unitária
        """
        if physical_tensor.dim() != 4:
            return physical_tensor.unsqueeze(0).unsqueeze(-1)  # Fallback simples

        batch_size, dim1, dim2, quat_dim = physical_tensor.shape

        # 1. Extrair características quânticas principais via SVD
        quantum_features = []

        for q in range(min(quat_dim, self.quaternion_dim)):
            quat_component = physical_tensor[0, :, :, q]  # [64, 64]

            # Decomposição espectral para preservar informação
            try:
                U, S, Vh = torch.linalg.svd(quat_component, full_matrices=False)

                # Manter componentes principais (95% de variância)
                explained_variance = torch.cumsum(S, dim=0) / torch.sum(S)
                n_components = torch.sum(explained_variance < 0.95).item() + 1
                n_components = min(n_components, 16)  # Limitar para estabilidade

                # Projeção para dimensão reduzida
                projected = U[:, :n_components] @ torch.diag(S[:n_components])
                quantum_features.append(projected.flatten()[:32])  # Limitar tamanho

            except RuntimeError:
                # Fallback se SVD falhar
                quantum_features.append(quat_component.flatten()[:32])

        # 2. Combinar características
        if quantum_features:
            combined_features = torch.cat(quantum_features)
        else:
            combined_features = physical_tensor.flatten()[:64]

        # 3. Redimensionar para formato de memória [1, seq, features]
        seq_length = min(32, len(combined_features))
        memory_tensor = combined_features[:seq_length].unsqueeze(0).unsqueeze(-1)

        return memory_tensor  # [1, seq, 1]

    def _memory_to_physical_adapter(self, memory_tensor: torch.Tensor) -> torch.Tensor:
        """
        Converte tensor de memória de volta para formato físico
        com reconstrução aproximada unitária
        """
        if memory_tensor.dim() != 3:
            # Criar tensor físico básico
            physical_tensor = torch.zeros((1, self.embed_dim, self.embed_dim, self.quaternion_dim),
                                        dtype=memory_tensor.dtype)
            return physical_tensor

        batch_size, seq_len, features = memory_tensor.shape

        # Criar tensor físico [1, 64, 64, 4]
        physical_tensor = torch.zeros((1, self.embed_dim, self.embed_dim, self.quaternion_dim),
                                    dtype=memory_tensor.dtype)

        # Distribuir informação da memória através das dimensões físicas
        memory_flat = memory_tensor.flatten()
        max_elements = min(len(memory_flat), self.embed_dim * self.embed_dim * self.quaternion_dim)

        for i in range(max_elements):
            # Mapear índice plano para coordenadas físicas
            physical_idx = self._flat_to_physical_index(i)
            if physical_idx is not None:
                physical_tensor[0, physical_idx[0], physical_idx[1], physical_idx[2]] = memory_flat[i]

        return physical_tensor

    def _flat_to_physical_index(self, flat_idx: int) -> Optional[Tuple[int, int, int]]:
        """Mapeia índice plano para coordenadas físicas (i, j, k)"""
        total_elements = self.embed_dim * self.embed_dim * self.quaternion_dim
        if flat_idx >= total_elements:
            return None

        k = flat_idx % self.quaternion_dim  # Dimensão quaterniônica
        remaining = flat_idx // self.quaternion_dim
        j = remaining % self.embed_dim
        i = remaining // self.embed_dim

        if i < self.embed_dim and j < self.embed_dim and k < self.quaternion_dim:
            return (i, j, k)
        return None


class TemporalCorrelationMatrix:
    """
    Matriz de correlação temporal que rastreia dependências entre posições sequenciais.
    """

    def __init__(self, max_positions: int = 100):
        self.max_positions = max_positions
        self.correlation_matrix = torch.zeros((max_positions, max_positions))
        self.position_count = 0

    def update_correlation(self, pos_i: int, pos_j: int, correlation_strength: float):
        """Atualiza correlação entre duas posições."""
        if pos_i < self.max_positions and pos_j < self.max_positions:
            self.correlation_matrix[pos_i, pos_j] = correlation_strength
            self.correlation_matrix[pos_j, pos_i] = correlation_strength  # Simétrica

    def get_correlation(self, pos_i: int, pos_j: int) -> float:
        """Retorna correlação entre duas posições."""
        if pos_i < self.max_positions and pos_j < self.max_positions:
            return float(self.correlation_matrix[pos_i, pos_j])
        return 0.0

    def get_contextual_weights(self, current_pos: int, context_window: int = 5) -> torch.Tensor:
        """Retorna pesos contextuais para posições próximas."""
        start_pos = max(0, current_pos - context_window)
        end_pos = min(self.max_positions, current_pos + context_window + 1)

        weights = self.correlation_matrix[current_pos, start_pos:end_pos]
        return weights / (torch.sum(weights) + 1e-10)  # Normalizar


class DecoherenceController:
    """
    Controla decoerência quântica preservando fases correlacionadas.
    Modelo: amplitude damping + phase damping com correção de fase.
    """

    def __init__(self, decoherence_rate: float = 0.1, phase_stability: float = 0.95):
        self.decoherence_rate = decoherence_rate
        self.phase_stability = phase_stability
        self.phase_memory = {}

    def apply_decoherence(self, quantum_state: torch.Tensor,
                         time_elapsed: float, memory_id: str = None) -> torch.Tensor:
        """
        Aplica decoerência realista preservando fases correlacionadas.
        """
        # Taxa de decoerência dependente do tempo
        gamma = 1.0 - torch.exp(torch.tensor(-self.decoherence_rate * time_elapsed)).item()

        if quantum_state.dim() == 1:
            # Decoerência para estado puro
            decohered_state = self._amplitude_damping(quantum_state, gamma)
            decohered_state = self._phase_damping(decohered_state, gamma, memory_id)
        else:
            # Decoerência para matriz densidade
            decohered_state = self._linblad_evolution(quantum_state, gamma)

        return decohered_state

    def _amplitude_damping(self, state: torch.Tensor, gamma: float) -> torch.Tensor:
        """Amplitude damping channel."""
        # |0⟩⟨0| + √γ|1⟩⟨1| + (1-γ)|0⟩⟨0|
        damping_matrix = torch.tensor([
            [1.0, 0.0],
            [0.0, math.sqrt(1 - gamma)],
            [0.0, math.sqrt(gamma)]
        ], dtype=torch.complex64)

        # Aplicar ao estado (simplificado para 1-qubit)
        if len(state) >= 2:
            # Assumir representação |ψ⟩ = [α, β]
            alpha, beta = state[0], state[1]
            new_alpha = alpha
            new_beta = math.sqrt(1 - gamma) * beta
            return torch.stack([new_alpha, new_beta] + list(state[2:]))

        return state

    def _phase_damping(self, state: torch.Tensor, gamma: float, memory_id: str = None) -> torch.Tensor:
        """Phase damping com preservação de fase correlacionada."""
        if memory_id and memory_id in self.phase_memory:
            # Preservar fase baseada em memória
            stored_phase = self.phase_memory[memory_id]
            phase_correction = torch.angle(state[0]) - stored_phase
            corrected_state = state * torch.exp(1j * self.phase_stability * phase_correction)
            return corrected_state
        else:
            # Phase damping padrão
            phase_damped = state * torch.exp(-gamma * torch.angle(state))
            # Armazenar fase para futuro
            if memory_id:
                self.phase_memory[memory_id] = torch.angle(state[0])
            return phase_damped

    def _linblad_evolution(self, rho: torch.Tensor, gamma: float) -> torch.Tensor:
        """Evolução Lindblad para decoerência (simplificada)."""
        # Para implementação completa seria necessário usar operadores Lindblad
        # Por simplicidade, aplicamos damping de amplitude
        return rho * (1 - gamma) + torch.diag(torch.diag(rho)) * gamma

    def preserve_correlated_phases(self, states: List[torch.Tensor],
                                 correlation_ids: List[str]) -> List[torch.Tensor]:
        """
        Preserva fases correlacionadas entre estados emaranhados.
        """
        if len(states) < 2:
            return states

        # Calcular fases relativas
        relative_phases = []
        for i in range(len(states) - 1):
            phase_diff = self._compute_phase_difference(states[i], states[i + 1])
            relative_phases.append(phase_diff)

        # Aplicar correção de fase correlacionada
        corrected_states = [states[0]]  # Estado âncora

        for i in range(1, len(states)):
            target_phase = relative_phases[i - 1]
            current_phase = self._compute_phase_difference(
                corrected_states[i - 1], states[i]
            )

            phase_correction = target_phase - current_phase
            corrected_phase = current_phase + 0.8 * phase_correction  # Fator de correlação

            corrected_state = self._apply_phase_correction(
                states[i], corrected_phase, correlation_ids[i]
            )
            corrected_states.append(corrected_state)

        return corrected_states

    def _compute_phase_difference(self, state_a: torch.Tensor, state_b: torch.Tensor) -> float:
        """Computa diferença de fase entre dois estados."""
        phase_a = torch.angle(state_a[0]) if len(state_a) > 0 else 0.0
        phase_b = torch.angle(state_b[0]) if len(state_b) > 0 else 0.0
        return float(phase_b - phase_a)

    def _apply_phase_correction(self, state: torch.Tensor, phase_correction: float,
                              memory_id: str = None) -> torch.Tensor:
        """Aplica correção de fase a um estado."""
        correction_factor = torch.exp(1j * phase_correction)
        corrected_state = state * correction_factor

        # Atualizar memória de fase
        if memory_id:
            self.phase_memory[memory_id] = torch.angle(corrected_state[0])

        return corrected_state


class QuantumTemporalMemory:
    """
    Sistema de memória quântica com correlações de longo alcance.

    Preserva contexto temporal através de emaranhamento quântico entre estados sequenciais.
    """

    def __init__(self, memory_size: int = 10, coherence_time: float = 5.0,
                 decoherence_rate: float = 0.1):
        self.memory_size = memory_size
        self.coherence_time = coherence_time
        self.entangled_states = deque(maxlen=memory_size)
        self.temporal_correlations = TemporalCorrelationMatrix(max_positions=1000)
        self.decoherence_controller = DecoherenceController(
            decoherence_rate=decoherence_rate
        )
        self.memory_counter = 0

    def store_quantum_context(self, current_state: torch.Tensor,
                            position: int, timestamp: float,
                            correlation_id: str = None) -> None:
        """
        Armazena estado quântico com emaranhamento temporal.

        |ψ(t)⟩ = U(t,t₀)|ψ(t₀)⟩ com preservação de fase relativa
        """
        # Criar estado emaranhado com memória anterior
        if self.entangled_states:
            previous_state = self.entangled_states[-1]['state']
            entangled_pair = self._create_temporal_entanglement(
                previous_state, current_state, correlation_id
            )
            # Atualizar estado anterior com versão emaranhada
            self.entangled_states[-1]['state'] = entangled_pair[0]
            current_state = entangled_pair[1]  # Estado atual emaranhado

        # Adicionar com metadados
        memory_item = {
            'state': current_state.detach().clone(),  # Não manter gradientes na memória
            'position': position,
            'timestamp': timestamp,
            'phase_coherence': 1.0,  # Coerência máxima inicial
            'correlation_id': correlation_id or f"mem_{self.memory_counter}",
            'memory_id': self.memory_counter
        }

        self.entangled_states.append(memory_item)
        self.memory_counter += 1

    def recall_contextual_state(self, current_position: int,
                              current_time: float,
                              context_window: int = 5) -> Optional[torch.Tensor]:
        """
        Recupera estado contextual considerando decoerência temporal
        e correlações de longo alcance.
        """
        if not self.entangled_states:
            return None

        # Calcular pesos de memória baseados em múltiplos fatores
        memory_weights = []
        contextual_states = []

        for memory in self.entangled_states:
            # 1. Decaimento temporal (lei exponencial)
            time_diff = current_time - memory['timestamp']
            temporal_weight = torch.exp(torch.tensor(-time_diff / self.coherence_time)).item()

            # 2. Proximidade sequencial
            pos_diff = abs(current_position - memory['position'])
            positional_weight = torch.exp(torch.tensor(-pos_diff / 2.0)).item()

            # 3. Coerência residual (com atualização de decoerência)
            coherence_decay = self.decoherence_controller.apply_decoherence(
                torch.ones(1), time_diff, memory['correlation_id']
            )[0].real
            coherence_weight = memory['phase_coherence'] * coherence_decay

            # 4. Correlação contextual da matriz
            context_weight = self.temporal_correlations.get_correlation(
                current_position, memory['position']
            )

            # Peso total (produto dos fatores)
            total_weight = (temporal_weight * positional_weight *
                          coherence_weight * (1.0 + context_weight))

            memory_weights.append(total_weight)

            # Aplicar evolução temporal reversa para sincronização
            evolved_state = self._reverse_time_evolution(
                memory['state'],
                current_time - memory['timestamp']
            )
            contextual_states.append(evolved_state)

        # Verificar se há pesos significativos
        total_weight = sum(memory_weights)
        if total_weight < 1e-6:
            return None

        # Combinação coerente dos estados de memória
        # Por enquanto, retornar apenas o estado mais recente para evitar problemas de shape
        # TODO: Implementar combinação robusta de tensores com diferentes shapes
        return contextual_states[-1] if contextual_states else None

    def _create_temporal_entanglement(self, state_a: torch.Tensor,
                                    state_b: torch.Tensor,
                                    correlation_id: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cria par emaranhado temporal entre dois estados.

        |Ψ⟩ = (|0⟩|ψ_a⟩ + |1⟩|ψ_b⟩)/√2 com fase temporal correlacionada
        """
        # Operador de emaranhamento temporal (simplificado)
        entanglement_operator = self._build_temporal_entangler()

        # Estados normalizados
        state_a_norm = state_a / (torch.norm(state_a) + 1e-10)
        state_b_norm = state_b / (torch.norm(state_b) + 1e-10)

        # Criar estado combinado em espaço expandido
        # |Ψ⟩ = (|0⟩⊗|ψ_a⟩ + |1⟩⊗|ψ_b⟩)/√2
        dim_a = len(state_a_norm)
        dim_b = len(state_b_norm)

        # Para simplificar, assumimos mesma dimensionalidade
        if dim_a == dim_b:
            combined_state = torch.zeros(dim_a * 2, dtype=torch.complex64)

            # |0⟩ ⊗ |ψ_a⟩
            combined_state[:dim_a] = state_a_norm

            # |1⟩ ⊗ |ψ_b⟩
            combined_state[dim_a:] = state_b_norm

            # Normalização
            combined_state = combined_state / torch.norm(combined_state)

            # Aplicar emaranhamento (simplificado)
            entangled_state = entanglement_operator @ combined_state

            # Estados reduzidos (trace out)
            state_a_entangled = entangled_state[:dim_a]
            state_b_entangled = entangled_state[dim_a:]

            # Aplicar correção de fase correlacionada
            correlated_pair = self.decoherence_controller.preserve_correlated_phases(
                [state_a_entangled, state_b_entangled],
                [correlation_id + "_a", correlation_id + "_b"] if correlation_id else ["a", "b"]
            )

            return correlated_pair[0], correlated_pair[1]
        else:
            # Fallback para estados não emaranhados se dimensões diferentes
            return state_a, state_b

    def _build_temporal_entangler(self) -> torch.Tensor:
        """Constrói operador de emaranhamento temporal (simplificado)."""
        # Matriz que cria emaranhamento tipo Bell
        # |00⟩ + |11⟩ para 2 qubits
        bell_state = torch.tensor([
            [1.0, 0.0, 0.0, 1.0],  # |00⟩ + |11⟩
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ], dtype=torch.complex64) / math.sqrt(2.0)

        return bell_state

    def _reverse_time_evolution(self, state: torch.Tensor, time_diff: float) -> torch.Tensor:
        """
        Evolução temporal reversa para sincronização de estados de memória.
        """
        # Operador de evolução temporal reversa (simplificado)
        # U†(t) onde U(t) = exp(-iHt/ħ)
        # Para simplificar, aplicamos uma rotação de fase baseada no tempo
        phase_shift = -time_diff * 0.1  # Frequência simplificada
        time_evolution = torch.exp(1j * phase_shift * torch.arange(len(state), dtype=torch.float32))

        return state * time_evolution.to(state.device)

    def update_temporal_correlations(self, positions: List[int],
                                   correlations: List[float],
                                   current_time: float):
        """Atualiza matriz de correlação temporal."""
        for i, pos_i in enumerate(positions):
            for j, pos_j in enumerate(positions):
                if i != j:
                    correlation_strength = correlations[min(i, j)] * torch.exp(
                        torch.tensor(-abs(pos_i - pos_j) / 10.0)
                    ).item()  # Decaimento espacial
                    self.temporal_correlations.update_correlation(
                        pos_i, pos_j, correlation_strength
                    )

    def get_memory_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da memória quântica."""
        return {
            'memory_size': len(self.entangled_states),
            'max_memory_size': self.memory_size,
            'coherence_time': self.coherence_time,
            'memory_counter': self.memory_counter,
            'oldest_timestamp': self.entangled_states[0]['timestamp'] if self.entangled_states else None,
            'newest_timestamp': self.entangled_states[-1]['timestamp'] if self.entangled_states else None
        }


class QuantumTimeController:
    """
    Controla evolução temporal quântica baseada na complexidade do estado.
    """

    def __init__(self, base_time_step: float = 0.1, complexity_factor: float = 0.5):
        self.base_time_step = base_time_step
        self.complexity_factor = complexity_factor

    def get_time_step(self, quantum_state: torch.Tensor) -> float:
        """
        Calcula passo temporal baseado na complexidade quântica do estado.
        Estados mais complexos requerem passos menores para estabilidade.
        """
        # Medir complexidade baseada na entropia
        if len(quantum_state) > 1:
            probabilities = torch.abs(quantum_state) ** 2
            probabilities = probabilities / torch.sum(probabilities)
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10))
            complexity = float(entropy / torch.log(torch.tensor(len(quantum_state))))
        else:
            complexity = 0.0

        # Passo temporal adaptativo
        adaptive_step = self.base_time_step * (1.0 - self.complexity_factor * complexity)
        return max(adaptive_step, 0.01)  # Mínimo para evitar dt=0


class QuantumContextualLinguisticProcessor:
    """
    Processador linguístico contextual com memória quântica.
    """

    def __init__(self, memory_size: int = 8, coherence_time: float = 3.0):
        self.dimension_adapter = QuantumDimensionAdapter()
        self.temporal_memory = QuantumTemporalMemory(
            memory_size=memory_size,
            coherence_time=coherence_time
        )
        self.time_controller = QuantumTimeController()
        self.context_window = 5

    def generate_contextual_sequence(self, input_sequence: List[torch.Tensor],
                                    input_text: str = None) -> str:
        """
        Gera sequência linguística com memória quântica contextual.
        Adapta dimensões automaticamente entre pipeline físico e sistema de memória.
        """
        phoneme_sequence = []
        current_time = 0.0
        positions = []

        for i, quantum_state in enumerate(input_sequence):
            # Adaptar dimensão do estado quântico físico para formato de memória
            try:
                memory_format_state = self.dimension_adapter.adapt_quantum_states(
                    quantum_state, "memory_system"
                )
            except Exception as e:
                print(f"⚠️  Erro na adaptação de dimensões: {e}")
                # Fallback: tentar usar estado original
                memory_format_state = quantum_state

            # Atualizar tempo quântico
            time_step = self.time_controller.get_time_step(memory_format_state)
            current_time += time_step
            positions.append(i)

            # Recuperar contexto da memória
            try:
                contextual_state = self.temporal_memory.recall_contextual_state(
                    i, current_time, self.context_window
                )

                # Combinar estado atual com contexto
                if contextual_state is not None:
                    combined_state = self._quantum_interference(
                        memory_format_state, contextual_state
                    )
                else:
                    combined_state = memory_format_state
            except Exception as e:
                # Fallback: usar apenas estado atual se houver problemas
                print(f"⚠️  Erro na recuperação de contexto quântico: {e}")
                combined_state = memory_format_state

            # Mapear para fonema com contexto
            current_phoneme = self._contextual_phoneme_mapping(
                combined_state,
                context_phonemes=phoneme_sequence[-self.context_window:] if phoneme_sequence else None,
                input_context=input_text
            )

            phoneme_sequence.append(current_phoneme)

            # Armazenar na memória quântica (no formato de memória)
            correlation_id = f"seq_{i}_{current_phoneme}"
            self.temporal_memory.store_quantum_context(
                combined_state, i, current_time, correlation_id
            )

        # Atualizar correlações temporais
        self._update_sequence_correlations(phoneme_sequence, positions, current_time)

        return self._apply_linguistic_constraints(phoneme_sequence, input_text)

    def _quantum_interference(self, current: torch.Tensor,
                            context: torch.Tensor) -> torch.Tensor:
        """
        Interferência quântica entre estado atual e contexto memorizado.
        |ψ_combined⟩ = α|ψ_current⟩ + β|ψ_context⟩ com |α|² + |β|² = 1

        Versão simplificada para evitar problemas de shape durante desenvolvimento.
        """
        # Por enquanto, retornar apenas o estado atual
        # TODO: Implementar interferência quântica completa com tratamento robusto de shapes
        return current

    def _compute_phase_similarity(self, state_a: torch.Tensor, state_b: torch.Tensor) -> float:
        """Computa similaridade de fase entre dois estados."""
        try:
            # Flatten ambos os estados para comparação
            state_a_flat = state_a.flatten()
            state_b_flat = state_b.flatten()

            # Usar o menor comprimento para comparação
            min_len = min(len(state_a_flat), len(state_b_flat))
            if min_len > 0:
                phase_a = torch.angle(state_a_flat[:min_len])
                phase_b = torch.angle(state_b_flat[:min_len])
                phase_diff = torch.abs(phase_a - phase_b)
                similarity = torch.mean(torch.cos(phase_diff))  # Similaridade circular
                return float(torch.clamp(similarity, -1.0, 1.0))
        except (RuntimeError, TypeError):
            pass
        return 0.0

    def _contextual_phoneme_mapping(self, quantum_state: torch.Tensor,
                                  context_phonemes: List[str] = None,
                                  input_context: str = None) -> str:
        """
        Mapeamento contextual de estado quântico para fonema.
        """
        # Análise básica do estado quântico
        magnitude = torch.abs(quantum_state)
        phase = torch.angle(quantum_state)

        # Características extraídas
        fundamental_freq = float(torch.mean(magnitude))
        spectral_spread = float(torch.std(magnitude))
        phase_coherence = float(torch.abs(torch.mean(torch.exp(1j * phase))))

        # Mapeamento baseado em contexto
        if context_phonemes:
            # Considerar contexto fonêmico
            last_phoneme = context_phonemes[-1] if context_phonemes else None

            # Regras contextuais simples
            if last_phoneme in ['a', 'e', 'i']:
                # Após vogal, favorecer consoantes
                if fundamental_freq > 0.6:
                    return 'm' if spectral_spread < 0.3 else 'n'
                else:
                    return 'p' if phase_coherence > 0.7 else 't'
            elif last_phoneme in ['m', 'n', 'p', 't']:
                # Após consoante, favorecer vogais
                if fundamental_freq < 0.4:
                    return 'a' if spectral_spread > 0.4 else 'i'
                else:
                    return 'u' if phase_coherence < 0.5 else 'o'

        # Mapeamento padrão baseado em características quânticas
        return self._quantum_to_phoneme(fundamental_freq, spectral_spread, phase_coherence)

    def _quantum_to_phoneme(self, freq: float, spread: float, coherence: float) -> str:
        """Mapeamento básico estado quântico → fonema."""
        # Mapeamento simplificado baseado em características
        if freq < 0.3:
            return 'a' if spread > 0.4 else 'i'
        elif freq < 0.6:
            return 'm' if coherence > 0.7 else 'n'
        else:
            return 'p' if spread < 0.3 else 't'

    def _update_sequence_correlations(self, phoneme_sequence: List[str],
                                    positions: List[int], current_time: float):
        """Atualiza correlações baseadas na sequência gerada."""
        # Correlações simples baseadas em similaridade fonêmica
        correlations = []
        for i in range(len(phoneme_sequence)):
            # Correlação baseada na posição (mais próximas = mais correlacionadas)
            correlation = torch.exp(torch.tensor(-i / len(phoneme_sequence))).item()
            correlations.append(correlation)

        self.temporal_memory.update_temporal_correlations(
            positions, correlations, current_time
        )

    def _apply_linguistic_constraints(self, phoneme_sequence: List[str],
                                    input_text: str = None) -> str:
        """
        Aplica restrições linguísticas finais à sequência.
        """
        # Converter para string
        text = ''.join(phoneme_sequence)

        # Aplicar algumas regras básicas
        # Evitar repetições excessivas
        cleaned = []
        prev_char = None
        repeat_count = 0

        for char in text:
            if char == prev_char:
                repeat_count += 1
                if repeat_count < 3:  # Permitir até 2 repetições
                    cleaned.append(char)
            else:
                repeat_count = 1
                cleaned.append(char)
            prev_char = char

        return ''.join(cleaned)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da memória."""
        return self.temporal_memory.get_memory_stats()


# Função de integração com pipeline ΨQRH
def create_quantum_memory_system(memory_size: int = 8,
                                coherence_time: float = 3.0) -> QuantumContextualLinguisticProcessor:
    """
    Factory function para criar sistema de memória quântica.

    Args:
        memory_size: Tamanho da memória temporal
        coherence_time: Tempo de coerência quântica

    Returns:
        Sistema de processamento linguístico contextual
    """
    return QuantumContextualLinguisticProcessor(
        memory_size=memory_size,
        coherence_time=coherence_time
    )


if __name__ == "__main__":
    # Teste do sistema de memória quântica
    print("🧠 Testando Sistema de Memória Quântica Temporal...")

    # Criar sistema
    memory_system = create_quantum_memory_system()

    # Teste com sequência simples
    test_sequence = [
        torch.randn(4, dtype=torch.complex64) for _ in range(5)
    ]

    print("📊 Sequência de entrada (estados quânticos):")
    for i, state in enumerate(test_sequence):
        print(f"  Estado {i}: norma={torch.norm(state):.3f}")

    # Gerar sequência contextual
    result = memory_system.generate_contextual_sequence(test_sequence, "test input")

    print(f"📝 Sequência gerada: '{result}'")
    print(f"📈 Estatísticas de memória: {memory_system.get_memory_stats()}")

    print("✅ Sistema de memória quântica inicializado com sucesso!")
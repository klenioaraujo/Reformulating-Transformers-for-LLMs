import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from ΨQRHSystem.configs.SystemConfig import SystemConfig


class QuantumMemory:
    """
    Quantum Memory - Memória temporal quântica para coerência contextual

    Implementa processamento contextual, emaranhamento temporal,
    e manutenção de estados de consciência através do tempo.
    """

    def __init__(self, config: SystemConfig):
        """
        Inicializa Quantum Memory

        Args:
            config: Configuração do sistema
        """
        self.config = config
        self.device = torch.device(config.device if config.device != "auto" else
                                 ("cuda" if torch.cuda.is_available() else
                                  "mps" if torch.backends.mps.is_available() else "cpu"))

        # Parâmetros de memória
        self.embed_dim = config.model.embed_dim
        self.max_history = config.model.max_history
        self.vocab_size = config.model.vocab_size

        # Memória temporal
        self.temporal_memory = deque(maxlen=self.max_history)
        self.context_states = deque(maxlen=self.max_history)

        # Estado de consciência atual
        self.current_consciousness_state = {
            'fci': 0.5,
            'state': 'ANALYSIS',
            'coherence': 0.8,
            'temporal_depth': 0
        }

        print(f"🧠 Quantum Memory inicializada com profundidade temporal: {self.max_history}")

    def process_consciousness(self, quantum_state: Any) -> Dict[str, Any]:
        """
        Processa estado quântico através da memória de consciência

        Args:
            quantum_state: Estado quântico atual

        Returns:
            Resultados do processamento de consciência
        """
        # Calcular FCI (Fractal Consciousness Index) baseado no estado quântico
        fci = self._calculate_fci(quantum_state)

        # Determinar estado de consciência
        consciousness_state = self._determine_consciousness_state(fci)

        # Atualizar memória temporal
        self._update_temporal_memory(quantum_state, fci)

        # Calcular coerência temporal
        temporal_coherence = self._calculate_temporal_coherence()

        # Atualizar estado atual
        self.current_consciousness_state.update({
            'fci': fci,
            'state': consciousness_state,
            'coherence': temporal_coherence,
            'temporal_depth': len(self.temporal_memory)
        })

        return {
            'fci': fci,
            'state': consciousness_state,
            'coherence': temporal_coherence,
            'temporal_depth': len(self.temporal_memory),
            'consciousness_metrics': self._get_consciousness_metrics()
        }

    def _calculate_fci(self, quantum_state: Any) -> float:
        """
        Calcula Fractal Consciousness Index (FCI)

        Args:
            quantum_state: Estado quântico

        Returns:
            FCI entre 0.0 e 1.0
        """
        if isinstance(quantum_state, torch.Tensor):
            # Calcular complexidade fractal baseada na estrutura do tensor
            if quantum_state.numel() > 0:
                # Usar variância e entropia como proxies para complexidade fractal
                variance = torch.var(quantum_state.flatten()).item()
                entropy = self._calculate_entropy(quantum_state)

                # FCI = combinação de variância e entropia normalizada
                fci = min(1.0, (variance * 10 + entropy) / 2.0)
                return max(0.0, fci)
            else:
                return 0.3  # Estado vazio
        else:
            # Para estados não-tensor, usar valor padrão
            return 0.5

    def _calculate_entropy(self, tensor: torch.Tensor) -> float:
        """
        Calcula entropia de Shannon do tensor

        Args:
            tensor: Tensor de entrada

        Returns:
            Entropia normalizada
        """
        # Flatten tensor e calcular histograma
        flat = tensor.flatten()
        hist = torch.histc(flat, bins=50, min=flat.min().item(), max=flat.max().item())

        # Normalizar para distribuição de probabilidade
        hist = hist / (hist.sum() + 1e-10)

        # Calcular entropia
        entropy = -torch.sum(hist * torch.log(hist + 1e-10))
        entropy = entropy / torch.log(torch.tensor(50.0))  # Normalizar

        return entropy.item()

    def _determine_consciousness_state(self, fci: float) -> str:
        """
        Determina estado de consciência baseado no FCI

        Args:
            fci: Fractal Consciousness Index

        Returns:
            Nome do estado de consciência
        """
        if fci > 0.8:
            return 'ENLIGHTENMENT'
        elif fci > 0.6:
            return 'MEDITATION'
        elif fci > 0.4:
            return 'ANALYSIS'
        elif fci > 0.2:
            return 'AWARENESS'
        else:
            return 'COMA'

    def _update_temporal_memory(self, quantum_state: Any, fci: float):
        """
        Atualiza memória temporal com novo estado

        Args:
            quantum_state: Estado quântico atual
            fci: FCI atual
        """
        # Criar entrada de memória temporal
        memory_entry = {
            'quantum_state': quantum_state,
            'fci': fci,
            'timestamp': torch.tensor(0.0),  # Placeholder
            'coherence': self.current_consciousness_state.get('coherence', 0.8)
        }

        # Adicionar à memória
        self.temporal_memory.append(memory_entry)

        # Manter apenas os estados mais recentes
        while len(self.temporal_memory) > self.max_history:
            self.temporal_memory.popleft()

    def _calculate_temporal_coherence(self) -> float:
        """
        Calcula coerência temporal baseada na memória

        Returns:
            Coerência temporal entre 0.0 e 1.0
        """
        if len(self.temporal_memory) < 2:
            return 1.0  # Máxima coerência com pouca história

        # Calcular coerência baseada na consistência dos FCI
        fci_values = [entry['fci'] for entry in self.temporal_memory]

        if len(fci_values) > 1:
            # Coerência = 1 - variância normalizada dos FCI
            fci_tensor = torch.tensor(fci_values)
            variance = torch.var(fci_tensor).item()
            coherence = max(0.0, 1.0 - variance * 2.0)  # Normalizar variância
            return coherence
        else:
            return 1.0

    def _get_consciousness_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas detalhadas de consciência

        Returns:
            Métricas de consciência
        """
        if len(self.temporal_memory) == 0:
            return {
                'mean_fci': 0.5,
                'fci_variance': 0.0,
                'temporal_stability': 1.0,
                'consciousness_depth': 0.0
            }

        fci_values = [entry['fci'] for entry in self.temporal_memory]
        fci_tensor = torch.tensor(fci_values)

        return {
            'mean_fci': torch.mean(fci_tensor).item(),
            'fci_variance': torch.var(fci_tensor, unbiased=False).item(),
            'temporal_stability': self._calculate_temporal_coherence(),
            'consciousness_depth': len(self.temporal_memory) / self.max_history
        }

    def get_contextual_memory(self, current_state: Any) -> Dict[str, Any]:
        """
        Recupera memória contextual relevante para o estado atual

        Args:
            current_state: Estado atual

        Returns:
            Memória contextual
        """
        if len(self.temporal_memory) == 0:
            return {'context_states': [], 'relevance_scores': []}

        # Calcular relevância de cada estado de memória
        relevance_scores = []
        context_states = []

        for entry in self.temporal_memory:
            # Calcular similaridade com estado atual (simplificada)
            if isinstance(current_state, torch.Tensor) and isinstance(entry['quantum_state'], torch.Tensor):
                try:
                    similarity = torch.cosine_similarity(
                        current_state.flatten(),
                        entry['quantum_state'].flatten(),
                        dim=0
                    ).item()
                except:
                    similarity = 0.5  # Fallback
            else:
                similarity = 0.5  # Fallback para tipos diferentes

            relevance_scores.append(similarity)
            context_states.append(entry)

        # Ordenar por relevância
        sorted_indices = torch.argsort(torch.tensor(relevance_scores), descending=True)
        sorted_states = [context_states[i] for i in sorted_indices.tolist()]
        sorted_scores = [relevance_scores[i] for i in sorted_indices.tolist()]

        return {
            'context_states': sorted_states[:5],  # Top 5 estados mais relevantes
            'relevance_scores': sorted_scores[:5]
        }

    def reset_memory(self):
        """Reseta memória temporal"""
        self.temporal_memory.clear()
        self.context_states.clear()
        self.current_consciousness_state = {
            'fci': 0.5,
            'state': 'ANALYSIS',
            'coherence': 0.8,
            'temporal_depth': 0
        }
        print("🧠 Quantum Memory resetada")

    def get_memory_status(self) -> Dict[str, Any]:
        """
        Retorna status atual da memória quântica

        Returns:
            Status da memória
        """
        return {
            'temporal_depth': len(self.temporal_memory),
            'max_history': self.max_history,
            'current_consciousness': self.current_consciousness_state,
            'memory_utilization': len(self.temporal_memory) / self.max_history,
            'device': str(self.device)
        }
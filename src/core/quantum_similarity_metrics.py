"""
Quantum Similarity Metrics - OPÇÃO 4 do Sistema de Calibração ΨQRH
===================================================================

Explora diferentes funções de similaridade além do cosine similarity:
- Cosine Similarity (baseline)
- Euclidean Distance
- Quantum Fidelity
- Hilbert-Schmidt Distance
- Bures Distance

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import math
from typing import Dict, List, Any, Optional, Tuple


class QuantumSimilarityMetrics:
    """
    OPÇÃO 4: Explorar diferentes funções de similaridade quântica

    Seleciona automaticamente a métrica mais apropriada baseada na
    estrutura do estado quântico atual.
    """

    def __init__(self):
        self.metrics = {
            'cosine': self._cosine_similarity,
            'euclidean': self._euclidean_distance,
            'quantum_fidelity': self._quantum_fidelity,
            'hilbert_schmidt': self._hilbert_schmidt_distance,
            'bures_distance': self._bures_distance
        }

        # Cache para evitar recálculo
        self.metric_cache = {}

    def select_optimal_metric(self, psi_state: torch.Tensor) -> str:
        """
        Seleciona automaticamente a métrica mais apropriada baseada na estrutura quântica.

        Args:
            psi_state: Estado quântico [embed_dim, 4] ou [batch, seq_len, embed_dim, 4]

        Returns:
            Nome da métrica selecionada
        """
        # Análise da estrutura do estado
        coherence = self._measure_coherence(psi_state)
        entanglement = self._measure_entanglement(psi_state)
        complexity = self._measure_complexity(psi_state)

        print(f"    🔬 [QuantumSimilarityMetrics] Analisando estrutura quântica:")
        print(f"      - Coerência: {coherence:.3f}")
        print(f"      - Emaranhamento: {entanglement:.3f}")
        print(f"      - Complexidade: {complexity:.3f}")

        # Lógica de seleção baseada na estrutura
        if coherence > 0.8:
            # Estados altamente coerentes → fidelidade quântica
            selected = 'quantum_fidelity'
            reason = "Estado altamente coerente - melhor usar fidelidade quântica"
        elif entanglement > 0.6:
            # Estados altamente emaranhados → distância Hilbert-Schmidt
            selected = 'hilbert_schmidt'
            reason = "Estado altamente emaranhado - melhor usar Hilbert-Schmidt"
        elif complexity > 0.7:
            # Estados complexos → distância euclidiana
            selected = 'euclidean'
            reason = "Estado complexo - melhor usar distância euclidiana"
        else:
            # Estados simples → cosine similarity (baseline)
            selected = 'cosine'
            reason = "Estado simples - usando cosine similarity como baseline"

        print(f"    🎯 [QuantumSimilarityMetrics] Métrica selecionada: {selected}")
        print(f"      - Justificativa: {reason}")

        return selected

    def compute_similarity(self, psi: torch.Tensor, char_pattern: torch.Tensor,
                          metric: Optional[str] = None) -> float:
        """
        Computa similaridade usando a métrica especificada ou a ótima.

        Args:
            psi: Estado quântico [embed_dim, 4]
            char_pattern: Padrão do caractere [embed_dim, 4]
            metric: Métrica específica ou None para automática

        Returns:
            Score de similaridade [0, 1]
        """
        if metric is None:
            metric = self.select_optimal_metric(psi)

        if metric not in self.metrics:
            print(f"    ⚠️  [QuantumSimilarityMetrics] Métrica '{metric}' não encontrada, usando cosine")
            metric = 'cosine'

        # Computa similaridade
        similarity = self.metrics[metric](psi, char_pattern)

        # Normaliza para [0, 1] (maior = mais similar)
        if metric in ['euclidean', 'hilbert_schmidt', 'bures_distance']:
            # Para distâncias: converte para similaridade (inverso)
            similarity = 1.0 / (1.0 + similarity)
        # Para cosine e fidelity: já está em [0, 1]

        return float(similarity)

    def _cosine_similarity(self, psi: torch.Tensor, char_pattern: torch.Tensor) -> float:
        """
        Similaridade do cosseno (baseline atual).
        """
        psi_flat = psi.flatten()
        char_flat = char_pattern.flatten()

        similarity = torch.nn.functional.cosine_similarity(
            psi_flat.unsqueeze(0),
            char_flat.unsqueeze(0),
            dim=1
        ).item()

        return max(0.0, min(1.0, similarity))  # Garante [0, 1]

    def _euclidean_distance(self, psi: torch.Tensor, char_pattern: torch.Tensor) -> float:
        """
        Distância euclidiana normalizada.
        Melhor para estados com alta variabilidade.
        """
        distance = torch.norm(psi - char_pattern).item()

        # Normaliza pela magnitude máxima possível
        max_possible_distance = math.sqrt(psi.numel() * 4.0)  # Assumindo valores ~[-2, 2]
        normalized_distance = distance / max_possible_distance

        return normalized_distance

    def _quantum_fidelity(self, psi: torch.Tensor, char_pattern: torch.Tensor) -> float:
        """
        Fidelidade quântica: |⟨ψ|φ⟩|²
        Melhor para estados coerentes puros.
        """
        # Trata como vetores complexos (parte real + i * parte imaginária)
        psi_complex = torch.complex(psi[..., 0], psi[..., 1])  # w + i*x
        char_complex = torch.complex(char_pattern[..., 0], char_pattern[..., 1])

        # Produto interno complexo
        fidelity = torch.abs(torch.sum(psi_complex * torch.conj(char_complex)))**2

        # Normaliza
        norm_psi = torch.sum(torch.abs(psi_complex)**2)
        norm_char = torch.sum(torch.abs(char_complex)**2)

        if norm_psi > 0 and norm_char > 0:
            fidelity = fidelity / (norm_psi * norm_char)

        return fidelity.item()

    def _hilbert_schmidt_distance(self, psi: torch.Tensor, char_pattern: torch.Tensor) -> float:
        """
        Distância Hilbert-Schmidt: ||ρ - σ||_HS
        Melhor para estados mistos e emaranhados.
        """
        # Computa como distância euclidiana dos vetores achatados
        # Para estados quânticos reais, seria traço de (ρ-σ)†(ρ-σ)
        diff = psi - char_pattern
        distance = torch.sqrt(torch.sum(diff ** 2)).item()

        return distance

    def _bures_distance(self, psi: torch.Tensor, char_pattern: torch.Tensor) -> float:
        """
        Distância de Bures: métrica quântica otimizada.
        Melhor para comparação de estados quânticos.
        """
        # Simplificação: usa fidelidade para computar distância de Bures
        fidelity = self._quantum_fidelity(psi, char_pattern)

        # Distância de Bures: sqrt(2 * (1 - sqrt(fidelity)))
        if fidelity >= 0:
            bures_distance = math.sqrt(2 * (1 - math.sqrt(fidelity)))
        else:
            bures_distance = 1.0  # Máxima distância

        return bures_distance

    def _measure_coherence(self, psi: torch.Tensor) -> float:
        """
        Mede coerência quântica do estado.
        """
        # Coerência baseada na "pureza" do estado
        # Estados coerentes têm baixa entropia

        # Simplificação: coerência baseada na variância
        psi_flat = psi.flatten()
        coherence = 1.0 / (1.0 + torch.std(psi_flat).item())

        return coherence

    def _measure_entanglement(self, psi: torch.Tensor) -> float:
        """
        Mede emaranhamento quântico aproximado.
        """
        # Medida simplificada baseada na correlação entre componentes
        w, x, y, z = psi[..., 0], psi[..., 1], psi[..., 2], psi[..., 3]

        # Correlação entre componentes
        corr_wx = torch.corrcoef(torch.stack([w.flatten(), x.flatten()]))[0, 1]
        corr_yz = torch.corrcoef(torch.stack([y.flatten(), z.flatten()]))[0, 1]

        # Emaranhamento aproximado
        entanglement = (abs(corr_wx.item()) + abs(corr_yz.item())) / 2.0

        return entanglement

    def _measure_complexity(self, psi: torch.Tensor) -> float:
        """
        Mede complexidade estrutural do estado quântico.
        """
        # Complexidade baseada na entropia espectral
        psi_flat = psi.flatten()

        # Análise de frequência
        spectrum = torch.abs(torch.fft.fft(psi_flat))
        spectrum = spectrum / torch.sum(spectrum)  # Normaliza

        # Entropia espectral
        entropy = -torch.sum(spectrum * torch.log(spectrum + 1e-10)).item()
        max_entropy = math.log(len(spectrum))

        complexity = entropy / max_entropy if max_entropy > 0 else 0.0

        return complexity

    def benchmark_metrics(self, psi: torch.Tensor, char_patterns: Dict[int, torch.Tensor],
                          n_samples: int = 100) -> Dict[str, Any]:
        """
        Benchmark das métricas para otimização.

        Args:
            psi: Estado quântico de teste
            char_patterns: Padrões de caracteres {ascii_code: pattern}
            n_samples: Número de amostras para benchmark

        Returns:
            Resultados do benchmark
        """
        print(f"    🔬 [QuantumSimilarityMetrics] Executando benchmark de métricas...")

        results = {}
        sample_chars = list(char_patterns.keys())[:min(n_samples, len(char_patterns))]

        for metric_name in self.metrics.keys():
            similarities = []

            for char_code in sample_chars:
                pattern = char_patterns[char_code]
                similarity = self.compute_similarity(psi, pattern, metric_name)
                similarities.append(similarity)

            # Estatísticas
            results[metric_name] = {
                'mean_similarity': sum(similarities) / len(similarities),
                'std_similarity': torch.std(torch.tensor(similarities)).item(),
                'max_similarity': max(similarities),
                'min_similarity': min(similarities)
            }

        # Encontra melhor métrica
        best_metric = max(results.keys(),
                         key=lambda m: results[m]['mean_similarity'])

        print(f"    🏆 [QuantumSimilarityMetrics] Melhor métrica no benchmark: {best_metric}")
        print(f"      - Similaridade média: {results[best_metric]['mean_similarity']:.3f}")

        return {
            'results': results,
            'best_metric': best_metric,
            'recommendation': f"Usar {best_metric} para estados similares"
        }


# Função de interface para integração
def create_quantum_similarity_metrics() -> QuantumSimilarityMetrics:
    """
    Factory function para criar instância das métricas de similaridade quântica.
    """
    return QuantumSimilarityMetrics()


# Teste das implementações
if __name__ == "__main__":
    # Teste básico
    metrics = create_quantum_similarity_metrics()

    # Estados de teste
    psi = torch.randn(64, 4)  # Estado quântico aleatório
    char_pattern = torch.randn(64, 4)  # Padrão de caractere aleatório

    # Testa seleção automática
    selected_metric = metrics.select_optimal_metric(psi)
    print(f"Métrica selecionada: {selected_metric}")

    # Testa computação
    similarity = metrics.compute_similarity(psi, char_pattern)
    print(f"Similaridade: {similarity:.4f}")

    # Testa todas as métricas
    print("\nTestando todas as métricas:")
    for metric_name in metrics.metrics.keys():
        sim = metrics.compute_similarity(psi, char_pattern, metric_name)
        print(f"  {metric_name}: {sim:.4f}")
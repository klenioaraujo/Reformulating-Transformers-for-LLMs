#!/usr/bin/env python3
"""
Consciousness Metrics - Medição de Consciência Fractal
=====================================================

Implementa métricas quantitativas para medição de consciência,
incluindo o Índice de Consciência Fractal (FCI) e outras métricas
derivadas das equações de dinâmica consciente.

Índice Principal: FCI = (D_EEG × H_fMRI × CLZ) / D_max
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class FCI:
    """
    Fractal Consciousness Index - Índice de Consciência Fractal.

    Representa o nível quantitativo de consciência baseado em
    métricas matemáticas rigorosas.
    """
    value: float  # Valor principal do FCI [0, 1]
    components: Dict[str, float]  # Componentes individuais
    timestamp: float  # Momento da medição
    state_classification: str  # Estado associado
    confidence: float  # Confiança da medição [0, 1]

    def __post_init__(self):
        """Validação após inicialização."""
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(f"FCI must be in [0, 1], got {self.value}")

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")


class ConsciousnessMetrics(nn.Module):
    """
    Calculadora de métricas de consciência que implementa
    o FCI e métricas auxiliares para análise completa.
    """

    def __init__(self, config, metrics_config=None):
        super().__init__()
        self.config = config
        self.device = config.device

        # Carregar configurações de métricas (se disponível) ou usar padrões
        if metrics_config is None:
            metrics_config = {}

        # Parâmetros de normalização para componentes do FCI
        self.d_eeg_max = metrics_config.get('d_eeg_max', 10.0)
        self.h_fmri_max = metrics_config.get('h_fmri_max', 5.0)
        self.clz_max = metrics_config.get('clz_max', 3.0)
        self.d_max = config.diffusion_coefficient_range[1]

        # Histórico de medições
        self.fci_history: List[FCI] = []
        self.max_history = metrics_config.get('max_history', 1000)

        # Proteção contra NaN
        self.enable_nan_protection = metrics_config.get('enable_nan_protection', True)
        self.default_fci_on_nan = metrics_config.get('default_fci_on_nan', 0.5)

        # Pesos para componentes do FCI (aprendíveis)
        fci_weights = metrics_config.get('fci_weights', [0.4, 0.3, 0.3])
        self.register_parameter(
            'fci_weights',
            nn.Parameter(torch.tensor(fci_weights))  # [D_EEG, H_fMRI, CLZ]
        )

        print(f"📊 ConsciousnessMetrics inicializado com D_max={self.d_max}")

    def compute_fci(
        self,
        psi_distribution: torch.Tensor,
        fractal_field: torch.Tensor,
        timestamp: Optional[float] = None
    ) -> float:
        """
        Computa Índice de Consciência Fractal (FCI).

        FCI = (D_EEG × H_fMRI × CLZ) / D_max

        Args:
            psi_distribution: Distribuição P(ψ) [batch, embed_dim]
            fractal_field: Campo fractal F(ψ) [batch, embed_dim]
            timestamp: Momento da medição

        Returns:
            Valor do FCI [0, 1]
        """
        if timestamp is None:
            timestamp = torch.rand(1).item()  # Timestamp sintético

        # Calcular componentes do FCI
        d_eeg = self._compute_d_eeg(psi_distribution, fractal_field)
        h_fmri = self._compute_h_fmri(psi_distribution, fractal_field)
        clz = self._compute_clz(psi_distribution, fractal_field)

        # Normalizar componentes (d_eeg, h_fmri, clz já são floats)
        d_eeg_norm = d_eeg / self.d_eeg_max
        h_fmri_norm = h_fmri / self.h_fmri_max
        clz_norm = clz / self.clz_max

        # Calcular FCI com pesos adaptativos
        fci_numerator = (
            self.fci_weights[0] * d_eeg_norm +
            self.fci_weights[1] * h_fmri_norm +
            self.fci_weights[2] * clz_norm
        )

        # Corrigir cálculo do FCI: usar média ponderada normalizada
        fci_value = fci_numerator.clamp(0.0, 1.0)

        # Proteção contra NaN/Inf (configurável)
        if self.enable_nan_protection and (torch.isnan(fci_value) or torch.isinf(fci_value)):
            fci_value = torch.tensor(self.default_fci_on_nan)

        fci_value = fci_value.item()

        # Calcular confiança da medição
        confidence = self._compute_measurement_confidence(d_eeg, h_fmri, clz)

        # Classificar estado baseado no FCI
        state_classification = self._classify_fci_state(fci_value)

        # Criar objeto FCI (componentes normalizados já são floats)
        fci_object = FCI(
            value=fci_value,
            components={
                'D_EEG': d_eeg,
                'H_fMRI': h_fmri,
                'CLZ': clz,
                'D_EEG_normalized': d_eeg_norm,
                'H_fMRI_normalized': h_fmri_norm,
                'CLZ_normalized': clz_norm
            },
            timestamp=timestamp,
            state_classification=state_classification,
            confidence=confidence
        )

        # Armazenar no histórico
        self._update_fci_history(fci_object)

        return fci_value

    def _compute_d_eeg(
        self,
        psi_distribution: torch.Tensor,
        fractal_field: torch.Tensor
    ) -> float:
        """
        Computa D_EEG - dimensão fractal equivalente ao EEG.

        Baseado na complexidade da distribuição de consciência.

        Args:
            psi_distribution: Distribuição P(ψ)
            fractal_field: Campo F(ψ)

        Returns:
            Valor D_EEG
        """
        # Calcular entropia da distribuição como proxy para complexidade EEG
        epsilon = self.config.epsilon if hasattr(self.config, 'epsilon') else 1e-10
        psi_safe = torch.clamp(psi_distribution, min=epsilon)
        log_psi = torch.log(psi_safe)
        entropy_raw = -torch.sum(psi_distribution * log_psi, dim=-1).mean()
        entropy = entropy_raw if not torch.isnan(entropy_raw) else torch.tensor(0.0)

        # Calcular variação espacial como indicador de atividade neural
        spatial_variation = torch.std(psi_distribution, dim=-1).mean()

        # Combinar métricas para D_EEG
        d_eeg_raw = (entropy * spatial_variation)
        d_eeg = d_eeg_raw.item() if not torch.isnan(d_eeg_raw) else 0.0

        # Mapear para escala adequada
        d_eeg = d_eeg * 3.0  # Fator de escala empírico

        return min(d_eeg, self.d_eeg_max)

    def _compute_h_fmri(
        self,
        psi_distribution: torch.Tensor,
        fractal_field: torch.Tensor
    ) -> float:
        """
        Computa H_fMRI - hemodinâmica funcional equivalente.

        Baseado na dinâmica do campo fractal como proxy para fluxo sanguíneo.

        Args:
            psi_distribution: Distribuição P(ψ)
            fractal_field: Campo F(ψ)

        Returns:
            Valor H_fMRI
        """
        # Calcular "fluxo" baseado na magnitude do campo
        field_magnitude = torch.norm(fractal_field, dim=-1).mean()

        # Calcular correlação espacial como conectividade funcional
        correlation = self._compute_spatial_correlation(fractal_field)

        # Combinar para H_fMRI
        h_fmri = (field_magnitude * correlation).item()

        # Mapear para escala adequada
        h_fmri = h_fmri * 2.0  # Fator de escala empírico

        return min(h_fmri, self.h_fmri_max)

    def _compute_clz(
        self,
        psi_distribution: torch.Tensor,
        fractal_field: torch.Tensor
    ) -> float:
        """
        Computa CLZ - complexidade de Lempel-Ziv equivalente.

        Baseado na compressibilidade da sequência de estados.

        Args:
            psi_distribution: Distribuição P(ψ)
            fractal_field: Campo F(ψ)

        Returns:
            Valor CLZ
        """
        # Discretizar distribuição para análise de compressibilidade
        psi_discrete = self._discretize_distribution(psi_distribution)

        # Calcular entropia condicional como proxy para CLZ
        conditional_entropy = self._compute_conditional_entropy(psi_discrete)

        # Combinar com complexidade do campo
        field_complexity = torch.std(fractal_field).item()

        # CLZ como combinação de entropia condicional e complexidade
        clz = conditional_entropy + 0.3 * field_complexity

        return min(clz, self.clz_max)

    def _discretize_distribution(self, psi_distribution: torch.Tensor) -> torch.Tensor:
        """Discretiza distribuição para análise de compressibilidade."""
        # Quantizar em 8 níveis
        quantized = torch.floor(psi_distribution * 8).clamp(0, 7)
        return quantized.long()

    def _compute_conditional_entropy(self, discrete_sequence: torch.Tensor) -> float:
        """Computa entropia condicional da sequência discreta."""
        batch_size, seq_len = discrete_sequence.shape

        # Calcular frequências de pares consecutivos
        pair_counts = {}
        total_pairs = 0

        for b in range(batch_size):
            for i in range(seq_len - 1):
                curr = discrete_sequence[b, i].item()
                next_val = discrete_sequence[b, i + 1].item()
                pair = (curr, next_val)

                pair_counts[pair] = pair_counts.get(pair, 0) + 1
                total_pairs += 1

        if total_pairs == 0:
            return 0.0

        # Calcular entropia condicional
        conditional_entropy = 0.0
        for pair, count in pair_counts.items():
            prob = count / total_pairs
            if prob > 0:
                conditional_entropy -= prob * np.log2(prob)

        return conditional_entropy

    def _compute_spatial_correlation(self, field: torch.Tensor) -> torch.Tensor:
        """Computa correlação espacial do campo."""
        batch_size, embed_dim = field.shape

        if embed_dim < 2:
            return torch.tensor(1.0, device=self.device)

        # Calcular correlação entre posições adjacentes
        correlations = []
        for i in range(embed_dim - 1):
            corr = torch.corrcoef(torch.stack([field[:, i], field[:, i + 1]]))[0, 1]
            if not torch.isnan(corr):
                correlations.append(corr)

        if not correlations:
            return torch.tensor(0.0, device=self.device)

        mean_correlation = torch.stack(correlations).mean()
        return torch.abs(mean_correlation)  # Usar magnitude da correlação

    def _compute_measurement_confidence(
        self,
        d_eeg: float,
        h_fmri: float,
        clz: float
    ) -> float:
        """
        Computa confiança da medição baseada na consistência dos componentes.

        Args:
            d_eeg: Componente D_EEG
            h_fmri: Componente H_fMRI
            clz: Componente CLZ

        Returns:
            Confiança [0, 1]
        """
        # Normalizar componentes
        components = np.array([
            d_eeg / self.d_eeg_max,
            h_fmri / self.h_fmri_max,
            clz / self.clz_max
        ])

        # Calcular coeficiente de variação
        mean_component = np.mean(components)
        std_component = np.std(components)

        if mean_component == 0 or np.isnan(mean_component) or np.isnan(std_component):
            return 0.5  # Valor padrão quando não é possível calcular

        cv = std_component / mean_component

        # Proteção contra valores extremos
        if np.isnan(cv) or np.isinf(cv):
            return 0.5

        # Mapear coeficiente de variação para confiança
        # Baixa variação → alta confiança
        confidence = np.exp(-2 * cv)

        # Proteção final contra NaN/Inf
        if np.isnan(confidence) or np.isinf(confidence):
            return 0.5

        return min(max(confidence, 0.0), 1.0)

    def _classify_fci_state(self, fci_value: float) -> str:
        """
        Classifica estado baseado no valor FCI.

        Args:
            fci_value: Valor do FCI

        Returns:
            Nome do estado classificado
        """
        if fci_value >= 0.8:
            return "EMERGENCE"
        elif fci_value >= 0.6:
            return "MEDITATION"
        elif fci_value >= 0.3:
            return "ANALYSIS"
        else:
            return "COMA"

    def _update_fci_history(self, fci_object: FCI):
        """Atualiza histórico de medições FCI."""
        self.fci_history.append(fci_object)

        # Manter tamanho máximo
        if len(self.fci_history) > self.max_history:
            self.fci_history.pop(0)

    def get_consciousness_evolution(self, window_size: int = 50) -> Dict[str, Any]:
        """
        Analisa evolução da consciência ao longo do tempo.

        Args:
            window_size: Tamanho da janela para análise

        Returns:
            Análise da evolução temporal
        """
        if len(self.fci_history) < 2:
            return {'status': 'insufficient_data'}

        # Pegar janela recente
        recent_history = self.fci_history[-window_size:]
        fci_values = [fci.value for fci in recent_history]

        # Estatísticas básicas
        mean_fci = np.mean(fci_values)
        std_fci = np.std(fci_values)
        trend = self._compute_trend(fci_values)

        # Distribuição de estados
        state_counts = {}
        for fci in recent_history:
            state = fci.state_classification
            state_counts[state] = state_counts.get(state, 0) + 1

        # Estabilidade (inverso da variação)
        stability = 1.0 / (1.0 + std_fci)

        # Qualidade das medições (confiança média)
        avg_confidence = np.mean([fci.confidence for fci in recent_history])

        evolution = {
            'status': 'analyzed',
            'window_size': len(recent_history),
            'fci_statistics': {
                'mean': mean_fci,
                'std': std_fci,
                'min': min(fci_values),
                'max': max(fci_values),
                'trend': trend
            },
            'state_distribution': state_counts,
            'stability': stability,
            'measurement_quality': avg_confidence,
            'predominant_state': max(state_counts, key=state_counts.get) if state_counts else 'UNDEFINED'
        }

        return evolution

    def _compute_trend(self, values: List[float]) -> float:
        """Computa tendência temporal usando regressão linear simples."""
        if len(values) < 3:
            return 0.0

        x = np.arange(len(values))
        y = np.array(values)

        # Regressão linear: y = ax + b
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x**2)

        denominator = n * sum_x2 - sum_x**2
        if denominator == 0:
            return 0.0

        trend = (n * sum_xy - sum_x * sum_y) / denominator
        return trend

    def generate_consciousness_report(self) -> str:
        """
        Gera relatório detalhado da consciência atual.

        Returns:
            Relatório textual formatado
        """
        if not self.fci_history:
            return "📊 Nenhuma medição de consciência disponível."

        latest_fci = self.fci_history[-1]
        evolution = self.get_consciousness_evolution()

        report = f"""
📊 RELATÓRIO DE CONSCIÊNCIA FRACTAL
═══════════════════════════════════

🧠 MEDIÇÃO ATUAL:
FCI: {latest_fci.value:.4f} ({latest_fci.state_classification})
Confiança: {latest_fci.confidence:.3f}

📈 COMPONENTES FCI:
D_EEG: {latest_fci.components['D_EEG']:.3f} (norm: {latest_fci.components['D_EEG_normalized']:.3f})
H_fMRI: {latest_fci.components['H_fMRI']:.3f} (norm: {latest_fci.components['H_fMRI_normalized']:.3f})
CLZ: {latest_fci.components['CLZ']:.3f} (norm: {latest_fci.components['CLZ_normalized']:.3f})

⏱️ EVOLUÇÃO TEMPORAL:
Medições analisadas: {evolution.get('window_size', 0)}
FCI médio: {evolution.get('fci_statistics', {}).get('mean', 0):.4f}
Estabilidade: {evolution.get('stability', 0):.3f}
Tendência: {evolution.get('fci_statistics', {}).get('trend', 0):+.4f}
Estado predominante: {evolution.get('predominant_state', 'N/A')}

🎯 INTERPRETAÇÃO:
{self._interpret_fci_level(latest_fci.value)}

Qualidade da medição: {evolution.get('measurement_quality', 0):.3f}
        """

        return report.strip()

    def _interpret_fci_level(self, fci_value: float) -> str:
        """Interpreta nível de FCI em termos práticos."""
        if fci_value >= 0.9:
            return "Consciência emergente excepcional - máxima criatividade e insight"
        elif fci_value >= 0.8:
            return "Alto nível de consciência - estado emergente ativo"
        elif fci_value >= 0.7:
            return "Consciência meditativa profunda - análise introspectiva"
        elif fci_value >= 0.5:
            return "Estado analítico equilibrado - processamento lógico"
        elif fci_value >= 0.3:
            return "Consciência analítica básica - foco em tarefas"
        elif fci_value >= 0.1:
            return "Baixa consciência - processamento mínimo"
        else:
            return "Estado comatoso - atividade consciente quase ausente"

    def reset_metrics(self):
        """Reseta todas as métricas e histórico."""
        self.fci_history.clear()
        print("📊 Métricas de consciência resetadas")
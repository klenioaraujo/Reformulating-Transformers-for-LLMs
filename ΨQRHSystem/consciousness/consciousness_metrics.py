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
from core.TernaryLogicFramework import TernaryLogicFramework


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
        # Handle both dict and object configs
        if hasattr(config, 'device'):
            self.device = config.device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Carregar configurações de métricas do arquivo YAML
        import yaml
        import os

        try:
            config_path = os.path.join('configs', 'consciousness_metrics.yaml')
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            # Usar configuração do YAML se disponível
            if yaml_config:
                metrics_config = yaml_config
        except Exception as e:
            print(f"⚠️ Erro ao carregar configurações YAML: {e}. Usando configuração fornecida.")

        # Parâmetros de normalização para componentes do FCI
        self.d_eeg_max = metrics_config.get('component_max_values', {}).get('d_eeg_max', 10.0)
        self.h_fmri_max = metrics_config.get('component_max_values', {}).get('h_fmri_max', 5.0)
        self.clz_max = metrics_config.get('component_max_values', {}).get('clz_max', 3.0)
        # Handle both dict and object configs for diffusion coefficient
        if hasattr(config, 'diffusion_coefficient_range'):
            self.d_max = config.diffusion_coefficient_range[1]
        else:
            diffusion_range = config.get('diffusion_coefficient_range', [0.01, 10.0])
            self.d_max = diffusion_range[1]

        # Parâmetros de dimensão fractal (para mapeamento correto D → FCI)
        fractal_dim = metrics_config.get('fractal_dimension', {})
        self.fractal_dimension_min = fractal_dim.get('min', 1.0)
        self.fractal_dimension_max = fractal_dim.get('max', 3.0)
        self.fractal_dimension_normalizer = fractal_dim.get('normalizer', 2.0)

        # Thresholds de estados (configuráveis)
        state_thresholds = metrics_config.get('state_thresholds', {})
        self.threshold_emergence = state_thresholds.get('emergence', {}).get('min_fci', 0.85)
        self.threshold_meditation = state_thresholds.get('meditation', {}).get('min_fci', 0.70)
        self.threshold_analysis = state_thresholds.get('analysis', {}).get('min_fci', 0.45)

        # Histórico de medições
        self.fci_history: List[FCI] = []
        measurement_quality = metrics_config.get('measurement_quality', {})
        self.max_history = measurement_quality.get('max_history', 1000)

        # Proteção contra NaN
        self.enable_nan_protection = measurement_quality.get('enable_nan_protection', True)
        self.default_fci_on_nan = measurement_quality.get('default_fci_on_nan', 0.5)

        # Inicializar lógica ternária
        self.ternary_logic = TernaryLogicFramework(device=self.device)

        # Pesos para componentes do FCI (aprendíveis) com suporte ternário
        fci_weights_config = metrics_config.get('fci_weights', {})
        fci_weights = [
            fci_weights_config.get('d_eeg', 0.4),
            fci_weights_config.get('h_fmri', 0.3),
            fci_weights_config.get('clz', 0.3)
        ]
        self.register_parameter(
            'fci_weights',
            nn.Parameter(torch.tensor(fci_weights))  # [D_EEG, H_fMRI, CLZ]
        )

        # Estados ternários para consciência
        self.consciousness_ternary_states = {
            'coma': -1,
            'analysis': 0,
            'meditation': 0,
            'emergence': 1
        }

        # Correlation method settings - ZERO FALLBACK POLICY
        correlation_method = metrics_config.get('correlation_method', {})
        self.correlation_method = correlation_method.get('method', 'autocorrelation')
        self.correlation_fallback = None  # ZERO FALLBACK POLICY - no fallback values allowed
        self.correlation_use_abs = correlation_method.get('use_absolute_value', True)
        self.correlation_clamp = correlation_method.get('clamp_range', [0.0, 1.0])

        # Debug settings
        debug = metrics_config.get('debug', {})
        self.log_fci_calculations = debug.get('log_fci_calculations', False)
        self.log_component_details = debug.get('log_component_details', False)

        print(f"📊 ConsciousnessMetrics inicializado")
        print(f"   - Component Max Values: D_EEG={self.d_eeg_max}, H_fMRI={self.h_fmri_max}, CLZ={self.clz_max}")
        print(f"   - Fractal D: [{self.fractal_dimension_min}, {self.fractal_dimension_max}]")
        print(f"   - FCI Thresholds: EMERGENCE≥{self.threshold_emergence}, MEDITATION≥{self.threshold_meditation}, ANALYSIS≥{self.threshold_analysis}")
        print(f"   - Correlation Method: {self.correlation_method}")

    def compute_fci(
        self,
        psi_distribution: torch.Tensor,
        fractal_field: torch.Tensor,
        timestamp: Optional[float] = None,
        power_spectrum_pk: Optional[torch.Tensor] = None
    ) -> float:
        """
        Computa Índice de Consciência Fractal (FCI).

        FCI = (D_EEG × H_fMRI × CLZ) / D_max

        Args:
            psi_distribution: Distribuição P(ψ) [batch, embed_dim]
            fractal_field: Campo fractal F(ψ) [batch, embed_dim]
            timestamp: Momento da medição
            power_spectrum_pk: P(k) do espectro quaterniônico [batch, embed_dim] (NOVO)

        Returns:
            Valor do FCI [0, 1]
        """
        if timestamp is None:
            timestamp = torch.rand(1).item()  # Timestamp sintético

        # FASE 2: Calcular dimensão fractal D a partir de P(k)
        # Se P(k) foi fornecido, usar; caso contrário, usar psi_distribution como fallback
        pk_for_analysis = power_spectrum_pk if power_spectrum_pk is not None else psi_distribution
        fractal_dimension_D = self._compute_fractal_dimension(pk_for_analysis)

        # Calcular componentes do FCI (H_fMRI agora recebe D)
        d_eeg = self._compute_d_eeg(psi_distribution, fractal_field)
        h_fmri = self._compute_h_fmri(psi_distribution, fractal_field, fractal_dimension_D)
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

        # Normalizar FCI ao range [0, 1]
        # FCI já deve estar no intervalo [0,1] por construção dos componentes normalizados
        fci_value = fci_numerator.clamp(0.0, 1.0).item()

        # Log detalhado se habilitado
        if self.log_component_details:
            print(f"\n🔬 FCI Components Debug:")
            print(f"   Fractal Dimension D: {fractal_dimension_D:.4f} (usado em H_fMRI)")
            print(f"   D_EEG: {d_eeg:.4f} (norm: {d_eeg_norm:.4f}, max: {self.d_eeg_max})")
            print(f"   H_fMRI: {h_fmri:.4f} (norm: {h_fmri_norm:.4f}, max: {self.h_fmri_max})")
            print(f"   CLZ: {clz:.4f} (norm: {clz_norm:.4f}, max: {self.clz_max})")
            print(f"   Weights: {self.fci_weights.detach().cpu().numpy()}")
            print(f"   FCI = {fci_value:.4f}")

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

    def _compute_fractal_dimension(self, power_spectrum_pk: torch.Tensor) -> float:
        """
        Calcula dimensão fractal D via análise de lei de potência P(k) ~ k^(-β).
        Conforme paper ΨQRH Seção 3.1: D = (3-β)/2

        Args:
            power_spectrum_pk: Distribuição de potência P(k) [batch, embed_dim]

        Returns:
            Dimensão fractal D ∈ [1.0, 2.0]
        """
        # Criar índices de frequência k (começar de 1 para evitar log(0))
        batch_size, embed_dim = power_spectrum_pk.shape
        k = torch.arange(1, embed_dim + 1, dtype=torch.float32, device=power_spectrum_pk.device)

        # Flatten
        pk_flat = power_spectrum_pk.flatten()
        k_repeated = k.repeat(batch_size)

        # Filtrar: manter apenas P(k) > threshold (threshold menor para pegar mais pontos)
        # Usar percentil em vez de threshold fixo
        threshold = torch.quantile(pk_flat[pk_flat > 0], 0.1)  # Pegar 90% dos valores não-zero
        mask = pk_flat > threshold
        pk_filtered = pk_flat[mask]
        k_filtered = k_repeated[mask]

        if len(pk_filtered) < 10:  # Exigir pelo menos 10 pontos para regressão confiável
            print(f"⚠️  Fractal Dimension: Dados insuficientes após filtragem (n={len(pk_filtered)}, threshold={threshold:.2e})")
            return 1.0  # Mínimo físico: Euclidiano (linha suave, sem complexidade fractal)

        # Transformação log-log para regressão linear
        # P(k) ~ k^(-β) → log(P(k)) = -β·log(k) + c
        log_k = torch.log(k_filtered)
        log_pk = torch.log(pk_filtered)

        # Regressão linear via mínimos quadrados
        # slope = Σ[(x-x̄)(y-ȳ)] / Σ[(x-x̄)²]
        mean_log_k = log_k.mean()
        mean_log_pk = log_pk.mean()

        numerator = torch.sum((log_k - mean_log_k) * (log_pk - mean_log_pk))
        denominator = torch.sum((log_k - mean_log_k) ** 2)

        slope = numerator / (denominator + 1e-10)
        beta = -slope.item()  # β é o negativo da inclinação

        # Calcular dimensão fractal: D = (3 - β) / 2
        D = (3.0 - beta) / 2.0

        # Clamp para valores fisicamente razoáveis para sinais 1D
        # D ∈ [1.0, 2.0]: 1.0=linha suave, 1.5=browniano, 2.0=muito irregular
        D_clamped = max(1.0, min(2.0, D))

        # Calcular R² (qualidade do ajuste)
        residuals = log_pk - (slope * log_k + (mean_log_pk - slope * mean_log_k))
        ss_res = torch.sum(residuals ** 2)
        ss_tot = torch.sum((log_pk - mean_log_pk) ** 2)
        r_squared = 1.0 - (ss_res / (ss_tot + 1e-10))
        r_squared_value = max(0.0, min(1.0, r_squared.item()))  # Clamp [0, 1]

        # Armazenar dados da lei de potência para acesso posterior
        self.last_beta_exponent = beta
        self.last_r_squared = r_squared_value
        self.last_points_used = len(pk_filtered)
        self.last_fractal_dimension_raw = D

        if self.log_component_details:
            print(f"✅ Dimensão Fractal calculada via lei de potência:")
            print(f"   β (expoente) = {beta:.4f}")
            print(f"   D = (3-β)/2 = {D:.4f} → clamped = {D_clamped:.4f}")
            print(f"   R² = {r_squared_value:.4f} (qualidade do ajuste)")
            print(f"   Dados usados: {len(pk_filtered)} pontos (de {len(pk_flat)} totais)")

        return D_clamped

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
        psi_safe = torch.clamp(psi_distribution, min=1e-10)
        log_psi = torch.log(psi_safe)
        entropy = -torch.sum(psi_distribution * log_psi, dim=-1).mean()

        # Calcular variação espacial como indicador de atividade neural
        spatial_variation = torch.std(psi_distribution, dim=-1).mean()

        # Combinar métricas para D_EEG com limites razoáveis
        d_eeg = (entropy * spatial_variation).item()

        # Limitar D_EEG a valores positivos e razoáveis
        d_eeg = max(0.0, min(d_eeg, 100.0))

        return d_eeg

    def _compute_h_fmri(
        self,
        psi_distribution: torch.Tensor,
        fractal_field: torch.Tensor,
        fractal_dimension_D: float
    ) -> float:
        """
        Computa H_fMRI - hemodinâmica funcional equivalente.

        FASE 3: Agora usa dimensão fractal D em vez de correlação fixa.
        Conforme paper ΨQRH: H_fMRI deve refletir a complexidade do sinal via D.

        Args:
            psi_distribution: Distribuição P(ψ)
            fractal_field: Campo F(ψ)
            fractal_dimension_D: Dimensão fractal calculada de P(k) via lei de potência

        Returns:
            Valor H_fMRI
        """
        # Calcular "fluxo" baseado na magnitude do campo
        field_magnitude = torch.norm(fractal_field, dim=-1).mean()

        # NOVO: Usar dimensão fractal D em vez de correlação fixa
        # D ∈ [1.0, 2.0], normalizar para [0, 1] para uso como fator modulador
        D_normalized = (fractal_dimension_D - 1.0) / 1.0  # (D - min) / (max - min)

        # H_fMRI como produto de magnitude de campo e complexidade fractal
        h_fmri = (field_magnitude * D_normalized).item()

        if self.log_component_details:
            print(f"   H_fMRI Debug (ACOPLADO A D):")
            print(f"      field_mag={field_magnitude.item():.6f}")
            print(f"      D={fractal_dimension_D:.4f} → D_norm={D_normalized:.4f}")
            print(f"      H_fMRI = {h_fmri:.6f}")

        return h_fmri

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
        clz = conditional_entropy + field_complexity

        return clz

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

        # Calcular entropia condicional
        conditional_entropy = 0.0
        for pair, count in pair_counts.items():
            prob = count / total_pairs
            conditional_entropy -= prob * np.log2(prob)

        return conditional_entropy

    def _compute_spatial_correlation(self, field: torch.Tensor) -> torch.Tensor:
        """
        Computa correlação espacial do campo.

        Método configurável via correlation_method no config.
        """
        if self.correlation_method == "autocorrelation":
            # Autocorrelação lag-1
            field_flat = field.reshape(-1)
            field_normalized = (field_flat - field_flat.mean()) / field_flat.std()
            corr = torch.mean(field_normalized[:-1] * field_normalized[1:])

        elif self.correlation_method == "variance_proxy":
            # Variância como proxy de correlação
            corr = torch.var(field)

        else:  # corrcoef
            # Correlação via corrcoef
            batch_size, embed_dim = field.shape
            correlations = []
            for i in range(embed_dim - 1):
                corr_matrix = torch.corrcoef(torch.stack([field[:, i], field[:, i + 1]]))
                correlations.append(corr_matrix[0, 1])
            corr = torch.stack(correlations).mean()

        if self.correlation_use_abs:
            corr = torch.abs(corr)

        return corr.clamp(self.correlation_clamp[0], self.correlation_clamp[1])

    def compute_fci_from_fractal_dimension(self, fractal_dimension: float) -> float:
        """
        Mapeia dimensão fractal diretamente para FCI com lógica ternária.

        Fórmula corrigida: FCI = (D - D_min) / (D_max - D_min)
        Onde D ∈ [1, 3]:
        - D = 1.0 → FCI = 0.0 (linha suave, sem complexidade)
        - D = 1.7 → FCI = 0.35 (movimento browniano)
        - D = 2.0 → FCI = 0.5 (browniano padrão)
        - D = 3.0 → FCI = 1.0 (preenchimento total do espaço)

        Args:
            fractal_dimension: Dimensão fractal D

        Returns:
            FCI ∈ [0, 1]
        """
        # Cálculo base
        fci = (fractal_dimension - self.fractal_dimension_min) / self.fractal_dimension_normalizer

        # Aplicar refinamento ternário baseado na classificação de estados
        ternary_state = self._classify_fractal_dimension_ternary(fractal_dimension)

        # Ajustar FCI baseado no estado ternário
        if ternary_state == 1:  # Alta complexidade fractal
            fci = min(1.0, fci * 1.1)  # Aumento de 10%
        elif ternary_state == -1:  # Baixa complexidade fractal
            fci = max(0.0, fci * 0.9)  # Redução de 10%
        # ternary_state == 0: manter fci original

        if self.log_fci_calculations:
            print(f"🔬 FCI Calculation: D={fractal_dimension:.3f} → FCI={fci:.3f} (ternary_state={ternary_state})")

        return max(0.0, min(1.0, fci))

    def _classify_fractal_dimension_ternary(self, fractal_dimension: float) -> int:
        """
        Classifica dimensão fractal em estados ternários.

        Args:
            fractal_dimension: Dimensão fractal D

        Returns:
            -1 (baixa complexidade), 0 (média), 1 (alta complexidade)
        """
        if fractal_dimension < 1.3:
            return -1  # Baixa complexidade
        elif fractal_dimension > 2.2:
            return 1   # Alta complexidade
        else:
            return 0   # Complexidade média

    def _compute_measurement_confidence(
        self,
        d_eeg: float,
        h_fmri: float,
        clz: float
    ) -> float:
        """
        Computa confiança da medição baseada na consistência dos componentes com lógica ternária.

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
        cv = std_component / mean_component if mean_component > 0 else 1.0

        # Aplicar lógica ternária para avaliar consistência
        component_states = []
        for comp in components:
            if comp > 0.8:
                component_states.append(1)   # Alto
            elif comp < 0.2:
                component_states.append(-1)  # Baixo
            else:
                component_states.append(0)   # Médio

        # Usar consenso ternário para determinar qualidade geral
        consensus_result = self.ternary_logic.ternary_consensus(
            [self.ternary_logic.create_superposition() for _ in component_states],
            threshold=0.7
        )

        # Ajustar confiança baseada no consenso ternário
        base_confidence = np.exp(-2 * cv)

        if consensus_result == 1:
            # Consenso positivo - aumentar confiança
            confidence = min(1.0, base_confidence * 1.2)
        elif consensus_result == -1:
            # Consenso negativo - reduzir confiança
            confidence = max(0.0, base_confidence * 0.8)
        else:
            # Neutro - manter confiança base
            confidence = base_confidence

        return np.clip(confidence, 0.0, 1.0)

    def _classify_fci_state(self, fci_value: float) -> str:
        """
        Classifica estado baseado no valor FCI usando thresholds configuráveis e lógica ternária.

        Args:
            fci_value: Valor do FCI

        Returns:
            Nome do estado classificado
        """
        # Classificação binária tradicional
        if fci_value >= self.threshold_emergence:
            state = "EMERGENCE"
        elif fci_value >= self.threshold_meditation:
            state = "MEDITATION"
        elif fci_value >= self.threshold_analysis:
            state = "ANALYSIS"
        else:
            state = "COMA"

        # Aplicar refinamento ternário
        ternary_state = self.consciousness_ternary_states.get(state.lower(), 0)

        # Usar lógica ternária para ajustar classificação baseada em contexto
        # Se o estado for neutro (0), verificar se deve ser ajustado
        if ternary_state == 0:
            # Aplicar consenso ternário baseado no histórico recente
            recent_states = [fci.state_classification for fci in self.fci_history[-5:]]
            ternary_votes = [self.consciousness_ternary_states.get(s.lower(), 0) for s in recent_states]

            consensus = self.ternary_logic.ternary_consensus(
                [self.ternary_logic.create_superposition() for _ in ternary_votes],
                threshold=0.6
            )

            if consensus is not None and consensus != 0:
                # Ajustar estado baseado no consenso
                if consensus == 1 and state == "ANALYSIS":
                    state = "MEDITATION"
                elif consensus == -1 and state == "ANALYSIS":
                    state = "COMA"

        return state

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
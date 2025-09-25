import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import math

from .gate_controller import GateController


class GateLevel(Enum):
    """Níveis hierárquicos dos gate controllers"""
    NUMERICAL = "numerical"      # Nível baixo: estabilidade numérica
    LOCAL_COHERENCE = "local"    # Nível médio: coerência local (frases)
    GLOBAL_RESONANCE = "global"  # Nível alto: ressonância global (parágrafos/documentos)
    SEMANTIC = "semantic"        # Nível superior: coerência semântica


class GateDecision(Enum):
    """Decisões expandidas do sistema de gates"""
    DELIVER = "DELIVER"          # Entregar resultado (alta qualidade)
    ABSTAIN = "ABSTAIN"          # Recusar processamento (erro crítico)
    CLARIFY = "CLARIFY"          # Solicitar esclarecimento (resultado incerto)
    AMPLIFY = "AMPLIFY"          # Amplificar sinal (alta ressonância)
    ATTENUATE = "ATTENUATE"      # Atenuar sinal (interferência destrutiva)


@dataclass
class ResonanceConfig:
    """Configuração para análise de ressonância"""
    embed_dim: int = 64
    num_resonance_modes: int = 8
    interference_threshold: float = 0.2  # More sensitive to destructive interference
    constructive_threshold: float = 0.6  # Lower threshold for constructive detection
    phase_tolerance: float = 0.15        # Slightly more tolerant
    amplitude_threshold: float = 1e-6
    coherence_window: int = 32
    # New optimized thresholds
    high_coherence_threshold: float = 0.75  # For DELIVER decisions
    low_coherence_threshold: float = 0.25   # For ABSTAIN decisions
    resonance_amplification_factor: float = 1.2  # For AMPLIFY decisions
    resonance_attenuation_factor: float = 0.8   # For ATTENUATE decisions


class NumericalGateController(GateController):
    """
    Gate Controller de baixo nível para validação de estabilidade numérica
    e coerência local dentro de uma frase ou pequeno contexto.
    """

    def __init__(self,
                 orthogonal_threshold: float = 1e-6,
                 energy_threshold: float = 0.1,
                 drift_threshold: float = 0.1,
                 numerical_stability_threshold: float = 1e-12):
        super().__init__(orthogonal_threshold, energy_threshold, drift_threshold)
        self.numerical_stability_threshold = numerical_stability_threshold
        self.level = GateLevel.NUMERICAL

    def validate_numerical_stability(self, tensor: torch.Tensor) -> Dict[str, float]:
        """
        Valida estabilidade numérica básica do tensor
        """
        stability_metrics = {}

        # Verificar NaN e Inf
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        stability_metrics['has_nan'] = float(has_nan)
        stability_metrics['has_inf'] = float(has_inf)

        # Verificar números muito pequenos que podem causar underflow
        min_abs_value = torch.min(torch.abs(tensor[tensor != 0])).item() if torch.any(tensor != 0) else 1.0
        stability_metrics['min_magnitude'] = min_abs_value
        stability_metrics['underflow_risk'] = float(min_abs_value < self.numerical_stability_threshold)

        # Verificar condicionamento (número de condição aproximado)
        if tensor.dim() >= 2:
            try:
                condition_number = torch.linalg.cond(tensor.view(-1, tensor.size(-1))).item()
                stability_metrics['condition_number'] = condition_number
                stability_metrics['ill_conditioned'] = float(condition_number > 1e12)
            except:
                stability_metrics['condition_number'] = float('inf')
                stability_metrics['ill_conditioned'] = 1.0
        else:
            stability_metrics['condition_number'] = 1.0
            stability_metrics['ill_conditioned'] = 0.0

        return stability_metrics

    def enhanced_gate_decision(self, receipts: Dict[str, float],
                              stability_metrics: Dict[str, float]) -> GateDecision:
        """
        Decisão de gate aprimorada considerando estabilidade numérica
        """
        # Primeiro, verificar problemas críticos de estabilidade
        if (stability_metrics['has_nan'] or stability_metrics['has_inf'] or
            stability_metrics['underflow_risk'] or stability_metrics['ill_conditioned']):
            return GateDecision.ABSTAIN

        # Usar lógica original para casos normais
        original_decision = self.decide_gate(receipts)

        return GateDecision(original_decision)


class LocalCoherenceGate(nn.Module):
    """
    Gate Controller de nível médio para análise de coerência local
    entre elementos próximos na sequência (nível de frase).
    """

    def __init__(self, config: ResonanceConfig):
        super().__init__()
        self.config = config
        self.level = GateLevel.LOCAL_COHERENCE
        self.embed_dim = config.embed_dim * 4

        # Análise de coerência local via convolução
        self.local_coherence_conv = nn.Conv1d(
            in_channels=self.embed_dim,
            out_channels=config.num_resonance_modes,
            kernel_size=config.coherence_window // 4,
            padding='same'
        )

        # Métricas de coerência
        self.coherence_scorer = nn.Sequential(
            nn.Linear(config.num_resonance_modes, config.num_resonance_modes // 2),
            nn.ReLU(),
            nn.Linear(config.num_resonance_modes // 2, 1),
            nn.Sigmoid()
        )

    def calculate_local_coherence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula coerência local usando análise convolucional

        Args:
            x: Input tensor [B, T, 4*D]
        Returns:
            coherence_scores: [B, T] scores de coerência local
        """
        batch_size, seq_len, embed_dim = x.shape

        # Transpor para formato de convolução [B, 4*D, T]
        x_conv = x.transpose(1, 2)

        # Aplicar convolução para detectar padrões locais
        local_features = self.local_coherence_conv(x_conv)  # [B, num_modes, T]

        # Transpor de volta e calcular scores
        local_features = local_features.transpose(1, 2)  # [B, T, num_modes]
        coherence_scores = self.coherence_scorer(local_features).squeeze(-1)  # [B, T]

        return coherence_scores

    def decide_local_gate(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decisão de gate baseada em coerência local OTIMIZADA
        """
        coherence_scores = self.calculate_local_coherence(x)

        # IMPROVEMENT: Use config-based optimized thresholds
        high_threshold = self.config.high_coherence_threshold
        low_threshold = self.config.low_coherence_threshold

        # More nuanced decision making
        very_high_coherence = (coherence_scores > high_threshold + 0.15).float()  # AMPLIFY
        high_coherence = ((coherence_scores > high_threshold) & (coherence_scores <= high_threshold + 0.15)).float()  # DELIVER
        medium_coherence = ((coherence_scores >= low_threshold) & (coherence_scores <= high_threshold)).float()  # CLARIFY
        low_coherence = ((coherence_scores >= low_threshold * 0.5) & (coherence_scores < low_threshold)).float()  # ATTENUATE
        very_low_coherence = (coherence_scores < low_threshold * 0.5).float()  # ABSTAIN

        gate_decisions = torch.zeros_like(coherence_scores, dtype=torch.long)
        gate_decisions += (high_coherence * 0).long()      # DELIVER = 0
        gate_decisions += (very_low_coherence * 1).long()  # ABSTAIN = 1
        gate_decisions += (medium_coherence * 2).long()    # CLARIFY = 2
        gate_decisions += (very_high_coherence * 3).long() # AMPLIFY = 3
        gate_decisions += (low_coherence * 4).long()       # ATTENUATE = 4

        metrics = {
            'coherence_scores': coherence_scores,
            'high_coherence_ratio': (high_coherence + very_high_coherence).mean(),
            'low_coherence_ratio': (low_coherence + very_low_coherence).mean(),
            'very_high_ratio': very_high_coherence.mean(),
            'very_low_ratio': very_low_coherence.mean()
        }

        return gate_decisions, metrics


class ResonanceAnalyzer(nn.Module):
    """
    Analisador de ressonância para detectar interferência construtiva e destrutiva
    entre diferentes partes do texto, implementando a análise de "ondas" quaterniônicas.
    """

    def __init__(self, config: ResonanceConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim * 4

        # Projeção para espaço de análise de ressonância
        self.resonance_projection = nn.Linear(self.embed_dim, config.num_resonance_modes)

        # Parâmetros para análise de fase
        self.phase_analyzers = nn.ModuleList([
            nn.Linear(config.num_resonance_modes, 1) for _ in range(4)  # Para cada componente quaterniônico
        ])

        # Detector de interferência construtiva/destrutiva
        self.interference_detector = nn.Sequential(
            nn.Linear(config.num_resonance_modes * 2, config.num_resonance_modes),
            nn.Tanh(),
            nn.Linear(config.num_resonance_modes, 3),  # [construtiva, destrutiva, neutra]
            nn.Softmax(dim=-1)
        )

    def extract_quaternion_phases(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrai fases dos quaternions para análise de ressonância

        Args:
            x: Tensor [B, T, 4*D] em representação quaterniônica
        Returns:
            phases: [B, T, D, 4] fases de cada componente quaterniônico
        """
        batch_size, seq_len, embed_dim = x.shape
        quat_dim = embed_dim // 4

        # Reshape para [B, T, D, 4] (quaternion components)
        x_quat = x.view(batch_size, seq_len, quat_dim, 4)

        # Calcular fases usando atan2 para cada quaternion
        phases = torch.zeros_like(x_quat)

        # Fase principal: arctan2(|v|, w) onde v = [x, y, z], w = componente escalar
        w = x_quat[..., 0]  # Componente escalar
        v_magnitude = torch.norm(x_quat[..., 1:], p=2, dim=-1)  # |v|

        main_phase = torch.atan2(v_magnitude, torch.abs(w) + 1e-8)
        phases[..., 0] = main_phase

        # Fases individuais para componentes imaginárias
        for i in range(1, 4):
            phases[..., i] = torch.atan2(x_quat[..., i], w + 1e-8)

        return phases

    def calculate_resonance_matrix(self, phases1: torch.Tensor,
                                  phases2: torch.Tensor) -> torch.Tensor:
        """
        Calcula matriz de ressonância entre duas sequências de fases

        Args:
            phases1, phases2: Tensores de fase [B, T, D, 4]
        Returns:
            resonance_matrix: [B, T, T] matriz de correlação de fase
        """
        batch_size, seq_len = phases1.shape[:2]

        # Flatten para cálculo de correlação
        phases1_flat = phases1.view(batch_size, seq_len, -1)  # [B, T, D*4]
        phases2_flat = phases2.view(batch_size, seq_len, -1)

        # Produto escalar normalizado para correlação de fase
        norm1 = torch.nn.functional.normalize(phases1_flat, p=2, dim=-1)
        norm2 = torch.nn.functional.normalize(phases2_flat, p=2, dim=-1)

        # Matriz de correlação cruzada
        resonance_matrix = torch.bmm(norm1, norm2.transpose(1, 2))  # [B, T, T]

        return resonance_matrix

    def detect_interference_patterns(self, resonance_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Detecta padrões de interferência construtiva e destrutiva
        """
        batch_size, seq_len, _ = resonance_matrix.shape

        # Interferência construtiva: correlações altas (próximas de 1)
        constructive_mask = (resonance_matrix > self.config.constructive_threshold).float()

        # Interferência destrutiva: correlações baixas/negativas (próximas de -1)
        destructive_mask = (resonance_matrix < -self.config.interference_threshold).float()

        # Métricas de interferência
        constructive_strength = torch.mean(constructive_mask, dim=[1, 2])  # [B]
        destructive_strength = torch.mean(destructive_mask, dim=[1, 2])    # [B]

        # Coerência global: razão entre interferência construtiva e destrutiva
        global_coherence = constructive_strength / (destructive_strength + 1e-8)

        return {
            'constructive_mask': constructive_mask,
            'destructive_mask': destructive_mask,
            'constructive_strength': constructive_strength,
            'destructive_strength': destructive_strength,
            'global_coherence': global_coherence,
            'resonance_matrix': resonance_matrix
        }


class HierarchicalGateSystem(nn.Module):
    """
    Sistema hierárquico de Gate Controllers com análise de ressonância,
    implementando múltiplos níveis de validação e controle de qualidade.
    """

    def __init__(self, resonance_config: ResonanceConfig):
        super().__init__()
        self.config = resonance_config

        # Hierarquia de gates
        self.numerical_gate = NumericalGateController()
        self.local_gate = LocalCoherenceGate(resonance_config)
        self.resonance_analyzer = ResonanceAnalyzer(resonance_config)

        # Gate controller de alto nível para decisões finais
        self.global_gate_controller = nn.Sequential(
            nn.Linear(resonance_config.num_resonance_modes + 6, resonance_config.num_resonance_modes),
            nn.ReLU(),
            nn.Linear(resonance_config.num_resonance_modes, 5),  # 5 decisões possíveis
            nn.Softmax(dim=-1)
        )

        # Parâmetros adaptativos para mistura de decisões
        self.decision_weights = nn.Parameter(torch.ones(4) / 4)  # [numerical, local, global, semantic]

    def process_through_hierarchy(self,
                                 input_tensor: torch.Tensor,
                                 output_tensor: torch.Tensor,
                                 rotation_params: Dict[str, torch.Tensor]) -> Dict[str, Union[torch.Tensor, str, Dict]]:
        """
        Processa entrada através da hierarquia completa de gates

        Args:
            input_tensor: Tensor de entrada [B, T, 4*D]
            output_tensor: Tensor de saída [B, T, 4*D]
            rotation_params: Parâmetros de rotação do QRH layer

        Returns:
            Resultado hierárquico com decisões e métricas
        """
        batch_size, seq_len = input_tensor.shape[:2]
        hierarchy_result = {}

        # 1. Gate Numérico (Nível mais baixo)
        numerical_receipts = self.numerical_gate.calculate_receipts(
            input_tensor, output_tensor, rotation_params
        )
        numerical_stability = self.numerical_gate.validate_numerical_stability(output_tensor)
        numerical_decision = self.numerical_gate.enhanced_gate_decision(
            numerical_receipts, numerical_stability
        )

        hierarchy_result['numerical'] = {
            'decision': numerical_decision,
            'receipts': numerical_receipts,
            'stability_metrics': numerical_stability
        }

        # Se gate numérico falha, interromper processamento
        if numerical_decision == GateDecision.ABSTAIN:
            hierarchy_result['final_decision'] = GateDecision.ABSTAIN
            hierarchy_result['processed_output'] = input_tensor  # Retorna entrada original
            return hierarchy_result

        # 2. Gate de Coerência Local (Nível médio)
        local_decisions, local_metrics = self.local_gate.decide_local_gate(output_tensor)
        hierarchy_result['local'] = {
            'decisions': local_decisions,
            'metrics': local_metrics
        }

        # 3. Análise de Ressonância Global (Nível alto)
        phases = self.resonance_analyzer.extract_quaternion_phases(output_tensor)

        # Calcular ressonância consigo mesmo (auto-correlação) e com entrada
        self_resonance = self.resonance_analyzer.calculate_resonance_matrix(phases, phases)

        input_phases = self.resonance_analyzer.extract_quaternion_phases(input_tensor)
        cross_resonance = self.resonance_analyzer.calculate_resonance_matrix(phases, input_phases)

        # Detectar padrões de interferência
        self_interference = self.resonance_analyzer.detect_interference_patterns(self_resonance)
        cross_interference = self.resonance_analyzer.detect_interference_patterns(cross_resonance)

        hierarchy_result['resonance'] = {
            'self_interference': self_interference,
            'cross_interference': cross_interference,
            'phases': phases
        }

        # 4. Decisão Global Integrada
        global_decision, processed_output = self._make_global_decision(
            input_tensor, output_tensor, hierarchy_result
        )

        hierarchy_result['final_decision'] = global_decision
        hierarchy_result['processed_output'] = processed_output

        return hierarchy_result

    def _make_global_decision(self,
                             input_tensor: torch.Tensor,
                             output_tensor: torch.Tensor,
                             hierarchy_result: Dict) -> Tuple[GateDecision, torch.Tensor]:
        """
        Toma decisão global baseada em todos os níveis da hierarquia
        """
        batch_size = input_tensor.shape[0]

        # Extrair features para decisão global
        numerical_features = torch.tensor([
            hierarchy_result['numerical']['receipts']['orthogonal_error'],
            hierarchy_result['numerical']['receipts']['energy_ratio'],
            hierarchy_result['numerical']['stability_metrics']['condition_number']
        ]).expand(batch_size, -1)

        local_features = torch.stack([
            hierarchy_result['local']['metrics']['coherence_scores'].mean(dim=1),
            hierarchy_result['local']['metrics']['high_coherence_ratio'].expand(batch_size),
            hierarchy_result['local']['metrics']['low_coherence_ratio'].expand(batch_size)
        ], dim=1)

        resonance_features = torch.stack([
            hierarchy_result['resonance']['self_interference']['global_coherence'],
            hierarchy_result['resonance']['cross_interference']['global_coherence']
        ], dim=1)

        # Concatenar todas as features
        global_features = torch.cat([numerical_features, local_features, resonance_features], dim=1)

        # Decisão via rede neural
        decision_probs = self.global_gate_controller(global_features)  # [B, 5]

        # Converter para decisão categórica
        decision_indices = torch.argmax(decision_probs, dim=1)
        decision_map = [GateDecision.DELIVER, GateDecision.ABSTAIN, GateDecision.CLARIFY,
                       GateDecision.AMPLIFY, GateDecision.ATTENUATE]

        # Para simplificar, usar decisão majoritária ao longo do batch
        majority_decision_idx = torch.mode(decision_indices).values.item()
        final_decision = decision_map[majority_decision_idx]

        # Aplicar política da decisão
        processed_output = self._apply_hierarchical_policy(
            final_decision, input_tensor, output_tensor, hierarchy_result
        )

        return final_decision, processed_output

    def _apply_hierarchical_policy(self,
                                  decision: GateDecision,
                                  input_tensor: torch.Tensor,
                                  output_tensor: torch.Tensor,
                                  hierarchy_result: Dict) -> torch.Tensor:
        """
        Aplica política baseada na decisão hierárquica
        """
        if decision == GateDecision.DELIVER:
            return output_tensor

        elif decision == GateDecision.ABSTAIN:
            return input_tensor

        elif decision == GateDecision.CLARIFY:
            # Mistura ponderada baseada em confiança local
            local_coherence = hierarchy_result['local']['metrics']['coherence_scores']
            confidence_weight = local_coherence.mean(dim=1, keepdim=True).unsqueeze(-1)
            return confidence_weight * output_tensor + (1 - confidence_weight) * input_tensor

        elif decision == GateDecision.AMPLIFY:
            # Amplifica baseado em ressonância construtiva - OTIMIZADO
            if 'resonance' in hierarchy_result and 'self_interference' in hierarchy_result['resonance']:
                constructive_strength = hierarchy_result['resonance']['self_interference']['constructive_strength']
                amplification_factor = self.config.resonance_amplification_factor * (
                    1.0 + constructive_strength.unsqueeze(1).unsqueeze(2)
                )
            else:
                amplification_factor = self.config.resonance_amplification_factor
            return amplification_factor * output_tensor

        elif decision == GateDecision.ATTENUATE:
            # Atenua baseado em interferência destrutiva - OTIMIZADO
            if 'resonance' in hierarchy_result and 'self_interference' in hierarchy_result['resonance']:
                destructive_strength = hierarchy_result['resonance']['self_interference']['destructive_strength']
                attenuation_factor = self.config.resonance_attenuation_factor * (
                    1.0 - 0.2 * destructive_strength.unsqueeze(1).unsqueeze(2)
                )
            else:
                attenuation_factor = self.config.resonance_attenuation_factor
            return attenuation_factor * output_tensor

        else:
            return output_tensor

    def get_hierarchy_health_report(self, hierarchy_result: Dict) -> Dict[str, float]:
        """
        Gera relatório de saúde do sistema hierárquico
        """
        health_report = {}

        # Saúde numérica
        numerical_health = 1.0 - float(hierarchy_result['numerical']['decision'] == GateDecision.ABSTAIN)
        health_report['numerical_health'] = numerical_health

        # Saúde de coerência local
        if 'local' in hierarchy_result and 'metrics' in hierarchy_result['local']:
            local_coherence_avg = hierarchy_result['local']['metrics']['coherence_scores'].mean().item()
            health_report['local_coherence_health'] = local_coherence_avg
        else:
            health_report['local_coherence_health'] = 0.5  # Default value

        # Saúde de ressonância global
        if 'resonance' in hierarchy_result and 'self_interference' in hierarchy_result['resonance']:
            global_resonance = hierarchy_result['resonance']['self_interference']['global_coherence'].mean().item()
            health_report['global_resonance_health'] = min(global_resonance, 2.0) / 2.0  # Normalizar

            # Métricas de interferência
            constructive_ratio = hierarchy_result['resonance']['self_interference']['constructive_strength'].mean().item()
            destructive_ratio = hierarchy_result['resonance']['self_interference']['destructive_strength'].mean().item()

            health_report['constructive_interference'] = constructive_ratio
            health_report['destructive_interference'] = destructive_ratio
            health_report['interference_balance'] = constructive_ratio / (destructive_ratio + 1e-8)
        else:
            health_report['global_resonance_health'] = 0.5  # Default value
            health_report['constructive_interference'] = 0.3
            health_report['destructive_interference'] = 0.3
            health_report['interference_balance'] = 1.0

        # Saúde geral do sistema
        local_coherence_value = health_report.get('local_coherence_health', 0.5)
        global_resonance_value = health_report.get('global_resonance_health', 0.5)

        health_report['overall_hierarchy_health'] = (
            0.3 * numerical_health +
            0.4 * local_coherence_value +
            0.3 * global_resonance_value
        )

        return health_report
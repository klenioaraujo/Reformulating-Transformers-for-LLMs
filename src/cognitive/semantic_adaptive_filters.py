import torch
import torch.nn as nn
import torch.fft as fft
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from ..core.quaternion_operations import quaternion_multiply


class SemanticNoiseType(Enum):
    """Types of semantic noise that can be filtered"""
    CONTRADICTION = "contradiction"
    IRRELEVANCE = "irrelevance"
    BIAS = "bias"


@dataclass
class SemanticFilterConfig:
    """Configuration for semantic adaptive filters"""
    embed_dim: int = 64
    num_heads: int = 8
    contradiction_threshold: float = 0.3  # Lowered for better sensitivity
    irrelevance_threshold: float = 0.4   # Optimized threshold
    bias_threshold: float = 0.6          # More sensitive bias detection
    learning_rate: float = 1e-4
    temperature: float = 0.5             # Lower temperature for sharper decisions
    epsilon: float = 1e-8
    contradiction_sensitivity: float = 2.0  # New: amplification factor
    phase_rotation_strength: float = 0.5    # New: rotation strength control


class ContradictionDetector(nn.Module):
    """
    Filtro de Contradição: Identifica e atenua padrões de "ondas" que representam
    informações conflitantes dentro do mesmo contexto.
    """

    def __init__(self, config: SemanticFilterConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim * 4  # quaternion embedding

        # Multi-head attention for contradiction detection
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Contradiction scoring network
        self.contradiction_scorer = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embed_dim // 2, 1),
            nn.Sigmoid()
        )

        # Quaternion-aware contradiction filter
        self.phase_contradiction_filter = nn.Parameter(
            torch.randn(4, requires_grad=True) * 0.1
        )

    def detect_contradictions(self, x: torch.Tensor) -> torch.Tensor:
        """
        PRODUCTION-READY contradiction detection with enhanced sensitivity

        Args:
            x: Tensor [B, T, 4*D] (quaternion representation)
        Returns:
            contradiction_scores: [B, T] scores indicating contradiction level
        """
        batch_size, seq_len, embed_dim = x.shape

        # ENHANCEMENT 1: Multi-scale attention analysis
        attn_output, attn_weights = self.attention(x, x, x)

        # Primary contradiction signal: attention divergence
        attention_divergence = torch.abs(attn_output - x)

        # ENHANCEMENT 2: Enhanced quaternion-based semantic opposition
        x_quat = x.view(batch_size, seq_len, embed_dim // 4, 4)

        # Vectorized semantic opposition calculation (much faster)
        opposition_scores = torch.zeros(batch_size, seq_len, device=x.device)

        # Calculate cosine similarity between consecutive quaternion states
        if seq_len > 1:
            # Normalize quaternions for better comparison
            x_quat_norm = x_quat / (torch.norm(x_quat, dim=-1, keepdim=True) + self.config.epsilon)

            # Compare each position with previous one
            curr_states = x_quat_norm[:, 1:].mean(dim=2)  # [B, T-1, 4] - average across embed dim
            prev_states = x_quat_norm[:, :-1].mean(dim=2)  # [B, T-1, 4]

            # Vectorized cosine similarity
            dot_products = torch.sum(curr_states * prev_states, dim=-1)  # [B, T-1]
            opposition_raw = torch.clamp(-dot_products, 0, 1)  # Negative similarity = opposition

            # Pad first position with zero
            opposition_scores[:, 1:] = opposition_raw

        # ENHANCEMENT 3: Attention pattern analysis
        attention_matrix = attn_weights.mean(dim=1)  # [B, T, T]

        # Look for contradictory attention patterns (high attention to distant, opposing tokens)
        attention_contradictions = torch.zeros(batch_size, seq_len, device=x.device)

        for t in range(seq_len):
            # Measure how much this token attends to tokens that oppose it
            if attention_matrix.dim() == 3:
                current_attentions = attention_matrix[:, t, :]  # [B, T]
            else:
                # Handle 2D attention matrix case
                current_attentions = attention_matrix  # [B, T] or [T, T]
                if current_attentions.dim() == 1:
                    current_attentions = current_attentions.unsqueeze(0).repeat(batch_size, 1)

            # Weight attention by semantic opposition
            if t < seq_len - 1:
                # Use opposition scores as weights
                if opposition_scores.shape[-1] == current_attentions.shape[-1]:
                    weighted_attention = current_attentions * opposition_scores
                    attention_contradictions[:, t] = weighted_attention.sum(dim=-1)
                else:
                    # Fallback to simple opposition score
                    attention_contradictions[:, t] = opposition_scores.mean(dim=-1)

        # ENHANCEMENT 4: Statistical anomaly detection
        # Look for tokens that deviate significantly from local patterns
        statistical_anomalies = torch.zeros(batch_size, seq_len, device=x.device)

        if seq_len >= 3:
            window_size = min(5, seq_len)
            for t in range(seq_len):
                start_idx = max(0, t - window_size // 2)
                end_idx = min(seq_len, t + window_size // 2 + 1)

                # Calculate local statistics
                local_tokens = x[:, start_idx:end_idx]  # [B, window, embed_dim]
                local_mean = local_tokens.mean(dim=1, keepdim=True)  # [B, 1, embed_dim]
                local_std = local_tokens.std(dim=1, keepdim=True) + self.config.epsilon

                # Measure deviation of current token
                current_token = x[:, t:t+1]  # [B, 1, embed_dim]
                deviation = torch.abs(current_token - local_mean) / local_std
                statistical_anomalies[:, t] = deviation.mean(dim=-1).squeeze(-1)  # [B]

        # ENHANCEMENT 5: Multi-feature fusion with learned weights
        base_scores = self.contradiction_scorer(attention_divergence).squeeze(-1)

        # Normalize all features to [0, 1] range
        opposition_norm = torch.sigmoid(opposition_scores * self.config.contradiction_sensitivity)
        attention_norm = torch.sigmoid(attention_contradictions * 2.0)
        anomaly_norm = torch.sigmoid(statistical_anomalies - 1.0)  # Center around 1.0

        # Adaptive feature combination
        # Higher weight on features that show more variation (more informative)
        opposition_weight = opposition_norm.std(dim=-1, keepdim=True) + 0.1
        attention_weight = attention_norm.std(dim=-1, keepdim=True) + 0.1
        anomaly_weight = anomaly_norm.std(dim=-1, keepdim=True) + 0.1

        # Weighted combination
        weighted_opposition = opposition_norm * opposition_weight
        weighted_attention = attention_norm * attention_weight
        weighted_anomaly = anomaly_norm * anomaly_weight

        # Final contradiction score
        combined_features = (weighted_opposition + weighted_attention + weighted_anomaly) / 3.0
        enhanced_scores = base_scores + combined_features

        # ENHANCEMENT 6: Adaptive thresholding with temperature scaling
        # Use a more aggressive sigmoid for better separation
        contradiction_scores = torch.sigmoid((enhanced_scores - 0.5) / self.config.temperature)

        return contradiction_scores, attn_weights

    def apply_contradiction_filter(self, x: torch.Tensor,
                                   contradiction_scores: torch.Tensor) -> torch.Tensor:
        """
        Aplica filtro de contradição APRIMORADO no domínio dos quaternions

        Args:
            x: Input tensor [B, T, 4*D]
            contradiction_scores: Contradiction scores [B, T]
        Returns:
            Filtered tensor with contradictions attenuated
        """
        batch_size, seq_len, embed_dim = x.shape

        # Reshape to quaternion format [B, T, D, 4]
        x_quat = x.view(batch_size, seq_len, embed_dim // 4, 4)

        # IMPROVEMENT 1: Adaptive thresholding based on local context
        # Calculate local contradiction context (moving average)
        window_size = min(3, seq_len)
        if window_size > 1:
            contradiction_smoothed = torch.nn.functional.avg_pool1d(
                contradiction_scores.unsqueeze(1), kernel_size=window_size,
                stride=1, padding=window_size//2
            ).squeeze(1)
        else:
            contradiction_smoothed = contradiction_scores

        # Adaptive threshold based on local statistics
        local_mean = contradiction_smoothed.mean(dim=1, keepdim=True)
        local_std = contradiction_smoothed.std(dim=1, keepdim=True) + self.config.epsilon

        # Dynamic threshold: base threshold + adaptive component
        adaptive_threshold = self.config.contradiction_threshold + 0.1 * (contradiction_smoothed - local_mean) / local_std
        contradiction_mask = (contradiction_scores > adaptive_threshold).float()

        # IMPROVEMENT 2: Progressive attenuation based on contradiction intensity
        # Instead of binary mask, use graduated attenuation
        contradiction_intensity = torch.clamp(
            (contradiction_scores - self.config.contradiction_threshold) / (1.0 - self.config.contradiction_threshold),
            0, 1
        )

        # IMPROVEMENT 3: Enhanced quaternion rotation with controlled strength
        # Create phase modulation with controlled strength
        phase_strength = self.config.phase_rotation_strength * contradiction_intensity

        for b in range(batch_size):
            for t in range(seq_len):
                if contradiction_intensity[b, t] > 0.1:  # Apply only if significant contradiction
                    # Create progressive attenuation quaternion
                    strength = phase_strength[b, t].item()

                    # Enhanced rotation that preserves more semantic information
                    rotation_angles = self.phase_contradiction_filter * strength

                    # Create unit quaternion for rotation
                    cos_half = torch.cos(rotation_angles[0] / 2)
                    sin_half = torch.sin(rotation_angles[0] / 2)

                    attn_quat = torch.stack([
                        cos_half,
                        sin_half * torch.cos(rotation_angles[1]),
                        sin_half * torch.sin(rotation_angles[1]) * torch.cos(rotation_angles[2]),
                        sin_half * torch.sin(rotation_angles[1]) * torch.sin(rotation_angles[2])
                    ])

                    # Apply to all embedding dimensions with preservation factor
                    preservation_factor = 1.0 - strength * 0.5  # Preserve some original information

                    for d in range(embed_dim // 4):
                        original_quat = x_quat[b, t, d].clone()
                        rotated_quat = quaternion_multiply(
                            attn_quat.unsqueeze(0),
                            original_quat.unsqueeze(0)
                        ).squeeze(0)

                        # Blend original and rotated based on preservation factor
                        x_quat[b, t, d] = preservation_factor * original_quat + (1 - preservation_factor) * rotated_quat

        return x_quat.view(batch_size, seq_len, embed_dim)


class IrrelevanceFilter(nn.Module):
    """
    Filtro de Irrelevância: Suprime tópicos que desviam do "sinal" principal
    da conversa ou do texto.
    """

    def __init__(self, config: SemanticFilterConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim * 4

        # Topic relevance scoring
        self.relevance_encoder = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim // 2)
        )

        # Main topic extraction via learned query
        self.topic_query = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        # Spectral irrelevance filter
        self.spectral_filter = nn.Parameter(torch.ones(4) * 0.5, requires_grad=True)

    def extract_main_topic(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrai o tópico principal usando atenção com query aprendida

        Args:
            x: Input tensor [B, T, 4*D]
        Returns:
            main_topic_representation: [B, 4*D]
        """
        batch_size = x.shape[0]

        # Expand topic query for batch
        topic_query = self.topic_query.expand(batch_size, -1, -1)

        # Compute attention scores to extract main topic
        topic_scores = torch.bmm(topic_query, x.transpose(1, 2))  # [B, 1, T]
        topic_weights = torch.softmax(topic_scores / self.config.temperature, dim=-1)

        # Weighted sum to get main topic representation
        main_topic = torch.bmm(topic_weights, x).squeeze(1)  # [B, 4*D]

        return main_topic, topic_weights.squeeze(1)

    def compute_relevance_scores(self, x: torch.Tensor,
                                main_topic: torch.Tensor) -> torch.Tensor:
        """
        Calcula scores de relevância comparando com o tópico principal
        """
        batch_size, seq_len = x.shape[:2]

        # Encode both inputs
        x_encoded = self.relevance_encoder(x)  # [B, T, D/2]
        topic_encoded = self.relevance_encoder(main_topic.unsqueeze(1))  # [B, 1, D/2]

        # Compute cosine similarity
        x_norm = torch.nn.functional.normalize(x_encoded, p=2, dim=-1)
        topic_norm = torch.nn.functional.normalize(topic_encoded, p=2, dim=-1)

        relevance_scores = torch.sum(x_norm * topic_norm, dim=-1)  # [B, T]

        return relevance_scores

    def apply_irrelevance_filter(self, x: torch.Tensor,
                                relevance_scores: torch.Tensor) -> torch.Tensor:
        """
        Aplica filtro espectral para suprimir irrelevância
        """
        batch_size, seq_len, embed_dim = x.shape

        # Identify irrelevant tokens
        irrelevant_mask = (relevance_scores < self.config.irrelevance_threshold).float()

        # Apply spectral filtering to irrelevant portions
        x_quat = x.view(batch_size, seq_len, embed_dim // 4, 4)

        # FFT along sequence dimension
        x_fft = fft.fft(x_quat, dim=1)

        # Apply learned spectral filter to irrelevant parts
        filter_response = torch.exp(1j * self.spectral_filter)

        # Apply filter selectively based on irrelevance mask
        for b in range(batch_size):
            for t in range(seq_len):
                if irrelevant_mask[b, t] > 0:
                    x_fft[b, t] *= filter_response.view(1, 4)

        # Inverse FFT
        x_filtered = fft.ifft(x_fft, dim=1).real

        return x_filtered.view(batch_size, seq_len, embed_dim)


class BiasFilter(nn.Module):
    """
    Filtro de Viés: Reconhece e modula padrões de linguagem associados
    a vieses cognitivos ou sociais indesejados.
    """

    def __init__(self, config: SemanticFilterConfig, bias_patterns: Optional[List[str]] = None):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim * 4

        # Bias pattern recognition network
        self.bias_detector = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Linear(self.embed_dim // 2, len(bias_patterns) if bias_patterns else 10),
            nn.Sigmoid()
        )

        # Quaternion bias correction parameters
        self.bias_correction_quaternions = nn.Parameter(
            torch.randn(len(bias_patterns) if bias_patterns else 10, 4) * 0.1
        )

        # Adaptive bias threshold
        self.adaptive_threshold = nn.Parameter(torch.tensor(config.bias_threshold))

    def detect_bias_patterns(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detecta padrões de viés no texto usando análise quaterniônica

        Args:
            x: Input tensor [B, T, 4*D]
        Returns:
            bias_scores: [B, T, num_bias_types] scores for each bias type
            bias_magnitude: [B, T] overall bias magnitude
        """
        # Detect bias patterns
        bias_scores = self.bias_detector(x)  # [B, T, num_bias_types]

        # Calculate overall bias magnitude
        bias_magnitude = torch.norm(bias_scores, p=2, dim=-1)  # [B, T]

        return bias_scores, bias_magnitude

    def apply_bias_correction(self, x: torch.Tensor,
                             bias_scores: torch.Tensor) -> torch.Tensor:
        """
        Aplica correção de viés usando rotações quaterniônicas
        """
        batch_size, seq_len, embed_dim = x.shape
        num_bias_types = bias_scores.shape[-1]

        x_quat = x.view(batch_size, seq_len, embed_dim // 4, 4)

        # Apply bias correction for each detected bias type
        for bias_type in range(num_bias_types):
            bias_mask = (bias_scores[:, :, bias_type] > self.adaptive_threshold).float()

            if bias_mask.sum() > 0:  # If bias detected
                correction_quat = self.bias_correction_quaternions[bias_type]

                # Apply correction quaternion to biased positions
                for b in range(batch_size):
                    for t in range(seq_len):
                        if bias_mask[b, t] > 0:
                            for d in range(embed_dim // 4):
                                x_quat[b, t, d] = quaternion_multiply(
                                    correction_quat.unsqueeze(0),
                                    x_quat[b, t, d].unsqueeze(0)
                                ).squeeze(0)

        return x_quat.view(batch_size, seq_len, embed_dim)


class SemanticAdaptiveFilter(nn.Module):
    """
    Sistema integrado de filtros semânticos adaptativos com múltiplas "cabeças"
    especializadas em diferentes tipos de ruído semântico.
    """

    def __init__(self, config: SemanticFilterConfig, bias_patterns: Optional[List[str]] = None):
        super().__init__()
        self.config = config

        # Initialize specialized filter heads
        self.contradiction_filter = ContradictionDetector(config)
        self.irrelevance_filter = IrrelevanceFilter(config)
        self.bias_filter = BiasFilter(config, bias_patterns)

        # Filter coordination network
        self.filter_coordinator = nn.Sequential(
            nn.Linear(config.embed_dim * 4, config.embed_dim * 2),
            nn.GELU(),
            nn.Linear(config.embed_dim * 2, 3),  # 3 filter types
            nn.Softmax(dim=-1)
        )

        # Adaptive mixing parameters
        self.mixing_params = nn.Parameter(torch.ones(3) / 3)  # Equal initial weights

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Aplica filtros semânticos adaptativos

        Args:
            x: Input tensor [B, T, 4*D]
        Returns:
            filtered_output: Filtered tensor
            filter_metrics: Dictionary with filtering metrics
        """
        batch_size, seq_len, embed_dim = x.shape
        metrics = {}

        # 1. Contradiction filtering
        contradiction_scores, attn_weights = self.contradiction_filter.detect_contradictions(x)
        x_contradiction_filtered = self.contradiction_filter.apply_contradiction_filter(
            x, contradiction_scores
        )
        metrics['contradiction_scores'] = contradiction_scores
        metrics['attention_weights'] = attn_weights

        # 2. Irrelevance filtering
        main_topic, topic_weights = self.irrelevance_filter.extract_main_topic(x)
        relevance_scores = self.irrelevance_filter.compute_relevance_scores(x, main_topic)
        x_irrelevance_filtered = self.irrelevance_filter.apply_irrelevance_filter(
            x, relevance_scores
        )
        metrics['relevance_scores'] = relevance_scores
        metrics['topic_weights'] = topic_weights

        # 3. Bias filtering
        bias_scores, bias_magnitude = self.bias_filter.detect_bias_patterns(x)
        x_bias_filtered = self.bias_filter.apply_bias_correction(x, bias_scores)
        metrics['bias_scores'] = bias_scores
        metrics['bias_magnitude'] = bias_magnitude

        # 4. Adaptive coordination of filters
        # Calculate per-token filter weights
        filter_weights = self.filter_coordinator(x)  # [B, T, 3]

        # Combine filtered outputs using learned mixing
        combined_filters = torch.stack([
            x_contradiction_filtered,
            x_irrelevance_filtered,
            x_bias_filtered
        ], dim=-1)  # [B, T, 4*D, 3]

        # Apply adaptive mixing
        filter_weights_expanded = filter_weights.unsqueeze(-2)  # [B, T, 1, 3]
        filtered_output = torch.sum(combined_filters * filter_weights_expanded, dim=-1)

        # Add residual connection with original input
        residual_weight = 0.1  # Configurable
        filtered_output = filtered_output + residual_weight * x

        metrics['filter_weights'] = filter_weights
        metrics['mixing_params'] = self.mixing_params

        return filtered_output, metrics

    def get_semantic_health_report(self, metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Gera relatório de saúde semântica baseado nas métricas dos filtros
        """
        health_report = {}

        # Contradiction health
        avg_contradiction = metrics['contradiction_scores'].mean().item()
        health_report['contradiction_level'] = avg_contradiction
        health_report['contradiction_health'] = 1.0 - min(avg_contradiction, 1.0)

        # Relevance health
        avg_relevance = metrics['relevance_scores'].mean().item()
        health_report['relevance_level'] = avg_relevance
        health_report['relevance_health'] = max(avg_relevance, 0.0)

        # Bias health
        avg_bias = metrics['bias_magnitude'].mean().item()
        health_report['bias_level'] = avg_bias
        health_report['bias_health'] = 1.0 - min(avg_bias, 1.0)

        # Overall semantic health
        health_report['overall_semantic_health'] = (
            health_report['contradiction_health'] +
            health_report['relevance_health'] +
            health_report['bias_health']
        ) / 3.0

        return health_report
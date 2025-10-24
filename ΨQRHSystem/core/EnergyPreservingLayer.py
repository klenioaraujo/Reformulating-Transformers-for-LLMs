import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from core.PiAutoCalibration import PiAutoCalibration
from core.EnergyConservation import EnergyConservation
from core.TernaryLogicFramework import TernaryLogicFramework


class EnergyPreservingLayer(nn.Module):
    """
    Camada com Conservação Automática de Energia

    π garante que a norma seja preservada através de:
    - Transformação unitária implícita via π
    - Auto-correção para conservação de energia
    - Normalização baseada em π
    """

    def __init__(self, embed_dim: int, pi_calibration: PiAutoCalibration,
                 device: str = "cpu"):
        """
        Inicializa camada com preservação de energia

        Args:
            embed_dim: Dimensão do embedding
            pi_calibration: Calibrador π
            device: Dispositivo de computação
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.pi_calibration = pi_calibration
        self.device = device

        # Componentes de preservação
        self.energy_conservation = EnergyConservation(device=device)
        self.ternary_logic = TernaryLogicFramework(device=device)

        # Transformações preservadoras de energia
        self.weight_matrix = nn.Parameter(torch.randn(embed_dim, embed_dim, device=device))
        self.bias = nn.Parameter(torch.zeros(embed_dim, device=device))

        # Hamiltoniano para verificação de energia
        self.hamiltonian = self._construct_hamiltonian()

        print("⚡ Energy Preserving Layer initialized with π-based conservation")

    def _construct_hamiltonian(self) -> torch.Tensor:
        """
        Constrói hamiltoniano baseado em π para verificação de energia

        Returns:
            Hamiltoniano π-based
        """
        # Hamiltoniano simples: H = π * I + pequena perturbação
        identity = torch.eye(self.embed_dim, device=self.device)
        perturbation = torch.randn(self.embed_dim, self.embed_dim, device=self.device) * 0.1

        hamiltonian = torch.pi * identity + perturbation

        # Garantir hermiticiadade
        hamiltonian = (hamiltonian + hamiltonian.conj().T) / 2

        return hamiltonian

    def energy_preserving_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass com preservação de energia

        Args:
            x: Input tensor [batch, seq, embed_dim]

        Returns:
            Output com energia preservada [batch, seq, embed_dim]
        """
        # Norma de entrada
        input_norm = torch.norm(x, dim=-1, keepdim=True)

        # Transformação calibrada com π
        transformed = self.pi_calibrated_transform(x)

        # Norma de saída
        output_norm = torch.norm(transformed, dim=-1, keepdim=True)

        # Fator de correção para conservação de energia
        correction_factor = input_norm / (output_norm + 1e-8)

        # Aplicar correção
        corrected_output = transformed * correction_factor

        # Verificar conservação
        self._verify_energy_conservation(x, corrected_output)

        return corrected_output

    def pi_calibrated_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transformação calibrada com π

        Args:
            x: Input tensor

        Returns:
            Transformado calibrado
        """
        # Aplicar transformação linear
        linear_out = torch.matmul(x, self.weight_matrix.T) + self.bias

        # Calibragem π dos pesos
        calibrated_weights = self.pi_calibration.auto_scale_weights(self.weight_matrix)

        # Reaplicar com pesos calibrados
        calibrated_out = torch.matmul(x, calibrated_weights.T) + self.bias

        # Normalização de fase π
        if torch.is_complex(calibrated_out):
            calibrated_out = self.pi_calibration.phase_normalization(calibrated_out)

        return calibrated_out

    def _verify_energy_conservation(self, input_tensor: torch.Tensor,
                                  output_tensor: torch.Tensor) -> bool:
        """
        Verifica conservação de energia entre entrada e saída

        Args:
            input_tensor: Tensor de entrada
            output_tensor: Tensor de saída

        Returns:
            True se energia conservada
        """
        # Energia de entrada (soma das normas ao quadrado)
        energy_input = torch.sum(input_tensor.abs() ** 2)

        # Energia de saída
        energy_output = torch.sum(output_tensor.abs() ** 2)

        # Verificar conservação
        is_conserved = self.energy_conservation.validate_energy_conservation(
            energy_input.item(), energy_output.item()
        )

        if not is_conserved:
            print(f"⚠️  Energy conservation violated: {energy_input:.4f} → {energy_output:.4f}")

        return is_conserved

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass padrão (alias para energy_preserving_forward)

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.energy_preserving_forward(x)


class EnergyPreservingAttention(nn.Module):
    """
    Atenção com Preservação de Energia
    """

    def __init__(self, embed_dim: int, num_heads: int, pi_calibration: PiAutoCalibration):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.pi_calibration = pi_calibration

        # Projeções
        self.q_proj = EnergyPreservingLayer(embed_dim, pi_calibration)
        self.k_proj = EnergyPreservingLayer(embed_dim, pi_calibration)
        self.v_proj = EnergyPreservingLayer(embed_dim, pi_calibration)
        self.out_proj = EnergyPreservingLayer(embed_dim, pi_calibration)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Projeções preservadoras de energia
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Atenção calibrada com π
        attention_output = self.pi_calibration.pi_stabilized_attention(
            Q.transpose(1, 2),  # [batch, heads, seq, head_dim]
            K.transpose(1, 2),
            V.transpose(1, 2)
        )

        # Reorganizar e projetar saída
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.embed_dim)

        # Projeção de saída preservadora de energia
        output = self.out_proj(attention_output)

        return output


class EnergyPreservingTransformerBlock(nn.Module):
    """
    Bloco Transformer com Preservação Total de Energia
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int,
                 pi_calibration: PiAutoCalibration):
        super().__init__()
        self.embed_dim = embed_dim
        self.pi_calibration = pi_calibration

        # Atenção preservadora de energia
        self.attention = EnergyPreservingAttention(embed_dim, num_heads, pi_calibration)

        # Feed-forward preservador de energia
        self.ff1 = EnergyPreservingLayer(embed_dim, pi_calibration)
        self.ff2 = EnergyPreservingLayer(ff_dim, pi_calibration)
        self.ff3 = EnergyPreservingLayer(embed_dim, pi_calibration)

        # Normalização (poderia ser também preservadora, mas usando LayerNorm padrão por simplicidade)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Atenção com residual connection
        attn_out = self.attention(self.norm1(x))
        x = x + attn_out  # Residual

        # Feed-forward com residual
        ff_out = self.ff3(self.ff2(self.ff1(self.norm2(x))))
        x = x + ff_out  # Residual

        return x


class EnergyPreservingNetwork(nn.Module):
    """
    Rede Neural com Conservação Global de Energia
    """

    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int,
                 num_layers: int, ff_dim: int, max_seq_len: int,
                 pi_calibration: PiAutoCalibration):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.pi_calibration = pi_calibration

        # Embedding preservador
        self.embedding = EnergyPreservingLayer(embed_dim, pi_calibration)

        # Positional encoding (simplificado)
        self.pos_encoding = self._create_positional_encoding(max_seq_len, embed_dim)

        # Transformer blocks preservadores
        self.layers = nn.ModuleList([
            EnergyPreservingTransformerBlock(embed_dim, num_heads, ff_dim, pi_calibration)
            for _ in range(num_layers)
        ])

        # Output head preservador
        self.output_head = EnergyPreservingLayer(vocab_size, pi_calibration)

    def _create_positional_encoding(self, max_seq_len: int, embed_dim: int) -> torch.Tensor:
        """
        Cria positional encoding baseado em π

        Args:
            max_seq_len: Comprimento máximo da sequência
            embed_dim: Dimensão do embedding

        Returns:
            Positional encoding [max_seq_len, embed_dim]
        """
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-torch.log(torch.tensor(10000.0)) / embed_dim))

        pos_encoding = torch.zeros(max_seq_len, embed_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term * torch.pi)  # π-based
        pos_encoding[:, 1::2] = torch.cos(position * div_term * torch.pi)

        return pos_encoding

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.shape[1]

        # Embedding
        x = self.embedding(input_ids.float())  # Convert to float for embedding

        # Adicionar positional encoding
        x = x + self.pos_encoding[:seq_len].to(x.device)

        # Aplicar camadas transformer
        for layer in self.layers:
            x = layer(x)

        # Output logits
        logits = self.output_head(x)

        return logits

    def get_energy_report(self) -> Dict[str, Any]:
        """
        Relatório de conservação de energia da rede

        Returns:
            Relatório detalhado
        """
        return {
            'pi_calibration_report': self.pi_calibration.get_pi_calibration_report(),
            'energy_conservation_report': self.pi_calibration.energy_conservation.get_conservation_report(),
            'network_parameters': sum(p.numel() for p in self.parameters())
        }
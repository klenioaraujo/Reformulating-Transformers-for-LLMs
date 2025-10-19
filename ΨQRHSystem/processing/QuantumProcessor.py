#!/usr/bin/env python3
"""
QuantumProcessor - Aplica transformações (filtros, rotações) ao estado quântico Ψ.
"""

import torch
from typing import Dict, Any, Optional

# Importar componentes necessários que antes estavam em psiqrh.py
# Em uma arquitetura real, eles viriam de seus próprios módulos.
try:
    from src.core.physical_fundamental_corrections import PhysicalHarmonicOrchestrator
    from src.core.quaternion_operations import OptimizedQuaternionOperations
except ImportError:
    PhysicalHarmonicOrchestrator = None
    OptimizedQuaternionOperations = None

class QuantumProcessor:
    """
    Responsável pela terceira etapa do pipeline: processamento do estado quântico Ψ.
    """
    def __init__(self, device: str = 'cpu', orchestrator: Optional[Any] = None):
        """
        Inicializa o processador quântico.

        Args:
            device: O dispositivo computacional (ex: 'cpu', 'cuda').
            orchestrator: Uma instância opcional do PhysicalHarmonicOrchestrator.
        """
        self.device = device
        self.orchestrator = orchestrator
        
        if OptimizedQuaternionOperations:
            self.quaternion_ops = OptimizedQuaternionOperations(device=self.device)
        else:
            self.quaternion_ops = None

        if self.orchestrator and self.quaternion_ops:
            print("✅ QuantumProcessor inicializado com Orquestrador e Operações Otimizadas.")
        else:
            print("✅ QuantumProcessor inicializado.")

    def apply_spectral_filtering(self, psi: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Aplica filtragem espectral ao estado Ψ.
        (Lógica migrada de psiqrh.py: _apply_spectral_filtering)
        """
        batch_size, seq_len, embed_dim, quat_dim = psi.shape

        if self.orchestrator:
            def base_spectral_filter(psi_tensor, alpha_param, resonance_mask=None, **kwargs):
                psi_fft = torch.fft.fft(psi_tensor, dim=2)
                if resonance_mask is not None:
                    if resonance_mask.shape[0] < embed_dim:
                        padding = torch.ones(embed_dim - resonance_mask.shape[0], device=resonance_mask.device)
                        resonance_mask = torch.cat([resonance_mask, padding])
                    mask_expanded = resonance_mask.view(1, 1, -1, 1).expand_as(psi_fft)
                    psi_fft = psi_fft * mask_expanded
                
                freqs = torch.fft.fftfreq(embed_dim, device=self.device)
                k = 2 * torch.pi * freqs.view(1, 1, -1, 1)
                filter_response = torch.exp(1j * alpha_param * torch.arctan(torch.log(torch.abs(k) + 1e-10)))
                psi_filtered_fft = psi_fft * filter_response.expand_as(psi_fft)
                return torch.fft.ifft(psi_filtered_fft, dim=2).real

            psi_filtered = self.orchestrator.orchestrate_transformation(
                psi.mean(dim=(0, 1, 3)),
                'spectral_filter',
                base_spectral_filter,
                psi_tensor=psi, alpha_param=alpha
            )
        else:
            psi_fft = torch.fft.fft(psi, dim=2)
            freqs = torch.fft.fftfreq(embed_dim, device=self.device)
            k = 2 * torch.pi * freqs.view(1, 1, -1, 1)
            filter_response = torch.exp(1j * alpha * torch.arctan(torch.log(torch.abs(k) + 1e-10)))
            psi_filtered_fft = psi_fft * filter_response.expand_as(psi_fft)
            psi_filtered = torch.fft.ifft(psi_filtered_fft, dim=2).real

        return psi_filtered

    def apply_so4_rotation(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Aplica rotações SO(4) unitárias ao estado Ψ.
        (Lógica migrada de psiqrh.py: _apply_so4_rotation)
        """
        if not self.quaternion_ops:
            # Fallback para operações básicas se otimizadas não estiverem disponíveis
            return psi

        if self.orchestrator:
            def base_so4_rotation(psi_tensor, rotation_angles, **kwargs):
                return self.quaternion_ops.so4_rotation(psi_tensor, rotation_angles)

            psi_rotated = self.orchestrator.orchestrate_transformation(
                psi.mean(dim=(0, 2, 3)),
                'so4_rotation',
                base_so4_rotation,
                psi_tensor=psi
            )
        else:
            batch_size, seq_len, embed_dim, _ = psi.shape
            # Legacy system uses only 3 parameters for left rotation
            rotation_angles = torch.tensor([0.1, 0.05, 0.02], device=self.device).expand(batch_size, seq_len, embed_dim, -1)
            psi_rotated = self.quaternion_ops.so4_rotation(psi, rotation_angles)

        return psi_rotated

    def process(self, psi: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Executa o pipeline de processamento quântico completo.
        """
        psi_filtered = self.apply_spectral_filtering(psi, alpha)
        psi_processed = self.apply_so4_rotation(psi_filtered)
        return psi_processed

# Exemplo de uso
if __name__ == '__main__':
    from TextProcessor import TextProcessor
    from QuantumMapper import QuantumMapper

    device = 'cpu'
    embedding_dimension = 64
    alpha_param = 1.2

    # 1. Gerar sinal e mapear para quaternions
    text_processor = TextProcessor(device=device)
    mapper = QuantumMapper(device=device)
    input_text = "Processamento quântico"
    fractal_signal, _ = text_processor.process(input_text, embedding_dimension)
    psi_initial = mapper.map_to_quaternions(fractal_signal, embedding_dimension)
    print(f"Estado quântico inicial Ψ com shape: {psi_initial.shape}")

    # 2. Processar o estado quântico
    processor = QuantumProcessor(device=device)
    psi_final = processor.process(psi_initial, alpha=alpha_param)

    print(f"Estado quântico final Ψ processado com shape: {psi_final.shape}")

    norm_initial = torch.norm(psi_initial)
    norm_final = torch.norm(psi_final)
    print(f"Norma inicial: {norm_initial:.4f}")
    print(f"Norma final: {norm_final:.4f} (deve ser similar, indicando conservação de energia)")

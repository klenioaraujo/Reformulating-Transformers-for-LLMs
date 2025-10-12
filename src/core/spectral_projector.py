#!/usr/bin/env python3
"""
ΨQRH Spectral Projector with Audit Framework
Enhanced with comprehensive auditing capabilities for debugging and analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
import hashlib


class AuditLogger:
    """Framework de auditoria para o pipeline ΨQRH"""

    def __init__(self, audit_dir: str = "audit_logs", enabled: bool = False):
        self.enabled = enabled
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(exist_ok=True)
        self.current_session = None
        self.session_data = {}

    def start_session(self, input_text: str, parameters: Dict[str, Any]) -> str:
        """Inicia uma nova sessão de auditoria"""
        if not self.enabled:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"audit_{timestamp}"

        self.current_session = session_id
        self.session_data = {
            "session_id": session_id,
            "timestamp": timestamp,
            "input_text": input_text,
            "parameters": parameters,
            "audit_trail": []
        }

        return session_id

    def log_tensor_state(self, step_name: str, psi_tensor: torch.Tensor,
                         additional_metrics: Optional[Dict[str, Any]] = None) -> str:
        """Registra o estado de um tensor quântico em um passo específico"""
        if not self.enabled or not self.current_session:
            return None

        # Calcular métricas padrão - lidar com tensores complexos
        if psi_tensor.is_complex():
            # Para tensores complexos, calcular métricas da magnitude
            magnitude = torch.abs(psi_tensor)
            metrics = {
                "norm_l2": float(torch.norm(magnitude).item()),
                "mean": float(torch.mean(magnitude).item()),
                "std": float(torch.std(magnitude).item()),
                "is_stable": not (torch.isinf(magnitude).any().item() or torch.isnan(magnitude).any().item()),
                "shape": list(psi_tensor.shape),
                "dtype": str(psi_tensor.dtype),
                "is_complex": True
            }
        else:
            metrics = {
                "norm_l2": float(torch.norm(psi_tensor).item()),
                "mean": float(torch.mean(psi_tensor).item()),
                "std": float(torch.std(psi_tensor).item()),
                "is_stable": not (torch.isinf(psi_tensor).any().item() or torch.isnan(psi_tensor).any().item()),
                "shape": list(psi_tensor.shape),
                "dtype": str(psi_tensor.dtype),
                "is_complex": False
            }

        # Adicionar métricas adicionais se fornecidas
        if additional_metrics:
            metrics.update(additional_metrics)

        # Salvar snapshot do tensor
        tensor_filename = f"{self.current_session}_{step_name}_tensor.pt"
        tensor_path = self.audit_dir / tensor_filename
        torch.save(psi_tensor.detach().cpu(), tensor_path)

        # Registrar no trail de auditoria
        audit_entry = {
            "step": step_name,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            "metrics": metrics,
            "tensor_snapshot": str(tensor_path)
        }

        self.session_data["audit_trail"].append(audit_entry)

        return str(tensor_path)

    def end_session(self, final_output: str = None) -> str:
        """Finaliza a sessão de auditoria e salva o log"""
        if not self.enabled or not self.current_session:
            return None

        if final_output:
            self.session_data["final_output"] = final_output

        # Salvar log JSON
        log_filename = f"{self.current_session}_log.json"
        log_path = self.audit_dir / log_filename

        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.session_data, f, indent=2, ensure_ascii=False)

        session_id = self.current_session
        self.current_session = None
        self.session_data = {}

        return str(log_path)


class SpectralQRHLayer(nn.Module):
    """
    Camada ΨQRH com capacidades de auditoria integradas
    """

    def __init__(self, embed_dim: int, alpha: float = 1.0, audit_logger: Optional[AuditLogger] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.audit_logger = audit_logger

        # Parâmetros aprendíveis para rotações quaterniônicas
        self.theta = nn.Parameter(torch.tensor(0.1))
        self.omega = nn.Parameter(torch.tensor(0.05))
        self.phi = nn.Parameter(torch.tensor(0.02))

    def get_rotation_quaternions(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gera quaternions de rotação unitários"""
        # Implementação simplificada - em produção seria mais sofisticada
        q_left = torch.tensor([1.0, self.theta, self.omega, self.phi], device=self.theta.device)
        q_right = torch.tensor([1.0, -self.theta, -self.omega, -self.phi], device=self.theta.device)

        # Normalizar para garantir unitariedade
        q_left = q_left / torch.norm(q_left)
        q_right = q_right / torch.norm(q_right)

        return q_left, q_right

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Transformação ΨQRH forward com auditoria e estabilização numérica
        """
        # Log estado inicial se auditoria estiver ativa
        if self.audit_logger:
            self.audit_logger.log_tensor_state("qrh_input", psi)

        # Aplicar rotações quaterniônicas
        q_left, q_right = self.get_rotation_quaternions()

        # Rotações simplificadas (implementação completa seria mais complexa)
        # Aqui apenas aplicamos uma transformação linear representativa

        # Aplicar FFT
        psi_fft = torch.fft.fft(psi, dim=-1)

        # Aplicar filtro espectral com clipping para prevenir overflow
        k = torch.arange(psi.shape[-1], dtype=torch.float32, device=psi.device)
        k = k + 1e-10  # Evitar divisão por zero

        # Clipping para prevenir valores extremos em log
        k_clipped = torch.clamp(k, min=1e-10, max=1e10)
        log_k = torch.log(k_clipped)

        # Clipping adicional para arctan
        log_k_clipped = torch.clamp(log_k, min=-10.0, max=10.0)
        phase = torch.arctan(log_k_clipped)

        # Clipping para o argumento da exponencial complexa
        alpha_phase = self.alpha * phase
        alpha_phase_clipped = torch.clamp(alpha_phase, min=-10.0, max=10.0)

        spectral_filter = torch.exp(1j * alpha_phase_clipped)

        psi_filtered = psi_fft * spectral_filter.unsqueeze(0).unsqueeze(0)

        # IFFT de volta
        psi_ifft = torch.fft.ifft(psi_filtered, dim=-1).real

        # NORMALIZAÇÃO L2: Preservar energia após IFFT
        psi_transformed = preserve_energy(psi_ifft, psi)

        # Log estado transformado
        if self.audit_logger:
            self.audit_logger.log_tensor_state("qrh_output", psi_transformed)

        return psi_transformed


def preserve_energy(tensor_out: torch.Tensor, tensor_in: torch.Tensor) -> torch.Tensor:
    """
    Renormaliza o tensor de saída para ter a mesma norma do tensor de entrada.
    Garante conservação de energia no pipeline quântico.
    """
    # Para tensores quaterniônicos [batch, seq, embed_dim, 4], preservar norma por posição
    # Calcular norma L2 sobre a dimensão dos componentes quaterniônicos (última dimensão)
    norm_in = torch.norm(tensor_in, p=2, dim=-1, keepdim=True)  # [batch, seq, embed_dim, 1]
    norm_out = torch.norm(tensor_out, p=2, dim=-1, keepdim=True)  # [batch, seq, embed_dim, 1]

    # Evitar divisão por zero e valores extremos
    epsilon = 1e-8
    norm_ratio = norm_in / (norm_out + epsilon)

    # Clipping para prevenir escalas extremas
    norm_ratio = torch.clamp(norm_ratio, min=0.1, max=10.0)

    # Renormalizar tensor de saída para preservar a norma quaterniônica
    tensor_renormalized = tensor_out * norm_ratio

    return tensor_renormalized


def invert_spectral_qrh(psi_qrh: torch.Tensor, qrh_layer: SpectralQRHLayer,
                        audit_logger: Optional[AuditLogger] = None) -> torch.Tensor:
    """
    Inversão da transformação ΨQRH com auditoria completa e estabilização numérica
    """
    if audit_logger:
        audit_logger.log_tensor_state("inversion_input", psi_qrh)

    # Passo 1: Inverter rotações (ordem reversa)
    q_left, q_right = qrh_layer.get_rotation_quaternions()
    q_left_inv = q_left.conj()  # Conjugado para inversão
    q_right_inv = q_right.conj()

    # Aplicar rotações inversas (simplificado)
    psi_unrotated = psi_qrh  # Implementação completa aplicaria rotações

    if audit_logger:
        audit_logger.log_tensor_state("after_rotation_inversion", psi_unrotated)

    # Passo 2: Aplicar FFT
    psi_fft = torch.fft.fft(psi_unrotated, dim=-1)

    if audit_logger:
        audit_logger.log_tensor_state("after_forward_fft", psi_fft)

    # Passo 3: Inverter filtro espectral com clipping para prevenir overflow
    k = torch.arange(psi_qrh.shape[-1], dtype=torch.float32, device=psi_qrh.device)
    k = k + 1e-10

    # Clipping para prevenir valores extremos em log
    k_clipped = torch.clamp(k, min=1e-10, max=1e10)
    log_k = torch.log(k_clipped)

    # Clipping adicional para arctan
    log_k_clipped = torch.clamp(log_k, min=-10.0, max=10.0)
    phase = torch.arctan(log_k_clipped)

    # Clipping para o argumento da exponencial complexa
    alpha_phase = qrh_layer.alpha * phase
    alpha_phase_clipped = torch.clamp(alpha_phase, min=-10.0, max=10.0)

    inverse_filter = torch.exp(-1j * alpha_phase_clipped)

    psi_filtered = psi_fft * inverse_filter.unsqueeze(0).unsqueeze(0)

    if audit_logger:
        audit_logger.log_tensor_state("after_filter_inversion", psi_filtered)

    # Passo 4: IFFT
    psi_ifft = torch.fft.ifft(psi_filtered, dim=-1).real

    # NORMALIZAÇÃO L2: Preservar energia após IFFT
    psi_inverted = preserve_energy(psi_ifft, psi_qrh)

    if audit_logger:
        audit_logger.log_tensor_state("final_inverted_output", psi_inverted)

    return psi_inverted


def create_audit_enabled_qrh_pipeline(embed_dim: int = 64, alpha: float = 1.0,
                                    audit_enabled: bool = False) -> Tuple[SpectralQRHLayer, AuditLogger]:
    """
    Factory function para criar pipeline ΨQRH com auditoria
    """
    audit_logger = AuditLogger(enabled=audit_enabled) if audit_enabled else None
    qrh_layer = SpectralQRHLayer(embed_dim=embed_dim, alpha=alpha, audit_logger=audit_logger)

    return qrh_layer, audit_logger
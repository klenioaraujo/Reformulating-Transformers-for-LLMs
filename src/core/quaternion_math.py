"""
Quaternion Math - Operações Matemáticas Puras (SEM torch.nn)
=============================================================

Implementa álgebra quaterniônica conforme doe.md Seção 2.1:
- Produto de Hamilton: q₁ ∗ q₂
- Criação de quaternion unitário via ângulos de Euler
- Operações SO(4) = (SU(2) × SU(2))/Z₂

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import math
from typing import Tuple


def hamilton_product(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Produto de Hamilton (doe.md Seção 2.1).

    q₁ ∗ q₂ = (w₁w₂ - x₁x₂ - y₁y₂ - z₁z₂) +
              (w₁x₂ + x₁w₂ + y₁z₂ - z₁y₂)i +
              (w₁y₂ - x₁z₂ + y₁w₂ + z₁x₂)j +
              (w₁z₂ + x₁y₂ - y₁x₂ + z₁w₂)k

    Args:
        q1: Primeiro quaternion [*, 4] onde componentes são [w, x, y, z]
        q2: Segundo quaternion [*, 4]

    Returns:
        Produto q₁ ∗ q₂ [*, 4]
    """
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)

    # Produto de Hamilton
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    Conjugado quaterniônico: q† = w - xi - yj - zk

    Args:
        q: Quaternion [*, 4]

    Returns:
        Conjugado q† [*, 4]
    """
    w, x, y, z = torch.unbind(q, dim=-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def quaternion_norm(q: torch.Tensor) -> torch.Tensor:
    """
    Norma quaterniônica: |q| = sqrt(w² + x² + y² + z²)

    Args:
        q: Quaternion [*, 4]

    Returns:
        Norma |q| [*]
    """
    return torch.sqrt(torch.sum(q ** 2, dim=-1))


def quaternion_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normaliza quaternion para norma unitária.

    Args:
        q: Quaternion [*, 4]
        eps: Epsilon para estabilidade numérica

    Returns:
        Quaternion unitário [*, 4]
    """
    norm = quaternion_norm(q).unsqueeze(-1)
    return q / (norm + eps)


def create_unit_quaternion(theta: float, omega: float, phi: float) -> torch.Tensor:
    """
    Cria quaternion unitário via ângulos de Euler (doe.md Seção 2.1).

    q = cos(θ/2) + sin(θ/2)[cos(ω)i + sin(ω)cos(φ)j + sin(ω)sin(φ)k]

    Args:
        theta: Ângulo de rotação principal
        omega: Ângulo azimutal
        phi: Ângulo polar

    Returns:
        Quaternion unitário [4]
    """
    half_theta = theta / 2.0

    w = math.cos(half_theta)
    x = math.sin(half_theta) * math.cos(omega)
    y = math.sin(half_theta) * math.sin(omega) * math.cos(phi)
    z = math.sin(half_theta) * math.sin(omega) * math.sin(phi)

    return torch.tensor([w, x, y, z], dtype=torch.float32)


def create_unit_quaternion_batch(thetas: torch.Tensor,
                                 omegas: torch.Tensor,
                                 phis: torch.Tensor) -> torch.Tensor:
    """
    Cria batch de quaternions unitários.

    Args:
        thetas: Ângulos θ [batch]
        omegas: Ângulos ω [batch]
        phis: Ângulos φ [batch]

    Returns:
        Quaternions unitários [batch, 4]
    """
    half_theta = thetas / 2.0

    w = torch.cos(half_theta)
    x = torch.sin(half_theta) * torch.cos(omegas)
    y = torch.sin(half_theta) * torch.sin(omegas) * torch.cos(phis)
    z = torch.sin(half_theta) * torch.sin(omegas) * torch.sin(phis)

    return torch.stack([w, x, y, z], dim=-1)


def so4_rotation(psi: torch.Tensor,
                 q_left: torch.Tensor,
                 q_right: torch.Tensor) -> torch.Tensor:
    """
    Rotação 4D completa via SO(4) (doe.md Seção 2.2).

    Ψ' = q_left ∗ Ψ ∗ q_right†

    Onde SO(4) ≅ (SU(2) × SU(2))/Z₂

    Args:
        psi: Estado quaterniônico [*, 4]
        q_left: Quaternion de rotação esquerda [4]
        q_right: Quaternion de rotação direita [4]

    Returns:
        Estado rotacionado [*, 4]
    """
    # Expandir quaternions para broadcast
    batch_shape = psi.shape[:-1]
    q_left_expanded = q_left.unsqueeze(0).expand(*batch_shape, -1)
    q_right_expanded = q_right.unsqueeze(0).expand(*batch_shape, -1)

    # Conjugado de q_right
    q_right_conj = quaternion_conjugate(q_right_expanded)

    # Aplicar rotação: q_left ∗ psi ∗ q_right†
    psi_intermediate = hamilton_product(q_left_expanded, psi)
    psi_rotated = hamilton_product(psi_intermediate, q_right_conj)

    return psi_rotated


class QuaternionOperations:
    """
    Classe utilitária para compatibilidade com código existente.
    Mantém interface, mas todas as operações são funções puras.
    """

    @staticmethod
    def multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Produto de Hamilton."""
        return hamilton_product(q1, q2)

    @staticmethod
    def conjugate(q: torch.Tensor) -> torch.Tensor:
        """Conjugado quaterniônico."""
        return quaternion_conjugate(q)

    @staticmethod
    def norm(q: torch.Tensor) -> torch.Tensor:
        """Norma quaterniônica."""
        return quaternion_norm(q)

    @staticmethod
    def normalize(q: torch.Tensor) -> torch.Tensor:
        """Normalização para quaternion unitário."""
        return quaternion_normalize(q)

    @staticmethod
    def create_unit_quaternion(theta: float, omega: float, phi: float) -> torch.Tensor:
        """Criação de quaternion unitário via Euler."""
        return create_unit_quaternion(theta, omega, phi)

    @staticmethod
    def create_unit_quaternion_batch(thetas: torch.Tensor,
                                     omegas: torch.Tensor,
                                     phis: torch.Tensor) -> torch.Tensor:
        """Criação batch de quaternions."""
        return create_unit_quaternion_batch(thetas, omegas, phis)

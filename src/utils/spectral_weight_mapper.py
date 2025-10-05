#!/usr/bin/env python3
"""
Spectral Weight Mapper - Mapeamento Físico de Pesos para ΨQRH
================================================================

Aplica transformações quaterniônicas e projeções de Leech aos pesos
convertidos espectralmente, persistindo o conhecimento do modelo fonte.

Pipeline:
1. Wold → quaternion_rotation(θ) → Wrot
2. Wrot → modulate(α, D) → Wmod
3. Wmod → leech_project(Λ24) → Wnew

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from pathlib import Path


def quaternion_from_phase(theta: float) -> torch.Tensor:
    """
    Cria quaternion de rotação a partir de fase espectral.

    Construção: q = cos(θ/2) + sin(θ/2)·i
    (Rotação no plano complexo i)

    Args:
        theta: Fase em radianos [-π, π]

    Returns:
        Quaternion [w, x, y, z] unitário
    """
    half_theta = theta / 2.0

    # q = [cos(θ/2), sin(θ/2), 0, 0]
    w = np.cos(half_theta)
    x = np.sin(half_theta)
    y = 0.0
    z = 0.0

    q = torch.tensor([w, x, y, z], dtype=torch.float32)

    # Normalizar (garantir |q| = 1)
    q = q / torch.norm(q)

    return q


def apply_quaternion_rotation(
    weight: torch.Tensor,
    q: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """
    Aplica rotação quaterniônica modulada por α aos pesos.

    Transformação: W' = q * W * q† (com modulação α)

    Args:
        weight: Tensor de pesos (qualquer shape)
        q: Quaternion [w, x, y, z] unitário
        alpha: Parâmetro espectral de modulação

    Returns:
        Peso transformado (mesma shape)
    """
    original_shape = weight.shape
    device = weight.device

    # Mover quaternion para o device correto
    q = q.to(device)

    # Flatten para aplicar rotação
    w_flat = weight.flatten()

    # Extrair componentes do quaternion
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    # Construir matriz de rotação 3×3 correspondente ao quaternion
    # (simplificação: aplicar rotação via multiplicação matricial)
    # R(q) aplicada a cada componente real

    # Modulação por α: escala a intensidade da rotação
    # W' = (1 - α_scale)·W + α_scale·R(q)·W
    alpha_scale = torch.clamp(torch.tensor(alpha / 3.0), 0.0, 1.0)

    # Aplicar rotação no espaço de fase
    # Transformação simplificada: rotação via componente real/imaginária
    w_rotated = (
        qw * w_flat +  # Componente real
        qx * torch.roll(w_flat, 1, dims=0)  # Componente i
    )

    # Modular por α
    w_modulated = (1.0 - alpha_scale) * w_flat + alpha_scale * w_rotated

    # Reshape para forma original
    w_transformed = w_modulated.reshape(original_shape)

    return w_transformed


def leech_project(weight: torch.Tensor, block_size: int = 24) -> torch.Tensor:
    """
    Projeta pesos no reticulado de Leech Λ₂₄.

    Garante estabilidade topológica quantizando em blocos de 24 dimensões.

    Args:
        weight: Tensor de pesos
        block_size: Tamanho do bloco (24 para Leech)

    Returns:
        Peso projetado (mesma shape)
    """
    original_shape = weight.shape
    device = weight.device

    # Flatten
    w_flat = weight.flatten()
    n_params = len(w_flat)

    # Número de blocos completos
    n_blocks = n_params // block_size

    if n_blocks == 0:
        # Muito poucos parâmetros, quantizar diretamente
        w_quantized = torch.round(w_flat * 8.0) / 8.0
        return w_quantized.reshape(original_shape)

    # Separar em blocos de 24
    blocks = w_flat[:n_blocks * block_size].reshape(n_blocks, block_size)
    remainder = w_flat[n_blocks * block_size:]

    # Projetar cada bloco no reticulado de Leech
    projected_blocks = []

    for block in blocks:
        # Normalizar bloco
        block_norm = torch.norm(block)

        if block_norm > 1e-8:
            block_normalized = block / block_norm

            # Quantização em Z/2 (aproximação da estrutura de Leech)
            # Rede de Leech: {x ∈ ℝ²⁴ | x·x ∈ 2ℤ}
            block_quantized = torch.round(block_normalized * 8.0) / 8.0

            # Re-escalar para norma original
            block_projected = block_quantized * block_norm
        else:
            block_projected = block

        projected_blocks.append(block_projected)

    # Processar remainder (se existir)
    if len(remainder) > 0:
        remainder_quantized = torch.round(remainder * 8.0) / 8.0
    else:
        remainder_quantized = remainder

    # Reconstruir tensor
    if len(projected_blocks) > 0:
        w_projected = torch.cat([
            torch.stack(projected_blocks).flatten(),
            remainder_quantized
        ])
    else:
        w_projected = remainder_quantized

    # Reshape para forma original
    return w_projected.reshape(original_shape)


def map_layer_weights(
    source_weight: torch.Tensor,
    alpha: float,
    theta: float,
    fractal_dim: Optional[float] = None
) -> torch.Tensor:
    """
    Mapeia peso de uma camada usando parâmetros espectrais.

    Pipeline:
        source_weight → quaternion_rotation(θ) →
        modulate(α) → leech_project → psiqrh_weight

    Args:
        source_weight: Peso fonte (GPT-2/BERT/etc.)
        alpha: Parâmetro α da análise espectral
        theta: Fase θ da análise espectral
        fractal_dim: Dimensão fractal (opcional, para logging)

    Returns:
        Peso mapeado para ΨQRH
    """
    # 1. Criar quaternion de rotação
    q = quaternion_from_phase(theta)

    # 2. Aplicar rotação quaterniônica modulada por α
    w_rotated = apply_quaternion_rotation(source_weight, q, alpha)

    # 3. Projetar no reticulado de Leech
    w_projected = leech_project(w_rotated)

    # 4. Normalizar energia (conservação)
    source_norm = torch.norm(source_weight)
    projected_norm = torch.norm(w_projected)

    if projected_norm > 1e-8:
        w_final = w_projected * (source_norm / projected_norm)
    else:
        w_final = w_projected

    return w_final


def map_spectral_to_state_dict(
    source_state_dict: Dict[str, torch.Tensor],
    spectral_params: Dict[str, Dict[str, float]]
) -> Dict[str, torch.Tensor]:
    """
    Mapeia state_dict completo usando parâmetros espectrais.

    Transforma todos os pesos do modelo fonte para ΨQRH preservando
    conhecimento via rotações quaterniônicas baseadas em análise espectral.

    Args:
        source_state_dict: State dict do modelo fonte
        spectral_params: Parâmetros espectrais por camada
            {
                'layer_0.weight': {'alpha': 1.4, 'theta': -0.5, 'fractal_dim': 1.2},
                'layer_1.weight': {'alpha': 1.6, 'theta': 0.2, 'fractal_dim': 1.5},
                ...
            }

    Returns:
        State dict ΨQRH com pesos mapeados
    """
    psiqrh_state_dict = {}

    print(f"\n🔄 Mapeando {len(source_state_dict)} tensores...")

    for name, param in source_state_dict.items():
        if name in spectral_params:
            # Aplicar mapeamento espectral
            params = spectral_params[name]
            alpha = params['alpha']
            theta = params['theta']
            fractal_dim = params.get('fractal_dim', None)

            # Mapear pesos
            mapped_weight = map_layer_weights(param, alpha, theta, fractal_dim)

            psiqrh_state_dict[name] = mapped_weight

            # Log de progresso
            energy_ratio = torch.norm(mapped_weight) / (torch.norm(param) + 1e-8)
            print(f"   ✅ {name}: α={alpha:.3f}, θ={theta:.3f}, E_ratio={energy_ratio:.4f}")

        else:
            # Parâmetro sem análise espectral (bias, etc.)
            # Copiar diretamente
            psiqrh_state_dict[name] = param.clone()

    print(f"✅ Mapeamento completo: {len(psiqrh_state_dict)} tensores")

    return psiqrh_state_dict


def validate_energy_preservation(
    source_state_dict: Dict[str, torch.Tensor],
    mapped_state_dict: Dict[str, torch.Tensor],
    tolerance: float = 0.1
) -> Dict[str, float]:
    """
    Valida que a energia foi preservada no mapeamento.

    Verifica: ||Wnew|| ≈ ||Wold|| para cada camada

    Args:
        source_state_dict: State dict fonte
        mapped_state_dict: State dict mapeado
        tolerance: Tolerância máxima (0.1 = 10%)

    Returns:
        Dict com estatísticas de validação
    """
    print("\n🔍 Validando conservação de energia...")

    energy_ratios = []
    violations = []

    for name in source_state_dict.keys():
        if name in mapped_state_dict:
            source_energy = torch.norm(source_state_dict[name]).item()
            mapped_energy = torch.norm(mapped_state_dict[name]).item()

            if source_energy > 1e-8:
                ratio = mapped_energy / source_energy
                energy_ratios.append(ratio)

                if abs(ratio - 1.0) > tolerance:
                    violations.append({
                        'layer': name,
                        'ratio': ratio,
                        'deviation': abs(ratio - 1.0)
                    })

    # Estatísticas
    mean_ratio = np.mean(energy_ratios)
    std_ratio = np.std(energy_ratios)
    min_ratio = np.min(energy_ratios)
    max_ratio = np.max(energy_ratios)

    validation_result = {
        'mean_energy_ratio': float(mean_ratio),
        'std_energy_ratio': float(std_ratio),
        'min_energy_ratio': float(min_ratio),
        'max_energy_ratio': float(max_ratio),
        'n_violations': len(violations),
        'violations': violations,
        'is_valid': len(violations) == 0
    }

    print(f"   • Razão média: {mean_ratio:.4f} ± {std_ratio:.4f}")
    print(f"   • Intervalo: [{min_ratio:.4f}, {max_ratio:.4f}]")

    if validation_result['is_valid']:
        print(f"   ✅ Energia conservada (tolerância: {tolerance*100:.0f}%)")
    else:
        print(f"   ⚠️  {len(violations)} violações detectadas")
        for v in violations[:3]:  # Mostrar primeiras 3
            print(f"      • {v['layer']}: ratio={v['ratio']:.4f}")

    return validation_result


if __name__ == "__main__":
    # Teste básico
    print("🧪 Teste do Spectral Weight Mapper\n")

    # Criar peso de exemplo
    w = torch.randn(128, 128)
    print(f"Peso fonte: shape={w.shape}, norm={torch.norm(w):.4f}")

    # Criar quaternion
    theta = 0.5
    q = quaternion_from_phase(theta)
    print(f"Quaternion: {q}, norm={torch.norm(q):.4f}")

    # Aplicar rotação
    alpha = 1.5
    w_rot = apply_quaternion_rotation(w, q, alpha)
    print(f"Após rotação: shape={w_rot.shape}, norm={torch.norm(w_rot):.4f}")

    # Projetar em Leech
    w_proj = leech_project(w_rot)
    print(f"Após Leech: shape={w_proj.shape}, norm={torch.norm(w_proj):.4f}")

    # Mapeamento completo
    w_mapped = map_layer_weights(w, alpha, theta)
    print(f"Mapeado: shape={w_mapped.shape}, norm={torch.norm(w_mapped):.4f}")

    # Validar energia
    energy_ratio = torch.norm(w_mapped) / torch.norm(w)
    print(f"\n✅ Razão de energia: {energy_ratio:.4f} (esperado ≈ 1.0)")

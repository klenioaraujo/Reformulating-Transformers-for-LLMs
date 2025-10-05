#!/usr/bin/env python3
"""
Embedding Spectral Converter - Conversão Física de Embeddings para ΨQRH
=========================================================================

Converte embedding layer do GPT-2 (W_e ∈ ℝ^{V×d}) para embeddings
quaterniônicos (Ψ_e ∈ ℍ^{V×d/4}) preservando semântica através de
análise espectral por token.

Pipeline:
1. Para cada token i ∈ [0, V):
   - Calcular FFT: ẽᵢ = F(eᵢ)
   - Espectro: Pᵢ(k) = |ẽᵢ(k)|²
   - Lei de potência: Pᵢ(k) ~ k^(-βᵢ)
   - Dimensão fractal: Dᵢ = (3-βᵢ)/2
   - Fase dominante: θᵢ = arg(ẽᵢ(k_dom))

2. Mapear eᵢ → Ψᵢ usando rotação quaterniônica baseada em Dᵢ e θᵢ

3. Salvar vocabulário e tokenizer do GPT-2

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional
import json
from pathlib import Path
from tqdm import tqdm
from collections import Counter


def fci_to_alpha(target_fci: float, fractal_dim: float, alpha_min: float = 0.1, alpha_max: float = 3.0) -> float:
    """
    Converte FCI alvo para α usando relação física.

    Args:
        target_fci: FCI alvo
        fractal_dim: Dimensão fractal
        alpha_min: α mínimo
        alpha_max: α máximo

    Returns:
        α calibrado
    """
    # Mapear FCI para α base (relação linear simplificada)
    alpha_base = 0.5 + (target_fci - 0.5) * 2.0  # FCI 0.5 → α 0.5, FCI 0.8 → α 1.1

    # Aplicar modulação por dimensão fractal
    d_eucl = 1.0
    lambda_coupling = 1.0
    alpha_target = alpha_base * (1.0 + lambda_coupling * (fractal_dim - d_eucl) / d_eucl)

    # Limitar ao intervalo permitido
    alpha_target = np.clip(alpha_target, alpha_min, alpha_max)

    return float(alpha_target)


def fit_power_law_exponent(power_spectrum: torch.Tensor) -> float:
    """
    Ajusta lei de potência P(k) ~ k^(-β) no espectro.

    Args:
        power_spectrum: Espectro de potência |F(x)|²

    Returns:
        Expoente β da lei de potência
    """
    # Converter para numpy
    ps = power_spectrum.cpu().numpy()

    # Frequências
    k = np.arange(1, len(ps) + 1)

    # Remover zeros e valores muito pequenos
    valid_mask = ps > 1e-12
    k_valid = k[valid_mask]
    ps_valid = ps[valid_mask]

    if len(k_valid) < 10:
        # Poucos pontos válidos, usar valor padrão
        return 1.0

    # Log-log space
    log_k = np.log(k_valid)
    log_ps = np.log(ps_valid + 1e-12)

    # Regressão linear
    try:
        coeffs = np.polyfit(log_k, log_ps, 1)
        beta = -coeffs[0]  # Inclinação negativa
        return float(np.clip(beta, 0.5, 2.5))
    except:
        return 1.0


def spectral_quaternion_map(
    embedding: torch.Tensor,
    fractal_dim: float,
    theta: float,
    alpha: float
) -> torch.Tensor:
    """
    Mapeia embedding clássico para quaterniônico usando parâmetros espectrais.

    Transformação:
    e ∈ ℝ^d → Ψ ∈ ℍ^{d/4}

    Args:
        embedding: Vetor de embedding [d]
        fractal_dim: Dimensão fractal D
        theta: Fase dominante θ
        alpha: Parâmetro espectral α

    Returns:
        Embedding quaterniônico [d/4, 4]
    """
    d = embedding.shape[0]
    assert d % 4 == 0, f"Dimensão {d} não é divisível por 4"

    # Reshape em grupos de 4 (componentes quaterniônicos)
    # [d] → [d/4, 4]
    quat_groups = embedding.reshape(-1, 4)

    # Normalizar cada quaternion
    norms = torch.norm(quat_groups, dim=-1, keepdim=True)
    quat_normalized = quat_groups / (norms + 1e-8)

    # Aplicar rotação baseada em theta e alpha
    # q_rot = [cos(θ/2), sin(θ/2), 0, 0]
    half_theta = theta / 2.0
    q_rot = torch.tensor([
        np.cos(half_theta),
        np.sin(half_theta),
        0.0,
        0.0
    ], dtype=embedding.dtype, device=embedding.device)

    # Rotação quaterniônica simplificada
    # q' = q_rot * q (multiplicação quaterniônica aproximada)
    alpha_scale = torch.clamp(torch.tensor(alpha / 3.0), 0.0, 1.0)

    quat_rotated = (
        (1.0 - alpha_scale) * quat_normalized +
        alpha_scale * (quat_normalized * q_rot[0] + torch.roll(quat_normalized, 1, dims=-1) * q_rot[1])
    )

    # Re-normalizar
    quat_final_norms = torch.norm(quat_rotated, dim=-1, keepdim=True)
    quat_final = quat_rotated / (quat_final_norms + 1e-8)

    # Re-escalar para norma original (conservação de energia)
    quat_final = quat_final * norms

    return quat_final


def convert_gpt2_embedding_to_psiqrh(
    gpt2_embedding_weight: torch.Tensor,
    calibration_config: Dict = None,
    semantic_categories: Dict = None,
    alpha_min: float = 0.1,
    alpha_max: float = 3.0,
    lambda_coupling: float = 1.0,
    d_euclidean: float = 1.0,
    verbose: bool = True
) -> Tuple[torch.Tensor, Dict]:
    """
    Converte embedding do GPT-2 para ΨQRH quaterniônico com calibração FCI.

    W_e ∈ ℝ^{V×d} → Ψ_e ∈ ℍ^{V×d/4}

    Args:
        gpt2_embedding_weight: Embedding GPT-2 [vocab_size, d_model]
        calibration_config: Configuração de calibração FCI
        semantic_categories: Categorias semânticas para modulação
        alpha_min: α mínimo
        alpha_max: α máximo
        lambda_coupling: Constante de acoplamento
        d_euclidean: Dimensão euclidiana de referência
        verbose: Mostrar progresso

    Returns:
        Tuple (psi_embeddings, metadata)
    """
    V, d = gpt2_embedding_weight.shape

    if verbose:
        print(f"\n🔄 Convertendo embedding GPT-2 → ΨQRH quaterniônico")
        print(f"   • Vocabulário: {V:,} tokens")
        print(f"   • Dimensão original: {d}")
        print(f"   • Dimensão quaterniônica: {d//4} (4 componentes)")
        if calibration_config:
            print(f"   • Modulação FCI: HABILITADA")
        if semantic_categories:
            print(f"   • Categorias semânticas: {len(semantic_categories)}")

    assert d % 4 == 0, f"Dimensão {d} não é divisível por 4"

    # Armazenar embeddings quaterniônicos
    psi_embeddings = []

    # Metadados por token
    token_metadata = {
        'betas': [],
        'fractal_dims': [],
        'thetas': [],
        'alphas': []
    }

    # Processar cada token
    iterator = tqdm(range(V), desc="Converting tokens") if verbose else range(V)

    for i in iterator:
        e_i = gpt2_embedding_weight[i]  # [d]

        # 1. FFT
        fft_e = torch.fft.fft(e_i)

        # 2. Espectro de potência (só frequências positivas)
        power = torch.abs(fft_e[:len(fft_e)//2])**2

        # 3. Ajustar lei de potência → β → D
        beta_i = fit_power_law_exponent(power)
        D_i = (3.0 - beta_i) / 2.0
        D_i = np.clip(D_i, 1.0, 2.0)

        # 4. Fase dominante
        dominant_idx = torch.argmax(torch.abs(fft_e[:len(fft_e)//2]))
        theta_i = float(torch.angle(fft_e[dominant_idx]))

        # 5. Mapear D → α (com modulação FCI se disponível)
        alpha_0 = (alpha_min + alpha_max) / 2.0
        alpha_geometric = alpha_0 * (1.0 + lambda_coupling * (D_i - d_euclidean) / d_euclidean)

        # Aplicar modulação FCI se disponível
        if calibration_config and semantic_categories:
            # Obter categoria semântica do token (simplificado)
            token_category = semantic_categories.get(str(i), 'neutral')
            if token_category in calibration_config.get('state_thresholds', {}):
                target_fci = calibration_config['state_thresholds'][token_category]['min_fci']
                # Converter FCI alvo para α
                alpha_calibrated = fci_to_alpha(target_fci, D_i, alpha_min, alpha_max)
                # Interpolar entre α geométrico e α calibrado
                alpha_i = 0.7 * alpha_geometric + 0.3 * alpha_calibrated
                if verbose and i % 1000 == 0:
                    print(f"   • Token {i}: {token_category} → FCI={target_fci:.3f}, α={alpha_i:.3f}")
            else:
                alpha_i = alpha_geometric
        else:
            alpha_i = alpha_geometric

        alpha_i = np.clip(alpha_i, alpha_min, alpha_max)

        # 6. Mapear para quaternião
        psi_i = spectral_quaternion_map(e_i, D_i, theta_i, alpha_i)

        psi_embeddings.append(psi_i)

        # Salvar metadata
        token_metadata['betas'].append(float(beta_i))
        token_metadata['fractal_dims'].append(float(D_i))
        token_metadata['thetas'].append(float(theta_i))
        token_metadata['alphas'].append(float(alpha_i))

        # Salvar categoria semântica se disponível
        if semantic_categories:
            token_category = semantic_categories.get(str(i), 'neutral')
            token_metadata.setdefault('semantic_categories', []).append(token_category)

    # Stack em tensor [V, d/4, 4]
    psi_embeddings_tensor = torch.stack(psi_embeddings)

    # 🔑 APLICAÇÃO ÚNICA DA LEI DE BENFORD: AUDITORIA E CORREÇÃO
    if verbose:
        print("\n🔍 Aplicando auditoria espectral Benford ao embedding convertido...")
    audit = benford_spectral_audit(psi_embeddings_tensor)

    if verbose:
        print(f"   • Conformidade Benford: {audit['benford_conformity']:.4f}")

    if not audit['is_conformant']:
        if verbose:
            print("🔧 Embedding não conforme. Aplicando re-normalização Benford...")
        psi_embeddings_tensor = spectral_benford_renormalization(psi_embeddings_tensor)
        # Re-auditar para confirmar
        audit = benford_spectral_audit(psi_embeddings_tensor)
        if verbose:
            print(f"   • Conformidade pós-correção: {audit['benford_conformity']:.4f}")

    # Estatísticas
    metadata = {
        'vocab_size': V,
        'd_model_original': d,
        'd_model_quaternion': d // 4,
        'mean_beta': float(np.mean(token_metadata['betas'])),
        'mean_fractal_dim': float(np.mean(token_metadata['fractal_dims'])),
        'mean_alpha': float(np.mean(token_metadata['alphas'])),
        'std_fractal_dim': float(np.std(token_metadata['fractal_dims'])),
        'token_metadata': token_metadata,
        'calibration_used': calibration_config is not None,
        'semantic_categories_used': semantic_categories is not None,
        'benford_audit': audit
    }

    # Adicionar estatísticas de categorias semânticas se disponível
    if semantic_categories and 'semantic_categories' in token_metadata:
        category_counts = Counter(token_metadata['semantic_categories'])
        metadata['semantic_category_distribution'] = dict(category_counts)

    if verbose:
        print(f"\n   ✅ Conversão completa:")
        print(f"      • β médio: {metadata['mean_beta']:.4f}")
        print(f"      • D médio: {metadata['mean_fractal_dim']:.4f} ± {metadata['std_fractal_dim']:.4f}")
        print(f"      • α médio: {metadata['mean_alpha']:.4f}")
        print(f"      • Shape: {psi_embeddings_tensor.shape}")
        print(f"      • Conformidade Benford: {audit['benford_conformity']:.4f}")
        if metadata['calibration_used']:
            print(f"      • Calibração FCI: HABILITADA")
        if metadata['semantic_categories_used']:
            print(f"      • Categorias semânticas: {len(metadata.get('semantic_category_distribution', {}))}")
            for category, count in metadata.get('semantic_category_distribution', {}).items():
                percentage = (count / V) * 100
                print(f"        └─ {category}: {count} ({percentage:.1f}%)")

    return psi_embeddings_tensor, metadata


def benford_spectral_audit(quaternion_embeddings: torch.Tensor) -> Dict[str, float]:
    """
    Auditoria espectral baseada na Lei de Benford generalizada.

    Verifica se a distribuição das magnitudes dos componentes
    segue uma lei de potência logarítmica esperada.

    Args:
        quaternion_embeddings: Tensor [V, d/4, 4]

    Returns:
        Dict com métricas de conformidade
    """
    # Magnitudes dos componentes quaterniônicos
    magnitudes = torch.norm(quaternion_embeddings, dim=-1)  # [V, d/4]

    # Distribuição logarítmica das magnitudes
    log_mags = torch.log10(magnitudes + 1e-10)
    fractional_parts = log_mags - torch.floor(log_mags)

    # Histograma dos dígitos significativos (Benford)
    hist = torch.histc(fractional_parts, bins=9, min=0, max=1)
    observed_probs = hist / hist.sum()

    # Lei de Benford teórica: P(d) = log10(1 + 1/d)
    theoretical_probs = torch.tensor([np.log10(1 + 1/d) for d in range(1, 10)], device=magnitudes.device)

    # Métrica de conformidade (KL divergence)
    kl_div = torch.sum(theoretical_probs * torch.log(theoretical_probs / (observed_probs + 1e-10)))

    return {
        'benford_conformity': float(kl_div.item()),
        'is_conformant': kl_div.item() < 0.5  # Limiar empírico
    }


def spectral_benford_renormalization(
    quaternion_embeddings: torch.Tensor,
    target_conformity: float = 0.3
) -> torch.Tensor:
    """
    Re-normaliza embeddings para respeitar a Lei de Benford espectral.

    Usa um mapeamento logarítmico adaptativo para ajustar magnitudes.

    Args:
        quaternion_embeddings: Tensor [V, d/4, 4]
        target_conformity: Conformidade alvo

    Returns:
        Tensor re-normalizado
    """
    V, d_quat, _ = quaternion_embeddings.shape

    # Reshape para componentes individuais [V*d_quat, 4]
    components = quaternion_embeddings.reshape(-1, 4)

    # Calcular magnitudes atuais
    mags = torch.norm(components, dim=-1, keepdim=True)

    # Gerar magnitudes alvo seguindo Lei de Benford
    num_components = components.shape[0]
    # Amostrar dígitos significativos da distribuição de Benford
    benford_probs = torch.tensor([np.log10(1 + 1/d) for d in range(1, 10)])
    digits = torch.multinomial(benford_probs, num_components, replacement=True).float() + 1.0

    # Mapear para magnitudes log-uniformes
    log_mags_target = torch.log10(digits.unsqueeze(1)) + torch.rand(num_components, 1)  # Parte fracionária aleatória
    mags_target = 10 ** log_mags_target

    # Re-normalizar componentes
    components_normalized = components / (mags + 1e-10)
    components_renorm = components_normalized * mags_target.to(components.device)

    # Reshape de volta
    return components_renorm.reshape(V, d_quat, 4)


def save_psiqrh_embedding(
    psi_embeddings: torch.Tensor,
    metadata: Dict,
    output_dir: Path
):
    """
    Salva embedding quaterniônico e metadados (SEM dependências externas).

    Args:
        psi_embeddings: Tensor [V, d/4, 4]
        metadata: Metadados da conversão
        output_dir: Diretório de saída
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Salvando embedding quaterniônico...")

    # Salvar tensor (formato compacto)
    embedding_path = output_dir / "quaternion_embedding.pt"
    torch.save(psi_embeddings, embedding_path)
    print(f"   ✅ Embedding salvo: {embedding_path}")
    print(f"      • Tamanho: {embedding_path.stat().st_size / (1024**2):.2f} MB")

    # Salvar metadata
    metadata_path = output_dir / "embedding_metadata.json"
    # Remover token_metadata para reduzir tamanho (pode ser muito grande)
    metadata_compact = {k: v for k, v in metadata.items() if k != 'token_metadata'}
    with open(metadata_path, 'w') as f:
        json.dump(metadata_compact, f, indent=2)
    print(f"   ✅ Metadata salva: {metadata_path}")
    print(f"   • Vocabulário: {metadata['vocab_size']:,} tokens GPT-2")
    print(f"   • Sistema 100% autônomo (sem transformers)")


def load_psiqrh_embedding(model_dir: Path) -> Tuple[torch.Tensor, Dict]:
    """
    Carrega embedding quaterniônico convertido.

    Args:
        model_dir: Diretório do modelo

    Returns:
        Tuple (embedding, metadata)
    """
    model_dir = Path(model_dir)

    # Carregar embedding
    embedding_path = model_dir / "quaternion_embedding.pt"
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding não encontrado: {embedding_path}")

    psi_embeddings = torch.load(embedding_path, map_location='cpu')

    # Carregar metadata
    metadata_path = model_dir / "embedding_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return psi_embeddings, metadata


if __name__ == "__main__":
    # Teste básico
    print("🧪 Teste do Embedding Spectral Converter\n")

    # Simular embedding GPT-2 (pequeno para teste)
    V_test = 100
    d_test = 768

    print(f"Criando embedding de teste: [{V_test}, {d_test}]")
    gpt2_embedding_test = torch.randn(V_test, d_test)

    # Converter
    psi_emb, metadata = convert_gpt2_embedding_to_psiqrh(
        gpt2_embedding_test,
        verbose=True
    )

    print(f"\n✅ Teste completo!")
    print(f"   • Shape original: {gpt2_embedding_test.shape}")
    print(f"   • Shape quaterniônico: {psi_emb.shape}")
    print(f"   • Redução de dimensão: {d_test} → {d_test//4} × 4")
    print(f"   • D médio: {metadata['mean_fractal_dim']:.4f}")

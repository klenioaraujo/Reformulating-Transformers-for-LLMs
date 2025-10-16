#!/usr/bin/env python3
"""
Sistema de similaridade quântica melhorado com múltiplas estratégias
para garantir discriminação adequada entre caracteres.
"""

import torch
import torch.nn.functional as F

class QuantumSimilarityEnhanced:
    """
    Sistema de similaridade quântica melhorado com múltiplas estratégias
    para garantir discriminação adequada entre caracteres.
    """

    @staticmethod
    def enhanced_quaternion_similarity(q1: torch.Tensor, q2: torch.Tensor) -> float:
        """
        Similaridade quântica aprimorada usando múltiplas métricas combinadas.

        Args:
            q1: Tensor quaterniónico [embed_dim, 4]
            q2: Tensor quaterniónico [embed_dim, 4]

        Returns:
            Similaridade normalizada [0, 1] onde 1 = idêntico, 0 = completamente diferente
        """
        # Estratégia 1: Similaridade Cosseno com normalização rigorosa
        q1_flat = q1.flatten()
        q2_flat = q2.flatten()

        # Normalização L2 rigorosa
        q1_norm = F.normalize(q1_flat, p=2, dim=0)
        q2_norm = F.normalize(q2_flat, p=2, dim=0)

        cosine_sim = torch.dot(q1_norm, q2_norm)

        # Estratégia 2: Distância Euclidiana normalizada
        euclidean_dist = torch.norm(q1_flat - q2_flat, p=2)
        max_possible_dist = torch.norm(q1_flat, p=2) + torch.norm(q2_flat, p=2)
        normalized_dist = euclidean_dist / (max_possible_dist + 1e-8)

        # Estratégia 3: Similaridade baseada em ângulos quaterniónicos
        quat_similarity = QuantumSimilarityEnhanced._quaternion_angle_similarity(q1, q2)

        # Combinação ponderada (favorecendo métricas mais discriminativas)
        final_similarity = (
            0.4 * cosine_sim +           # Similaridade cosseno
            0.3 * (1 - normalized_dist) + # Inverso da distância euclidiana
            0.3 * quat_similarity        # Similaridade quaterniónica
        )

        # Garantir que está no intervalo [0, 1]
        final_similarity = torch.clamp(final_similarity, 0.0, 1.0)

        return float(final_similarity)

    @staticmethod
    def _quaternion_angle_similarity(q1: torch.Tensor, q2: torch.Tensor) -> float:
        """
        Calcula similaridade baseada no ângulo entre quatérnios.
        """
        # Tratar cada quaternião como um vetor 4D e calcular o ângulo
        q1_4d = q1.view(-1, 4).mean(dim=0)  # Média sobre todas as dimensões de embedding
        q2_4d = q2.view(-1, 4).mean(dim=0)

        # Normalizar
        q1_norm = F.normalize(q1_4d, p=2, dim=0)
        q2_norm = F.normalize(q2_4d, p=2, dim=0)

        # Produto escalar = cos(ângulo)
        dot_product = torch.dot(q1_norm, q2_norm)

        # Mapear para [0, 1] onde 1 = mesmo ângulo, 0 = ângulos ortogonais
        angle_similarity = (dot_product + 1.0) / 2.0

        return float(angle_similarity)

    @staticmethod
    def debug_similarity_analysis(q1: torch.Tensor, q2: torch.Tensor, char1: str, char2: str):
        """
        Função de debug para analisar componentes da similaridade.
        """
        q1_flat = q1.flatten()
        q2_flat = q2.flatten()

        # Componentes individuais
        cosine_sim = torch.dot(F.normalize(q1_flat, p=2, dim=0),
                              F.normalize(q2_flat, p=2, dim=0))

        euclidean_dist = torch.norm(q1_flat - q2_flat, p=2)
        max_dist = torch.norm(q1_flat, p=2) + torch.norm(q2_flat, p=2)
        dist_similarity = 1 - (euclidean_dist / max_dist)

        quat_sim = QuantumSimilarityEnhanced._quaternion_angle_similarity(q1, q2)

        final_sim = QuantumSimilarityEnhanced.enhanced_quaternion_similarity(q1, q2)

        print(f"\n🔍 DEBUG Similaridade {char1}-{char2}:")
        print(f"   Similaridade Cosseno: {cosine_sim:.4f}")
        print(f"   Similaridade Distância: {dist_similarity:.4f}")
        print(f"   Similaridade Quaternião: {quat_sim:.4f}")
        print(f"   Similaridade Final: {final_sim:.4f}")
        print(f"   Norma q1: {torch.norm(q1_flat):.4f}, Norma q2: {torch.norm(q2_flat):.4f}")
        print(f"   Distância Euclidiana: {euclidean_dist:.4f}")
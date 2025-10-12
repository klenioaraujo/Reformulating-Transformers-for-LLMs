#!/usr/bin/env python3
"""
ΨQRH Audit Analyzer
Framework de análise para logs de auditoria do pipeline ΨQRH
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import argparse
from datetime import datetime

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  Matplotlib não disponível - gráficos serão desabilitados")


class ΨQRHAuditAnalyzer:
    """Analisador principal de logs de auditoria ΨQRH"""

    def __init__(self, audit_dir: str = "audit_logs"):
        self.audit_dir = Path(audit_dir)
        self.ascii_codes = list(range(32, 127))  # Caracteres ASCII imprimíveis

    def load_audit_log(self, log_file: str) -> Dict[str, Any]:
        """Carrega um arquivo de log de auditoria"""
        log_path = Path(log_file)
        if not log_path.exists():
            raise FileNotFoundError(f"Arquivo de log não encontrado: {log_file}")

        with open(log_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def analyze_reconstruction_error(self, log_file: str) -> Dict[str, float]:
        """
        Análise de Corrupção de Sinal: Calcula erro de reconstrução
        Compara psi_input vs psi_inverted
        """
        log_data = self.load_audit_log(log_file)

        # Encontrar snapshots relevantes
        psi_input_path = None
        psi_inverted_path = None

        for entry in log_data["audit_trail"]:
            if entry["step"] == "qrh_input":
                psi_input_path = entry["tensor_snapshot"]
            elif entry["step"] == "final_inverted_output":
                psi_inverted_path = entry["tensor_snapshot"]

        if not psi_input_path or not psi_inverted_path:
            raise ValueError("Snapshots necessários não encontrados no log")

        # Carregar tensores
        psi_input = torch.load(psi_input_path)
        psi_inverted = torch.load(psi_inverted_path)

        # Garantir que têm o mesmo shape para comparação
        min_seq_len = min(psi_input.shape[1], psi_inverted.shape[1])
        psi_input = psi_input[:, :min_seq_len]
        psi_inverted = psi_inverted[:, :min_seq_len]

        # Calcular métricas de erro
        mse_error = F.mse_loss(psi_input, psi_inverted).item()

        # Similaridade de cosseno (flatten para comparação global)
        psi_input_flat = psi_input.flatten()
        psi_inverted_flat = psi_inverted.flatten()

        cos_similarity = F.cosine_similarity(
            psi_input_flat.unsqueeze(0),
            psi_inverted_flat.unsqueeze(0)
        ).item()

        # Norma relativa (conservação de energia)
        energy_preservation = torch.norm(psi_inverted) / torch.norm(psi_input)

        return {
            "mse_error": mse_error,
            "cosine_similarity": cos_similarity,
            "energy_preservation": energy_preservation.item(),
            "input_norm": torch.norm(psi_input).item(),
            "inverted_norm": torch.norm(psi_inverted).item()
        }

    def generate_ascii_probes(self, embed_dim: int, device: str = "cpu") -> torch.Tensor:
        """
        Gera probes quânticos para todos os caracteres ASCII
        Simplificação: usa embeddings baseados em códigos ASCII
        """
        n_chars = len(self.ascii_codes)
        probes = torch.zeros(n_chars, embed_dim, dtype=torch.float32, device=device)

        for i, ascii_code in enumerate(self.ascii_codes):
            # Embedding simples baseado no código ASCII
            base_value = ascii_code / 127.0  # Normalizar para [0, 1]

            # Criar padrão único para cada caractere
            for j in range(embed_dim):
                probes[i, j] = base_value * torch.sin(torch.tensor(2 * np.pi * j * base_value))

        return probes

    def analyze_embedding_space(self, embed_dim: int, save_heatmap: bool = True) -> Dict[str, Any]:
        """
        Análise de Discriminabilidade: Examina o espaço de embedding dos caracteres
        """
        probes = self.generate_ascii_probes(embed_dim)

        # Calcular matriz de similaridade de cosseno
        n_chars = len(probes)
        similarity_matrix = torch.zeros(n_chars, n_chars)

        for i in range(n_chars):
            for j in range(n_chars):
                if i != j:
                    similarity_matrix[i, j] = F.cosine_similarity(
                        probes[i].unsqueeze(0),
                        probes[j].unsqueeze(0)
                    ).item()

        # Encontrar pares mais similares (mais problemáticos)
        similarity_flat = similarity_matrix.flatten()
        top_similar_indices = torch.topk(similarity_flat, 10).indices

        problematic_pairs = []
        for idx in top_similar_indices:
            i = idx // n_chars
            j = idx % n_chars
            if i < j:  # Evitar duplicatas
                char_i = chr(self.ascii_codes[i])
                char_j = chr(self.ascii_codes[j])
                similarity = similarity_matrix[i, j]
                problematic_pairs.append((char_i, char_j, similarity))

        # Calcular estatísticas de separabilidade
        # Distância média para o vizinho mais próximo
        min_distances = []
        for i in range(n_chars):
            distances = []
            for j in range(n_chars):
                if i != j:
                    dist = torch.norm(probes[i] - probes[j]).item()
                    distances.append(dist)
            min_distances.append(min(distances))

        avg_min_distance = np.mean(min_distances)
        std_min_distance = np.std(min_distances)

        # Gerar heatmap se solicitado e matplotlib disponível
        if save_heatmap and HAS_MATPLOTLIB:
            plt.figure(figsize=(12, 10))
            char_labels = [chr(code) for code in self.ascii_codes]

            # Mostrar apenas uma amostra para visualização (muitos caracteres)
            sample_size = min(50, n_chars)
            sample_indices = np.linspace(0, n_chars-1, sample_size, dtype=int)
            sample_matrix = similarity_matrix[sample_indices][:, sample_indices]
            sample_labels = [char_labels[i] for i in sample_indices]

            sns.heatmap(sample_matrix, xticklabels=sample_labels, yticklabels=sample_labels,
                       cmap='coolwarm', center=0, annot=False)
            plt.title(f'ΨQRH Embedding Space Similarity (embed_dim={embed_dim})')
            plt.tight_layout()
            plt.savefig(f'embedding_similarity_heatmap_{embed_dim}.png', dpi=150, bbox_inches='tight')
            plt.close()
        elif save_heatmap and not HAS_MATPLOTLIB:
            print("⚠️  Matplotlib não disponível - heatmap não será gerado")

        return {
            "embed_dim": embed_dim,
            "avg_min_distance": avg_min_distance,
            "std_min_distance": std_min_distance,
            "most_problematic_pairs": problematic_pairs[:5],  # Top 5
            "similarity_matrix_shape": list(similarity_matrix.shape),
            "heatmap_saved": save_heatmap
        }

    def analyze_contextual_interference(self, log_file: str) -> Dict[str, float]:
        """
        Análise de Interferência Contextual: Examina correlações entre posições adjacentes
        """
        log_data = self.load_audit_log(log_file)

        # Encontrar tensor de input
        psi_input_path = None
        for entry in log_data["audit_trail"]:
            if entry["step"] == "qrh_input":
                psi_input_path = entry["tensor_snapshot"]
                break

        if not psi_input_path:
            raise ValueError("Tensor de input não encontrado no log")

        # Carregar tensor
        psi_sequence = torch.load(psi_input_path)  # Shape: [batch, seq_len, embed_dim]

        if psi_sequence.dim() not in [3, 4]:
            raise ValueError(f"Tensor deve ter 3 ou 4 dimensões, tem {psi_sequence.dim()}")

        if psi_sequence.dim() == 4:
            # Para tensores quaterniônicos [batch, seq_len, embed_dim, 4], reduzir para [batch, seq_len, embed_dim]
            # Usando a magnitude dos quaternions
            psi_sequence = torch.norm(psi_sequence, dim=-1)

        batch_size, seq_len, embed_dim = psi_sequence.shape

        # Calcular autocorrelação entre posições adjacentes
        autocorrelations = []

        for b in range(batch_size):
            for pos in range(seq_len - 1):
                # Estados em posições adjacentes
                psi_current = psi_sequence[b, pos]    # [embed_dim]
                psi_next = psi_sequence[b, pos + 1]   # [embed_dim]

                # Correlação de Pearson
                corr = torch.corrcoef(torch.stack([psi_current, psi_next]))[0, 1]
                autocorrelations.append(corr.item())

        # Estatísticas da autocorrelação
        autocorrelations = np.array(autocorrelations)
        mean_autocorr = np.mean(np.abs(autocorrelations))  # Usar valor absoluto
        std_autocorr = np.std(autocorrelations)
        max_autocorr = np.max(np.abs(autocorrelations))

        # Análise de independência
        # Se autocorrelação > 0.5, considerar alta dependência contextual
        high_correlation_ratio = np.mean(np.abs(autocorrelations) > 0.5)

        return {
            "mean_abs_autocorrelation": mean_autocorr,
            "std_autocorrelation": std_autocorr,
            "max_abs_autocorrelation": max_autocorr,
            "high_correlation_ratio": high_correlation_ratio,
            "sequence_length": seq_len,
            "independence_assumption_valid": mean_autocorr < 0.3  # Threshold arbitrário
        }

    def generate_diagnostic_report(self, log_file: str, embed_dim: int = 64) -> str:
        """
        Gera relatório completo de diagnóstico em Markdown
        """
        log_data = self.load_audit_log(log_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Executar todas as análises
        reconstruction_analysis = self.analyze_reconstruction_error(log_file)
        embedding_analysis = self.analyze_embedding_space(embed_dim)
        contextual_analysis = self.analyze_contextual_interference(log_file)

        # Criar relatório
        parameters_str = json.dumps(log_data.get('parameters', {}), indent=2)
        report = f"""# Relatório de Diagnóstico do Pipeline ΨQRH

**Timestamp:** {timestamp}
**Log File:** {log_file}
**Input Text:** "{log_data.get('input_text', 'N/A')}"
**Parameters:** {parameters_str}

## Análise de Fidelidade da Reconstrução

- **Erro Quadrático Médio (Input vs. Inverted):** {reconstruction_analysis['mse_error']:.6f}
- **Similaridade de Cosseno (Input vs. Inverted):** {reconstruction_analysis['cosine_similarity']:.6f}
- **Preservação de Energia:** {reconstruction_analysis['energy_preservation']:.6f}
- **Norma Input:** {reconstruction_analysis['input_norm']:.6f}
- **Norma Inverted:** {reconstruction_analysis['inverted_norm']:.6f}

### Diagnóstico de Reconstrução
"""

        # Diagnóstico baseado nos valores
        mse = reconstruction_analysis['mse_error']
        cos_sim = reconstruction_analysis['cosine_similarity']
        energy = reconstruction_analysis['energy_preservation']

        if mse < 0.01 and cos_sim > 0.95 and 0.95 <= energy <= 1.05:
            report += "**✅ EXCELENTE:** Reconstrução quase perfeita. Perda mínima de informação.\n"
        elif mse < 0.1 and cos_sim > 0.8 and 0.9 <= energy <= 1.1:
            report += "**⚠️ MODERADO:** Perda de informação detectada. Ciclo de transformação não é perfeitamente reversível.\n"
        else:
            report += "**❌ CRÍTICO:** Perda significativa de informação. Problemas graves de estabilidade numérica.\n"

        report += f"""

## Análise do Espaço de Embedding (dim={embed_dim})

- **Distância Média Mínima:** {embedding_analysis['avg_min_distance']:.6f}
- **Desvio Padrão das Distâncias:** {embedding_analysis['std_min_distance']:.6f}

### Pares de Caracteres Mais Problemáticos
"""

        for char1, char2, similarity in embedding_analysis['most_problematic_pairs']:
            report += f"- **('{char1}', '{char2}')**: Similaridade = {similarity:.6f}\n"

        # Diagnóstico de embedding
        avg_min_dist = embedding_analysis['avg_min_distance']
        if avg_min_dist > 1.0:
            report += "\n### Diagnóstico de Embedding\n**✅ BOM:** Boa separabilidade entre caracteres.\n"
        elif avg_min_dist > 0.5:
            report += "\n### Diagnóstico de Embedding\n**⚠️ MODERADO:** Separabilidade adequada, mas alguns caracteres similares podem causar confusão.\n"
        else:
            report += "\n### Diagnóstico de Embedding\n**❌ CRÍTICO:** Baixa separabilidade. Espaço de embedding muito 'lotado', causando mapeamentos incorretos.\n"

        report += f"""

## Análise de Interferência Contextual

- **Autocorrelação Média (Absoluta):** {contextual_analysis['mean_abs_autocorrelation']:.6f}
- **Desvio Padrão da Autocorrelação:** {contextual_analysis['std_autocorrelation']:.6f}
- **Autocorrelação Máxima (Absoluta):** {contextual_analysis['max_abs_autocorrelation']:.6f}
- **Razão de Alta Correlação (>0.5):** {contextual_analysis['high_correlation_ratio']:.6f}
- **Assunção de Independência Válida:** {contextual_analysis['independence_assumption_valid']}

### Diagnóstico Contextual
"""

        mean_autocorr = contextual_analysis['mean_abs_autocorrelation']
        independence_valid = contextual_analysis['independence_assumption_valid']

        if mean_autocorr < 0.2 and independence_valid:
            report += "**✅ BOM:** Baixa interferência contextual. Assunção de independência é válida.\n"
        elif mean_autocorr < 0.5:
            report += "**⚠️ MODERADO:** Interferência contextual moderada. Método de probing pode ter limitações.\n"
        else:
            report += "**❌ CRÍTICO:** Alta interferência contextual. Assunção de independência é **inválida**. Estados quânticos contêm fortes 'ecos' de vizinhos.\n"

        # Conclusão
        report += f"""

## Conclusão e Recomendações

### Problemas Identificados
"""

        issues = []
        recommendations = []

        # Análise de reconstrução
        if reconstruction_analysis['mse_error'] > 0.1:
            issues.append("Perda significativa de informação na reconstrução")
            recommendations.append("Investigar acumulação de erros numéricos em operações FFT/filtro")

        # Análise de embedding
        if embedding_analysis['avg_min_distance'] < 0.5:
            issues.append("Baixa separabilidade no espaço de embedding")
            recommendations.append("Aumentar embed_dim ou implementar melhor estratégia de embedding")

        # Análise contextual
        if not contextual_analysis['independence_assumption_valid']:
            issues.append("Interferência contextual viola assunção de independência")
            recommendations.append("Implementar probing contextual que considere dependências sequenciais")

        if not issues:
            report += "- ✅ Nenhum problema crítico identificado\n"
        else:
            for issue in issues:
                report += f"- ❌ {issue}\n"

        report += "\n### Recomendações\n"
        if not recommendations:
            report += "- ✅ Sistema funcionando adequadamente\n"
        else:
            for rec in recommendations:
                report += f"- 🔧 {rec}\n"

        # Salvar relatório
        report_filename = f"diagnostic_report_{timestamp}.md"
        report_path = Path(f"reports/{report_filename}")
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"Relatório salvo em: {report_path}")
        return report


def main():
    """Função principal para linha de comando"""
    parser = argparse.ArgumentParser(description="ΨQRH Audit Analyzer")
    parser.add_argument("log_file", help="Arquivo de log de auditoria para analisar")
    parser.add_argument("--embed-dim", type=int, default=64, help="Dimensão do embedding para análise")
    parser.add_argument("--no-heatmap", action="store_true", help="Não gerar heatmap de similaridade")

    args = parser.parse_args()

    analyzer = ΨQRHAuditAnalyzer()

    try:
        # Executar análise completa
        report = analyzer.generate_diagnostic_report(
            args.log_file,
            embed_dim=args.embed_dim
        )

        print("Análise completa executada com sucesso!")
        print("Verifique o relatório gerado para detalhes.")

    except Exception as e:
        print(f"Erro durante análise: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
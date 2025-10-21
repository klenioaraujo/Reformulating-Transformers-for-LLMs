#!/usr/bin/env python3
"""
Benchmark PsiQRH com Semântica Bilateral (biPsiQRH)

Este script executa benchmarks completos do pipeline PsiQRH
com análise semântica bilateral nos datasets GLUE.
"""

import torch
import json
from typing import Dict, List
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from psi_qrh_semantic_pipeline import PsiQRHSemanticPipeline
from semantic_processing_pipeline import SemanticProcessingPipeline
from psi_qrh_evaluator import PsiQRHEvaluator

def main():
    # Configuração e execução do pipeline
    print("🚀 Inicializando Pipeline biPsiQRH...")

    # Inicialização
    pipeline = PsiQRHSemanticPipeline('psi_qrh_config.json')
    semantic_pipeline = SemanticProcessingPipeline(pipeline)
    evaluator = PsiQRHEvaluator(semantic_pipeline)

    # Processamento de exemplo
    text = "O banco tem capital suficiente para investimentos"
    result = semantic_pipeline.process_text(text, 'ag_news')

    print(f"Predição: {result['prediction']}")
    print(f"Método: {result['method']}")
    print(f"Explicação: {result['explanation']}")

    # Avaliação comparativa
    print("\n📊 Executando avaliação comparativa...")
    comparison = evaluator.comparative_analysis()

    print("\nAnálise Comparativa:")
    total_improvement = 0
    for dataset, stats in comparison.items():
        improvement_pct = stats['improvement']
        semantic_rate_pct = stats['semantic_usage_rate'] * 100
        print(f"{dataset}: {improvement_pct:.2f}% de melhoria")
        print(".2f")
        total_improvement += improvement_pct

    avg_improvement = total_improvement / len(comparison)
    print(".2f")

    # Análise detalhada do RTE e WNLI
    print("\n🎯 Análise de Desambiguização Semântica:")
    for dataset in ['rte', 'wnli']:
        if dataset in comparison:
            stats = comparison[dataset]
            print(f"{dataset.upper()}:")
            print(".2f")
            print(".1f")

    print("\n✅ Benchmark biPsiQRH concluído!")

if __name__ == "__main__":
    main()
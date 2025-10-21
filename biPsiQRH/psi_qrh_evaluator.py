import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np
from semantic_processing_pipeline import SemanticProcessingPipeline

class PsiQRHEvaluator:
    def __init__(self, pipeline: SemanticProcessingPipeline):
        self.pipeline = pipeline
        self.datasets = {
            'ag_news': self._load_ag_news,
            'imdb': self._load_imdb,
            'rte': self._load_rte,
            'wnli': self._load_wnli
        }

    def evaluate_on_dataset(self, dataset_name: str, samples: int = 1000) -> Dict:
        """Avaliação completa no dataset específico"""

        dataset = self.datasets[dataset_name]()
        results = {
            'accuracy': 0,
            'f1_score': 0,
            'semantic_usage': 0,  # % de vezes que usou análise semântica
            'improvement_over_baseline': 0
        }

        correct = 0
        semantic_decisions = 0

        for i, (text, label) in enumerate(dataset[:samples]):
            prediction = self.pipeline.process_text(text, dataset_name)

            if prediction['prediction'] == label:
                correct += 1

            if prediction['method'] == 'semantic':
                semantic_decisions += 1

        results['accuracy'] = correct / samples
        results['semantic_usage'] = semantic_decisions / samples

        return results

    def comparative_analysis(self) -> Dict:
        """Análise comparativa entre abordagens"""
        baseline_results = self._load_baseline_results()  # Resultados originais do JSON

        comparison = {}
        for dataset in ['ag_news', 'imdb', 'rte', 'wnli']:
            new_results = self.evaluate_on_dataset(dataset)
            old_accuracy = baseline_results[dataset]['accuracy']

            comparison[dataset] = {
                'old_accuracy': old_accuracy,
                'new_accuracy': new_results['accuracy'],
                'improvement': ((new_results['accuracy'] - old_accuracy) / old_accuracy) * 100 if old_accuracy > 0 else 0,
                'semantic_usage_rate': new_results['semantic_usage']
            }

        return comparison

    def _load_ag_news(self) -> List[Tuple[str, int]]:
        """Carrega dataset AG News (simulado)"""
        # Simulação de dados do AG News
        return [
            ("O banco tem capital suficiente para investimentos", 2),  # Business
            ("Gato persegue rato no jardim", 0),  # World
            ("Novo avanço em tecnologia quântica", 3),  # Sci/Tech
            ("Time vence campeonato nacional", 1),  # Sports
        ] * 250  # Repetir para ter 1000 amostras

    def _load_imdb(self) -> List[Tuple[str, int]]:
        """Carrega dataset IMDB (simulado)"""
        return [
            ("Este filme é excelente, recomendo muito", 1),  # Positive
            ("Filme terrível, não perca seu tempo", 0),  # Negative
            ("Atuação incrível e roteiro envolvente", 1),  # Positive
            ("Produção ruim e história previsível", 0),  # Negative
        ] * 250

    def _load_rte(self) -> List[Tuple[str, int]]:
        """Carrega dataset RTE (Recognizing Textual Entailment)"""
        return [
            ("O banco tem dinheiro. O banco é rico.", 1),  # Entailment
            ("O gato caça ratos. O gato é predador.", 1),  # Entailment
            ("O banco está vazio. O banco tem clientes.", 0),  # Not entailment
            ("O rato foge do gato. O rato é presa.", 1),  # Entailment
        ] * 250

    def _load_wnli(self) -> List[Tuple[str, int]]:
        """Carrega dataset WNLI (Winograd NLI)"""
        return [
            ("João ama Maria porque ela é inteligente. Ela é amada por João.", 1),  # Correct
            ("O banco está fechado porque é feriado. Ele não abre hoje.", 1),  # Correct
            ("O rato foi pego pelo gato. Ele estava com medo.", 0),  # Incorrect
            ("Maria comprou o livro porque estava barato. Ele custava pouco.", 1),  # Correct
        ] * 250

    def _load_baseline_results(self) -> Dict:
        """Carrega resultados baseline do JSON"""
        # Simulação dos resultados originais
        return {
            'ag_news': {'accuracy': 0.85},
            'imdb': {'accuracy': 0.82},
            'rte': {'accuracy': 0.65},
            'wnli': {'accuracy': 0.70}
        }
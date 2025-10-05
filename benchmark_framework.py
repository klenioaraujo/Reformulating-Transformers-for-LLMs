#!/usr/bin/env python3
"""
Framework de Benchmark Comparativo ΨQRH vs Baseline Transformer

Implementa comparação justa entre:
- Baseline Transformer (GPT-2 small, ~82M parâmetros)
- ΨQRH Transformer (mesma capacidade, ~82M parâmetros)

Usa mesmo dataset, tokenizer, hardware e métricas.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime

from src.core.qrh_layer import QRHLayer, QRHConfig
from src.core.negentropy_transformer_block import NegentropyTransformerBlock

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuração comum para ambos os modelos"""
    vocab_size: int = 50257  # GPT-2 vocab size
    max_seq_len: int = 1024
    num_layers: int = 12
    num_heads: int = 12
    embed_dim: int = 768
    ff_dim: int = 3072
    dropout: float = 0.1

class BaselineTransformer(nn.Module):
    """Baseline Transformer baseado no GPT-2 small (~82M parâmetros)"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

        # Camadas Transformer
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.embed_dim,
                nhead=config.num_heads,
                dim_feedforward=config.ff_dim,
                dropout=config.dropout,
                batch_first=True
            )
            for _ in range(config.num_layers)
        ])

        # Layer normalization final
        self.layer_norm = nn.LayerNorm(config.embed_dim)

        # Head de saída
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(pos_ids)

        x = token_embeds + pos_embeds

        # Aplicar camadas Transformer
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=attention_mask)

        x = self.layer_norm(x)

        # Head de linguagem
        logits = self.lm_head(x)

        return logits

    @property
    def num_parameters(self) -> int:
        """Calcula número total de parâmetros"""
        return sum(p.numel() for p in self.parameters())

class PsiQRHTransformer(nn.Module):
    """ΨQRH Transformer com mesma capacidade que baseline (~82M parâmetros)"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

        # Configuração ΨQRH
        qrh_config = QRHConfig(
            embed_dim=config.embed_dim // 4,  # Dividir por 4 para quaternions
            alpha=1.0,
            use_learned_rotation=True
        )

        # Camadas ΨQRH Transformer
        self.layers = nn.ModuleList([
            NegentropyTransformerBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                ff_dim=config.ff_dim,
                dropout=config.dropout,
                qrh_config=qrh_config
            )
            for _ in range(config.num_layers)
        ])

        # Layer normalization final
        self.layer_norm = nn.LayerNorm(config.embed_dim)

        # Head de saída
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(pos_ids)

        x = token_embeds + pos_embeds

        # Aplicar camadas ΨQRH Transformer
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        x = self.layer_norm(x)

        # Head de linguagem
        logits = self.lm_head(x)

        return logits

    @property
    def num_parameters(self) -> int:
        """Calcula número total de parâmetros"""
        return sum(p.numel() for p in self.parameters())

@dataclass
class TrainingMetrics:
    """Métricas de treinamento"""
    step: int
    loss: float
    perplexity: float
    learning_rate: float
    tokens_per_second: float
    gpu_memory_mb: float
    timestamp: str

class BenchmarkTrainer:
    """Treinador para benchmark comparativo"""

    def __init__(self, model: nn.Module, config: ModelConfig, device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Otimizador
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=5e-5,
            weight_decay=0.01
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_seq_len * 10000  # Aproximação
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Métricas
        self.metrics: List[TrainingMetrics] = []

    def calculate_perplexity(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Calcula perplexidade"""
        with torch.no_grad():
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            perplexity = torch.exp(loss).item()
        return perplexity

    def train_step(self, batch: Dict[str, torch.Tensor]) -> TrainingMetrics:
        """Executa um passo de treinamento"""
        start_time = time.time()

        # Mover dados para dispositivo
        input_ids = batch['input_ids'].to(self.device)
        targets = batch['labels'].to(self.device) if 'labels' in batch else input_ids

        # Forward pass
        self.optimizer.zero_grad()
        logits = self.model(input_ids)

        # Calcular loss
        loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Calcular métricas
        step_time = time.time() - start_time
        tokens_processed = input_ids.numel()
        tokens_per_second = tokens_processed / step_time if step_time > 0 else 0

        perplexity = self.calculate_perplexity(logits, targets)

        # Memória GPU
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        else:
            gpu_memory = 0.0

        metrics = TrainingMetrics(
            step=len(self.metrics) + 1,
            loss=loss.item(),
            perplexity=perplexity,
            learning_rate=self.optimizer.param_groups[0]['lr'],
            tokens_per_second=tokens_per_second,
            gpu_memory_mb=gpu_memory,
            timestamp=datetime.now().isoformat()
        )

        self.metrics.append(metrics)
        return metrics

class BenchmarkEvaluator:
    """Avaliador para benchmark"""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Avalia modelo no dataset de validação"""
        total_loss = 0.0
        total_tokens = 0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                targets = batch['labels'].to(self.device) if 'labels' in batch else input_ids

                logits = self.model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

                total_loss += loss.item() * input_ids.numel()
                total_tokens += input_ids.numel()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)

        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'total_tokens': total_tokens
        }

class FairComparisonBenchmark:
    """Benchmark de comparação justa"""

    def __init__(self, config: ModelConfig, device: torch.device):
        self.config = config
        self.device = device

        # Criar modelos
        self.baseline_model = BaselineTransformer(config)
        self.psiqrh_model = PsiQRHTransformer(config)

        # Verificar tamanho dos modelos
        baseline_params = self.baseline_model.num_parameters
        psiqrh_params = self.psiqrh_model.num_parameters

        logger.info(f"Baseline Transformer: {baseline_params:,} parâmetros")
        logger.info(f"ΨQRH Transformer: {psiqrh_params:,} parâmetros")

        # Verificar se estão dentro de 5% de diferença
        param_ratio = abs(baseline_params - psiqrh_params) / baseline_params
        if param_ratio > 0.05:
            logger.warning(f"Diferença de parâmetros: {param_ratio:.2%} (limite: 5%)")

    def run_training_benchmark(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
                              num_steps: int = 1000) -> Dict[str, Any]:
        """Executa benchmark de treinamento"""
        logger.info("Iniciando benchmark de treinamento...")

        results = {
            'baseline': {},
            'psiqrh': {},
            'comparison': {}
        }

        # Benchmark Baseline Transformer
        logger.info("Treinando Baseline Transformer...")
        baseline_trainer = BenchmarkTrainer(self.baseline_model, self.config, self.device)
        baseline_evaluator = BenchmarkEvaluator(self.baseline_model, self.device)

        baseline_metrics = self._train_model(baseline_trainer, train_dataloader, num_steps)
        baseline_val_results = baseline_evaluator.evaluate(val_dataloader)

        results['baseline'] = {
            'training_metrics': baseline_metrics,
            'validation_results': baseline_val_results,
            'num_parameters': self.baseline_model.num_parameters
        }

        # Benchmark ΨQRH Transformer
        logger.info("Treinando ΨQRH Transformer...")
        psiqrh_trainer = BenchmarkTrainer(self.psiqrh_model, self.config, self.device)
        psiqrh_evaluator = BenchmarkEvaluator(self.psiqrh_model, self.device)

        psiqrh_metrics = self._train_model(psiqrh_trainer, train_dataloader, num_steps)
        psiqrh_val_results = psiqrh_evaluator.evaluate(val_dataloader)

        results['psiqrh'] = {
            'training_metrics': psiqrh_metrics,
            'validation_results': psiqrh_val_results,
            'num_parameters': self.psiqrh_model.num_parameters
        }

        # Comparação
        results['comparison'] = self._compare_results(results['baseline'], results['psiqrh'])

        return results

    def _train_model(self, trainer: BenchmarkTrainer, dataloader: DataLoader, num_steps: int) -> List[TrainingMetrics]:
        """Treina modelo e coleta métricas"""
        metrics = []

        # Criar iterador infinito do dataloader
        data_iter = iter(dataloader)

        for step in range(num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            metrics_step = trainer.train_step(batch)
            metrics.append(metrics_step)

            if step % 100 == 0:
                logger.info(f"Step {step}: Loss={metrics_step.loss:.4f}, PPL={metrics_step.perplexity:.2f}")

        return metrics

    def _compare_results(self, baseline: Dict[str, Any], psiqrh: Dict[str, Any]) -> Dict[str, Any]:
        """Compara resultados dos dois modelos"""
        comparison = {}

        # Comparar perplexidade final
        baseline_ppl = baseline['validation_results']['perplexity']
        psiqrh_ppl = psiqrh['validation_results']['perplexity']
        comparison['perplexity_ratio'] = psiqrh_ppl / baseline_ppl

        # Comparar velocidade de treinamento
        baseline_tps = np.mean([m.tokens_per_second for m in baseline['training_metrics'][-100:]])
        psiqrh_tps = np.mean([m.tokens_per_second for m in psiqrh['training_metrics'][-100:]])
        comparison['speed_ratio'] = psiqrh_tps / baseline_tps

        # Comparar uso de memória
        baseline_mem = np.mean([m.gpu_memory_mb for m in baseline['training_metrics'][-100:]])
        psiqrh_mem = np.mean([m.gpu_memory_mb for m in psiqrh['training_metrics'][-100:]])
        comparison['memory_ratio'] = psiqrh_mem / baseline_mem

        # Comparar convergência
        baseline_final_loss = baseline['training_metrics'][-1].loss
        psiqrh_final_loss = psiqrh['training_metrics'][-1].loss
        comparison['loss_ratio'] = psiqrh_final_loss / baseline_final_loss

        return comparison

class BenchmarkReportGenerator:
    """Gerador de relatórios de benchmark"""

    def __init__(self, output_dir: str = "benchmark_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_report(self, benchmark_results: Dict[str, Any], config: ModelConfig) -> str:
        """Gera relatório completo de benchmark"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"benchmark_report_{timestamp}.json"

        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'framework': 'ΨQRH Fair Comparison Benchmark',
                'model_config': {
                    'vocab_size': config.vocab_size,
                    'max_seq_len': config.max_seq_len,
                    'num_layers': config.num_layers,
                    'num_heads': config.num_heads,
                    'embed_dim': config.embed_dim,
                    'ff_dim': config.ff_dim
                }
            },
            'results': benchmark_results
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=self._json_serializer)

        # Gerar resumo em markdown
        self._generate_markdown_summary(report, timestamp)

        logger.info(f"Relatório de benchmark gerado: {report_path}")
        return str(report_path)

    def _json_serializer(self, obj):
        """Serializador customizado para objetos complexos"""
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)

    def _generate_markdown_summary(self, report: Dict[str, Any], timestamp: str):
        """Gera resumo em markdown"""
        md_path = self.output_dir / f"benchmark_summary_{timestamp}.md"

        results = report['results']
        comparison = results['comparison']

        with open(md_path, 'w') as f:
            f.write("# Relatório de Benchmark ΨQRH vs Baseline\n\n")
            f.write(f"**Data**: {report['metadata']['timestamp']}\n\n")

            f.write("## Resumo Executivo\n\n")
            f.write(f"- **Baseline Parameters**: {results['baseline']['num_parameters']:,}\\n")
            f.write(f"- **ΨQRH Parameters**: {results['psiqrh']['num_parameters']:,}\\n")
            f.write(f"- **Perplexity Ratio**: {comparison['perplexity_ratio']:.3f}\\n")
            f.write(f"- **Speed Ratio**: {comparison['speed_ratio']:.3f}\\n")
            f.write(f"- **Memory Ratio**: {comparison['memory_ratio']:.3f}\n\n")

            f.write("## Resultados Detalhados\n\n")
            f.write("### Baseline Transformer\n\n")
            baseline_val = results['baseline']['validation_results']
            f.write(f"- **Final Perplexity**: {baseline_val['perplexity']:.2f}\\n")
            f.write(f"- **Final Loss**: {baseline_val['loss']:.4f}\n\n")

            f.write("### ΨQRH Transformer\n\n")
            psiqrh_val = results['psiqrh']['validation_results']
            f.write(f"- **Final Perplexity**: {psiqrh_val['perplexity']:.2f}\\n")
            f.write(f"- **Final Loss**: {psiqrh_val['loss']:.4f}\n\n")

            f.write("## Análise de Trade-offs\n\n")
            f.write("| Métrica | Baseline | ΨQRH | Ratio |\n")
            f.write("|---------|----------|------|-------|\n")
            f.write(f"| Perplexity | {baseline_val['perplexity']:.2f} | {psiqrh_val['perplexity']:.2f} | {comparison['perplexity_ratio']:.3f} |\n")
            f.write(f"| Speed (tokens/s) | {np.mean([m.tokens_per_second for m in results['baseline']['training_metrics'][-100:]]):.0f} | {np.mean([m.tokens_per_second for m in results['psiqrh']['training_metrics'][-100:]]):.0f} | {comparison['speed_ratio']:.3f} |\n")
            f.write(f"| Memory (MB) | {np.mean([m.gpu_memory_mb for m in results['baseline']['training_metrics'][-100:]]):.0f} | {np.mean([m.gpu_memory_mb for m in results['psiqrh']['training_metrics'][-100:]]):.0f} | {comparison['memory_ratio']:.3f} |\n")

def main():
    """Função principal de demonstração"""
    # Configuração
    config = ModelConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Framework de Benchmark ΨQRH")
    print("=" * 50)
    print(f"Dispositivo: {device}")

    # Criar benchmark
    benchmark = FairComparisonBenchmark(config, device)

    # TODO: Implementar dataloader real com OpenWebText
    # Por enquanto, usar dados dummy para demonstração
    print("\n⚠️  Benchmark usando dados dummy (implementar dataloader real)")

    # Gerar relatório de configuração
    report_generator = BenchmarkReportGenerator()

    # Relatório básico de configuração
    basic_results = {
        'baseline': {'num_parameters': benchmark.baseline_model.num_parameters},
        'psiqrh': {'num_parameters': benchmark.psiqrh_model.num_parameters},
        'comparison': {'parameter_difference_ratio':
            abs(benchmark.baseline_model.num_parameters - benchmark.psiqrh_model.num_parameters) /
            benchmark.baseline_model.num_parameters}
    }

    report_path = report_generator.generate_report(basic_results, config)
    print(f"Relatório gerado: {report_path}")

if __name__ == "__main__":
    main()
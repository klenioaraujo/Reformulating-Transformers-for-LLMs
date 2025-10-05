#!/usr/bin/env python3
"""
Benchmark Training and Evaluation Pipeline for ΨQRH Transformer

This script trains and evaluates PsiQRHTransformer and baseline models
on GLUE (classification) and WikiText-103 (language modeling) benchmarks.

Usage examples:
  # Train ΨQRH on WikiText-103
  python3 run_benchmark.py --model_name psiqrh --benchmark wikitext --d_model 256 --n_layers 4

  # Train standard transformer on GLUE SST-2
  python3 run_benchmark.py --model_name standard --benchmark glue --task_name sst2 --d_model 256 --n_layers 4
"""

import argparse
import os
import sys
import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, DataCollatorWithPadding
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.architecture.psiqrh_transformer import PsiQRHTransformer


class PsiQRHForLanguageModeling(nn.Module):
    """
    Wrapper for PsiQRHTransformer for language modeling tasks.
    Compatible with Hugging Face Trainer.
    """

    def __init__(self, vocab_size: int, **model_kwargs):
        super().__init__()
        self.transformer = PsiQRHTransformer(vocab_size=vocab_size, **model_kwargs)
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Forward pass for language modeling.

        Args:
            input_ids: Token indices [batch_size, seq_len]
            labels: Target token indices [batch_size, seq_len]

        Returns:
            dict with loss and logits
        """
        logits = self.transformer(input_ids)  # [batch_size, seq_len, vocab_size]

        loss = None
        if labels is not None:
            # Shift logits and labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size),
                           shift_labels.view(-1))

        return {'loss': loss, 'logits': logits}


class PsiQRHForSequenceClassification(nn.Module):
    """
    Wrapper for PsiQRHTransformer for sequence classification tasks.
    Compatible with Hugging Face Trainer.
    """

    def __init__(self, vocab_size: int, num_labels: int, **model_kwargs):
        super().__init__()
        self.transformer = PsiQRHTransformer(vocab_size=vocab_size, **model_kwargs)
        self.num_labels = num_labels

        # Classification head
        d_model = model_kwargs.get('d_model', 512)
        quaternion_multiplier = model_kwargs.get('quaternion_multiplier', 4)
        self.classifier = nn.Linear(d_model * quaternion_multiplier, num_labels)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        Forward pass for sequence classification.

        Args:
            input_ids: Token indices [batch_size, seq_len]
            labels: Target labels [batch_size]

        Returns:
            dict with loss and logits
        """
        # Get transformer output
        transformer_output = self.transformer(input_ids)  # [batch_size, seq_len, vocab_size]

        # Use [CLS] token representation for classification
        # For now, use mean pooling across sequence
        pooled_output = transformer_output.mean(dim=1)  # [batch_size, d_model * quaternion_multiplier]

        # Classification layer
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {'loss': loss, 'logits': logits}


class StandardTransformerForLanguageModeling(nn.Module):
    """
    Standard Transformer for language modeling (baseline).
    """

    def __init__(self, vocab_size: int, d_model: int = 512, n_layers: int = 6,
                 n_heads: int = 8, dim_feedforward: int = 2048, max_seq_length: int = 1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Standard transformer components
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_length, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        seq_len = input_ids.size(1)

        # Embedding + positional encoding
        x = self.embedding(input_ids)
        x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer
        x = self.transformer(x)

        # Output projection
        logits = self.output_projection(x)

        loss = None
        if labels is not None:
            # Shift for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size),
                           shift_labels.view(-1))

        return {'loss': loss, 'logits': logits}


class StandardTransformerForSequenceClassification(nn.Module):
    """
    Standard Transformer for sequence classification (baseline).
    """

    def __init__(self, vocab_size: int, num_labels: int, d_model: int = 512,
                 n_layers: int = 6, n_heads: int = 8, dim_feedforward: int = 2048,
                 max_seq_length: int = 1024):
        super().__init__()
        self.num_labels = num_labels
        self.d_model = d_model

        # Standard transformer components
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_length, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        seq_len = input_ids.size(1)

        # Embedding + positional encoding
        x = self.embedding(input_ids)
        x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer
        x = self.transformer(x)

        # Classification (use [CLS] equivalent - mean pooling)
        pooled_output = x.mean(dim=1)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return {'loss': loss, 'logits': logits}


def load_and_prepare_data(args):
    """
    Load and preprocess data for the specified benchmark and task.

    Args:
        args: Command line arguments

    Returns:
        tokenizer, tokenized_datasets, data_collator
    """

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.benchmark == 'wikitext':
        # Load WikiText-103 dataset
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')

        # Tokenization function for language modeling
        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, max_length=args.max_seq_length)

        # Tokenize dataset
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )

        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,  # Causal language modeling
            return_tensors='pt'
        )

        return tokenizer, tokenized_datasets, data_collator

    elif args.benchmark == 'glue':
        # Load GLUE dataset
        dataset = load_dataset('glue', args.task_name)

        # Get sentence columns based on task
        if args.task_name in ['cola', 'sst2']:
            sentence_keys = ['sentence']
        elif args.task_name in ['mrpc', 'qqp', 'stsb']:
            sentence_keys = ['sentence1', 'sentence2']
        elif args.task_name in ['mnli', 'mnli_mismatched', 'mnli_matched']:
            sentence_keys = ['premise', 'hypothesis']
        elif args.task_name in ['qnli', 'rte']:
            sentence_keys = ['question', 'sentence']
        else:
            sentence_keys = ['sentence1', 'sentence2']  # fallback

        # Tokenization function for classification
        def tokenize_function(examples):
            if len(sentence_keys) == 1:
                return tokenizer(examples[sentence_keys[0]], truncation=True, max_length=args.max_seq_length)
            else:
                return tokenizer(examples[sentence_keys[0]], examples[sentence_keys[1]],
                               truncation=True, max_length=args.max_seq_length)

        # Tokenize dataset
        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names
        )

        # Data collator for classification
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        return tokenizer, tokenized_datasets, data_collator

    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")


def compute_metrics_lm(eval_pred):
    """
    Compute metrics for language modeling (perplexity).
    """
    predictions, labels = eval_pred

    # Calculate perplexity
    loss_fct = nn.CrossEntropyLoss()
    shift_logits = predictions[..., :-1, :]
    shift_labels = labels[..., 1:]

    loss = loss_fct(shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1))
    perplexity = torch.exp(loss).item()

    return {'perplexity': perplexity, 'loss': loss.item()}


def compute_metrics_classification(eval_pred):
    """
    Compute metrics for classification tasks.
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')

    return {'accuracy': accuracy, 'f1': f1}


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate models on NLP benchmarks')

    # Model selection
    parser.add_argument('--model_name', type=str, required=True,
                       choices=['psiqrh', 'standard'],
                       help='Model to train (psiqrh or standard)')

    # Benchmark selection
    parser.add_argument('--benchmark', type=str, required=True,
                       choices=['wikitext', 'glue'],
                       help='Benchmark to evaluate on')

    # GLUE task (only for glue benchmark)
    parser.add_argument('--task_name', type=str, default='sst2',
                       choices=['cola', 'sst2', 'mrpc', 'qqp', 'stsb',
                               'mnli', 'mnli_mismatched', 'mnli_matched',
                               'qnli', 'rte', 'wnli', 'ax'],
                       help='GLUE task name (only for glue benchmark)')

    # Model architecture
    parser.add_argument('--d_model', type=int, default=256,
                       help='Model dimension')
    parser.add_argument('--n_layers', type=int, default=4,
                       help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--dim_feedforward', type=int, default=1024,
                       help='Feed-forward dimension')
    parser.add_argument('--max_seq_length', type=int, default=128,
                       help='Maximum sequence length')

    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')

    # Output
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f"Starting benchmark: {args.model_name} on {args.benchmark}")

    # Load and prepare data
    logger.info("Loading and preprocessing data...")
    tokenizer, tokenized_datasets, data_collator = load_and_prepare_data(args)

    # Determine vocab size and number of labels
    vocab_size = len(tokenizer)
    num_labels = None
    if args.benchmark == 'glue':
        # Get number of labels from dataset
        num_labels = len(set(tokenized_datasets['train']['label']))

    # Create model
    logger.info("Creating model...")
    model_kwargs = {
        'd_model': args.d_model,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'dim_feedforward': args.dim_feedforward,
        'max_seq_length': args.max_seq_length
    }

    if args.model_name == 'psiqrh':
        if args.benchmark == 'wikitext':
            model = PsiQRHForLanguageModeling(vocab_size=vocab_size, **model_kwargs)
        else:
            model = PsiQRHForSequenceClassification(
                vocab_size=vocab_size, num_labels=num_labels, **model_kwargs
            )
    else:  # standard
        if args.benchmark == 'wikitext':
            model = StandardTransformerForLanguageModeling(
                vocab_size=vocab_size, **model_kwargs
            )
        else:
            model = StandardTransformerForSequenceClassification(
                vocab_size=vocab_size, num_labels=num_labels, **model_kwargs
            )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=100,
        eval_steps=500,
        save_steps=1000,
        evaluation_strategy='steps',
        save_strategy='steps',
        load_best_model_at_end=True,
        metric_for_best_model='loss' if args.benchmark == 'wikitext' else 'accuracy',
        greater_is_better=False if args.benchmark == 'wikitext' else True,
    )

    # Select compute_metrics function
    if args.benchmark == 'wikitext':
        compute_metrics = compute_metrics_lm
    else:
        compute_metrics = compute_metrics_classification

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train model
    logger.info("Starting training...")
    trainer.train()

    # Evaluate model
    logger.info("Evaluating model...")
    eval_results = trainer.evaluate()

    # Save results
    results_file = os.path.join(args.output_dir, 'results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Benchmark: {args.benchmark}\n")
        if args.benchmark == 'glue':
            f.write(f"Task: {args.task_name}\n")
        f.write(f"Architecture: d_model={args.d_model}, layers={args.n_layers}, heads={args.n_heads}\n")
        f.write(f"Training: lr={args.learning_rate}, batch_size={args.batch_size}, epochs={args.num_epochs}\n")
        f.write("\nEvaluation Results:\n")
        for key, value in eval_results.items():
            f.write(f"{key}: {value}\n")

    logger.info(f"Training and evaluation completed! Results saved to {results_file}")
    logger.info(f"Final results: {eval_results}")


if __name__ == '__main__':
    main()
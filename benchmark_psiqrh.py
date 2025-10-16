#!/usr/bin/env python3
"""
ΨQRH LLM Benchmark Evaluation Script

Evaluates the ΨQRH language model on GLUE tasks and perplexity benchmarks.
Compatible with the model implementation in psiqrh_llm.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import math
from tqdm import tqdm

# Import ΨQRH model
from psiqrh_llm import PsiQRHConfig, PsiQRHForCausalLM
from psiqrh_tokenizer import create_psiqrh_tokenizer


class GLUEDataset(Dataset):
    """Dataset wrapper for GLUE tasks"""

    def __init__(self, dataset, tokenizer, task_name: str, max_length: int = 512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.max_length = max_length

        # Task-specific configurations
        self.task_configs = {
            'cola': {'sentence_keys': ['sentence'], 'num_labels': 2},
            'sst2': {'sentence_keys': ['sentence'], 'num_labels': 2},
            'mrpc': {'sentence_keys': ['sentence1', 'sentence2'], 'num_labels': 2},
            'qqp': {'sentence_keys': ['question1', 'question2'], 'num_labels': 2},
            'stsb': {'sentence_keys': ['sentence1', 'sentence2'], 'num_labels': 1},
            'mnli': {'sentence_keys': ['premise', 'hypothesis'], 'num_labels': 3},
            'qnli': {'sentence_keys': ['question', 'sentence'], 'num_labels': 2},
            'rte': {'sentence_keys': ['sentence1', 'sentence2'], 'num_labels': 2},
            'wnli': {'sentence_keys': ['sentence1', 'sentence2'], 'num_labels': 2}
        }

        if task_name not in self.task_configs:
            raise ValueError(f"Unsupported task: {task_name}")

        self.config = self.task_configs[task_name]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Extract sentences based on task
        if len(self.config['sentence_keys']) == 1:
            text = item[self.config['sentence_keys'][0]]
        else:
            text = item[self.config['sentence_keys'][0]] + " [SEP] " + item[self.config['sentence_keys'][1]]

        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }


class WikiTextDataset(Dataset):
    """Dataset for WikiText perplexity evaluation"""

    def __init__(self, dataset, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Process dataset
        self.sequences = []
        for item in dataset:
            text = item['text']
            if text.strip():  # Skip empty lines
                tokens = tokenizer.encode(text.strip())
                if len(tokens) >= 50:  # Minimum sequence length
                    self.sequences.append(tokens)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.sequences[idx]

        # Truncate if too long
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        # Pad to max_length if needed
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))

        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(tokens, dtype=torch.long)
        }


def load_glue_data(task_name: str, tokenizer, batch_size: int = 32, max_length: int = 512):
    """Load GLUE dataset for evaluation"""
    print(f"Loading GLUE task: {task_name}")

    # Load dataset
    dataset = load_dataset('glue', task_name)

    # Create datasets
    train_dataset = GLUEDataset(dataset['train'], tokenizer, task_name, max_length)
    val_dataset = GLUEDataset(dataset['validation'], tokenizer, task_name, max_length)
    test_dataset = GLUEDataset(dataset['test'], tokenizer, task_name, max_length)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_wikitext_data(tokenizer, batch_size: int = 8, max_length: int = 1024):
    """Load WikiText-103 for perplexity evaluation"""
    print("Loading WikiText-103 dataset")

    # Load dataset
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')

    # Create datasets
    val_dataset = WikiTextDataset(dataset['validation'], tokenizer, max_length)
    test_dataset = WikiTextDataset(dataset['test'], tokenizer, max_length)

    # Create data loaders
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return val_loader, test_loader


def create_model_for_glue(vocab_size: int, num_labels: int, model_config: Dict[str, Any]):
    """Create ΨQRH model adapted for GLUE classification"""
    config_dict = model_config.copy()
    config_dict['vocab_size'] = vocab_size
    config = PsiQRHConfig(**config_dict)

    # Create base model
    model = PsiQRHForCausalLM(config)

    # Add classification head
    class PsiQRHForSequenceClassification(nn.Module):
        def __init__(self, base_model, num_labels):
            super().__init__()
            self.base_model = base_model
            self.num_labels = num_labels
            # Use the embed_dim from the model config, not n_embd
            embed_dim = base_model.model.config.embed_dim
            self.classifier = nn.Linear(embed_dim, num_labels)

        def forward(self, input_ids, attention_mask=None, labels=None):
            # Get hidden states without lm_head (for classification)
            hidden_states = self.base_model.model.get_hidden_states(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Use [CLS] token (first token) for classification
            pooled = hidden_states[:, 0]  # [batch_size, n_embd]

            logits = self.classifier(pooled)

            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            return {'loss': loss, 'logits': logits}

    return PsiQRHForSequenceClassification(model, num_labels)


def evaluate_glue_task(model, data_loader, device, task_name: str):
    """Evaluate model on GLUE task"""
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    task_configs = {
        'cola': {'num_labels': 2, 'metric': 'matthews_corrcoef'},
        'sst2': {'num_labels': 2, 'metric': 'accuracy'},
        'mrpc': {'num_labels': 2, 'metric': 'f1'},
        'qqp': {'num_labels': 2, 'metric': 'f1'},
        'stsb': {'num_labels': 1, 'metric': 'spearmanr'},
        'mnli': {'num_labels': 3, 'metric': 'accuracy'},
        'qnli': {'num_labels': 2, 'metric': 'accuracy'},
        'rte': {'num_labels': 2, 'metric': 'accuracy'},
        'wnli': {'num_labels': 2, 'metric': 'accuracy'}
    }

    config = task_configs[task_name]

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {task_name}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs['logits']
            loss = outputs['loss']

            if loss is not None:
                total_loss += loss.item()
            num_batches += 1

            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    results = {}

    if config['metric'] == 'accuracy':
        results['accuracy'] = accuracy_score(all_labels, all_predictions)
    elif config['metric'] == 'f1':
        results['f1'] = f1_score(all_labels, all_predictions, average='macro')
        results['accuracy'] = accuracy_score(all_labels, all_predictions)
    elif config['metric'] == 'matthews_corrcoef':
        results['matthews_corrcoef'] = matthews_corrcoef(all_labels, all_predictions)
        results['accuracy'] = accuracy_score(all_labels, all_predictions)
    elif config['metric'] == 'spearmanr':
        # For STS-B, we need to handle regression
        from scipy.stats import spearmanr
        results['spearmanr'] = spearmanr(all_labels, all_predictions)[0]

    if num_batches > 0:
        results['loss'] = total_loss / num_batches

    return results


def evaluate_perplexity(model, data_loader, device):
    """Evaluate perplexity on WikiText"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating perplexity"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Get model outputs
            outputs = model.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            if loss is not None:
                # Calculate number of tokens (excluding padding)
                attention_mask = (input_ids != 0).float()
                num_tokens = attention_mask.sum().item()

                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
    else:
        perplexity = float('inf')

    return {'perplexity': perplexity, 'loss': avg_loss}


def load_model_checkpoint(checkpoint_path: str, model_config: Dict[str, Any], task_type: str, num_labels: int = None):
    """Load model from checkpoint"""
    if task_type == 'glue':
        vocab_size = 50257  # GPT-2 vocab size
        model = create_model_for_glue(vocab_size, num_labels, model_config)
    else:  # language modeling
        config = PsiQRHConfig(**model_config)
        model = PsiQRHForCausalLM(config)

    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if task_type == 'glue':
            # For GLUE, we need to handle the wrapped model
            model.base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found, using random initialization")

    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate ΨQRH model on benchmarks')

    # Model configuration
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to model checkpoint')
    parser.add_argument('--model_config', type=str, default=None,
                       help='Path to model config JSON')

    # Benchmark selection
    parser.add_argument('--benchmark', type=str, required=True,
                       choices=['glue', 'ppl', 'all'],
                       help='Benchmark to run')
    parser.add_argument('--glue_task', type=str, default='sst2',
                       choices=['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'qnli', 'rte', 'wnli'],
                       help='GLUE task to evaluate')

    # Evaluation settings
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, default='./benchmark_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model configuration
    if args.model_config:
        with open(args.model_config, 'r') as f:
            model_config = json.load(f)
    else:
        # Default ΨQRH config
        model_config = {
            'vocab_size': 50257,
            'n_positions': 1024,
            'n_embd': 768,
            'n_layer': 12,
            'n_head': 12,
            'embed_dim': 64,
            'num_heads': 8,
            'hidden_dim': 512,
            'alpha': 1.0,
            'beta': 0.5,
            'kuramoto_coupling': 0.1,
            'kuramoto_frequency': 1.0
        }

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # Run GLUE evaluation
    if args.benchmark in ['glue', 'all']:
        print(f"\n=== Evaluating GLUE task: {args.glue_task} ===")

        # Get number of labels for task
        task_num_labels = {
            'cola': 2, 'sst2': 2, 'mrpc': 2, 'qqp': 2, 'stsb': 1,
            'mnli': 3, 'qnli': 2, 'rte': 2, 'wnli': 2
        }[args.glue_task]

        # Load model
        model = load_model_checkpoint(args.checkpoint_path, model_config, 'glue', task_num_labels)
        model = model.to(device)

        # Load data
        _, val_loader, test_loader = load_glue_data(args.glue_task, tokenizer, args.batch_size, args.max_length)

        # Evaluate on validation set
        val_results = evaluate_glue_task(model, val_loader, device, args.glue_task)
        print(f"Validation results: {val_results}")

        # Evaluate on test set
        test_results = evaluate_glue_task(model, test_loader, device, args.glue_task)
        print(f"Test results: {test_results}")

        results[args.glue_task] = {
            'validation': val_results,
            'test': test_results
        }

    # Run perplexity evaluation
    if args.benchmark in ['ppl', 'all']:
        print("\n=== Evaluating Perplexity on WikiText-103 ===")

        # Load model
        model = load_model_checkpoint(args.checkpoint_path, model_config, 'lm')
        model = model.to(device)

        # Load data
        val_loader, test_loader = load_wikitext_data(tokenizer, args.batch_size, args.max_length)

        # Evaluate perplexity
        val_ppl = evaluate_perplexity(model, val_loader, device)
        test_ppl = evaluate_perplexity(model, test_loader, device)

        print(f"Validation perplexity: {val_ppl}")
        print(f"Test perplexity: {test_ppl}")

        results['perplexity'] = {
            'validation': val_ppl,
            'test': test_ppl
        }

    # Save results
    results_file = os.path.join(args.output_dir, 'benchmark_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Print summary
    print("\n=== Benchmark Results Summary ===")
    for benchmark, benchmark_results in results.items():
        print(f"\n{benchmark.upper()}:")
        for split, metrics in benchmark_results.items():
            print(f"  {split}: {metrics}")


if __name__ == '__main__':
    main()
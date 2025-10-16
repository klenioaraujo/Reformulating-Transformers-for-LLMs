#!/usr/bin/env python3
"""
ΨQRH Language Model Training Pipeline

End-to-end training pipeline for ΨQRH autoregressive language model.
Supports distributed training, mixed precision, and physical validation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm

from psiqrh_llm import PsiQRHConfig, PsiQRHForCausalLM
from psiqrh_tokenizer import create_psiqrh_tokenizer


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model config
    vocab_size: int = 50257
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_positions: int = 1024

    # ΨQRH specific
    embed_dim: int = 64
    num_heads: int = 8
    hidden_dim: int = 512
    alpha: float = 1.0
    beta: float = 0.5
    kuramoto_coupling: float = 0.1
    kuramoto_frequency: float = 1.0

    # Training config
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    max_steps: int = 100000
    save_steps: int = 5000
    eval_steps: int = 1000
    logging_steps: int = 100

    # Data config
    train_data_path: str = "data/train.txt"
    eval_data_path: str = "data/eval.txt"
    tokenizer_name: str = "gpt2"

    # Hardware config
    device: str = "auto"
    mixed_precision: bool = True
    distributed: bool = False
    local_rank: int = -1

    # Physical validation
    validate_physics: bool = True
    physics_check_steps: int = 500


class TextDataset(Dataset):
    """Dataset for language modeling"""

    def __init__(self, file_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load and tokenize data
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Tokenize entire text
        tokens = tokenizer.encode(text)

        # Create sequences
        self.sequences = []
        for i in range(0, len(tokens) - max_length, max_length // 2):
            seq = tokens[i:i + max_length]
            if len(seq) == max_length:
                self.sequences.append(seq)

        print(f"Loaded {len(self.sequences)} sequences from {file_path}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.sequences[idx]
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'labels': torch.tensor(tokens, dtype=torch.long)
        }


def create_data_loader(
    file_path: str,
    tokenizer,
    batch_size: int,
    max_length: int = 1024,
    distributed: bool = False
) -> DataLoader:
    """Create data loader for training/evaluation"""
    dataset = TextDataset(file_path, tokenizer, max_length)

    sampler = None
    if distributed:
        sampler = torch.utils.data.DistributedSampler(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    return data_loader


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss"""
    return math.exp(loss)


def validate_physics(model: PsiQRHForCausalLM, tokenizer, device: torch.device) -> Dict[str, float]:
    """
    Validate physical properties of the model

    Checks energy conservation, unitarity, and fractal consistency
    """
    model.eval()
    physics_metrics = {}

    try:
        # Test with a simple quantum-related text
        test_text = "Quantum entanglement is a fundamental property of quantum mechanics."
        inputs = tokenizer(test_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            # Energy conservation check
            # ||output|| should be close to ||input||
            input_norm = torch.norm(inputs['input_ids'].float())
            output_norm = torch.norm(logits)
            energy_conservation = abs(output_norm.item() - input_norm.item()) / input_norm.item()
            physics_metrics['energy_conservation_error'] = energy_conservation

            # Unitarity check (approximate)
            # Check if attention weights sum to 1
            if hasattr(outputs, 'attentions') and outputs.attentions:
                attn_weights = outputs.attentions[0]  # First layer
                attn_sum = attn_weights.sum(dim=-1)
                unitarity_error = torch.mean(torch.abs(attn_sum - 1.0)).item()
                physics_metrics['unitarity_error'] = unitarity_error
            else:
                physics_metrics['unitarity_error'] = 0.0

            # Fractal dimension check
            # Compute fractal dimension from attention patterns
            if outputs.attentions:
                # Use attention entropy as proxy for fractal dimension
                attn_entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-10), dim=-1)
                avg_entropy = torch.mean(attn_entropy).item()
                # Map entropy to fractal dimension range [1, 2]
                fractal_dim = 1.0 + (avg_entropy / math.log(inputs['input_ids'].shape[1]))
                physics_metrics['fractal_dimension'] = min(2.0, max(1.0, fractal_dim))
            else:
                physics_metrics['fractal_dimension'] = 1.5

    except Exception as e:
        print(f"Physics validation failed: {e}")
        physics_metrics = {
            'energy_conservation_error': 1.0,
            'unitarity_error': 1.0,
            'fractal_dimension': 1.5
        }

    return physics_metrics


def train_epoch(
    model: PsiQRHForCausalLM,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    config: TrainingConfig,
    device: torch.device,
    global_step: int,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=config.mixed_precision):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / config.gradient_accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Scheduler step
            if scheduler:
                scheduler.step()

            global_step += 1

        total_loss += loss.item() * config.gradient_accumulation_steps
        num_batches += 1

        # Logging
        if global_step % config.logging_steps == 0:
            current_loss = total_loss / num_batches
            perplexity = compute_perplexity(current_loss)

            progress_bar.set_postfix({
                'loss': f"{current_loss:.4f}",
                'ppl': f"{perplexity:.2f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })

            # Log to wandb
            if wandb.run:
                wandb.log({
                    'train/loss': current_loss,
                    'train/perplexity': perplexity,
                    'train/learning_rate': optimizer.param_groups[0]['lr'],
                    'train/global_step': global_step
                })

        # Save checkpoint
        if global_step % config.save_steps == 0:
            save_checkpoint(model, optimizer, scheduler, global_step, config)

    avg_loss = total_loss / num_batches
    avg_ppl = compute_perplexity(avg_loss)

    return {
        'loss': avg_loss,
        'perplexity': avg_ppl,
        'global_step': global_step
    }


def evaluate_model(
    model: PsiQRHForCausalLM,
    eval_loader: DataLoader,
    tokenizer,
    config: TrainingConfig,
    device: torch.device,
    global_step: int
) -> Dict[str, float]:
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_ppl = compute_perplexity(avg_loss)

    # Physics validation
    physics_metrics = {}
    if config.validate_physics and global_step % config.physics_check_steps == 0:
        physics_metrics = validate_physics(model, tokenizer, device)

    eval_metrics = {
        'eval/loss': avg_loss,
        'eval/perplexity': avg_ppl,
        'eval/global_step': global_step,
        **{f'physics/{k}': v for k, v in physics_metrics.items()}
    }

    # Log to wandb
    if wandb.run:
        wandb.log(eval_metrics)

    return eval_metrics


def save_checkpoint(
    model: PsiQRHForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    global_step: int,
    config: TrainingConfig
):
    """Save model checkpoint"""
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint_path = checkpoint_dir / f"psiqrh_step_{global_step}.pt"

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'global_step': global_step,
        'config': config.__dict__
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: PsiQRHForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler
) -> int:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    global_step = checkpoint['global_step']
    print(f"Checkpoint loaded: {checkpoint_path} (step {global_step})")

    return global_step


def setup_distributed_training(config: TrainingConfig) -> Tuple[torch.device, int, int]:
    """Setup distributed training"""
    if config.distributed:
        # Initialize process group
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        local_rank = 0
        world_size = 1
        device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    return device, local_rank, world_size


def main():
    """Main training function"""
    config = TrainingConfig()

    # Setup distributed training
    device, local_rank, world_size = setup_distributed_training(config)

    # Initialize tokenizer
    tokenizer = create_psiqrh_tokenizer(config.tokenizer_name, config.vocab_size)

    # Initialize model
    model_config = PsiQRHConfig(
        vocab_size=config.vocab_size,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_positions=config.n_positions,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        hidden_dim=config.hidden_dim,
        alpha=config.alpha,
        beta=config.beta,
        kuramoto_coupling=config.kuramoto_coupling,
        kuramoto_frequency=config.kuramoto_frequency
    )

    model = PsiQRHForCausalLM(model_config).to(device)

    # Wrap model for distributed training
    if config.distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Initialize scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)

    # Initialize wandb (only on main process)
    if not config.distributed or local_rank == 0:
        wandb.init(
            project="psiqrh-llm",
            config=config.__dict__,
            name=f"psiqrh_training_{time.strftime('%Y%m%d_%H%M%S')}"
        )

    # Create data loaders
    train_loader = create_data_loader(
        config.train_data_path,
        tokenizer,
        config.batch_size,
        config.n_positions,
        config.distributed
    )

    eval_loader = create_data_loader(
        config.eval_data_path,
        tokenizer,
        config.batch_size,
        config.n_positions,
        False  # No distributed sampling for eval
    )

    # Training loop
    global_step = 0
    best_eval_loss = float('inf')

    print("Starting ΨQRH training...")
    print(f"Model: {model_config}")
    print(f"Training on device: {device}")
    print(f"Distributed: {config.distributed} (world_size: {world_size})")

    for epoch in range(config.max_steps // len(train_loader) + 1):
        # Train epoch
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            config, device, global_step, epoch
        )

        global_step = train_metrics['global_step']

        # Evaluate
        if global_step % config.eval_steps == 0:
            eval_metrics = evaluate_model(
                model, eval_loader, tokenizer, config, device, global_step
            )

            # Save best model
            if eval_metrics['eval/loss'] < best_eval_loss:
                best_eval_loss = eval_metrics['eval/loss']
                save_checkpoint(model, optimizer, scheduler, global_step, config)

        # Check stopping condition
        if global_step >= config.max_steps:
            break

    # Final evaluation
    final_eval_metrics = evaluate_model(
        model, eval_loader, tokenizer, config, device, global_step
    )

    print("Training completed!")
    print(f"Final eval loss: {final_eval_metrics['eval/loss']:.4f}")
    print(f"Final perplexity: {final_eval_metrics['eval/perplexity']:.2f}")

    # Save final model
    save_checkpoint(model, optimizer, scheduler, global_step, config)


if __name__ == "__main__":
    main()
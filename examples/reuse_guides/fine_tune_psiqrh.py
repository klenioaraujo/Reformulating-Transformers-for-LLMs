#!/usr/bin/env python3
"""
Fine-tuning Guide for ΨQRH Transformer

This script demonstrates how to fine-tune the ΨQRH transformer
on a custom task or dataset.

Copyright (C) 2025 Klenio Araujo Padilha
Licensed under GNU GPLv3 - see LICENSE file

DOI: https://zenodo.org/records/17171112
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.architecture.psiqrh_transformer import PsiQRHTransformer


class CustomDataset(Dataset):
    """
    Example custom dataset for fine-tuning.

    Replace this with your own dataset implementation.
    """

    def __init__(self, data_path, max_length=512):
        self.max_length = max_length
        # Load your data here
        self.data = self._load_data(data_path)

    def _load_data(self, data_path):
        """Load and preprocess your data"""
        # Implement your data loading logic
        # This is a placeholder
        return [
            {"input_ids": torch.randint(0, 1000, (self.max_length,)),
             "labels": torch.randint(0, 1000, (self.max_length,))}
            for _ in range(100)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_pretrained_model(checkpoint_path=None, config_path="configs/qrh_config.yaml"):
    """
    Load a pre-trained ΨQRH model.

    Args:
        checkpoint_path: Path to model checkpoint (optional)
        config_path: Path to model configuration

    Returns:
        model: Loaded ΨQRH model
        config: Model configuration
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize model
    model = PsiQRHTransformer(config)

    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")

    return model, config


def freeze_layers(model, freeze_until_layer=None):
    """
    Freeze specific layers for transfer learning.

    Args:
        model: ΨQRH model
        freeze_until_layer: Freeze all layers up to this layer number
                           None = don't freeze anything
    """
    if freeze_until_layer is None:
        return model

    for name, param in model.named_parameters():
        # Example: freeze embedding and early transformer layers
        should_freeze = 'embedding' in name or any(
            f'layer.{i}' in name for i in range(freeze_until_layer)
        )
        if should_freeze:
            param.requires_grad = False
            print(f"Frozen: {name}")

    return model


def fine_tune(
    model,
    train_dataloader,
    val_dataloader,
    device='cuda',
    epochs=3,
    learning_rate=1e-5,
    save_path='models/finetuned_psiqrh.pt'
):
    """
    Fine-tune the ΨQRH model on custom data.

    Args:
        model: ΨQRH model to fine-tune
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        device: Device to train on ('cuda' or 'cpu')
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        save_path: Path to save fine-tuned model
    """
    model = model.to(device)
    model.train()

    # Setup optimizer (use lower learning rate for fine-tuning)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * len(train_dataloader)
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*50}")

        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0

        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids)

            # Calculate loss
            loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1)
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_steps += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_dataloader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

        avg_train_loss = train_loss / train_steps
        print(f"\nAverage Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids)
                loss = criterion(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1)
                )

                val_loss += loss.item()
                val_steps += 1

        avg_val_loss = val_loss / val_steps
        print(f"Average Validation Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, save_path)
            print(f"✓ Saved best model to {save_path}")

    print(f"\n{'='*50}")
    print("Fine-tuning completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*50}")


def main():
    """Main fine-tuning pipeline"""

    # Configuration
    config_path = "configs/qrh_config.yaml"
    checkpoint_path = None  # Set to pre-trained model path if available
    data_path = "data/custom_dataset"  # Your dataset path

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    print("Loading ΨQRH model...")
    model, config = load_pretrained_model(checkpoint_path, config_path)

    # Optionally freeze layers for transfer learning
    # model = freeze_layers(model, freeze_until_layer=6)

    # Prepare data
    print("Preparing datasets...")
    train_dataset = CustomDataset(data_path)
    val_dataset = CustomDataset(data_path)  # Use separate validation data

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )

    # Fine-tune
    print("Starting fine-tuning...")
    fine_tune(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        epochs=3,
        learning_rate=1e-5,
        save_path='models/finetuned_psiqrh.pt'
    )

    print("\n✓ Fine-tuning guide completed!")
    print("\nNext steps:")
    print("1. Replace CustomDataset with your own dataset")
    print("2. Adjust hyperparameters (learning rate, batch size, epochs)")
    print("3. Consider freezing layers for transfer learning")
    print("4. Evaluate on your test set")
    print("5. Export model for deployment (see convert_psiqrh_to_onnx.py)")


if __name__ == "__main__":
    main()
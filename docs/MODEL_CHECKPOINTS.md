# ΨQRH Model Checkpoints Guide

This guide explains how to create, save, and manage model checkpoints.

**Note:** The `models/` directory currently has root permissions. Run:
```bash
sudo chown -R $USER:$USER models/
```

## Directory Structure

```
models/
├── pretrained/          # Pre-trained model weights
├── finetuned/          # Fine-tuned models
└── checkpoints/        # Training checkpoints
```

## Creating Checkpoints

See full documentation at: `models/README.md` (to be created)

## Quick Start

```python
import torch

# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'models/checkpoints/checkpoint.pt')

# Load checkpoint
checkpoint = torch.load('models/checkpoints/checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

For complete guide, fix permissions and see models/README.md
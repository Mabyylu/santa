"""
Training loop for QAP-SA model.

Implements the full training pipeline with support for:
- Multimodal loss (action prediction + codebook regularization)
- Learning rate scheduling
- Checkpointing
- Metrics tracking
"""

import os
import time
import json
from typing import Dict, Optional
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from .action_generator import CompositionalActionGenerator, create_model
from .data import create_dataloader


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Model parameters
    visual_output_dim: int = 512
    language_output_dim: int = 512
    action_dim: int = 7
    hash_bits: int = 12
    num_gaussians: int = 3
    vocab_size: int = 10000
    image_size: int = 128
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 50
    warmup_epochs: int = 5
    
    # Loss weights
    action_loss_weight: float = 1.0
    diversity_loss_weight: float = 0.1
    commitment_loss_weight: float = 0.25
    
    # Data parameters
    num_train_samples: int = 10000
    num_val_samples: int = 1000
    seq_len: int = 10
    
    # Logging and checkpointing
    log_interval: int = 100
    checkpoint_interval: int = 5
    output_dir: str = './outputs'
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42


class Trainer:
    """Training manager for QAP-SA model."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seeds
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        # Initialize model
        self.model = create_model(asdict(config)).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = None  # Will be initialized after warmup
        
        # Loss tracking
        self.train_losses = []
        self.val_losses = []
        self.metrics_history = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        
    def _warmup_lr(self, epoch: int) -> float:
        """Linear warmup learning rate."""
        if epoch < self.config.warmup_epochs:
            return self.config.learning_rate * (epoch + 1) / self.config.warmup_epochs
        return self.config.learning_rate
    
    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        actions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss from model outputs."""
        
        losses = {}
        
        # Action prediction loss (negative log likelihood)
        action_loss = outputs.get('action_loss', torch.tensor(0.0))
        losses['action_loss'] = action_loss * self.config.action_loss_weight
        
        # Codebook diversity loss
        aux_loss = outputs.get('aux_loss', {})
        if 'diversity' in aux_loss:
            diversity_loss = aux_loss['diversity']
            losses['diversity_loss'] = diversity_loss * self.config.diversity_loss_weight
        
        # Commitment loss
        if 'commitment' in aux_loss:
            commitment_loss = aux_loss['commitment']
            losses['commitment_loss'] = commitment_loss * self.config.commitment_loss_weight
        
        # Total loss
        losses['total_loss'] = sum(losses.values())
        
        return losses
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        
        # Apply warmup
        lr = self._warmup_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        total_losses = {
            'total_loss': 0.0,
            'action_loss': 0.0,
            'diversity_loss': 0.0,
            'commitment_loss': 0.0
        }
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            images = batch['images'].to(self.device)
            actions = batch['actions'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            token_masks = batch['token_masks'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                images=images,
                tokens=tokens,
                token_mask=token_masks,
                actions=actions,
                training=True
            )
            
            # Compute loss
            losses = self._compute_loss(outputs, actions)
            total_loss = losses['total_loss']
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track losses
            for key in total_losses:
                if key in losses:
                    total_losses[key] += losses[key].item()
            
            num_batches += 1
            
            # Logging
            if (batch_idx + 1) % self.config.log_interval == 0:
                avg_losses = {k: v / num_batches for k, v in total_losses.items()}
                print(f"Epoch {epoch}, Batch {batch_idx + 1}: "
                      f"Total Loss: {avg_losses['total_loss']:.4f}")
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Validate model."""
        
        self.model.eval()
        
        total_losses = {
            'total_loss': 0.0,
            'action_loss': 0.0,
            'diversity_loss': 0.0,
            'commitment_loss': 0.0
        }
        num_batches = 0
        
        for batch in dataloader:
            # Move to device
            images = batch['images'].to(self.device)
            actions = batch['actions'].to(self.device)
            tokens = batch['tokens'].to(self.device)
            token_masks = batch['token_masks'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                images=images,
                tokens=tokens,
                token_mask=token_masks,
                actions=actions,
                training=False
            )
            
            # Compute loss
            losses = self._compute_loss(outputs, actions)
            
            # Track losses
            for key in total_losses:
                if key in losses:
                    total_losses[key] += losses[key].item()
            
            num_batches += 1
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        
        return avg_losses
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ):
        """Full training loop."""
        
        print(f"Starting training on {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"Validation samples: {len(val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_losses)
            
            # Validate
            if val_loader:
                val_losses = self.validate(val_loader)
                self.val_losses.append(val_losses)
                
                # Save best model
                if val_losses['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total_loss']
                    self.save_checkpoint('best_model.pt')
                
                # Print validation results
                print(f"Epoch {epoch} - Val Loss: {val_losses['total_loss']:.4f}")
            
            epoch_time = time.time() - epoch_start
            
            # Print epoch summary
            print(f"Epoch {epoch} completed in {epoch_time:.1f}s")
            print(f"  Train Loss: {train_losses['total_loss']:.4f}")
            print(f"  Learning Rate: {train_losses.get('lr', self.optimizer.param_groups[0]['lr']):.6f}")
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
            
            # Track metrics
            self.metrics_history.append({
                'epoch': epoch,
                'train_loss': train_losses['total_loss'],
                'val_loss': val_losses['total_loss'] if val_loader else None,
                'time': epoch_time
            })
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        
        # Save training history
        self.save_training_history()
        
        return self.metrics_history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.config.output_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint_path = os.path.join(self.config.output_dir, filename)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def save_training_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(self.config.output_dir, 'training_history.json')
        
        # Convert tensors to floats for JSON serialization
        history = []
        for entry in self.metrics_history:
            history.append({
                'epoch': entry['epoch'],
                'train_loss': float(entry['train_loss']),
                'val_loss': float(entry['val_loss']) if entry['val_loss'] else None,
                'time': float(entry['time'])
            })
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Saved training history to {history_path}")


def run_training(config: Optional[TrainingConfig] = None):
    """Run training with default or custom config."""
    
    if config is None:
        config = TrainingConfig()
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(config.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = create_dataloader(
        num_samples=config.num_train_samples,
        batch_size=config.batch_size,
        image_size=config.image_size,
        action_dim=config.action_dim,
        seq_len=config.seq_len,
        shuffle=True
    )
    
    val_loader = create_dataloader(
        num_samples=config.num_val_samples,
        batch_size=config.batch_size,
        image_size=config.image_size,
        action_dim=config.action_dim,
        seq_len=config.seq_len,
        shuffle=False
    )
    
    # Create trainer
    trainer = Trainer(config)
    
    # Run training
    metrics = trainer.train(train_loader, val_loader)
    
    return trainer, metrics


if __name__ == '__main__':
    # Quick test with small config
    config = TrainingConfig(
        num_epochs=2,
        num_train_samples=100,
        num_val_samples=50,
        batch_size=8,
        log_interval=10
    )
    
    trainer, metrics = run_training(config)
    print("Training test completed!")

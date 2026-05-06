"""
Inference and evaluation utilities for QAP-SA.

Provides interfaces for:
- Loading trained models
- Generating actions from observations
- Evaluating model performance
- Visualizing results
"""

import os
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from .action_generator import CompositionalActionGenerator
from .data import SyntheticVisuomotorDataset


class InferenceEngine:
    """
    Inference engine for QAP-SA model.
    
    Handles model loading, preprocessing, and action generation.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config: Optional[Dict] = None,
        device: str = None
    ):
        """
        Initialize inference engine.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config: Model configuration (if not loading from checkpoint)
            device: Device to run inference on
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load config
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.config = checkpoint.get('config', {})
            self.model = self._build_model(self.config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {checkpoint_path}")
        elif config:
            self.config = config
            self.model = self._build_model(config)
            print("Initialized model from config")
        else:
            raise ValueError("Must provide either checkpoint_path or config")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Statistics
        self.num_params = sum(p.numel() for p in self.model.parameters())
        
    def _build_model(self, config: Dict) -> CompositionalActionGenerator:
        """Build model from config."""
        return CompositionalActionGenerator(
            visual_output_dim=config.get('visual_output_dim', 512),
            language_output_dim=config.get('language_output_dim', 512),
            action_dim=config.get('action_dim', 7),
            hash_bits=config.get('hash_bits', 12),
            num_gaussians=config.get('num_gaussians', 3),
            vocab_size=config.get('vocab_size', 10000),
            image_size=config.get('image_size', 128)
        )
    
    @torch.no_grad()
    def generate_actions(
        self,
        image: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate action sequence from observation and instruction.
        
        Args:
            image: Observation image [channels, height, width] or [batch, channels, h, w]
            tokens: Instruction tokens [seq_len] or [batch, seq_len]
            token_mask: Token mask
            num_steps: Number of action steps to generate
            temperature: Sampling temperature
            
        Returns:
            actions: Generated actions [num_steps, action_dim] or [batch, num_steps, action_dim]
        """
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if token_mask is not None and token_mask.dim() == 1:
            token_mask = token_mask.unsqueeze(0)
        
        # Move to device
        image = image.to(self.device)
        tokens = tokens.to(self.device)
        if token_mask is not None:
            token_mask = token_mask.to(self.device)
        
        # Generate actions
        actions = self.model.generate_actions(
            images=image,
            tokens=tokens,
            token_mask=token_mask,
            num_steps=num_steps,
            temperature=temperature
        )
        
        return actions
    
    @torch.no_grad()
    def get_action_primitive_codes(
        self,
        image: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get the discrete action primitive codes selected by the model.
        
        Args:
            image: Observation image
            tokens: Instruction tokens
            token_mask: Token mask
            
        Returns:
            hash_codes: Binary hash codes representing action primitives
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if token_mask is not None and token_mask.dim() == 1:
            token_mask = token_mask.unsqueeze(0)
        
        image = image.to(self.device)
        tokens = tokens.to(self.device)
        if token_mask is not None:
            token_mask = token_mask.to(self.device)
        
        # Forward pass
        outputs = self.model(
            images=image,
            tokens=tokens,
            token_mask=token_mask,
            training=False
        )
        
        return outputs['hash_codes']
    
    def get_codebook_stats(self) -> Dict:
        """Get statistics about codebook usage."""
        return self.model.get_codebook_stats()


class Evaluator:
    """
    Evaluation utilities for QAP-SA model.
    """
    
    def __init__(self, inference_engine: InferenceEngine):
        self.engine = inference_engine
        self.device = self.engine.device
        
    @torch.no_grad()
    def evaluate_on_dataset(
        self,
        dataset: SyntheticVisuomotorDataset,
        num_samples: int = 100,
        num_steps: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            dataset: Dataset to evaluate on
            num_samples: Number of samples to evaluate
            num_steps: Number of action steps to generate
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        self.engine.model.eval()
        
        total_action_loss = 0.0
        total_position_error = 0.0
        total_orientation_error = 0.0
        
        num_evaluated = 0
        
        for idx in range(min(num_samples, len(dataset))):
            sample = dataset[idx]
            
            image = sample['images'].to(self.device)
            actions_gt = sample['actions'].to(self.device)
            tokens = sample['tokens'].to(self.device)
            token_mask = sample['token_masks'].to(self.device)
            
            # Generate actions
            actions_pred = self.engine.generate_actions(
                image=image,
                tokens=tokens,
                token_mask=token_mask,
                num_steps=num_steps,
                temperature=0.0  # Deterministic for evaluation
            )[0]  # Remove batch dimension
            
            # Compute action loss (NLL)
            outputs = self.engine.model(
                images=image.unsqueeze(0),
                tokens=tokens.unsqueeze(0),
                token_mask=token_mask.unsqueeze(0),
                actions=actions_gt.unsqueeze(0),
                training=False
            )
            action_loss = -outputs['action_log_prob'][0].mean().item()
            total_action_loss += action_loss
            
            # Compute position error (first 3 dimensions)
            position_error = torch.mean(
                torch.sqrt(torch.sum((actions_pred[:, :3] - actions_gt[:, :3]) ** 2, dim=1))
            ).item()
            total_position_error += position_error
            
            # Compute orientation error (last 4 dimensions)
            # Using cosine similarity for quaternion-like orientations
            orient_pred = actions_pred[:, 3:]
            orient_gt = actions_gt[:, 3:]
            orient_pred_norm = orient_pred / (torch.norm(orient_pred, dim=1, keepdim=True) + 1e-8)
            orient_gt_norm = orient_gt / (torch.norm(orient_gt, dim=1, keepdim=True) + 1e-8)
            orientation_error = torch.mean(1 - torch.sum(orient_pred_norm * orient_gt_norm, dim=1)).item()
            total_orientation_error += orientation_error
            
            num_evaluated += 1
        
        metrics = {
            'action_loss': total_action_loss / num_evaluated,
            'position_error': total_position_error / num_evaluated,
            'orientation_error': total_orientation_error / num_evaluated,
            'num_samples': num_evaluated
        }
        
        return metrics
    
    @torch.no_grad()
    def evaluate_diversity(
        self,
        dataset: SyntheticVisuomotorDataset,
        num_samples: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate diversity of action primitives used.
        
        Args:
            dataset: Dataset to evaluate on
            num_samples: Number of samples
            
        Returns:
            diversity_metrics: Dictionary of diversity metrics
        """
        self.engine.model.eval()
        
        all_hash_codes = []
        
        for idx in range(min(num_samples, len(dataset))):
            sample = dataset[idx]
            
            image = sample['images'].to(self.device).unsqueeze(0)
            tokens = sample['tokens'].to(self.device).unsqueeze(0)
            token_mask = sample['token_masks'].to(self.device).unsqueeze(0)
            
            hash_codes = self.engine.get_action_primitive_codes(
                image=image,
                tokens=tokens,
                token_mask=token_mask
            )
            
            all_hash_codes.append(hash_codes.cpu())
        
        # Concatenate all hash codes
        all_codes = torch.cat(all_hash_codes, dim=0).view(-1, all_hash_codes[0].shape[-1])
        
        # Convert binary codes to integers
        powers = 2 ** torch.arange(all_codes.shape[1])
        code_indices = (all_codes * powers).sum(dim=1).long()
        
        # Compute unique codes
        unique_codes = torch.unique(code_indices)
        
        # Compute entropy
        code_counts = torch.bincount(code_indices, minlength=2 ** self.engine.config.get('hash_bits', 12))
        code_probs = code_counts.float() / code_counts.sum()
        entropy = -(code_probs[code_probs > 0] * torch.log(code_probs[code_probs > 0])).sum().item()
        
        diversity_metrics = {
            'unique_primitives': len(unique_codes),
            'total_possible': 2 ** self.engine.config.get('hash_bits', 12),
            'usage_entropy': entropy,
            'coverage_ratio': len(unique_codes) / (2 ** self.engine.config.get('hash_bits', 12))
        }
        
        return diversity_metrics


def run_inference_demo():
    """Run a demo of inference capabilities."""
    
    print("=" * 60)
    print("QAP-SA Inference Demo")
    print("=" * 60)
    
    # Create a small test model
    from .train import TrainingConfig, Trainer
    from .data import create_dataloader
    
    config = TrainingConfig(
        num_epochs=1,
        num_train_samples=100,
        num_val_samples=50,
        batch_size=8,
        hash_bits=8,  # Smaller for demo
        output_dir='./outputs_demo'
    )
    
    # Quick training to get a usable model
    print("\nTraining quick demo model...")
    trainer = Trainer(config)
    
    train_loader = create_dataloader(
        num_samples=config.num_train_samples,
        batch_size=config.batch_size,
        image_size=config.image_size,
        action_dim=config.action_dim,
        seq_len=config.seq_len
    )
    
    val_loader = create_dataloader(
        num_samples=config.num_val_samples,
        batch_size=config.batch_size,
        image_size=config.image_size,
        action_dim=config.action_dim,
        seq_len=config.seq_len
    )
    
    trainer.train(train_loader, val_loader)
    
    # Create inference engine
    print("\nLoading model for inference...")
    engine = InferenceEngine(
        checkpoint_path=os.path.join(config.output_dir, 'final_model.pt')
    )
    
    print(f"Model parameters: {engine.num_params:,}")
    
    # Create evaluator
    evaluator = Evaluator(engine)
    
    # Create test dataset
    from .data import SyntheticVisuomotorDataset
    test_dataset = SyntheticVisuomotorDataset(num_samples=50)
    
    # Run evaluation
    print("\nRunning evaluation...")
    metrics = evaluator.evaluate_on_dataset(test_dataset, num_samples=50)
    
    print("\nEvaluation Results:")
    print(f"  Action Loss: {metrics['action_loss']:.4f}")
    print(f"  Position Error: {metrics['position_error']:.4f}")
    print(f"  Orientation Error: {metrics['orientation_error']:.4f}")
    
    # Diversity evaluation
    diversity_metrics = evaluator.evaluate_diversity(test_dataset, num_samples=50)
    
    print("\nDiversity Metrics:")
    print(f"  Unique Primitives Used: {diversity_metrics['unique_primitives']}")
    print(f"  Coverage Ratio: {diversity_metrics['coverage_ratio']:.4f}")
    print(f"  Usage Entropy: {diversity_metrics['usage_entropy']:.4f}")
    
    # Codebook stats
    codebook_stats = engine.get_codebook_stats()
    print("\nCodebook Statistics:")
    print(f"  Active Codes: {codebook_stats['active_codes']}")
    print(f"  Total Codes: {codebook_stats['total_codes']}")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
    
    return engine, evaluator, metrics


if __name__ == '__main__':
    engine, evaluator, metrics = run_inference_demo()

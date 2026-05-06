"""
Semantic Hash Network - Core innovation for binary action primitive addressing.

This module implements a differentiable hashing mechanism that maps continuous
visual-linguistic features to discrete binary codes for efficient action retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SemanticHashNetwork(nn.Module):
    """
    Generates binary hash codes from visual-linguistic features.
    
    Key Innovation: Uses a variational bottleneck with straight-through estimator
    to learn discrete binary representations that can be used as addresses into
    a codebook of action primitives.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hash_bits: int = 16,
        temperature: float = 1.0,
        decay: float = 0.99
    ):
        super().__init__()
        self.hash_bits = hash_bits
        self.temperature = temperature
        self.decay = decay
        
        # Hash projection layers
        self.hash_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, hash_bits)
        )
        
        # Codebook for action primitives (2^hash_bits entries)
        self.num_codes = 2 ** hash_bits
        self.codebook = nn.Embedding(self.num_codes, input_dim)
        
        # Code usage tracking for load balancing
        self.register_buffer('code_usage', torch.ones(self.num_codes))
        self.register_buffer('ema_count', torch.ones(self.num_codes))
        
    def _straight_through_sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        """Straight-through estimator for binary quantization."""
        sigmoid = torch.sigmoid(x / self.temperature)
        binary = (sigmoid > 0.5).float()
        # Straight-through gradient
        return binary + sigmoid - sigmoid.detach()
    
    def _binary_to_index(self, binary: torch.Tensor) -> torch.Tensor:
        """Convert binary vector to integer index."""
        powers = 2 ** torch.arange(self.hash_bits, device=binary.device)
        return (binary * powers).sum(dim=-1, keepdim=True).long()
    
    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            x: Input features [batch, seq_len, input_dim] or [batch*seq_len, input_dim]
            training: Whether in training mode
            
        Returns:
            quantized: Quantized features from codebook
            hash_codes: Binary hash codes
            aux_loss: Auxiliary losses for training
        """
        # Handle both 2D and 3D inputs
        original_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add sequence dimension
        
        batch_size, seq_len, _ = x.shape
        
        # Generate hash logits
        hash_logits = self.hash_proj(x)  # [batch, seq_len, hash_bits]
        
        # Binary quantization with straight-through
        if training:
            hash_codes = self._straight_through_sigmoid(hash_logits)
        else:
            hash_codes = (torch.sigmoid(hash_logits) > 0.5).float()
        
        # Convert to indices
        indices = self._binary_to_index(hash_codes)  # [batch, seq_len, 1]
        indices = indices.squeeze(-1)  # [batch, seq_len]
        
        # Retrieve from codebook
        quantized = self.codebook(indices)  # [batch, seq_len, input_dim]
        
        # Compute auxiliary losses
        aux_loss = {}
        
        # Diversity loss: encourage uniform code usage
        if training:
            code_counts = torch.bincount(
                indices.flatten(), 
                minlength=self.num_codes
            ).float()
            code_probs = code_probs = code_counts / (code_counts.sum() + 1e-8)
            
            # Target uniform distribution
            target_prob = 1.0 / self.num_codes
            diversity_loss = F.kl_div(
                (code_probs + 1e-8).log(),
                torch.full_like(code_probs, target_prob),
                reduction='batchmean'
            )
            aux_loss['diversity'] = diversity_loss
            
            # Commitment loss: ensure encoder commits to codebook
            commitment_loss = F.mse_loss(x.detach(), quantized)
            aux_loss['commitment'] = commitment_loss
        
        # Update code usage statistics
        with torch.no_grad():
            code_counts = torch.bincount(
                indices.flatten(),
                minlength=self.num_codes
            ).float()
            self.code_usage = self.decay * self.code_usage + (1 - self.decay) * code_counts
        
        return quantized, hash_codes, aux_loss
    
    def get_code_usage_stats(self) -> dict:
        """Return statistics about codebook usage."""
        total_usage = self.code_usage.sum()
        usage_ratio = self.code_usage / (total_usage + 1e-8)
        
        # Percentage of codes used at least once
        active_codes = (self.code_usage > 1.0).sum().item()
        
        return {
            'active_codes': active_codes,
            'total_codes': self.num_codes,
            'usage_entropy': -(usage_ratio * (usage_ratio + 1e-8).log()).sum().item(),
            'max_usage_ratio': usage_ratio.max().item(),
            'min_usage_ratio': usage_ratio.min().item()
        }


class ActionPrimitiveDecoder(nn.Module):
    """
    Decodes quantized representations into continuous action parameters.
    
    Uses a mixture density approach where each action primitive can generate
    a distribution over possible action parameters.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        action_dim: int = 7,  # e.g., [x, y, z, qx, qy, qz, qw] for robot arm
        num_gaussians: int = 3
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_gaussians = num_gaussians
        
        # Mixture density parameters
        self.pi_head = nn.Linear(input_dim, num_gaussians)  # Mixing coefficients
        self.mu_head = nn.Linear(input_dim, num_gaussians * action_dim)  # Means
        self.sigma_head = nn.Linear(input_dim, num_gaussians * action_dim)  # Std devs
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features [batch, seq_len, input_dim]
            
        Returns:
            pi: Mixing coefficients [batch, seq_len, num_gaussians]
            mu: Means [batch, seq_len, num_gaussians, action_dim]
            sigma: Standard deviations [batch, seq_len, num_gaussians, action_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        pi = F.softmax(self.pi_head(x), dim=-1)
        mu = self.mu_head(x).view(batch_size, seq_len, self.num_gaussians, self.action_dim)
        sigma = F.softplus(self.sigma_head(x)).view(
            batch_size, seq_len, self.num_gaussians, self.action_dim
        )
        
        return pi, mu, sigma
    
    def sample(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sample actions from the mixture density."""
        pi, mu, sigma = self.forward(x)
        
        # Select component
        component = torch.multinomial(
            pi.view(-1, self.num_gaussians),
            num_samples=1
        ).view(pi.shape[:-1])
        
        # Gather parameters for selected component
        batch_size, seq_len = pi.shape[:2]
        batch_idx = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, seq_len)
        seq_idx = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        mu_selected = mu[batch_idx, seq_idx, component]
        sigma_selected = sigma[batch_idx, seq_idx, component]
        
        # Sample from Gaussian
        action = mu_selected + sigma_selected * torch.randn_like(mu_selected) * temperature
        
        return action
    
    def log_probability(
        self,
        x: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probability of actions under the predicted distribution.
        
        Args:
            x: Input features [batch, seq_len, input_dim]
            actions: Target actions [batch, seq_len, action_dim] or [batch, action_dim]
            
        Returns:
            log_prob: Log probability [batch, seq_len]
        """
        pi, mu, sigma = self.forward(x)
        
        # Handle action dimension mismatch - truncate or pad as needed
        if actions.dim() == 2:
            # Actions are [batch, action_dim], expand to match sequence
            actions = actions.unsqueeze(1).expand(-1, mu.shape[1], -1)
        
        # Ensure sequence lengths match
        if actions.shape[1] != mu.shape[1]:
            # Truncate or repeat actions to match
            if actions.shape[1] < mu.shape[1]:
                actions = actions.repeat(1, mu.shape[1] // actions.shape[1] + 1, 1)[:, :mu.shape[1], :]
            else:
                actions = actions[:, :mu.shape[1], :]
        
        # Compute Gaussian log probabilities for each component
        diff = actions.unsqueeze(2) - mu  # [batch, seq, num_gauss, action_dim]
        log_prob_components = -0.5 * ((diff / sigma) ** 2 + 2 * sigma.log())
        log_prob_components = log_prob_components.sum(dim=-1)  # Sum over action dim
        
        # Weight by mixing coefficients and sum
        log_prob = torch.logsumexp(
            pi.log() + log_prob_components,
            dim=-1
        )
        
        return log_prob

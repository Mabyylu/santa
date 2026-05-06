"""
Compositional Action Generator - Core VLA model.

This module implements the main Vision-Language-Action model that combines
visual features, language embeddings, and semantic hashing to generate
compositional action sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from .semantic_hash import SemanticHashNetwork, ActionPrimitiveDecoder


class VisualEncoder(nn.Module):
    """
    Lightweight visual encoder for processing image observations.
    
    Uses a small CNN backbone suitable for models under 1B parameters.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 512,
        image_size: int = 128
    ):
        super().__init__()
        self.image_size = image_size
        
        # Simple but effective CNN backbone
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, image_size, image_size)
            flat_size = self.conv_layers(dummy_input).shape[1]
        
        self.proj = nn.Linear(flat_size, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Images [batch, channels, height, width]
            
        Returns:
            features: Visual features [batch, output_dim]
        """
        features = self.conv_layers(x)
        return self.proj(features)


class LanguageEncoder(nn.Module):
    """
    Lightweight language encoder for processing instructions.
    
    Can use either learned embeddings or interface with pretrained models.
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        max_seq_len: int = 64,
        num_layers: int = 2
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02
        )
        
        # Simple transformer-style self-attention
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.proj = nn.Linear(embed_dim, hidden_dim)
        
    def forward(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tokens: Token IDs [batch, seq_len]
            mask: Attention mask [batch, seq_len]
            
        Returns:
            features: Language features [batch, hidden_dim]
        """
        batch_size, seq_len = tokens.shape
        
        # Embedding + positional encoding
        x = self.embedding(tokens) + self.positional_encoding[:, :seq_len, :]
        
        # Apply attention layers
        for layer in self.attention_layers:
            x = layer(x, src_key_padding_mask=mask)
        
        # Global pooling (mean over non-padded positions)
        if mask is not None:
            mask_float = (~mask).float().unsqueeze(-1)
            x = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        return self.proj(x)


class CompositionalActionGenerator(nn.Module):
    """
    Main VLA model combining vision, language, and action generation.
    
    Architecture:
    1. Visual encoder processes observation images
    2. Language encoder processes task instructions
    3. Multimodal fusion combines both modalities
    4. Semantic hash network generates discrete action primitive codes
    5. Action decoder produces continuous action parameters
    
    Key Innovation: The semantic hash creates a compositional action space
    where complex behaviors emerge from combinations of learned primitives.
    """
    
    def __init__(
        self,
        visual_output_dim: int = 512,
        language_output_dim: int = 512,
        action_dim: int = 7,
        hash_bits: int = 12,  # 2^12 = 4096 primitives
        num_gaussians: int = 3,
        vocab_size: int = 10000,
        image_size: int = 128
    ):
        super().__init__()
        
        self.visual_encoder = VisualEncoder(
            output_dim=visual_output_dim,
            image_size=image_size
        )
        
        self.language_encoder = LanguageEncoder(
            vocab_size=vocab_size,
            hidden_dim=language_output_dim
        )
        
        # Multimodal fusion
        self.fusion = nn.Sequential(
            nn.Linear(visual_output_dim + language_output_dim, 
                     (visual_output_dim + language_output_dim) // 2),
            nn.ReLU(),
            nn.Linear((visual_output_dim + language_output_dim) // 2, visual_output_dim),
            nn.LayerNorm(visual_output_dim)
        )
        
        # Temporal integration (for action sequences)
        self.temporal_encoder = nn.GRU(
            input_size=visual_output_dim,
            hidden_size=visual_output_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Core innovation: semantic hashing for action primitives
        self.semantic_hash = SemanticHashNetwork(
            input_dim=visual_output_dim,
            hash_bits=hash_bits
        )
        
        # Action parameter decoder
        self.action_decoder = ActionPrimitiveDecoder(
            input_dim=visual_output_dim,
            action_dim=action_dim,
            num_gaussians=num_gaussians
        )
        
        # Value head for training stability (optional RL component)
        self.value_head = nn.Sequential(
            nn.Linear(visual_output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def encode_multimodal(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode and fuse visual and language features.
        
        Args:
            images: [batch, channels, height, width] or [batch, seq, channels, h, w]
            tokens: [batch, seq_len]
            token_mask: [batch, seq_len]
            
        Returns:
            fused_features: [batch, seq_len, dim]
        """
        # Handle temporal dimension
        if images.dim() == 5:
            batch_size, seq_len = images.shape[:2]
            images_flat = images.view(batch_size * seq_len, *images.shape[2:])
            visual_features = self.visual_encoder(images_flat)
            visual_features = visual_features.view(batch_size, seq_len, -1)
        else:
            visual_features = self.visual_encoder(images)
            # Expand for sequence if needed
            if tokens.dim() > 2:
                seq_len = tokens.shape[1]
                visual_features = visual_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Language encoding
        lang_features = self.language_encoder(tokens, token_mask)
        
        # Broadcast language features to match visual sequence
        if visual_features.dim() > 2:
            seq_len = visual_features.shape[1]
            lang_features = lang_features.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate and fuse
        combined = torch.cat([visual_features, lang_features], dim=-1)
        fused = self.fusion(combined)
        
        return fused
    
    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training and inference.
        
        Args:
            images: Observation images
            tokens: Instruction tokens
            token_mask: Attention mask for tokens
            actions: Target actions (for training)
            training: Whether in training mode
            
        Returns:
            outputs: Dictionary containing predictions and losses
        """
        # Encode multimodal input
        fused_features = self.encode_multimodal(images, tokens, token_mask)
        
        # Temporal integration
        temporal_features, _ = self.temporal_encoder(fused_features)
        
        # Generate action primitives via semantic hashing
        quantized, hash_codes, hash_aux_loss = self.semantic_hash(
            temporal_features, 
            training=training
        )
        
        # Decode action parameters
        pi, mu, sigma = self.action_decoder(quantized)
        
        # Compute value estimates
        values = self.value_head(quantized.mean(dim=1))
        
        outputs = {
            'pi': pi,
            'mu': mu,
            'sigma': sigma,
            'hash_codes': hash_codes,
            'quantized': quantized,
            'values': values,
            'aux_loss': hash_aux_loss
        }
        
        # Compute action loss if targets provided
        if actions is not None:
            action_log_prob = self.action_decoder.log_probability(quantized, actions)
            outputs['action_log_prob'] = action_log_prob
            
            # Negative log likelihood loss
            outputs['action_loss'] = -action_log_prob.mean()
        
        return outputs
    
    @torch.no_grad()
    def generate_actions(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate action sequence autoregressively.
        
        Args:
            images: Initial observation
            tokens: Task instruction
            token_mask: Token mask
            num_steps: Number of action steps to generate
            temperature: Sampling temperature
            
        Returns:
            actions: Generated action sequence [batch, num_steps, action_dim]
        """
        batch_size = images.shape[0] if images.dim() == 4 else images.shape[0]
        device = images.device
        
        # Initialize action sequence
        all_actions = []
        
        # For autoregressive generation, we update the visual input
        current_images = images
        
        for step in range(num_steps):
            # Forward pass
            outputs = self.forward(
                current_images,
                tokens,
                token_mask,
                training=False
            )
            
            # Sample action
            action = self.action_decoder.sample(
                outputs['quantized'],
                temperature=temperature
            )[:, 0, :]  # Take first timestep
            
            all_actions.append(action)
            
            # In a real system, we would update current_images based on 
            # predicted action (e.g., through environment simulation)
            # Here we just reuse the same image for simplicity
        
        return torch.stack(all_actions, dim=1)
    
    def get_codebook_stats(self) -> dict:
        """Get statistics about action primitive codebook usage."""
        return self.semantic_hash.get_code_usage_stats()


def create_model(
    config: dict
) -> CompositionalActionGenerator:
    """Factory function to create model from config."""
    return CompositionalActionGenerator(
        visual_output_dim=config.get('visual_output_dim', 512),
        language_output_dim=config.get('language_output_dim', 512),
        action_dim=config.get('action_dim', 7),
        hash_bits=config.get('hash_bits', 12),
        num_gaussians=config.get('num_gaussians', 3),
        vocab_size=config.get('vocab_size', 10000),
        image_size=config.get('image_size', 128)
    )

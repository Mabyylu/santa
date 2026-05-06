"""
Test suite for QAP-SA model.

Run with: pytest tests/ -v
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.semantic_hash import SemanticHashNetwork, ActionPrimitiveDecoder
from src.action_generator import (
    VisualEncoder, 
    LanguageEncoder, 
    CompositionalActionGenerator,
    create_model
)
from src.data import SyntheticVisuomotorDataset, create_dataloader


class TestSemanticHashNetwork:
    """Tests for semantic hashing module."""
    
    def test_hash_network_forward(self):
        """Test forward pass of semantic hash network."""
        batch_size = 4
        seq_len = 10
        input_dim = 512
        hash_bits = 8
        
        network = SemanticHashNetwork(input_dim=input_dim, hash_bits=hash_bits)
        x = torch.randn(batch_size, seq_len, input_dim)
        
        quantized, hash_codes, aux_loss = network(x, training=True)
        
        assert quantized.shape == (batch_size, seq_len, input_dim)
        assert hash_codes.shape == (batch_size, seq_len, hash_bits)
        assert 'diversity' in aux_loss
        assert 'commitment' in aux_loss
    
    def test_binary_quantization(self):
        """Test that hash codes are binary."""
        network = SemanticHashNetwork(input_dim=256, hash_bits=16)
        x = torch.randn(2, 5, 256)
        
        _, hash_codes, _ = network(x, training=False)
        
        # All values should be 0 or 1
        unique_values = torch.unique(hash_codes)
        assert all(v in [0.0, 1.0] for v in unique_values.tolist())
    
    def test_codebook_lookup(self):
        """Test codebook indexing."""
        hash_bits = 6
        network = SemanticHashNetwork(input_dim=128, hash_bits=hash_bits)
        
        # Check codebook size
        assert network.codebook.num_embeddings == 2 ** hash_bits


class TestActionPrimitiveDecoder:
    """Tests for action primitive decoder."""
    
    def test_mixture_density_output(self):
        """Test mixture density network outputs."""
        batch_size = 4
        seq_len = 10
        input_dim = 512
        action_dim = 7
        num_gaussians = 3
        
        decoder = ActionPrimitiveDecoder(
            input_dim=input_dim,
            action_dim=action_dim,
            num_gaussians=num_gaussians
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        pi, mu, sigma = decoder(x)
        
        assert pi.shape == (batch_size, seq_len, num_gaussians)
        assert mu.shape == (batch_size, seq_len, num_gaussians, action_dim)
        assert sigma.shape == (batch_size, seq_len, num_gaussians, action_dim)
        
        # Pi should sum to 1
        pi_sum = pi.sum(dim=-1)
        assert torch.allclose(pi_sum, torch.ones_like(pi_sum), atol=1e-5)
        
        # Sigma should be positive
        assert torch.all(sigma > 0)
    
    def test_action_sampling(self):
        """Test action sampling."""
        decoder = ActionPrimitiveDecoder(input_dim=256, action_dim=7)
        x = torch.randn(2, 5, 256)
        
        actions = decoder.sample(x, temperature=1.0)
        
        assert actions.shape == (2, 5, 7)
    
    def test_log_probability(self):
        """Test log probability computation."""
        decoder = ActionPrimitiveDecoder(input_dim=256, action_dim=7)
        x = torch.randn(2, 5, 256)
        actions = torch.randn(2, 5, 7)
        
        log_prob = decoder.log_probability(x, actions)
        
        assert log_prob.shape == (2, 5)
        assert torch.all(torch.isfinite(log_prob))


class TestVisualEncoder:
    """Tests for visual encoder."""
    
    def test_visual_encoding(self):
        """Test visual feature extraction."""
        batch_size = 4
        image_size = 128
        output_dim = 512
        
        encoder = VisualEncoder(output_dim=output_dim, image_size=image_size)
        images = torch.randn(batch_size, 3, image_size, image_size)
        
        features = encoder(images)
        
        assert features.shape == (batch_size, output_dim)


class TestLanguageEncoder:
    """Tests for language encoder."""
    
    def test_language_encoding(self):
        """Test language feature extraction."""
        batch_size = 4
        seq_len = 16
        vocab_size = 1000
        hidden_dim = 512
        
        encoder = LanguageEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            max_seq_len=seq_len
        )
        
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        
        features = encoder(tokens, mask)
        
        assert features.shape == (batch_size, hidden_dim)


class TestCompositionalActionGenerator:
    """Tests for main VLA model."""
    
    def test_model_forward(self):
        """Test full model forward pass."""
        batch_size = 2
        image_size = 128
        seq_len = 16
        vocab_size = 1000
        action_dim = 7
        
        model = CompositionalActionGenerator(
            vocab_size=vocab_size,
            image_size=image_size,
            action_dim=action_dim,
            hash_bits=8
        )
        
        images = torch.randn(batch_size, 3, image_size, image_size)
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        token_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        actions = torch.randn(batch_size, 10, action_dim)
        
        outputs = model(
            images=images,
            tokens=tokens,
            token_mask=token_mask,
            actions=actions,
            training=True
        )
        
        assert 'pi' in outputs
        assert 'mu' in outputs
        assert 'sigma' in outputs
        assert 'hash_codes' in outputs
        assert 'action_loss' in outputs
    
    def test_action_generation(self):
        """Test autoregressive action generation."""
        batch_size = 2
        image_size = 128
        seq_len = 16
        vocab_size = 1000
        num_steps = 10
        
        model = CompositionalActionGenerator(
            vocab_size=vocab_size,
            image_size=image_size,
            hash_bits=8
        )
        
        images = torch.randn(batch_size, 3, image_size, image_size)
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            actions = model.generate_actions(
                images=images,
                tokens=tokens,
                num_steps=num_steps,
                temperature=0.5
            )
        
        assert actions.shape == (batch_size, num_steps, 7)
    
    def test_parameter_count(self):
        """Test that model is under 1B parameters."""
        model = CompositionalActionGenerator(hash_bits=12)
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should be well under 1B (around 10-20M expected)
        assert total_params < 1_000_000_000
        print(f"Model parameters: {total_params:,} ({total_params / 1e6:.2f}M)")


class TestDataset:
    """Tests for data generation."""
    
    def test_dataset_creation(self):
        """Test synthetic dataset creation."""
        dataset = SyntheticVisuomotorDataset(num_samples=100)
        
        assert len(dataset) == 100
        
        sample = dataset[0]
        assert 'images' in sample
        assert 'actions' in sample
        assert 'tokens' in sample
        assert 'task_labels' in sample
    
    def test_dataloader(self):
        """Test data loader."""
        dataloader = create_dataloader(
            num_samples=100,
            batch_size=8,
            shuffle=True
        )
        
        batch = next(iter(dataloader))
        
        assert batch['images'].shape[0] == 8
        assert batch['actions'].shape[0] == 8
        assert len(batch['images'].shape) == 4  # [B, C, H, W]
    
    def test_task_variety(self):
        """Test that dataset contains multiple task types."""
        dataset = SyntheticVisuomotorDataset(num_samples=1000)
        
        unique_tasks = torch.unique(dataset.task_labels)
        
        # Should have at least 3 different task types
        assert len(unique_tasks) >= 3


class TestIntegration:
    """Integration tests."""
    
    def test_train_step(self):
        """Test a single training step."""
        model = CompositionalActionGenerator(hash_bits=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Create dummy batch
        images = torch.randn(4, 3, 128, 128)
        tokens = torch.randint(0, 1000, (4, 16))
        token_mask = torch.zeros(4, 16, dtype=torch.bool)
        actions = torch.randn(4, 10, 7)
        
        # Forward pass
        outputs = model(
            images=images,
            tokens=tokens,
            token_mask=token_mask,
            actions=actions,
            training=True
        )
        
        loss = outputs['action_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Loss should be finite
        assert torch.isfinite(loss)
        assert loss.item() > 0
    
    def test_model_save_load(self):
        """Test model checkpointing."""
        import tempfile
        import os
        
        model = CompositionalActionGenerator(hash_bits=8)
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test.pt')
            torch.save(model.state_dict(), checkpoint_path)
            
            # Load
            model2 = CompositionalActionGenerator(hash_bits=8)
            model2.load_state_dict(torch.load(checkpoint_path))
            
            # Verify weights match
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(),
                model2.named_parameters()
            ):
                assert torch.allclose(p1, p2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Data generation and loading utilities for QAP-SA.

Generates synthetic visuomotor control data for training and evaluation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader


class SyntheticVisuomotorDataset(Dataset):
    """
    Synthetic dataset for visuomotor control tasks.
    
    Generates image-like observations and corresponding action sequences
    for simple manipulation tasks. This allows rapid prototyping without
    requiring real robot data.
    """
    
    def __init__(
        self,
        num_samples: int = 10000,
        image_size: int = 128,
        action_dim: int = 7,
        seq_len: int = 10,
        task_types: List[str] = None,
        noise_level: float = 0.1
    ):
        super().__init__()
        self.num_samples = num_samples
        self.image_size = image_size
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.task_types = task_types or ['reach', 'grasp', 'push', 'place']
        self.noise_level = noise_level
        
        # Pre-generate all data for efficiency
        self._generate_all_data()
        
    def _generate_all_data(self):
        """Generate all synthetic data upfront."""
        self.images = []
        self.actions = []
        self.tokens = []
        self.token_masks = []
        self.task_labels = []
        
        for i in range(self.num_samples):
            img, act, tok, tok_mask, task = self._generate_single_sample()
            self.images.append(img)
            self.actions.append(act)
            self.tokens.append(tok)  # Already a tensor
            self.token_masks.append(tok_mask)  # Already a tensor
            self.task_labels.append(task)
            
        # Convert to tensors using stack (not tensor constructor)
        self.images = torch.stack(self.images)
        self.actions = torch.stack(self.actions)
        self.tokens = torch.stack(self.tokens)
        self.token_masks = torch.stack(self.token_masks)
        self.task_labels = torch.tensor(self.task_labels, dtype=torch.long)
        
    def _generate_single_sample(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Generate a single training sample."""
        
        # Select task type
        task_idx = np.random.randint(len(self.task_types))
        task_type = self.task_types[task_idx]
        
        # Generate initial state (represented as image features)
        start_pos = np.random.uniform(-0.5, 0.5, size=3)  # 3D position
        start_orient = np.random.randn(4)
        start_orient = start_orient / (np.linalg.norm(start_orient) + 1e-8)
        
        # Generate goal based on task type
        if task_type == 'reach':
            goal_offset = np.random.uniform(-0.3, 0.3, size=3)
        elif task_type == 'grasp':
            goal_offset = np.random.uniform(-0.2, 0.2, size=3)
        elif task_type == 'push':
            goal_offset = np.random.uniform(-0.4, 0.4, size=3)
        else:  # place
            goal_offset = np.random.uniform(-0.3, 0.3, size=3)
        
        goal_pos = start_pos + goal_offset
        
        # Generate action sequence (smooth trajectory with noise)
        t = np.linspace(0, 1, self.seq_len)[:, np.newaxis]
        
        # Smooth interpolation
        positions = start_pos * (1 - t) + goal_pos * t
        
        # Add some curvature for realism
        control_point = (start_pos + goal_pos) / 2 + np.random.uniform(-0.1, 0.1, size=3)
        positions = (
            start_pos * (1 - t) ** 2 + 
            2 * control_point * t * (1 - t) + 
            goal_pos * t ** 2
        )
        
        # Orientation changes
        orientations = np.tile(start_orient, (self.seq_len, 1))
        orientations += np.random.randn(self.seq_len, 4) * 0.1
        
        # Combine into actions
        actions = np.concatenate([positions, orientations], axis=1)
        
        # Add noise
        actions += np.random.randn(*actions.shape) * self.noise_level
        
        # Generate synthetic "image" observation
        # Encode state information in pixel patterns
        image = self._state_to_image(start_pos, goal_pos, task_idx)
        
        # Generate token sequence (simple encoding of task)
        # In real system, this would be actual language tokens
        tokens, token_mask = self._task_to_tokens(task_type, goal_pos)
        
        return (
            torch.FloatTensor(image),
            torch.FloatTensor(actions),
            torch.tensor(tokens, dtype=torch.long),  # Convert to list first
            torch.tensor(token_mask, dtype=torch.bool),
            task_idx
        )
    
    def _state_to_image(
        self,
        start_pos: np.ndarray,
        goal_pos: np.ndarray,
        task_idx: int
    ) -> np.ndarray:
        """Convert state to synthetic image representation."""
        image = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
        
        # Map positions to image coordinates
        start_x = int((start_pos[0] + 0.5) * self.image_size)
        start_y = int((start_pos[1] + 0.5) * self.image_size)
        goal_x = int((goal_pos[0] + 0.5) * self.image_size)
        goal_y = int((goal_pos[1] + 0.5) * self.image_size)
        
        # Clip to valid range
        start_x = np.clip(start_x, 0, self.image_size - 1)
        start_y = np.clip(start_y, 0, self.image_size - 1)
        goal_x = np.clip(goal_x, 0, self.image_size - 1)
        goal_y = np.clip(goal_y, 0, self.image_size - 1)
        
        # Draw start position (red blob)
        radius = max(3, int(self.image_size * 0.05))
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    x, y = start_x + dx, start_y + dy
                    if 0 <= x < self.image_size and 0 <= y < self.image_size:
                        image[0, y, x] = 1.0  # Red channel
                        image[1, y, x] = 0.3
                        image[2, y, x] = 0.3
        
        # Draw goal position (green blob)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    x, y = goal_x + dx, goal_y + dy
                    if 0 <= x < self.image_size and 0 <= y < self.image_size:
                        image[0, y, x] = 0.3
                        image[1, y, x] = 1.0  # Green channel
                        image[2, y, x] = 0.3
        
        # Add task-specific color coding
        task_colors = [
            [1.0, 0.5, 0.0],  # reach - orange
            [0.5, 0.0, 1.0],  # grasp - purple
            [0.0, 1.0, 1.0],  # push - cyan
            [1.0, 1.0, 0.0],  # place - yellow
        ]
        color = task_colors[task_idx % len(task_colors)]
        for i in range(3):
            image[i, 0:5, :] = color[i]
        
        # Add some background noise for realism
        image += np.random.randn(*image.shape) * 0.05
        image = np.clip(image, 0, 1)
        
        return image
    
    def _task_to_tokens(
        self,
        task_type: str,
        goal_pos: np.ndarray
    ) -> Tuple[List[int], List[bool]]:
        """Convert task description to token sequence."""
        # Simple vocabulary mapping
        vocab = {
            '<pad>': 0,
            '<bos>': 1,
            '<eos>': 2,
            'reach': 3,
            'grasp': 4,
            'push': 5,
            'place': 6,
            'object': 7,
            'position': 8,
            'move': 9,
            'to': 10,
        }
        
        # Create token sequence (as Python list, not tensor)
        tokens = [vocab['<bos>']]
        
        if task_type == 'reach':
            tokens.extend([vocab['reach'], vocab['to'], vocab['position']])
        elif task_type == 'grasp':
            tokens.extend([vocab['grasp'], vocab['object']])
        elif task_type == 'push':
            tokens.extend([vocab['push'], vocab['object'], vocab['to'], vocab['position']])
        else:
            tokens.extend([vocab['place'], vocab['object'], vocab['to'], vocab['position']])
        
        tokens.append(vocab['<eos>'])
        
        # Pad to fixed length
        max_len = 16
        while len(tokens) < max_len:
            tokens.append(vocab['<pad>'])
        
        # Create mask (True for padding)
        token_mask = [t == vocab['<pad>'] for t in tokens]
        
        return tokens[:max_len], token_mask[:max_len]
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'images': self.images[idx],
            'actions': self.actions[idx],
            'tokens': self.tokens[idx],
            'token_masks': self.token_masks[idx],
            'task_labels': self.task_labels[idx]
        }


def create_dataloader(
    num_samples: int = 10000,
    batch_size: int = 32,
    image_size: int = 128,
    action_dim: int = 7,
    seq_len: int = 10,
    num_workers: int = 0,  # Changed to 0 to avoid multiprocessing issues
    shuffle: bool = True
) -> DataLoader:
    """Create a data loader for synthetic visuomotor data."""
    
    dataset = SyntheticVisuomotorDataset(
        num_samples=num_samples,
        image_size=image_size,
        action_dim=action_dim,
        seq_len=seq_len
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


if __name__ == '__main__':
    # Test data generation
    print("Testing synthetic data generation...")
    
    dataset = SyntheticVisuomotorDataset(num_samples=100)
    print(f"Generated {len(dataset)} samples")
    
    sample = dataset[0]
    print(f"Image shape: {sample['images'].shape}")
    print(f"Action shape: {sample['actions'].shape}")
    print(f"Token shape: {sample['tokens'].shape}")
    
    dataloader = create_dataloader(num_samples=100, batch_size=8)
    batch = next(iter(dataloader))
    print(f"Batch images: {batch['images'].shape}")
    print(f"Batch actions: {batch['actions'].shape}")
    print("Data generation test passed!")

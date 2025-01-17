import string
import random
import torch
from torch.utils.data import Dataset

class BrainrotXDataset(Dataset):
    """Custom Dataset for generating random text sequences with rotation."""
    def __init__(self, seq_length, num_samples, rot=26):
        """
        Initialize the dataset.
        
        Args:
            seq_length (int): Length of the input sequence
            num_samples (int): Number of random samples to generate
            rot (int): Rotation amount for the output text
        """
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.rot = rot
        
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return self.num_samples
        
    def __getitem__(self, _):
        """
        Generate a random text sequence and its rotated version.
        
        Args:
            _: Index of the dataset item
            
        Returns:
            tuple: (input_indices, target_indices)
        """
        text = ''.join(random.choice(string.ascii_lowercase) for _ in range(self.seq_length))
        
        input_indices = torch.tensor([ord(c) - ord('a') for c in text], dtype=torch.long)
        
        target_text = ''.join(chr(((ord(c) - ord('a') + self.rot) % 26) + ord('a')) for c in text)
        target_indices = torch.tensor([ord(c) - ord('a') for c in target_text], dtype=torch.long)
        
        return input_indices, target_indices
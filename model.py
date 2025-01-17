import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer model."""
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1):
        """
        Positional encoding for the transformer model.
        
        Args:
            d_model: The number of expected features in the input
            max_seq_length: The maximum length of the input sequence
            dropout: The dropout value
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass of the positional encoding.
        
        Args:
            x: The input tensor
            
        Returns:
            Tensor: Positionally encoded tensor
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class BrainrotX(nn.Module):
    """Transformer model for text sequence transformation."""
    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        """
        Initialize the transformer model.
        
        Args:
            d_model: The number of expected features in the input
            nhead: The number of heads in the multiheadattention models
            num_layers: The number of sub-encoder-layers in the encoder
            dim_feedforward: The dimension of the feedforward network model
            dropout: The dropout value
        """
        super().__init__()
        
        self.embedding = nn.Embedding(26, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        self.output_layer = nn.Linear(d_model, 26)
        self.d_model = d_model
        self.init_weights()
    
    def init_weights(self):
        """
        Initialize model weights.
        """
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src, src_mask=None):
        """
        Forward pass of the transformer model.
        
        Args:
            src: Tensor, shape [batch_size, seq_len]
            src_mask: Tensor, shape [seq_len, seq_len]
            
        Returns:
            Tensor: Output tensor
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.output_layer(output)
        return output
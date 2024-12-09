import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        max_len: int = 50,
        scale: int = 100,
    ):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.Tensor([scale])) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class RotaryEncoding(nn.Module):
    """
    Module for applying rotary positional encoding to input sequences.

    This module incorporates positional information into input sequences using rotary positional encoding.
    It supports two modes: concatenation and addition. In concatenation mode, the positional embeddings
    are concatenated with the input, followed by a linear transformation. In addition mode, the positional
    embeddings are added directly to the input.

    Args:
    - config (object): Configuration object with the following attributes:
        - position_concatenation (bool): Whether to concatenate positional embeddings with input.
        - maxlen (int): Maximum length of input sequences.
        - embedding_d (int): Dimensionality of the input embeddings.

    Attributes:
    - concat (bool): Whether to concatenate positional embeddings with input.
    - position_embeddings (nn.Embedding): Embedding layer for positional embeddings.
    - encoding (nn.Linear): Linear layer for concatenation mode.

    Methods:
    - forward(x: Tensor) -> Tensor: Apply rotary positional encoding to input tensor.

    Example:
    >> config = Configuration(position_concatenation=True, maxlen=100, embedding_d=512)
    >> rotary_encoder = RotaryPositionalEncoding(config)
    >> input_tensor = torch.rand((batch_size, sequence_length, embedding_dim))
    >> output_tensor = rotary_encoder(input_tensor)
    """

    def __init__(self, config):
        """
        Initialize the RotaryPositionalEncoding module.
        Args:
        - config (object): Configuration object with required attributes.

        """
        super().__init__()
        self.concat = config.position_concatenation
        L, H = config.maxlen, config.embedding_d
        self.position_embeddings = nn.Embedding(L, H)
        if self.concat:
            self.encoding = nn.Linear(H * 2, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary positional encoding to the input tensor.

        Args:
        - x (Tensor): Input tensor with shape (batch_size, sequence_length, embedding_dim).

        Returns:
        Tensor: Output tensor after applying rotary positional encoding.

        """
        # position_ids => L x H, rows [ 0, 1, 2, ...,H]
        position_ids = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        position_embeddings = self.position_embeddings(position_ids)

        # Rotary Positional Encoding
        angles = position_embeddings / 10000.0
        angle_rads = angles[:, :, 0::2] * 2 * math.pi

        sin_angles = torch.sin(angle_rads)
        cos_angles = torch.cos(angle_rads)

        # Add rotation
        sin_angles = sin_angles * torch.tensor([(-1) ** i for i in range(sin_angles.size(-1))], device=x.device)

        # Combine sine and cosine embeddings
        position_embeddings[:, :, 0::2] = sin_angles
        position_embeddings[:, :, 1::2] = cos_angles

        if not self.concat:
            x = x + position_embeddings
        else:
            x = torch.cat([x, position_embeddings], -1)
            x = self.encoding(x)
        return x

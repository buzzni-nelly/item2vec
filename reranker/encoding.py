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

    def forward(self, x, combine: bool = True):
        if combine:
            return x + self.pe[:, : x.size(1)]
        return self.pe[:, : x.size(1)]


class RotaryEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        """
        Rotary Encoding 모듈
        Args:
            dim (int): 임베딩 차원 (홀수 차원은 지원하지 않음)
            max_len (int): 최대 길이
        """
        super(RotaryEncoding, self).__init__()
        assert dim % 2 == 0, "Dimension must be even for Rotary Encoding."

        self.dim = dim
        self.max_len = max_len

        # precompute sin and cos for efficiency
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))  # (dim // 2,)
        self.register_buffer("sin", torch.sin(position * div_term))  # (max_len, dim // 2)
        self.register_buffer("cos", torch.cos(position * div_term))  # (max_len, dim // 2)

    def forward(self, embeddings: torch.Tensor):
        """
        Args:
            embeddings (torch.Tensor): (batch_size, seq_len, dim)
        Returns:
            torch.Tensor: Rotary Encoding이 적용된 embeddings
        """
        seq_len = embeddings.size(1)
        assert seq_len <= self.max_len, f"Sequence length ({seq_len}) exceeds maximum length ({self.max_len})."

        # Split into even and odd dimensions
        embeddings_odd = embeddings[..., 0::2]
        embeddings_even = embeddings[..., 1::2]

        # Apply rotary encoding
        sin = self.sin[:seq_len, :].unsqueeze(0)  # (1, seq_len, dim // 2)
        cos = self.cos[:seq_len, :].unsqueeze(0)  # (1, seq_len, dim // 2)
        embeddings_rotated = torch.cat(
            [
                embeddings_odd * cos - embeddings_even * sin,
                embeddings_odd * sin + embeddings_even * cos,
            ],
            dim=-1,
        )

        return embeddings_rotated


class RelativePosition(nn.Module):
    """
    Module for generating relative positional embeddings.

    This module computes relative positional embeddings for sequences of given lengths.
    It utilizes a learnable embeddings table that is initialized with Xavier uniform initialization.

    Args:
    - num_units (int): The number of embedding units for each position.
    - max_relative_position (int): The maximum relative position allowed.

    Attributes:
    - num_units (int): The number of embedding units for each position.
    - max_relative_position (int): The maximum relative position allowed.
    - embeddings_table (nn.Parameter): Learnable parameter representing the embeddings table.

    Methods:
    - forward(length_q, length_k): Compute relative positional embeddings for given sequence lengths.

    Example:
    >> relative_position = RelativePosition(num_units=512, max_relative_position=128)
    >> embeddings = relative_position(10, 12)
    """

    def __init__(self, num_units, max_relative_position):
        """
        Initialize the RelativePosition module.

        Args:
        - num_units (int): The number of embedding units for each position.
        - max_relative_position (int): The maximum relative position allowed.

        """
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        """
        Compute relative positional embeddings for given sequence lengths.

        Args:
        - length_q (int): Length of the query sequence.
        - length_k (int): Length of the key sequence.

        Returns:
        torch.Tensor: Relative positional embeddings for the given lengths.

        """
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()
        return embeddings

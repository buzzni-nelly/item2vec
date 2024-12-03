import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        max_len: int = 50,
        scale: int = 100,
        dropout: float = 0.05,
    ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * -(torch.log(torch.Tensor([scale])) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = sqrt(embed_dim / num_heads)

    def forward(self, Q, K, V, mask=None):
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, V)
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))


class BERT4Rec(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_layers=2,
        num_heads=2,
        ff_dim=512,
        max_len=50,
        dropout=0.1,
    ):
        super(BERT4Rec, self).__init__()
        self.max_len = max_len

        # Embedding layers
        self.item_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)

        # Transformer Encoder
        self.attention = ScaledDotProductAttention(embed_dim, num_heads)
        self.feed_forward = PositionwiseFeedForward(embed_dim, ff_dim)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )

        # Output layers with two-layer feed-forward network
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, mask=None):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Item and position embeddings
        item_embeddings = self.item_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = self.dropout(item_embeddings + position_embeddings)

        # Transformer Encoder with attention and feed-forward
        for layer in self.layers:
            attn_output = self.attention(embeddings, embeddings, embeddings, mask)
            embeddings = layer(attn_output + embeddings)
            embeddings = self.feed_forward(embeddings)

        # Output layer processing
        hidden_states = self.projection(embeddings)
        hidden_states = F.gelu(hidden_states)
        logits = self.output_layer(hidden_states)

        return logits

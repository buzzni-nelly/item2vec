import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

from item2vec.models import GraphItem2Vec


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


class BERT4Rec(nn.Module):
    def __init__(
        self,
        item2vec: GraphItem2Vec,
        vocab_size: int,
        embed_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 2,
        max_len: int = 50,
        dropout: float = 0.1,
        freeze_item_embeddings: bool = True,
    ):
        super(BERT4Rec, self).__init__()
        self.max_len = max_len

        # 여기서 pretrained 된 item2vec.embeddings 를 계승해서 쓰고 싶음
        self.item_embedding = nn.Embedding(vocab_size, embed_dim)
        self.item_embedding.weight.data.copy_(item2vec.embeddings)
        self.item_embedding.weight.requires_grad = not freeze_item_embeddings

        self.position_embedding = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.projection = nn.Linear(embed_dim, embed_dim // 2)
        self.output_layer = nn.Linear(embed_dim // 2, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, src_key_padding_mask: torch.Tensor = None):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        item_embeddings = self.item_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = self.dropout(item_embeddings + position_embeddings)

        transformer_output = self.transformer_encoder(embeddings, src_key_padding_mask=src_key_padding_mask)

        hidden_states = self.projection(transformer_output)
        hidden_states = F.gelu(hidden_states)
        logits = self.output_layer(hidden_states)
        return logits

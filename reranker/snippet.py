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
    def __init__(self, embed_dim: int, num_heads: int):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = sqrt(embed_dim / num_heads)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None):
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

    def forward(self, input_ids, mask: torch.Tensor | None = None):
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


if __name__ == "__main__":
    embed_dim = 10
    num_heads = 2
    batch_size = 1
    seq_length = 5
    Q = torch.randn(batch_size, seq_length, embed_dim)
    K = torch.randn(batch_size, seq_length, embed_dim)
    V = torch.randn(batch_size, seq_length, embed_dim)

    mask = torch.ones(batch_size, seq_length, seq_length)
    mask[-1][-1][-1] = 0
    mask[-1][-1][-2] = 0
    mask[-1][-1][-3] = 0
    mask[-1][-1][-4] = 0
    mask[-1][-1][-5] = 0

    attention_layer = ScaledDotProductAttention(embed_dim, num_heads)
    output = attention_layer(Q, K, V, mask=mask)

    print("Output shape:", output.shape)
    print("Output:", output)


def generate_masked_input(input_seq, pad_token_idx=0, mask_token_idx=10001, mask_prob=0.15):
    """
    입력 시퀀스에서 마스크를 생성하고 레이블을 반환합니다.
    :param input_seq: [batch_size, max_len] - 입력 시퀀스
    :param pad_token_idx: 패딩 토큰 ID
    :param mask_token_idx: 마스크 토큰 ID
    :param mask_prob: 마스크 확률
    :return: 마스크된 입력 시퀀스, 레이블, 레이블 마스크
    """
    # NOTE: softmax 써야 함..
    input_seq = input_seq.clone()
    labels = input_seq.clone()
    mask = (torch.rand(input_seq.shape) < mask_prob) & (input_seq != pad_token_idx)

    # 입력에서 마스크 위치를 마스크 토큰으로 대체
    input_seq[mask] = mask_token_idx

    # 마스크된 위치만 레이블로 유지, 나머지는 -100으로 설정 (PyTorch의 ignore_index)
    labels[~mask] = -100

    return input_seq, labels


# if __name__ == "__main__":
#     torch.manual_seed(42)
#
#     num_items = 10
#     mask_token_idx = num_items + 0
#     pad_token_idx = num_items + 1
#     embed_dim = 4
#     num_heads = 2
#     num_layers = 2
#     max_len = 5
#     dropout = 0.1
#
#     model = BERT4Rec(
#         num_items=num_items,
#         embed_dim=embed_dim,
#         num_heads=num_heads,
#         num_layers=num_layers,
#         max_len=max_len,
#         dropout=dropout,
#     )
#
#     batch_size = 2
#     input_seqs = torch.randint(0, num_items, (batch_size, max_len))
#     input_seqs[0][-1] = pad_token_idx
#     input_seqs[0][-2] = pad_token_idx
#     input_seqs[1][-1] = pad_token_idx
#     padding_mask = input_seqs == pad_token_idx
#
#     batch_indices = torch.arange(input_seqs.size(0))
#     last_indices = (~padding_mask).sum(dim=1) - 1
#     input_seqs[batch_indices, last_indices] = mask_token_idx
#
#     positive_idx = torch.randint(0, num_items, (batch_size,))
#     negative_idx = torch.randint(0, num_items, (batch_size,))
#
#     logits = model(input_seqs, padding_mask)
#     output = logits[torch.arange(logits.size(0)), last_indices, :]
#
#     positive_embeddings = model.item_embedding(positive_idx)
#     negative_embeddings = model.item_embedding(negative_idx)
#     positive_scores = torch.sum(output * positive_embeddings, dim=-1)
#     negative_scores = torch.sum(output * negative_embeddings, dim=-1)
#
#     loss = bpr_loss(positive_scores, negative_scores)
#     print(f"BPR Loss: {loss.item()}")

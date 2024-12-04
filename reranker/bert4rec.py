import torch
import torch.nn as nn


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
    def __init__(self, num_items: int, embed_dim: int, num_heads: int, num_layers: int, max_len: int, dropout=0.1):
        super(BERT4Rec, self).__init__()
        self.num_items = num_items
        self.mask_token_idx = self.num_items + 0
        self.pad_token_idx = self.num_items + 1
        self.embed_dim = embed_dim
        self.max_len = max_len

        self.item_embedding = nn.Embedding(num_items + 2, embed_dim, padding_idx=self.pad_token_idx)
        self.position_embedding = PositionalEncoding(embed_dim=embed_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_seq: torch.Tensor, padding_mask: torch.Tensor=None):
        embeddings = self.item_embedding(input_seq)
        embeddings = self.position_embedding(embeddings)
        encoder_output = self.transformer_encoder(embeddings, src_key_padding_mask=padding_mask)
        return encoder_output


def bpr_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor):
    loss = -torch.mean(torch.log(torch.sigmoid(positive_scores - negative_scores)))
    return loss


if __name__ == "__main__":
    torch.manual_seed(42)

    num_items = 10
    mask_token_idx = num_items + 0
    pad_token_idx = num_items + 1
    embed_dim = 4
    num_heads = 2
    num_layers = 2
    max_len = 5
    dropout = 0.1

    model = BERT4Rec(
        num_items=num_items,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_len=max_len,
        dropout=dropout,
    )

    batch_size = 2
    input_seqs = torch.randint(0, num_items, (batch_size, max_len))
    input_seqs[0][-1] = pad_token_idx
    input_seqs[0][-2] = pad_token_idx
    input_seqs[1][-1] = pad_token_idx
    padding_mask = input_seqs == pad_token_idx

    batch_indices = torch.arange(input_seqs.size(0))
    last_indices = (~padding_mask).sum(dim=1) - 1
    input_seqs[batch_indices, last_indices] = mask_token_idx

    positive_idx = torch.randint(0, num_items, (batch_size,))
    negative_idx = torch.randint(0, num_items, (batch_size,))

    logits = model(input_seqs, padding_mask)
    output = logits[torch.arange(logits.size(0)), last_indices, :]

    positive_embeddings = model.item_embedding(positive_idx)
    negative_embeddings = model.item_embedding(negative_idx)
    positive_scores = torch.sum(output * positive_embeddings, dim=-1)
    negative_scores = torch.sum(output * negative_embeddings, dim=-1)

    loss = bpr_loss(positive_scores, negative_scores)
    print(f"BPR Loss: {loss.item()}")


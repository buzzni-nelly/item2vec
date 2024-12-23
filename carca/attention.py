import copy
from typing import Callable, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, ModuleList
from torch.nn import LayerNorm
from torch.nn import Linear
from torch.nn import MultiheadAttention


class CrossAttentionEncoderLayer(nn.Module):

    __constants__ = ["norm_first"]

    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(
            embed_dim,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(embed_dim, dim_feedforward, bias=bias)
        self.linear2 = Linear(dim_feedforward, embed_dim, bias=bias)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(embed_dim, elementwise_affine=False)
        self.norm2 = LayerNorm(embed_dim, elementwise_affine=False)
        self.dropout = Dropout(dropout)
        self.activation = activation

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize self attention
        torch.nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
        torch.nn.init.constant_(self.self_attn.in_proj_bias, 0)
        torch.nn.init.xavier_uniform_(self.self_attn.out_proj.weight)
        torch.nn.init.constant_(self.self_attn.out_proj.bias, 0)

        # Initialize linear layers
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.constant_(self.linear1.bias, 0)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.constant_(self.linear2.bias, 0)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.gelu

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        residual_strategy_1: Literal["sum", "multiply", "none"] = "sum",
        residual_strategy_2: Literal["sum", "multiply", "none"] = "sum",
    ) -> tuple[Tensor, Tensor]:

        assert q.dtype == k.dtype == v.dtype, "k, q, v must have the same dtype"

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(None),
            other_name="src_mask",
            target_type=q.dtype,
        )

        q = self.norm1(q)
        sa, w = self._sa_block(
            q,
            k,
            v,
            src_key_padding_mask,
            need_weights=need_weights,
        )

        if residual_strategy_1 == "sum":
            x = self.norm2(q + sa)
        elif residual_strategy_1 == "multiply":
            x = self.norm2(q * sa)
        elif residual_strategy_1 == "none":
            x = sa
        else:
            raise ValueError(f"Invalid residual operator: {residual_strategy_1}")

        if residual_strategy_2 == "sum":
            x = x + self._ff_block(x)
        elif residual_strategy_2 == "multiply":
            x = x + self._ff_block(x)
        elif residual_strategy_2 == "none":
            x = sa
        else:
            raise ValueError(f"Invalid residual operator: {residual_strategy_2}")

        x = self.dropout(x)
        return x, w

    # self-attention block
    def _sa_block(
        self,
        k: Tensor,
        q: Tensor,
        v: Tensor,
        key_padding_mask: Optional[Tensor],
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor]:
        x, weights = self.self_attn(
            k,
            q,
            v,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        return x, weights

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class CrossAttentionDecoderLayer(nn.Module):

    __constants__ = ["norm_first"]

    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        batch_first: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = MultiheadAttention(
            embed_dim,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
        )
        self.linear1 = Linear(embed_dim, dim_feedforward, bias=bias)
        self.linear2 = Linear(dim_feedforward, embed_dim, bias=bias)

        self.dropout = Dropout(dropout)
        self.activation = activation

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize self attention
        torch.nn.init.xavier_uniform_(self.self_attn.in_proj_weight)
        torch.nn.init.constant_(self.self_attn.in_proj_bias, 0)
        torch.nn.init.xavier_uniform_(self.self_attn.out_proj.weight)
        torch.nn.init.constant_(self.self_attn.out_proj.bias, 0)

        # Initialize linear layers
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.constant_(self.linear1.bias, 0)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.constant_(self.linear2.bias, 0)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.gelu

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        residual_strategy_1: Literal["sum", "multiply", "none"] = "sum",
    ) -> tuple[Tensor, Tensor]:

        assert q.dtype == k.dtype == v.dtype, "k, q, v must have the same dtype"

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(None),
            other_name="src_mask",
            target_type=q.dtype,
        )

        sa, w = self._sa_block(
            q,
            k,
            v,
            src_key_padding_mask,
            need_weights=need_weights,
        )

        if residual_strategy_1 == "sum":
            x = q + sa
        elif residual_strategy_1 == "multiply":
            x = q * sa
        elif residual_strategy_1 == "none":
            x = sa
        else:
            raise ValueError(f"Invalid residual operator: {residual_strategy_1}")

        x = self._ff_block(x)
        # x = self.dropout(x)
        return x, w

    # self-attention block
    def _sa_block(
        self,
        k: Tensor,
        q: Tensor,
        v: Tensor,
        key_padding_mask: Optional[Tensor],
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor]:
        x, weights = self.self_attn(
            k,
            q,
            v,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        return x, weights

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class CrossAttentionEncoder(nn.Module):

    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer: "CrossAttentionEncoderLayer",
        num_layers: int,
        mask_check: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.layers = _get_clones(encoder_layer, self.num_layers)
        self.mask_check = mask_check

    def forward(
        self,
        q: Tensor,  # query: item embeddings
        k: Tensor,  # key: item embeddings
        v: Tensor,  # value: item embeddings
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, list[Tensor]]:

        assert q.dtype == k.dtype == v.dtype, "k, q, v must have the same dtype"

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(None),
            other_name="mask",
            target_type=q.dtype,
        )

        output = q
        weights = []
        for mod in self.layers:
            output, weight = mod(
                q,
                k,
                v,
                src_key_padding_mask=src_key_padding_mask,
                residual_strategy_1="sum",
                residual_strategy_2="sum",
            )
            q, k, v = output, output, output
            weights.append(weight)

        return output, weights


class CrossAttentionDecoder(nn.Module):

    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer: "CrossAttentionDecoderLayer",
        num_layers: int,
        mask_check: bool = True,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.mask_check = mask_check

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, list[Tensor]]:

        assert q.dtype == k.dtype == v.dtype, "k, q, v must have the same dtype"

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(None),
            other_name="mask",
            target_type=q.dtype,
        )

        weights = []
        output = q
        for mod in self.layers:
            output, weight = mod(
                q,
                k,
                v,
                src_key_padding_mask=src_key_padding_mask,
            )
            q = output
            weights.append(weight)

        return output, weights


class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_len: int = 50,
        dropout=0.1,
    ):
        super(CrossAttention, self).__init__()

        self.Hd = embed_dim // num_heads
        # self.rotary_encoding_kv = RotaryEncoding(embed_dim, max_len)

        encoder_layer = CrossAttentionEncoderLayer(
            embed_dim=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            activation=F.selu,
            batch_first=True,
        )
        decoder_layer = CrossAttentionDecoderLayer(
            embed_dim=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            activation=F.selu,
            batch_first=True,
        )
        self.transformer_encoder = CrossAttentionEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_decoder = CrossAttentionDecoder(decoder_layer, num_layers=1)
        self.norm = LayerNorm(embed_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor) -> tuple[Tensor, Tensor]:
        encoder_output, encoder_weights = self.transformer_encoder(
            q=x,  # query
            k=x,  # key
            v=x,  # value
            src_key_padding_mask=src_key_padding_mask,
        )
        kv = self.norm(encoder_output)
        decoder_output, decoder_weights = self.transformer_decoder(
            q=x,
            k=kv,
            v=kv,
            src_key_padding_mask=src_key_padding_mask,
        )
        return decoder_output, decoder_weights


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])

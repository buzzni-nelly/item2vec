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

from carca.encoding import RotaryEncoding


class TransformerEncoderLayer(nn.Module):

    __constants__ = ["norm_first"]

    def __init__(
        self,
        embed_dim: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(
            embed_dim,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(embed_dim, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, embed_dim, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(embed_dim, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(embed_dim, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "activation"):
            self.activation = F.gelu

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
        residual_strategy_1: Literal["sum", "multiply", "none"] = "sum",
        residual_strategy_2: Literal["sum", "multiply", "none"] = "sum",
    ) -> tuple[Tensor, Tensor]:

        assert q.dtype == k.dtype == v.dtype, "k, q, v must have the same dtype"

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=q.dtype,
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=q.dtype,
            check_other=False,
        )

        sa, w = self._sa_block(
            q,
            k,
            v,
            src_mask,
            src_key_padding_mask,
            is_causal=is_causal,
            need_weights=need_weights,
        )

        if residual_strategy_1 == "sum":
            x = self.norm1(q + sa)
        elif residual_strategy_1 == "multiply":
            x = self.norm1(q * sa)
        elif residual_strategy_1 == "none":
            x = sa
        else:
            raise ValueError(f"Invalid residual operator: {residual_strategy_1}")

        if residual_strategy_2 == "sum":
            x = self.norm2(x + self._ff_block(x))
        elif residual_strategy_2 == "multiply":
            x = self.norm2(x * self._ff_block(x))
        elif residual_strategy_2 == "none":
            x = sa
        else:
            raise ValueError(f"Invalid residual operator: {residual_strategy_2}")

        return x, w

    # self-attention block
    def _sa_block(
        self,
        k: Tensor,
        q: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> tuple[Tensor, Tensor]:
        x, weights = self.self_attn(
            k,
            q,
            v,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            is_causal=is_causal,
        )
        x = self.dropout(x)
        return x, weights

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class TransformerEncoder(nn.Module):

    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer: "TransformerEncoderLayer",
        num_layers: int,
        norm: Optional[nn.Module] = None,
        mask_check: bool = True,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        # item2vec pretrained 된 임베딩을 사용하기 때문에 필요 없는듯.. (성능 하락)
        # self.layer_norm = LayerNorm(128)
        self.norm = norm
        self.mask_check = mask_check

    def forward(
        self,
        q: Tensor,  # query: item embeddings
        k: Tensor,  # key: item embeddings
        v: Tensor,  # value: item embeddings
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
    ) -> tuple[Tensor, list[Tensor]]:

        assert q.dtype == k.dtype == v.dtype, "k, q, v must have the same dtype"

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=q.dtype,
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=q.dtype,
            check_other=False,
        )

        first_layer = self.layers[0]
        batch_first = first_layer.self_attn.batch_first

        seq_len = _get_seq_len(q, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        output = q

        weights = []
        for mod in self.layers:
            # q = self.layer_norm(q)
            output, weight = mod(
                q,
                k,
                v,
                src_mask=mask,
                is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask,
                residual_strategy_1="sum",
                residual_strategy_2="sum",
            )
            q, k, v = output, output, output
            weights.append(weight)

        if self.norm is not None:
            output = self.norm(output)

        return output, weights


class TransformerDecoder(nn.Module):

    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer: "TransformerEncoderLayer",
        num_layers: int,
        norm: Optional[nn.Module] = None,
        mask_check: bool = True,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.mask_check = mask_check

    def forward(
        self,
        q: Tensor,  # query: item embeddings
        k: Tensor,  # key: TransformerEncoder logit
        v: Tensor,  # value: TransformerEncoder logit
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
    ) -> tuple[Tensor, list[Tensor]]:

        assert q.dtype == k.dtype == v.dtype, "k, q, v must have the same dtype"

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=q.dtype,
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=q.dtype,
            check_other=False,
        )

        first_layer = self.layers[0]
        batch_first = first_layer.self_attn.batch_first

        seq_len = _get_seq_len(k, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        weights = []
        output = q
        for mod in self.layers:
            output, weight = mod(
                q,
                k,
                v,
                src_mask=mask,
                is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask,
                residual_strategy_1="sum",
                residual_strategy_2="sum",
            )
            q = output
            weights.append(weight)

        if self.norm is not None:
            output = self.norm(output)

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

        encoder_layer = TransformerEncoderLayer(
            embed_dim=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
        )
        decoder_layer = TransformerEncoderLayer(
            embed_dim=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor) -> tuple[Tensor, Tensor]:
        encoder_output, encoder_weights = self.transformer_encoder(
            x,  # query
            x,  # key
            x,  # value
            src_key_padding_mask=src_key_padding_mask,
        )

        k = encoder_output  # self.rotary_encoding_kv(encoder_output)  # 성능 하락함.
        v = encoder_output  # self.rotary_encoding_kv(encoder_output)  # 성능 하락함.
        decoder_output, decoder_weights = self.transformer_decoder(
            x,  # query
            k,  # key
            v,  # value
            src_key_padding_mask=src_key_padding_mask,
        )
        return decoder_output, decoder_weights


def _generate_square_subsequent_mask(
    sz: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )


def _get_seq_len(src: Tensor, batch_first: bool) -> Optional[int]:
    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _detect_is_causal_mask(
    mask: Optional[Tensor],
    is_causal: Optional[bool] = None,
    size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = is_causal is True

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    encoder_n_layers: int = 8  # number of encoder layers
    decoder_memory_start: int = 0  # feed encoder output from Xth layer of decoder
    decoder_memory_end: int = 32  # feed encoder output up to the Xth layer of decoder
    encoder_dim: int = 128  # reduce computational complexity, = dim // n_heads
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_dec_seq_len: int = 2048
    max_enc_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads  # n_heads per gpu
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False
        )

        self.cache_k = torch.zeros(
            (args.max_batch_size, 512, self.n_local_heads, self.head_dim)
        )
        self.cache_v = torch.zeros(
            (args.max_batch_size, 512, self.n_local_heads, self.head_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        is_eval: bool = False
    ):
        bsz, seqlen, _ = x.shape
        original_seqlen = seqlen
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if is_eval:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            keys = xk
            values = xv

        if encoder_output is not None:
            encoder_output = encoder_output.view(bsz, -1, self.head_dim * self.n_local_heads)
            encoder_k = self.wk(encoder_output)
            encoder_k = encoder_k.view(bsz, -1, self.n_local_heads, self.head_dim)
            encoder_v = self.wv(encoder_output)
            encoder_v = encoder_v.view(bsz, -1, self.n_local_heads, self.head_dim)
            keys = torch.cat([encoder_k, keys], dim=1)
            values = torch.cat([encoder_v, values], dim=1)
            if seqlen != 1:
                extra_mask = torch.zeros(1, 1, seqlen, encoder_v.size(1)).to(mask)
                mask = torch.cat([extra_mask, mask], dim=-1)
                # FIXME: should xk, xv length be capped?
                # mask = mask[:, :, :, : min(self.cache_k.size(1), mask.size(-1))]
                # seqlen = min(self.cache_k.size(1) - start_pos, seqlen + encoder_k.size(1))

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, original_seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        encoder_output: Optional[torch.Tensor] = None,
        is_eval=False
    ):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask, encoder_output, is_eval=is_eval)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.encoder_n_layers = params.encoder_n_layers
        self.decoder_memory_start = params.decoder_memory_start
        self.decoder_memory_end = params.decoder_memory_end

        self.tok_embeddings = nn.Embedding(
            params.vocab_size, params.dim
        )
        self.enc_tok_embeddings = nn.Embedding(
            params.vocab_size, params.encoder_dim
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(
            params.dim, params.vocab_size
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads, params.max_dec_seq_len * 2
        )
        self.memory_freqs_cis = precompute_freqs_cis(
            params.encoder_dim // params.n_heads, params.max_enc_seq_len * 2
        )

        encoder_params = params
        params.n_layers = params.encoder_n_layers
        self.encoder = torch.nn.ModuleList()
        params.dim = params.encoder_dim
        for layer_id in range(params.encoder_n_layers):
            self.encoder.insert(0, TransformerBlock(self.n_layers + layer_id, encoder_params))

    def forward(self, tokens: torch.Tensor, start_pos: int, memory_tokens: torch.Tensor = None, memory_start_pos: int = 0, is_eval=False):
        bsz, seqlen = tokens.shape

        # freeze weights
        with torch.no_grad():
            h = self.tok_embeddings(tokens)
            self.freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        if memory_tokens is not None:
            h_memory = self.enc_tok_embeddings(memory_tokens)
            self.memory_freqs_cis = self.memory_freqs_cis.to(h_memory.device)
            memory_freqs_cis = self.memory_freqs_cis[memory_start_pos: memory_start_pos + memory_tokens.size(1)]

            for layer in self.encoder:
                h_memory = layer(h_memory, memory_start_pos, memory_freqs_cis)

        for layer in self.layers:
            if memory_tokens is not None and layer.layer_id >= self.decoder_memory_start and layer.layer_id < self.decoder_memory_end:
                # If paired with encoder layer
                h = layer(h, start_pos, freqs_cis, mask, encoder_output=h_memory, is_eval=is_eval)
            else:
                with torch.no_grad():
                    h = layer(h, start_pos, freqs_cis, mask, is_eval=is_eval)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float(), self.output(h).view(bsz, self.vocab_size, -1)

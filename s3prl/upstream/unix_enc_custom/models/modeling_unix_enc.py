import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchaudio.compliance.kaldi import fbank
from transformers import PreTrainedModel
from upstream.unix_enc_custom.models.configuration_unix_enc import UnixEncConfig
from fairseq.data.data_utils import compute_mask_indices
from typing import Dict, List, Optional, Tuple
import numpy as np
from fairseq.modules import LayerNorm, SamePad, TransposeLast
from fairseq.utils import index_put
import math
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.wav2vec.utils import pad_to_multiple
from fairseq import utils


class MultiheadAttention(nn.Module):
    """Drop-in for fairseq MultiheadAttention with checkpoint-compatible weight names.
    Uses PyTorch's fused in_proj path so it works under bf16/fp16 autocast on sm_120.

    The s3prl upstream substitutes this for fairseq's native MultiheadAttention on the
    attn_impl="fairseq" path, so fairseq-trained checkpoints run under bf16/fp16 on sm_120.
    It is numerically identical and weight-compatible with fairseq's MultiheadAttention.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, self_attention=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        in_proj_weight = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
        in_proj_bias = torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias])
        return F.multi_head_attention_forward(
            query, key, value,
            self.embed_dim, self.num_heads,
            in_proj_weight, in_proj_bias,
            None, None, False,
            self.dropout if self.training else 0.0,
            self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
        )


def make_conv_pos(e, k, g, is_batch_norm=False):
    pos_conv = nn.Conv1d(
        e,
        e,
        kernel_size=k,
        padding=k // 2,
        groups=g,
    )
    dropout = 0
    std = math.sqrt((4 * (1.0 - dropout)) / (k * e))
    nn.init.normal_(pos_conv.weight, mean=0, std=std)
    nn.init.constant_(pos_conv.bias, 0)

    if not is_batch_norm:
        pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
        pos_conv = nn.Sequential(pos_conv, SamePad(k), nn.GELU())
    else:
        batch_norm = nn.BatchNorm1d(e)
        pos_conv = nn.Sequential(batch_norm, pos_conv, SamePad(k), nn.GELU())

    return pos_conv

class ChannelAverager(nn.Module):
    def forward(self, x):
        return x.mean(dim=1)

class Channel0(nn.Module):
    def forward(self, x):
        return x[:, 0, :, :]

class ChannelAttnPool(nn.Module):
    """Learned attention pooling over the channel axis.

    Collapses [B, C, T, D] -> [B, T, D] by letting a learned per-head query
    attend across the C channels independently at every (B, T) position. This
    replaces the non-learned `ch0`/`avg` collapse and stays channel-count
    agnostic: it works for any C (including C=1) since the softmax is taken over
    whatever channels are present.
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0.0, attn_impl="fairseq"):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.attn_impl = attn_impl

        self.query = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, C, T, D]
        B, C, T, D = x.shape
        H, hd = self.num_heads, self.head_dim
        k = self.k_proj(x).view(B, C, T, H, hd)
        v = self.v_proj(x).view(B, C, T, H, hd)

        if self.attn_impl == "sdpa":
            # SDPA over the channel axis: single learned query (Lq=1), C keys.
            # Batch the (B, T) positions and heads: N = B*T.
            N = B * T
            q = self.query.view(1, H, 1, hd).expand(N, H, 1, hd)            # [N, H, 1, hd]
            k_s = k.permute(0, 2, 3, 1, 4).reshape(N, H, C, hd)             # [N, H, C, hd]
            v_s = v.permute(0, 2, 3, 1, 4).reshape(N, H, C, hd)
            # SDPA applies the 1/sqrt(hd) scaling internally (== self.scaling).
            out = F.scaled_dot_product_attention(
                q, k_s, v_s,
                dropout_p=self.dropout_p if self.training else 0.0,
            )                                                              # [N, H, 1, hd]
            out = out.reshape(B, T, D)
        else:
            # scores over channels: [B, T, H, C]
            scores = torch.einsum("hd,bcthd->bthc", self.query, k) * self.scaling
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            # weighted sum over channels -> [B, T, H, hd] -> [B, T, D]
            out = torch.einsum("bthc,bcthd->bthd", attn, v).reshape(B, T, D)
        return self.out_proj(out)

class SDPAMultiheadAttention(nn.Module):
    """Multi-head attention backed by ``F.scaled_dot_product_attention``.

    Drop-in replacement for the subset of fairseq ``MultiheadAttention`` used by
    ``MchTransformerEncoderLayer``: it takes/returns sequence-first tensors
    ``(L, N, D)`` and accepts ``query``/``key``/``value`` (allowing cross
    attention) plus an optional ``key_padding_mask`` of shape ``(N, L_k)`` where
    ``True`` marks padding. SDPA dispatches to the flash / memory-efficient
    kernels, which never materialize the ``L_q x L_k`` score matrix — the source
    of the OOM in the multi-channel (C>1) layers.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None):
        Lq, N, _ = query.shape
        Lk = key.shape[0]
        H, hd = self.num_heads, self.head_dim

        # (L, N, D) -> (N, H, L, hd)
        q = self.q_proj(query).view(Lq, N, H, hd).permute(1, 2, 0, 3)
        k = self.k_proj(key).view(Lk, N, H, hd).permute(1, 2, 0, 3)
        v = self.v_proj(value).view(Lk, N, H, hd).permute(1, 2, 0, 3)

        sdpa_mask = None
        if key_padding_mask is not None:
            # fairseq convention: True == pad. SDPA boolean mask: True == attend.
            sdpa_mask = (~key_padding_mask).view(N, 1, 1, Lk)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        # (N, H, Lq, hd) -> (Lq, N, D)
        out = out.permute(2, 0, 1, 3).reshape(Lq, N, self.embed_dim)
        out = self.out_proj(out)
        # attn weights not returned (callers always pass need_weights=False)
        return out, None

class MchTransformerEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
        layer_type: str = "crosschannel",
        context_size: int = 3,
        attn_impl: str = "fairseq",
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.layer_type = layer_type
        self.attn_impl = attn_impl

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        if self.layer_type == 'crosschannel':
            self_attention=False
        else:
            self_attention=True
        if attn_impl == "sdpa":
            self.self_attn = SDPAMultiheadAttention(
                self.embedding_dim,
                num_attention_heads,
                dropout=attention_dropout,
            )
        elif attn_impl == "fairseq":
            # s3prl upstream: use the bf16/fp16-safe drop-in in place of fairseq's
            # native MultiheadAttention (numerically identical, weight-compatible).
            self.self_attn = MultiheadAttention(
                self.embedding_dim,
                num_attention_heads,
                dropout=attention_dropout,
                self_attention=self_attention,
            )
        else:
            raise ValueError(f"Unknown attn_impl: {attn_impl}")
        #self.self_attn = torch.nn.MultiheadAttention(
        #    self.embedding_dim,
        #    num_attention_heads, 
        #    dropout=attention_dropout,
        #    batch_first=True,
        #)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)
        self.context_size = context_size

    def expand_context(
        self, 
        x, 
        context_size
    ):
        B, C, T, D = x.shape
        assert context_size % 2 == 1

        pad_size = context_size // 2
        x_padded = torch.nn.functional.pad(x, (0, 0, pad_size, pad_size), mode='constant')
        x_context_list = [x_padded[:, c:c+1, i:T + i, :] for c in range(C) for i in range(context_size)]
        x_context = torch.cat(x_context_list, dim=1)
        return x_context

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        if self.layer_type == 'crosschannel':
            # B x C x T x D
            B, C, T, D = x.size()
            residual = x

            if self.layer_norm_first:
                raise
                #x = self.self_attn_layer_norm(x)
                #x, attn = self.self_attn(
                #    query=x,
                #    key=x,
                #    value=x,
                #    key_padding_mask=self_attn_padding_mask,
                #    attn_mask=self_attn_mask,
                #    need_weights=False,
                #)
                #x = self.dropout1(x)
                #x = residual + x

                #residual = x
                #x = self.final_layer_norm(x)
                #x = self.activation_fn(self.fc1(x))
                #x = self.dropout2(x)
                #x = self.fc2(x)

                #layer_result = x

                #x = self.dropout3(x)
                #x = residual + x
            else:
                x_context = self.expand_context(x, context_size=self.context_size)
                #q = x.permute(0, 2, 1, 3).reshape(-1, x.shape[1], x.shape[3]) 
                #k = x_context.permute(0, 2, 1, 3).reshape(-1, x_context.shape[1], x_context.shape[3])
                #v = x_context.permute(0, 2, 1, 3).reshape(-1, x_context.shape[1], x_context.shape[3])
                q = x.permute(1, 0, 2, 3).reshape(x.shape[1], -1, x.shape[3]) # C x (B x T) x D
                k = x_context.permute(1, 0, 2, 3).reshape(x_context.shape[1], -1, x_context.shape[3])
                v = x_context.permute(1, 0, 2, 3).reshape(x_context.shape[1], -1, x_context.shape[3])
                #print("q", q.size(), "k", k.size(), "v", v.size())
                x, attn = self.self_attn(
                    query=q,
                    key=k,
                    value=v,
                    need_weights=False,
                )
                #print("attn", attn.shape)
                #print("x1", x.shape)
                x = (x.reshape(x.shape[0], B, T, x.shape[2])).permute(1, 0, 2, 3)
                #print("x2", x.shape)

                x = self.dropout1(x)
                x = residual + x

                x = self.self_attn_layer_norm(x)

                residual = x
                x = self.activation_fn(self.fc1(x))
                x = self.dropout2(x)
                x = self.fc2(x)

                layer_result = x

                x = self.dropout3(x)
                x = residual + x
                x = self.final_layer_norm(x)
        elif self.layer_type == 'crossframe':
            if len(x.size()) == 4:
                # B x C x T x D
                B, C, T, D = x.size()
                residual = x

                if self.layer_norm_first:
                    raise
                    #x = self.self_attn_layer_norm(x)
                    #x, attn = self.self_attn(
                    #    query=x,
                    #    key=x,
                    #    value=x,
                    #    key_padding_mask=self_attn_padding_mask,
                    #    attn_mask=self_attn_mask,
                    #    need_weights=False,
                    #)
                    #x = self.dropout1(x)
                    #x = residual + x

                    #residual = x
                    #x = self.final_layer_norm(x)
                    #x = self.activation_fn(self.fc1(x))
                    #x = self.dropout2(x)
                    #x = self.fc2(x)

                    #layer_result = x

                    #x = self.dropout3(x)
                    #x = residual + x
                else:
                    x = x.reshape(-1, x.size(2), x.size(3)) # (B x C) x T x D
                    x = x.permute(1, 0, 2) # T x (B x C) x D
                    self_attn_padding_mask = self_attn_padding_mask.unsqueeze(1).expand(-1, C, -1)
                    self_attn_padding_mask = self_attn_padding_mask.reshape(-1, self_attn_padding_mask.size(2))
                    #print("input self attn x", x.shape)
                    x, attn = self.self_attn(
                        query=x,
                        key=x,
                        value=x,
                        key_padding_mask=self_attn_padding_mask,
                        need_weights=False,
                    )
                    #print("output self attn x", x.shape)
                    #print("attn", attn.shape)
                    x = x.permute(1, 0, 2)
                    x = x.reshape(B, C, T, D)

                    x = self.dropout1(x)
                    x = residual + x

                    x = self.self_attn_layer_norm(x)

                    residual = x
                    x = self.activation_fn(self.fc1(x))
                    x = self.dropout2(x)
                    x = self.fc2(x)

                    layer_result = x

                    x = self.dropout3(x)
                    x = residual + x
                    x = self.final_layer_norm(x)
            elif len(x.size()) == 3:
                # B x T x D
                B, T, D = x.size()
                residual = x

                if self.layer_norm_first:
                    raise
                    #x = self.self_attn_layer_norm(x)
                    #x, attn = self.self_attn(
                    #    query=x,
                    #    key=x,
                    #    value=x,
                    #    key_padding_mask=self_attn_padding_mask,
                    #    attn_mask=self_attn_mask,
                    #    need_weights=False,
                    #)
                    #x = self.dropout1(x)
                    #x = residual + x

                    #residual = x
                    #x = self.final_layer_norm(x)
                    #x = self.activation_fn(self.fc1(x))
                    #x = self.dropout2(x)
                    #x = self.fc2(x)

                    #layer_result = x

                    #x = self.dropout3(x)
                    #x = residual + x
                else:
                    x = x.permute(1, 0, 2)
                    #print("input self attn x", x.shape)
                    x, attn = self.self_attn(
                        query=x,
                        key=x,
                        value=x,
                        key_padding_mask=self_attn_padding_mask,
                        need_weights=False,
                    )
                    #print("output self attn x", x.shape)
                    #print("attn", attn.shape)
                    x = x.permute(1, 0, 2)

                    x = self.dropout1(x)
                    x = residual + x

                    x = self.self_attn_layer_norm(x)

                    residual = x
                    x = self.activation_fn(self.fc1(x))
                    x = self.dropout2(x)
                    x = self.fc2(x)

                    layer_result = x

                    x = self.dropout3(x)
                    x = residual + x
                    x = self.final_layer_norm(x)

        return x, (attn, layer_result)

class MchTransformerEncoder(nn.Module):
    def __init__(self, args, skip_pos_conv: bool = False):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim
        self.required_seq_len_multiple = args.required_seq_len_multiple

        pos_conv_depth = getattr(args, "pos_conv_depth", 1)
        if pos_conv_depth > 1:
            num_layers = args.pos_conv_depth
            k = max(3, args.conv_pos // num_layers)

            def make_conv_block(e, k, g, l):
                return nn.Sequential(
                    *[
                        nn.Sequential(
                            nn.Conv1d(
                                e,
                                e,
                                kernel_size=k,
                                padding=k // 2,
                                groups=g,
                            ),
                            SamePad(k),
                            TransposeLast(),
                            LayerNorm(e, elementwise_affine=False),
                            TransposeLast(),
                            nn.GELU(),
                        )
                        for _ in range(l)
                    ]
                )

            self.pos_conv = make_conv_block(
                self.embedding_dim, k, args.conv_pos_groups, num_layers
            )
        elif skip_pos_conv:
            self.pos_conv = None
        else:
            self.pos_conv = make_conv_pos(
                self.embedding_dim,
                args.conv_pos,
                args.conv_pos_groups,
                is_batch_norm=args.conv_pos_batch_norm
                if hasattr(args, "conv_pos_batch_norm")
                else False,
            )

        if args.channel_pos_enc == 'none':
            self.channel_pos_enc = None
        elif args.channel_pos_enc == 'learnable':
            self.channel_pos_enc = nn.Parameter(torch.zeros(args.max_num_chans, args.encoder_embed_dim))
            nn.init.xavier_uniform_(self.channel_pos_enc)

        encoder_layers = (args.encoder_layers).split(',')
        layer_list = []
        for layer in encoder_layers:
            if layer.startswith('cc'):
                layer_list.append(
                    MchTransformerEncoderLayer(
                        embedding_dim=args.encoder_embed_dim,
                        ffn_embedding_dim=args.encoder_ffn_embed_dim,
                        num_attention_heads=args.encoder_attention_heads,
                        dropout=args.dropout,
                        attention_dropout=args.attention_dropout,
                        activation_dropout=args.activation_dropout,
                        activation_fn=args.activation_fn,
                        layer_norm_first=args.layer_norm_first,
                        layer_type='crosschannel',
                        context_size=int(layer.lstrip('cc')),
                        attn_impl=getattr(args, "attn_impl", "fairseq"),
                    )
                )
            elif layer == 'cf':
                layer_list.append(
                    MchTransformerEncoderLayer(
                        embedding_dim=args.encoder_embed_dim,
                        ffn_embedding_dim=args.encoder_ffn_embed_dim,
                        num_attention_heads=args.encoder_attention_heads,
                        dropout=args.dropout,
                        attention_dropout=args.attention_dropout,
                        activation_dropout=args.activation_dropout,
                        activation_fn=args.activation_fn,
                        layer_norm_first=args.layer_norm_first,
                        layer_type='crossframe',
                        attn_impl=getattr(args, "attn_impl", "fairseq"),
                    )
                )
            elif layer == 'avg':
                layer_list.append(ChannelAverager())
            elif layer == 'ch0':
                layer_list.append(Channel0())
            elif layer == 'attnpool':
                layer_list.append(
                    ChannelAttnPool(
                        embed_dim=args.encoder_embed_dim,
                        num_heads=args.encoder_attention_heads,
                        dropout=args.attention_dropout,
                        attn_impl=getattr(args, "attn_impl", "fairseq"),
                    )
                )

        
        self.layers = nn.ModuleList(layer_list)
        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop
        self.checkpoint_activations = getattr(args, "checkpoint_activations", False)

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, layer=None, corpus_key=None):
        x, layer_results = self.extract_features(
            x, padding_mask, layer, corpus_key=corpus_key
        )

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(
        self,
        x,
        padding_mask=None,
        tgt_layer=None,
        min_layer=0,
        corpus_key=None,
    ):

        x = x * (~padding_mask).unsqueeze(1).unsqueeze(-1)

        B, C, T, D = x.size(0), x.size(1), x.size(2), x.size(3)

        if self.channel_pos_enc is not None:
            x = x + ((self.channel_pos_enc[:C, :].to(x.dtype)).unsqueeze(1)).unsqueeze(0)

        x = x.view(B * C, T, D)

        if self.pos_conv is not None:
            x_conv = self.pos_conv(x.transpose(1, 2))
            x_conv = x_conv.transpose(1, 2)
            x = x + x_conv

        x = x.view(B, C, T, D)

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        # pad to the sequence length dimension
        x, pad_length = pad_to_multiple(
            x, self.required_seq_len_multiple, dim=-2, value=0
        )
        if pad_length > 0 and padding_mask is None:
            padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
            padding_mask[:, -pad_length:] = True
        else:
            padding_mask, _ = pad_to_multiple(
                padding_mask, self.required_seq_len_multiple, dim=-1, value=True
            )
        x = F.dropout(x, p=self.dropout, training=self.training) # B x C x T x D

        layer_results = []
        r = None

        for i, layer in enumerate(self.layers):
            if isinstance(layer, (ChannelAverager, Channel0, ChannelAttnPool)):
                x = layer(x)
                #print("Layer", i, x.size())
                continue
            dropout_probability = np.random.random() if self.layerdrop > 0 else 1
            if not self.training or (dropout_probability > self.layerdrop):
                if self.checkpoint_activations and self.training:
                    x, (z, lr) = checkpoint(
                        layer,
                        x,
                        self_attn_padding_mask=padding_mask,
                        need_weights=False,
                        use_reentrant=False,
                    )
                else:
                    x, (z, lr) = layer(
                        x, self_attn_padding_mask=padding_mask, need_weights=False
                    )
                if i >= min_layer:
                    layer_results.append((x, z, lr))
                #print("Layer", i, x.size())
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # undo paddding
        if pad_length > 0:
            if len(x.size()) == 4:
                x = x[:, :, :-pad_length, :]
            elif len(x.size()) == 3:
                x = x[:, :-pad_length, :]

            def undo_pad(a, b, c):
                if len(a.size()) == 4:
                    return (
                        a[:, :, :-pad_length],
                        b[:, :, :-pad_length] if b is not None else b,
                        c[:, :, :-pad_length],
                    )
                elif len(a.size()) == 3: 
                    return (
                        a[:, :-pad_length],
                        b[:, :-pad_length] if b is not None else b,
                        c[:, :-pad_length],
                    )

            layer_results = [undo_pad(*u) for u in layer_results]

        return x, layer_results

class WhisperConvBlock(nn.Module):
    def __init__(self):
        super(WhisperConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(80, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
    def forward(self, x, padding_mask): # [B, C, T, D]
        B, C, T, D = x.size()
        x = x.view(B * C, T, D)
        x = x.permute(0, 2, 1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = x.view(B, C, T, x.size(-1))
        return x, padding_mask

class WhisperConvBlock2(nn.Module):
    def __init__(self, in_dim=514):
        super(WhisperConvBlock2, self).__init__()
        self.conv1 = nn.Conv1d(in_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
    def forward(self, x, padding_mask): # [B, C, T, D]
        B, C, T, D = x.size()
        x = x.view(B * C, T, D)
        x = x.permute(0, 2, 1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        x = x.view(B, C, x.size(-2), x.size(-1))
        x = x[:, :, 1:-1, :]
        feat_length = torch.sum(~padding_mask, 1)
        feat_length_after = feat_length // 2 - 1
        padding_mask_new = torch.zeros((B, x.size(2)), device=padding_mask.device, dtype=padding_mask.dtype)
        for i in range(len(feat_length_after)):
            padding_mask_new[i, feat_length_after[i]:] = 1
        return x, padding_mask_new

class Conv2dSubsampling2(nn.Module):
    def __init__(self):
        super(Conv2dSubsampling2, self).__init__()
        self.output_dim = 64
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, self.output_dim, 3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(self.output_dim, self.output_dim, 3, stride=1),
            torch.nn.ReLU(),
        )
        self.fc = torch.nn.Linear(self.output_dim * (((257 - 1) // 2 - 2)), self.output_dim)

    def forward(self, x, padding_mask):
        B, C, K, T, D = x.size()
        x = x.view(B * C, K, T, D)
        x = self.conv(x)
        _, C1, T1, D1 = x.size()
        x = (x.transpose(2, 3)).reshape(B*C, T1, (C1*D1))
        x = self.fc(x)
        x = x.reshape(B, C, T1, -1)
        feat_length = torch.sum(~padding_mask, 1)
        feat_length_after = (feat_length - 1) // 2 - 2
        padding_mask_new = torch.zeros((B, T1), device=padding_mask.device, dtype=padding_mask.dtype)
        for i in range(len(feat_length_after)):
            padding_mask_new[i, feat_length_after[i]:] = 1
        return x, padding_mask_new

class UnixEncModel(PreTrainedModel):
    config_class = UnixEncConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.n_encoder_layers = config.encoder_layers

        feat_emb_dim = config.feat_emb_dim
        if 'conv_frontend' in self.config:
            if config.conv_frontend == 'none':
                self.feature_extractor = None
            elif config.conv_frontend == 'whisper':
                self.feature_extractor = WhisperConvBlock()
                feat_emb_dim = 256
            elif config.conv_frontend == 'whisper2':
                self.feature_extractor = WhisperConvBlock2(in_dim=getattr(config, 'conv_frontend_in_dim', 514))
                feat_emb_dim = 256
            elif config.conv_frontend == 'conv2d2':
                self.feature_extractor = Conv2dSubsampling2()
                feat_emb_dim = 64
        else:
            self.feature_extractor = None
        self.pre_extract_proj = (
            nn.Linear(feat_emb_dim, config.encoder_embed_dim)
            if feat_emb_dim != config.encoder_embed_dim
            else None
        )

        if config.channel_encoder_layers == '':
            self.channel_encoder = None
        else:
            channel_encoder_layers = (config.channel_encoder_layers).split(',')
            layer_list = []
            for layer in channel_encoder_layers:
                if layer.startswith('cc'):
                    layer_list.append(
                        MchTransformerEncoderLayer(
                            embedding_dim=config.encoder_embed_dim,
                            ffn_embedding_dim=config.encoder_ffn_embed_dim,
                            num_attention_heads=config.encoder_attention_heads,
                            dropout=config.dropout,
                            attention_dropout=config.attention_dropout,
                            activation_dropout=config.activation_dropout,
                            activation_fn=config.activation_fn,
                            layer_norm_first=config.layer_norm_first,
                            layer_type='crosschannel',
                            context_size=int(layer.lstrip('cc')),
                            attn_impl=getattr(config, "attn_impl", "fairseq"),
                        )
                    )
                else:
                    raise ValueError
            self.channel_encoder = nn.ModuleList(layer_list)

        self.encoder = MchTransformerEncoder(config)

        assert not self.config.mask_before_proj
        if self.config.learnable_mask_emb:
            if not self.config.mask_before_proj: 
                self.mask_emb = nn.Parameter(
                    torch.FloatTensor(config.encoder_embed_dim).uniform_()
                )
        else:
            if not self.config.mask_before_proj:
                self.mask_emb = 0

        self.final_proj = nn.Linear(config.encoder_embed_dim, config.num_cluster)

        self.bilabel = config.bi_label
        if self.bilabel:
            self.final_proj_sec = nn.Linear(config.encoder_embed_dim, config.num_cluster + 1)

    def apply_mask(self, x, padding_mask):
        B, C, T, D = x.shape
        if self.config.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.config.mask_prob,
                self.config.mask_length,
                self.config.mask_selection,
                self.config.mask_other,
                min_masks=2,
                no_overlap=self.config.no_mask_overlap,
                min_space=self.config.mask_min_space,
            )
            #print("mask_indices", mask_indices.shape)
            #print("mask_indices", 100.0 * np.sum(mask_indices) / (B * T))
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            #print("mask_indices", mask_indices.size())
            x[(mask_indices.unsqueeze(1)).expand(-1, C, -1)] = (self.mask_emb).to(x.device).to(x.dtype)
        else:
            mask_indices =  None

        #if self.mask_channel_prob > 0:
        #    mask_channel_indices = compute_mask_indices(
        #        (B, D),
        #        None,
        #        self.mask_channel_prob,
        #        self.mask_channel_length,
        #        self.mask_channel_selection,
        #        self.mask_channel_other,
        #        no_overlap=self.no_mask_channel_overlap,
        #        min_space=self.mask_channel_min_space,
        #    )
        #    mask_channel_indices = (
        #        torch.from_numpy(mask_channel_indices)
        #        .to(x.device)
        #        .unsqueeze(1)
        #        .expand(-1, T, -1)
        #    )
        #    x[mask_channel_indices] = 0
        return x, mask_indices

    def forward(
        self, 
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        sec_labels: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
        output_layer: Optional[int] = None,
    ):
        if self.bilabel:
            assert sec_labels is not None
        padding_mask = ~attention_mask

        if input_values.dim() == 3:
            input_values = input_values.unsqueeze(1)

        if self.feature_extractor is not None:
            input_values, padding_mask = self.feature_extractor(input_values, padding_mask)

        if self.pre_extract_proj is not None:
            features = self.pre_extract_proj(input_values)

        if self.channel_encoder is not None:
            for i, layer in enumerate(self.channel_encoder):
                if isinstance(layer, (ChannelAverager, Channel0, ChannelAttnPool)):
                    features = layer(features)
                elif isinstance(layer, MchTransformerEncoderLayer):
                    features, (z, lr) = layer(
                        features, self_attn_padding_mask=padding_mask, need_weights=False
                    )
                else:
                    raise

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
        else:
            x = features
            mask_indices = None
        
        x, _ = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )
        
        if features_only:
            return {"x": x, "padding_mask": padding_mask, "features": features}

        assert (x.size(1) - labels.size(1)) <= 3
        assert labels.size(1) >= x.size(1) # TODO: add support for another case
        labels = labels[:, :x.size(1)]
        if self.bilabel:
            sec_labels = sec_labels[:, :x.size(1)]

        masked_indices = torch.logical_and(~padding_mask, mask_indices)
        if self.bilabel:
            logit_m = self.final_proj(x[masked_indices])
            label_m = labels[masked_indices]
            logit_sec_m = self.final_proj_sec(x[masked_indices])
            label_sec_m = sec_labels[masked_indices]
            pri_loss = F.cross_entropy(logit_m, label_m)

            weights = torch.ones(logit_sec_m.size(-1), device=x.device)
            weights[-1] = 1e-10
            sec_loss = F.cross_entropy(logit_sec_m, label_sec_m, weight=weights)
            loss = pri_loss + sec_loss
            assert not torch.isnan(loss)
            return {"loss": loss, "pri_loss": pri_loss, "sec_loss": sec_loss, "logits_m": logit_m, "labels_m": label_m}
        else:
            logit_m = self.final_proj(x[masked_indices])
            label_m = labels[masked_indices]
            loss = F.cross_entropy(logit_m, label_m)
            return {"loss": loss, "logits_m": logit_m, "labels_m": label_m}

    def extract_features(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        padding_mask = ~attention_mask
        hidden_states_raw = []

        if input_values.dim() == 3:
            input_values = input_values.unsqueeze(1)

        if self.feature_extractor is not None:
            input_values, padding_mask = self.feature_extractor(input_values, padding_mask)

        if self.pre_extract_proj is not None:
            features = self.pre_extract_proj(input_values)

        if self.channel_encoder is not None:
            for i, layer in enumerate(self.channel_encoder):
                if isinstance(layer, (ChannelAverager, Channel0, ChannelAttnPool)):
                    features = layer(features)
                    hidden_states_raw.append(features)
                elif isinstance(layer, MchTransformerEncoderLayer):
                    features, (z, lr) = layer(
                        features, self_attn_padding_mask=padding_mask, need_weights=False
                    )
                    hidden_states_raw.append(features)
                else:
                    raise

        x = features

        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None,
        )

        for res in layer_results:
            hidden_states_raw.append(res[0])

        hidden_states = []
        for h_state in hidden_states_raw:
            if len(h_state.size()) == 4:
                hidden_states.append(h_state.mean(1))
            elif len(h_state.size()) == 3:
                hidden_states.append(h_state)
        return {"x": x, "hidden_states": hidden_states}

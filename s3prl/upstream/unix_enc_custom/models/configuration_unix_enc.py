import torch
from transformers import PreTrainedModel, PretrainedConfig

class UnixEncConfig(PretrainedConfig):
    def __init__(
        self, 
        feat_emb_dim: int = 40,
        pos_emb_type: str = "conv",
        pos_conv_depth: int = 1,
        conv_pos: int = 128,
        conv_pos_groups: int = 16,
        channel_pos_enc: str = 'none',
        max_num_chans: int = 8,
        channel_encoder_layers: str = '',
        encoder_layers: str = 'cc,cc,cc,avg,cf,cf,cf,cf,cf,cf,cf,cf,cf',
        encoder_embed_dim: int = 768,
        encoder_ffn_embed_dim: int = 3072,
        encoder_attention_heads: int = 12,
        activation_fn: str = "gelu",
        layer_norm_first: bool = False,
        num_cluster: int = 512,

        mask_prob: float = 0.8,
        mask_length: int = 10,
        mask_selection: str = 'static',
        mask_other: float = 0.0,
        no_mask_overlap: bool = False,
        mask_min_space: int = 1,
        mask_channel_prob: float = 0.0,

        learnable_mask_emb: bool = False,
        mask_before_proj: bool = True,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        encoder_layerdrop: float = 0.0,
        required_seq_len_multiple: int = 2,
        layer_type: str = "transformer",
        attn_impl: str = "fairseq",
        checkpoint_activations: bool = False,
        conv_frontend: str = "none",
        conv_frontend_in_dim: int = 514,
        feats_type: str = None,
        bi_label: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.feat_emb_dim = feat_emb_dim
        self.pos_emb_type = pos_emb_type
        self.pos_conv_depth = pos_conv_depth
        self.conv_pos = conv_pos
        self.conv_pos_groups = conv_pos_groups
        self.channel_pos_enc = channel_pos_enc
        self.max_num_chans = max_num_chans
        self.channel_encoder_layers = channel_encoder_layers
        self.encoder_layers = encoder_layers
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.activation_fn = activation_fn
        self.layer_norm_first = layer_norm_first
        self.num_cluster = num_cluster

        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_selection = mask_selection
        self.mask_other = mask_other
        self.no_mask_overlap = no_mask_overlap
        self.mask_min_space = mask_min_space
        self.mask_channel_prob = mask_channel_prob

        self.learnable_mask_emb = learnable_mask_emb
        self.mask_before_proj = mask_before_proj
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.required_seq_len_multiple = required_seq_len_multiple
        self.layer_type = layer_type
        self.attn_impl = attn_impl
        self.checkpoint_activations = checkpoint_activations
        self.conv_frontend = conv_frontend
        self.conv_frontend_in_dim = conv_frontend_in_dim
        self.feats_type = feats_type
        self.bi_label = bi_label

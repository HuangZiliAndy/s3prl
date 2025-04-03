import torch
from transformers import PreTrainedModel, PretrainedConfig

class MchMelHuBERTConfig(PretrainedConfig):
    def __init__(
        self, 
        feat_emb_dim: int = 40,
        pos_emb_type: str = "conv",
        pos_conv_depth: int = 1,
        conv_pos: int = 128,
        conv_pos_groups: int = 16,
        encoder_layers: int = 12,
        encoder_embed_dim: int = 768,
        encoder_ffn_embed_dim: int = 3072,
        encoder_attention_heads: int = 12,
        activation_fn: str = "gelu",
        layer_norm_first: bool = False,
        num_cluster: int = 512,
        pred_masked_weight: float = 1.0,
        pred_nomask_weight: float = 0.0,
        mask_prob: float = 0.8,
        mask_length: int = 10,
        mask_selection: str = 'static',
        mask_other: float = 0.0,
        no_mask_overlap: bool = False,
        mask_min_space: int = 1,
        mask_channel_prob: float = 0.0,
        skip_masked: bool = False,
        skip_nomask: bool = True,
        learnable_mask_emb: bool = False,
        mask_before_proj: bool = True,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        encoder_layerdrop: float = 0.0,
        required_seq_len_multiple: int = 2,
        layer_type: str = "transformer",
        checkpoint_activations: bool = False,
        conv_frontend: str = "none",
        n_channels: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.feat_emb_dim = feat_emb_dim
        self.pos_emb_type = pos_emb_type
        self.pos_conv_depth = pos_conv_depth
        self.conv_pos = conv_pos
        self.conv_pos_groups = conv_pos_groups
        self.encoder_layers = encoder_layers
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.activation_fn = activation_fn
        self.layer_norm_first = layer_norm_first
        self.num_cluster = num_cluster
        self.pred_masked_weight = pred_masked_weight
        self.pred_nomask_weight = pred_nomask_weight
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_selection = mask_selection
        self.mask_other = mask_other
        self.no_mask_overlap = no_mask_overlap
        self.mask_min_space = mask_min_space
        self.mask_channel_prob = mask_channel_prob
        self.skip_masked = skip_masked
        self.skip_nomask = skip_nomask
        self.learnable_mask_emb = learnable_mask_emb
        self.mask_before_proj = mask_before_proj
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.required_seq_len_multiple = required_seq_len_multiple
        self.layer_type = layer_type
        self.checkpoint_activations = checkpoint_activations
        self.conv_frontend = conv_frontend
        self.n_channels = n_channels

import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.compliance.kaldi import fbank
from transformers import PreTrainedModel
from upstream.mel_hubert_custom.models.configuration_mel_hubert import MelHuBERTConfig
from fairseq.models.wav2vec.wav2vec2 import TransformerEncoder
from fairseq.data.data_utils import compute_mask_indices
from typing import Dict, List, Optional, Tuple

class WhisperConvBlock(nn.Module):
    def __init__(self):
        super(WhisperConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(80, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        return x

class MelHuBERTModel(PreTrainedModel):
    config_class = MelHuBERTConfig

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
        else:
            self.feature_extractor = None
        self.pre_extract_proj = (
            nn.Linear(feat_emb_dim, config.encoder_embed_dim)
            if feat_emb_dim != config.encoder_embed_dim
            else None
        )
        self.encoder = TransformerEncoder(config)

        if self.config.learnable_mask_emb:
            self.mask_emb = nn.Parameter(
                torch.FloatTensor(config.encoder_embed_dim).uniform_()
            )
        else:
            self.mask_emb = 0

        self.final_proj = nn.Linear(config.encoder_embed_dim, config.num_cluster)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        if self.config.bi_label:
            # +1 for empty_token (= num_cluster), which represents "no interfering speaker"
            self.final_proj_sec = nn.Linear(config.encoder_embed_dim, config.num_cluster + 1)
            self.criterion_sec = nn.CrossEntropyLoss(ignore_index=-100)

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
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
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = (self.mask_emb).to(x.device).to(x.dtype)
        else:
            mask_indices =  None
        assert self.config.mask_channel_prob == 0
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
        padding_mask = ~attention_mask

        if self.feature_extractor is not None:
            input_values = self.feature_extractor(input_values)

        features = self.pre_extract_proj(input_values) if self.pre_extract_proj is not None else input_values

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

        masked_indices = torch.logical_and(~padding_mask, mask_indices)
        logit_m = self.final_proj(x[masked_indices])
        label_m = labels[masked_indices]
        loss = self.criterion(logit_m, label_m)

        if self.config.bi_label and sec_labels is not None:
            logit_sec_m = self.final_proj_sec(x[masked_indices])
            label_sec_m = sec_labels[masked_indices]
            loss = loss + self.criterion_sec(logit_sec_m, label_sec_m)

        return {"loss": loss, "logits_m": logit_m, "labels_m": label_m}

    def extract_features(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        padding_mask = ~attention_mask

        if self.feature_extractor is not None:
            input_values = self.feature_extractor(input_values)

        features = self.pre_extract_proj(input_values) if self.pre_extract_proj is not None else input_values

        x = features

        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=len(self.encoder.layers) - 1,
        )
        return {"x": x, "hidden_states": [res[0].transpose(0, 1) for res in layer_results]}

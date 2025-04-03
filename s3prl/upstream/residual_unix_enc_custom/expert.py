import os
import logging
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from upstream.interfaces import UpstreamBase

SAMPLE_RATE = 16000
EXAMPLE_SEC = 5

logger = logging.getLogger(__name__)


class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwds):
        super().__init__()
        from upstream.residual_unix_enc_custom.models.modeling_residual_unix_enc import ResidualUnixEncModel
        from upstream.residual_unix_enc_custom.models.configuration_residual_unix_enc import ResidualUnixEncConfig
        from safetensors.torch import load_file
        config_path = "{}/config.json".format(ckpt)
        config = ResidualUnixEncConfig.from_json_file(config_path)
        config.encoder_layerdrop = 0.0
        self.feats_type = 'STFT'

        self.model = ResidualUnixEncModel(config)
        model_weights_path = "{}/model.safetensors".format(ckpt)
        model_weights = load_file(model_weights_path)
        missing, unexpected = self.model.load_state_dict(model_weights)
        assert len(missing + unexpected) == 0

        self.n_chans = -1

    def get_downsample_rates(self, key: str = None) -> int:
        return 320

    def _create_attention_mask(self, lengths, max_length) -> torch.Tensor:
        attn_mask = torch.zeros((len(lengths), max_length), dtype=torch.bool)
        for i, length in enumerate(lengths):
            attn_mask[i, :length] = 1
        return attn_mask

    def _pad_to_multiple(self, tensor, pad_to_multiple_of) -> torch.Tensor:
        current_length = tensor.size(0)
        target_length = (current_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of
        pad_length = target_length - current_length
        return F.pad(tensor, (0, 0, 0, pad_length))

    def _extract_feats(self, waveform):
        if self.feats_type == 'STFT':
            stft_feats = torch.stft(waveform, n_fft=512, hop_length=160, win_length=400, window=torch.hann_window(400).to(waveform.device), return_complex=True)
            return stft_feats.transpose(-2, -1)

    def forward(self, wavs):
        device = wavs[0].device
        input_features, lengths = [], []
        for waveform in wavs:
            feats = self._extract_feats(waveform.transpose(0, 1))
            input_features.append(feats)
            lengths.append(feats.size(-2))

        max_T = max(lengths)
        batch_features = torch.zeros(len(input_features), input_features[0].size(0), max_T, input_features[0].size(2))
        for i in range(len(input_features)):
            batch_features[i, :, :lengths[i], :] = input_features[i]
        attn_mask = self._create_attention_mask(lengths, batch_features.size(2))
        batch = {
            "input_values": batch_features.to(device),
            "attention_mask": attn_mask.to(device),
        }

        output = self.model.extract_features(**batch)
        return {"hidden_states": output["hidden_states"]}

if __name__ == '__main__':
    device = torch.device("cuda")
    ckpt = '/export/c02/hzili1/workspace/s3prl/s3prl/upstream/residual_unix_enc_custom/exp/cfg39/checkpoint-200000'
    upstream = UpstreamExpert(ckpt)
    upstream = upstream.to(device)
    wavs = (torch.rand(7, 48501, 4)).to(device)
    output = upstream(wavs)
    print(len(output['hidden_states']))
    for h_state in output['hidden_states']:
        print(h_state.size())

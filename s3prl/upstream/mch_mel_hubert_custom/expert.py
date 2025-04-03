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
        from upstream.mch_mel_hubert_custom.models.modeling_mch_mel_hubert import MchMelHuBERTModel
        from upstream.mch_mel_hubert_custom.models.configuration_mch_mel_hubert import MchMelHuBERTConfig
        from safetensors.torch import load_file
        config_path = "{}/config.json".format(ckpt)
        config = MchMelHuBERTConfig.from_json_file(config_path)
        config.encoder_layerdrop = 0.0

        #mean_std_file = "/export/c02/hzili1/tools/s3prl/s3prl/upstream/mel_hubert_custom/libri-960-mean-std.npy"
        #mean, std = np.load(mean_std_file)
        #self.mean, self.std = torch.from_numpy(mean), torch.from_numpy(std)

        #assert os.path.exists("{}/combine_v0.npz".format(ckpt))
        #stats_file = "{}/combine_v0.npz".format(ckpt)
        #stats_file = np.load(stats_file)
        #self.mean, self.std = stats_file['mean'], stats_file['std']
        #print('mean')
        #print(self.mean)
        #print('std')
        #print(self.std)
        #self.mean, self.std = torch.from_numpy(self.mean), torch.from_numpy(self.std)

        self.model = MchMelHuBERTModel(config)
        model_weights_path = "{}/model.safetensors".format(ckpt)
        model_weights = load_file(model_weights_path)
        missing, unexpected = self.model.load_state_dict(model_weights)
        assert len(missing + unexpected) == 0

        self.n_chans = 4

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

    def forward(self, wavs):
        device = wavs[0].device
        input_features, lengths = [], []
        for waveform in wavs:
            assert len(waveform.size()) == 2
            waveform = waveform.transpose(0, 1)
            x_stft = torch.stft(
                        waveform, n_fft = 512, win_length = 400, hop_length = 160, window = torch.hann_window(400).to(device), center = True, return_complex = False
                     )
            specgram = x_stft[0:1, :, :, 0]**2 + x_stft[0:1, :, :, 1]**2
            mel_scale = torchaudio.transforms.MelScale(n_mels=80, sample_rate=16000, n_stft=257).to(device)
            log_mel_specgram = torch.log1p(mel_scale(specgram))

            phase = torch.atan2(x_stft[..., 1], x_stft[..., 0])
            ref_phase = phase[0:1, :, :]
            ipd = phase[1:, :, :] - ref_phase
            cos_ipd = torch.cos(ipd)
            cos_ipd_transform = mel_scale(cos_ipd)
            stack_feats = torch.cat([log_mel_specgram, cos_ipd_transform], 0)
            stack_feats = stack_feats.transpose(1, 2) # [channels, T, F]

            stack_feats = stack_feats[:, 1:-1, :]
            if stack_feats.size(1) % 2 == 1:
                stack_feats = stack_feats[:, :-1, :]
            stack_feats = stack_feats.view(stack_feats.size(0), int(stack_feats.size(1) / 2), 2, stack_feats.size(2))
            stack_feats = stack_feats.reshape(stack_feats.size(0), stack_feats.size(1), 2 * stack_feats.size(3))
            input_features.append(stack_feats)
            lengths.append(stack_feats.size(1))

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
    ckpt = '/export/c02/hzili1/workspace/s3prl/s3prl/upstream/mch_mel_hubert_custom/exp/cfg21/checkpoint-200000'
    upstream = UpstreamExpert(ckpt)
    upstream = upstream.to(device)
    wavs = (torch.rand(7, 4, 48501)).to(device)
    upstream(wavs)

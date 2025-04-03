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
        from upstream.mel_hubert_custom.models.modeling_mel_hubert import MelHuBERTModel
        from upstream.mel_hubert_custom.models.configuration_mel_hubert import MelHuBERTConfig
        from safetensors.torch import load_file
        config_path = "{}/config.json".format(ckpt)
        config = MelHuBERTConfig.from_json_file(config_path)
        config.encoder_layerdrop = 0.0

        #mean_std_file = "/export/c02/hzili1/tools/s3prl/s3prl/upstream/mel_hubert_custom/libri-960-mean-std.npy"
        #mean, std = np.load(mean_std_file)
        #self.mean, self.std = torch.from_numpy(mean), torch.from_numpy(std)

        assert os.path.exists("{}/libri-960.npz".format(ckpt)) or os.path.exists("{}/combine_v0.npz".format(ckpt))
        if os.path.exists("{}/libri-960.npz".format(ckpt)):
            stats_file = "{}/libri-960.npz".format(ckpt)
            self.scale = True
        elif os.path.exists("{}/combine_v0.npz".format(ckpt)):
            stats_file = "{}/combine_v0.npz".format(ckpt)
            self.scale = False
        else:
            stats_file = None
        stats_file = np.load(stats_file)
        self.mean, self.std = stats_file['mean'], stats_file['std']
        print('mean')
        print(self.mean)
        print('std')
        print(self.std)
        print("self.scale", self.scale)
        self.mean, self.std = torch.from_numpy(self.mean), torch.from_numpy(self.std)

        self.model = MelHuBERTModel(config)
        model_weights_path = "{}/model.safetensors".format(ckpt)
        model_weights = load_file(model_weights_path)
        missing, unexpected = self.model.load_state_dict(model_weights)
        assert len(missing + unexpected) == 0

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
        if self.scale:
            wavs = [wav * (2 ** 15) for wav in wavs]
        input_features, lengths = [], []
        for audio in wavs:
            fbank_feats = torchaudio.compliance.kaldi.fbank(
                        audio.unsqueeze(0),
                        num_mel_bins=40,
                        sample_frequency=16000,
                        window_type='hamming',
                        frame_length=25,
                        frame_shift=10
                    )
            fbank_feats = (fbank_feats - self.mean.to(device)) / self.std.to(device)
            T, D = fbank_feats.size()
            if T % 2 != 0:
                fbank_feats = torch.cat([fbank_feats, torch.zeros(1, D, device=fbank_feats.device, dtype=fbank_feats.dtype)], dim=0)
            fbank_feats = fbank_feats.view(-1, 2, D).reshape(-1, 2 * D)

            input_features.append(fbank_feats)
            lengths.append(len(fbank_feats))

        batch_features = torch.nn.utils.rnn.pad_sequence(input_features, batch_first=True).float()
        attn_mask = self._create_attention_mask(lengths, batch_features.size(1))
        batch = {
            "input_values": batch_features.to(device),
            "attention_mask": attn_mask.to(device),
        }

        output = self.model.extract_features(**batch)
        return {"hidden_states": output["hidden_states"]}

if __name__ == '__main__':
    device = torch.device("cuda")
    ckpt = '/export/c02/hzili1/tools/s3prl/s3prl/upstream/mel_hubert_custom/exp/cfg0/checkpoint-159000'
    upstream = UpstreamExpert(ckpt)
    upstream = upstream.to(device)
    wavs = (torch.rand(4, 48501)).to(device)
    upstream(wavs)

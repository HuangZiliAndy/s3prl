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

        stats_path = "{}/stats.npz".format(ckpt)
        assert os.path.exists(stats_path)
        stats = np.load(stats_path)
        self.register_buffer('mean', torch.from_numpy(stats['mean']).float())
        self.register_buffer('std', torch.from_numpy(stats['std']).float())

        self.model = MelHuBERTModel(config)
        model_weights_path = "{}/model.safetensors".format(ckpt)
        model_weights = load_file(model_weights_path)
        missing, unexpected = self.model.load_state_dict(model_weights)
        assert len(missing + unexpected) == 0

    def get_collate_fn(self):
        def fbank_collate(wavs):
            results = []
            for wav in wavs:
                fbank = torchaudio.compliance.kaldi.fbank(
                    wav.unsqueeze(0).float(),
                    num_mel_bins=40,
                    sample_frequency=16000,
                    window_type='hamming',
                    frame_length=25,
                    frame_shift=10,
                )
                results.append(fbank)
            return results
        return fbank_collate

    def get_downsample_rates(self, key: str = None) -> int:
        return 320

    def _create_attention_mask(self, lengths, max_length) -> torch.Tensor:
        attn_mask = torch.zeros((len(lengths), max_length), dtype=torch.bool)
        for i, length in enumerate(lengths):
            attn_mask[i, :length] = 1
        return attn_mask

    def forward(self, wavs):
        device = wavs[0].device

        if wavs[0].dim() == 2:
            # Fast path: fbank pre-computed by DataLoader workers on CPU.
            # wavs is a list of (T_i, 40) tensors already on device.
            # Pad → normalize → stride, all as single batched GPU ops.
            fbank_list = wavs
            pre_stride_lengths = [f.size(0) for f in fbank_list]

            batch_features = torch.nn.utils.rnn.pad_sequence(
                fbank_list, batch_first=True
            ).float()  # (B, T_max, 40)

            batch_features = (batch_features - self.mean) / self.std

            B, T, D = batch_features.shape
            if T % 2 != 0:
                batch_features = torch.cat([
                    batch_features,
                    torch.zeros(B, 1, D, device=device, dtype=batch_features.dtype),
                ], dim=1)
                T += 1
            batch_features = batch_features.view(B, T // 2, 2, D).reshape(B, T // 2, 2 * D)
            lengths = [(l + 1) // 2 for l in pre_stride_lengths]
            attn_mask = self._create_attention_mask(lengths, batch_features.size(1))
            batch = {
                "input_values": batch_features.to(device),
                "attention_mask": attn_mask.to(device),
            }
            output = self.model.extract_features(**batch)
            return {"hidden_states": output["hidden_states"], "output_lengths": lengths}
        else:
            # Original path: raw 1-D audio tensors, compute fbank per sample on device.
            input_features, lengths = [], []
            for audio in wavs:
                fbank_feats = torchaudio.compliance.kaldi.fbank(
                    audio.unsqueeze(0),
                    num_mel_bins=40,
                    sample_frequency=16000,
                    window_type='hamming',
                    frame_length=25,
                    frame_shift=10,
                )
                fbank_feats = (fbank_feats - self.mean) / self.std
                T, D = fbank_feats.size()
                if T % 2 != 0:
                    fbank_feats = torch.cat(
                        [fbank_feats, torch.zeros(1, D, device=device, dtype=fbank_feats.dtype)],
                        dim=0,
                    )
                fbank_feats = fbank_feats.view(-1, 2, D).reshape(-1, 2 * D)
                input_features.append(fbank_feats)
                lengths.append(len(fbank_feats))
            batch_features = torch.nn.utils.rnn.pad_sequence(
                input_features, batch_first=True
            ).float()

        attn_mask = self._create_attention_mask(lengths, batch_features.size(1))
        batch = {
            "input_values": batch_features.to(device),
            "attention_mask": attn_mask.to(device),
        }

        output = self.model.extract_features(**batch)
        return {"hidden_states": output["hidden_states"], "output_lengths": lengths}

if __name__ == '__main__':
    device = torch.device("cuda")
    ckpt = '/workspace/workspace/MelHuBERT/exp/cfg2/checkpoint-141000'
    upstream = UpstreamExpert(ckpt)
    upstream = upstream.to(device)
    upstream.eval()
    wavs = (torch.rand(4, 48501)).to(device)
    output = upstream(wavs)
    print([hidden.shape for hidden in output["hidden_states"]])

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
        from upstream.unix_enc_custom.models.modeling_unix_enc import UnixEncModel
        from upstream.unix_enc_custom.models.configuration_unix_enc import UnixEncConfig
        from safetensors.torch import load_file
        config_path = "{}/config.json".format(ckpt)
        config = UnixEncConfig.from_json_file(config_path)
        config.encoder_layerdrop = 0.0

        if config.conv_frontend == 'whisper2':
            self.feats_type = 'STFT_CAT'
        else:
            self.feats_type = 'FBANK'

        if self.feats_type == 'FBANK':
            assert os.path.exists("{}/combine_v0.npz".format(ckpt))
            stats_file = "{}/combine_v0.npz".format(ckpt)
            stats_file = np.load(stats_file)
            self.mean, self.std = stats_file['mean'], stats_file['std']
            print('mean')
            print(self.mean)
            print('std')
            print(self.std)
            self.mean, self.std = torch.from_numpy(self.mean), torch.from_numpy(self.std)

        self.model = UnixEncModel(config)
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
        if self.feats_type == 'LFB+IPD':
            x_stft = torch.stft(
                waveform, n_fft = 512, win_length = 400, hop_length = 160, window = torch.hann_window(400), center = True, return_complex = False
            ) # [channels, F, T, real/img]
            specgram = x_stft[0:1, :, :, 0]**2 + x_stft[0:1, :, :, 1]**2
            mel_scale = torchaudio.transforms.MelScale(n_mels=80, sample_rate=16000, n_stft=257)
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
            return stack_feats
        elif self.feats_type == 'FBANK':
            fbank_feats = torch.stack([torchaudio.compliance.kaldi.fbank(
                            waveform[i:i+1, :],
                            num_mel_bins=40,
                            sample_frequency=16000,
                            window_type='hamming',
                            frame_length=25,
                            frame_shift=10
                        ) for i in range(len(waveform))])
            if self.mean is not None and self.std is not None:
                fbank_feats = (fbank_feats - self.mean) / self.std
            C, T, D = fbank_feats.size()
            if T % 2 != 0:
                fbank_feats = torch.cat([fbank_feats, torch.zeros((C, 1, D), device=fbank_feats.device, dtype=fbank_feats.dtype)], dim=1)
            fbank_feats = fbank_feats.view(C, -1, 2, D).reshape(C, -1, 2 * D)
            return fbank_feats
        elif self.feats_type == 'STFT':
            stft_feats = torch.stft(waveform, n_fft=512, hop_length=160, win_length=400, window=torch.hann_window(400).to(waveform.device), return_complex=True)
            stft_mag = torch.log1p(torch.abs(stft_feats))
            stft_phase_cos = torch.cos(torch.angle(stft_feats))
            stft_phase_sin = torch.sin(torch.angle(stft_feats))
            stft_feats = torch.stack([stft_mag, stft_phase_cos, stft_phase_sin], dim=1)
            return stft_feats.transpose(-2, -1)
        elif self.feats_type == 'STFT_CAT':
            stft_feats = torch.stft(waveform, n_fft=512, hop_length=160, win_length=400, window=torch.hann_window(400).to(waveform.device), return_complex=True)
            #stft_feats = torch.from_numpy(librosa.stft(waveform.numpy(), n_fft=512, hop_length=160, win_length=400, window='hann', center=True))
            #stft_feats = torch.cat([torch.stft(waveform[i:i+1, :], n_fft=512, hop_length=160, win_length=400, window=torch.hann_window(400), return_complex=True) for i in range(len(waveform))])
            stft_mag = torch.log1p(torch.abs(stft_feats))
            stft_phase_cos = torch.cos(torch.angle(stft_feats))
            stft_feats = torch.cat([stft_mag, stft_phase_cos], dim=-2)
            stft_feats = stft_feats.transpose(-2, -1)
            return stft_feats

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

    def freeze_all_but_channel_pos(self):
        self.model.encoder.channel_pos_enc
        for name, param in self.model.named_parameters():
            if "encoder.channel_pos_enc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        return 0

if __name__ == '__main__':
    device = torch.device("cuda")
    ckpt = '/export/c02/hzili1/workspace/s3prl/s3prl/upstream/unix_enc_custom/exp/cfg42/checkpoint-200000'
    upstream = UpstreamExpert(ckpt)
    upstream = upstream.to(device)
    for name, param in upstream.model.named_parameters():
        print('-' * 80)
        print(name)
        print(param)
    #wavs = (torch.rand(7, 48501, 4)).to(device)
    #output = upstream(wavs)
    #for h_state in output['hidden_states']:
    #    print(h_state.size())

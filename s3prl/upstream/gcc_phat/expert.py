# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/log_stft/expert.py ]
#   Synopsis     [ the wrapper for STFT magnitude ]
#   Author       [ Zili Huang ]
"""*********************************************************************************************"""


import math

###############
# IMPORTATION #
###############
import os
import random

# -------------#
import torch
import torch.nn as nn
import yaml
from torch.nn.utils.rnn import pad_sequence
import librosa
import numpy as np

# -------------#

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(nn.Module):
    """
    Extract spectrogram features from wavforms with torchaudio
    """

    def __init__(self, model_config=None, **kwargs):
        super(UpstreamExpert, self).__init__()

        with open(model_config, "r") as file:
            self.config = yaml.load(file, Loader=yaml.FullLoader)
        self.nfft = self.config['n_fft']
        self.hop_length = self.config['hop_length']
        self.n_chans = self.config['n_chans']
        self.output_dim = 49 * int(self.n_chans * (self.n_chans - 1) / 2)
        self.downsample_rate = self.hop_length 

    def get_downsample_rates(self, key: str) -> int:
        return self.downsample_rate

    def _extractor_forward(self, wavs):
        feats = []
        for wav in wavs:
            feats.append(torch.from_numpy(self.extract_gcc_phat(wav.transpose(0, 1).data.cpu().numpy())).float())
        return feats

    def gcc_phat(self, sig, refsig):
        Px = librosa.stft(y=sig,
                        n_fft=self.nfft,
                        hop_length=self.hop_length,
                        center=True,
                        window='hann',
                        pad_mode='reflect')
        Px_ref = librosa.stft(y=refsig,
                            n_fft=self.nfft,
                            hop_length=self.hop_length,
                            center=True,
                            window='hann',
                            pad_mode='reflect')
        R = Px*np.conj(Px_ref)
        n_frames = R.shape[1]
        gcc_phat = []
        max_shift = 24
        for i in range(n_frames):
            spec = R[:, i].flatten()
            cc = np.fft.irfft(np.exp(1.j*np.angle(spec)))
            cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
            gcc_phat.append(cc)
        gcc_phat = np.array(gcc_phat)
        return gcc_phat

    def extract_gcc_phat(self, waveform):
        C, T = waveform.shape
        gcc_features = []

        # Compute GCC-PHAT for all channel pairs
        for i in range(C):
            for j in range(i + 1, C):
                gcc = self.gcc_phat(waveform[i], waveform[j])
                gcc_features.append(gcc)

        gcc_features = np.stack(gcc_features)
        K, T, D = gcc_features.shape
        gcc_features = (gcc_features.transpose(1, 0, 2)).reshape(T, K * D)
        return gcc_features

    def forward(self, wavs):
        device = wavs[0].device
        feats = self._extractor_forward(wavs)
        feats = (pad_sequence(feats, batch_first=True)).to(device)
        return {"last_hidden_state": [feats], "hidden_states": [feats]}

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/wavlm/expert.py ]
#   Synopsis     [ the WavLM wrapper ]
#   Author       [ Microsoft ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
from .model import Whisper, ModelDimensions, log_mel_spectrogram 

############
# CONSTANT #
############
SAMPLE_RATE = 16000
EXAMPLE_SEC = 5


###################
# UPSTREAM EXPERT #
###################
class UpstreamExpert(UpstreamBase):
    def __init__(self, ckpt, **kwargs):
        super().__init__(**kwargs)

        checkpoint = torch.load(ckpt)
        dims = ModelDimensions(**checkpoint["dims"])
        self.model = Whisper(dims)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def get_downsample_rates(self, key: str) -> int:
        return 320

    def forward(self, wavs):
        device = wavs[0].device
        mels = [(log_mel_spectrogram(wav, n_mels=self.model.dims.n_mels).transpose(0, 1)).to(device) for wav in wavs]
        mels = (pad_sequence(mels, batch_first=True)).transpose(1, 2)
        output = self.model.encoder(mels)
        return {"x": output["x"], "hidden_states": output["hidden_states"]}

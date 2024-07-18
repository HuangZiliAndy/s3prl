# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ upstream/mockingjay/expert.py ]
#   Synopsis     [ the mockingjay wrapper ]
#   Author       [ Andy T. Liu (https://github.com/andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


from collections import OrderedDict
from typing import List, Tuple

import torch
import yaml
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from ..interfaces import UpstreamBase
import dac

class UpstreamExpert(UpstreamBase):
    """
    The DAC wrapper
    """

    def __init__(self, ckpt, downsampling=320, model_type="DAC", z="z", **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.z = z
        if model_type == "DAC":
            model = dac.DAC.load(ckpt)
        elif model_type == "DACMultiRes":
            model = dac.model.DACMultiRes.load(ckpt)
        elif model_type == "DACVqVae2":
            model = dac.model.DACVqVae2.load(ckpt)
        elif model_type == "DACKD":
            model = dac.model.DACKD.load(ckpt)
        else:
            raise ValueError("Unsuported model type (DAC or DACMultiRes)")
        self.model = model
        self.downsampling = int(downsampling)
        print(f"Model: {self.model_type} Features type: {self.z}")
        #self.quantize = quantize

    def get_downsample_rates(self, key: str) -> int:
        return self.downsampling

    def forward(self, wavs):
        device = wavs[0].device
        wav_lengths = torch.LongTensor([len(wav) for wav in wavs]).to(device)
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0).to(device),
            wav_lengths.unsqueeze(1),
        )
        padded_wav = pad_sequence(wavs, batch_first=True)
        padded_wav = padded_wav.unsqueeze(1)
        #if self.quantize:
        #    z, codes, latents, commitment_loss, codebook_loss = self.model.encode(padded_wav)
        #else:
        #    z = self.model.encoder(padded_wav)

        out = self.model(padded_wav, 16000)
        z = out[self.z]

        return {
            "last_hidden_state": z.transpose(1, 2),
            "hidden_states": [z.transpose(1, 2)],
        }

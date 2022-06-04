# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ loss.py ]
#   Synopsis     [ the objective functions for speech separation ]
#   Source       [ Use some code from https://github.com/funcwj/uPIT-for-speech-separation and https://github.com/asteroid-team/asteroid ]
#   Author       [ Zili Huang ]
#   Copyright    [ Copyright(c), Johns Hopkins University ]
"""*********************************************************************************************"""

import torch
from itertools import permutations
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SepLoss(object):
    def __init__(self, num_srcs, loss_type, mask_type, log='none'):
        """
        Args:
            num_srcs (int):
                number of sources

            mask_type (str):
                type of mask to approach, currently supporting AM, PSM and
                NPSM. Please see Kolbæk M, Yu D, Tan Z H, et al
                Multitalker speech separation with utterance-level permutation
                invariant training of deep recurrent neural network
                for details
        """
        self.num_srcs = num_srcs
        self.loss_type = loss_type
        self.mask_type = mask_type
        assert self.loss_type in ["MSE", "L1"]
        assert self.mask_type in ["AM", "PSM", "NPSM"]
        if self.loss_type == "MSE":
            self.loss = torch.nn.MSELoss(reduction='none')
        elif self.loss_type == "L1":
            self.loss = torch.nn.L1Loss(reduction='none')
        self.log = log

    def compute_loss(self, masks, feat_length, source_attr, target_attr):
        feat_length = feat_length.to(device)
        mixture_spect = source_attr["magnitude"].to(device)
        targets_spect = torch.stack(target_attr["magnitude"], dim=1).to(device)
        mixture_phase = source_attr["phase"].to(device)
        targets_phase = torch.stack(target_attr["phase"], dim=1).to(device)
        mask = torch.stack(masks, dim=1)
        perm_list = [list(perm) for perm in list(permutations(range(self.num_srcs)))]
        targets_spect = torch.stack([targets_spect[:, perm, :, :] for perm in perm_list], dim=0)
        targets_phase = torch.stack([targets_phase[:, perm, :, :] for perm in perm_list], dim=0)
        if self.mask_type == "AM":
            refer_spect = targets_spect
        elif self.mask_type == "PSM":
            refer_spect = targets_spect * torch.cos(mixture_phase.unsqueeze(0).unsqueeze(2) - targets_phase)
        elif self.mask_type == "NPSM":
            refer_spect = targets_spect * F.relu(torch.cos(mixture_phase.unsqueeze(0).unsqueeze(2) - targets_phase))
        else:
            raise ValueError("Mask type not defined.")

        if self.log == 'none':
            pass
        elif self.log == 'log1p':
            mixture_spect = torch.log1p(mixture_spect)
            refer_spect = torch.log1p(refer_spect)
        else:
            raise ValueError("Log type not defined.")

        predict_spect = mask * torch.unsqueeze(mixture_spect, 1)
        loss = self.loss(predict_spect.expand_as(refer_spect), refer_spect)
        loss = torch.sum(loss, dim=(2, 3, 4))
        loss, _ = torch.min(loss, dim=0)
        loss = loss / (feat_length * self.num_srcs)
        loss = torch.mean(loss)
        return loss

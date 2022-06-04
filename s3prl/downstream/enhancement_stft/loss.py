# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ loss.py ]
#   Synopsis     [ the objective functions for speech separation ]
#   Source       [ Use some code from https://github.com/funcwj/uPIT-for-speech-separation and https://github.com/asteroid-team/asteroid ]
#   Author       [ Zili Huang ]
#   Copyright    [ Copyright(c), Johns Hopkins University ]
"""*********************************************************************************************"""

import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EnhLoss(object):
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
        targets_spect = target_attr["magnitude"][0].to(device)
        mixture_phase = source_attr["phase"].to(device)
        targets_phase = target_attr["phase"][0].to(device)

        if self.mask_type == "AM":
            refer_spect = targets_spect
        elif self.mask_type == "PSM":
            refer_spect = targets_spect * torch.cos(mixture_phase - targets_phase)
        elif self.mask_type == "NPSM":
            refer_spect = targets_spect * F.relu(torch.cos(mixture_phase - targets_phase))
        else:
            raise ValueError("Mask type not defined.")

        if self.log == 'none':
            pass
        elif self.log == 'log1p':
            mixture_spect = torch.log1p(mixture_spect)
            refer_spect = torch.log1p(refer_spect)
        else:
            raise ValueError("Log type not defined.")

        loss = self.loss(masks[0] * mixture_spect, refer_spect)
        loss = torch.sum(loss, dim=(1, 2))
        loss = loss / feat_length
        loss = torch.mean(loss)
        return loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations

EPS = 1e-8

class MaskLoss(nn.Module):
    def __init__(self, mask_type, loss_type, num_srcs):
        super(MaskLoss, self).__init__()
        self.mask_type = mask_type
        self.loss_type = loss_type
        self.num_srcs = num_srcs
        assert self.loss_type in ["L1", "MSE"]
        if self.loss_type == "L1":
            self.loss = torch.nn.L1Loss(reduction='none')
        elif self.loss_type == "MSE":
            self.loss = torch.nn.MSELoss(reduction='none')

    def forward(self, pred_mask, src_stft_mag, tgt_stft_mag, src_stft_phase, tgt_stft_phase, src_audio, tgt_audios):
        pred_stft_mag = src_stft_mag.unsqueeze(1) * pred_mask.permute(0, 2, 1, 3)
        perm_list = [list(perm) for perm in list(permutations(range(self.num_srcs)))] 
        tgt_stft_mag = torch.stack([tgt_stft_mag[:, perm, :, :] for perm in perm_list], dim=1) 
        tgt_stft_phase = torch.stack([tgt_stft_phase[:, perm, :, :] for perm in perm_list], dim=1)
        if self.mask_type == "AM":
            ref_stft_mag = tgt_stft_mag
        elif self.mask_type == "PSM":
            ref_stft_mag = tgt_stft_mag * torch.cos(torch.angle(src_stft_phase.unsqueeze(1).unsqueeze(2)) - torch.angle(tgt_stft_phase))
        elif self.mask_type == "NPSM":
            ref_stft_mag = tgt_stft_mag * F.relu(torch.cos(torch.angle(src_stft_phase.unsqueeze(1).unsqueeze(2)) - torch.angle(tgt_stft_phase)))
        else:
            raise ValueError("Mask type not defined.")

        #print("self.mask_type", self.mask_type)
        #print("ref_stft_mag", ref_stft_mag.shape, torch.min(ref_stft_mag), torch.max(ref_stft_mag))
        #print("pred_stft_mag", pred_stft_mag.shape, torch.min(pred_stft_mag), torch.max(pred_stft_mag))
        loss = self.loss(pred_stft_mag.unsqueeze(1).expand_as(ref_stft_mag), ref_stft_mag)
        #print("loss", loss.shape)
        loss = torch.mean(torch.sum(loss, dim=4), dim=(2,3))
        #print("loss", loss.shape, loss)
        loss, min_idx = torch.min(loss, 1)
        #print("loss", loss, "min_idx", min_idx)
        loss = torch.mean(loss)
        return loss, [perm_list[idx] for idx in min_idx], ref_stft_mag

class SISDRLoss(nn.Module):
    def __init__(self, loss_type, num_srcs, n_fft, hop_length, win_length):
        super(SISDRLoss, self).__init__()
        self.num_srcs = num_srcs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.zero_mean = True

    def forward(self, pred_mask, src_stft_mag, tgt_stft_mag, src_stft_phase, tgt_stft_phase, src_audio, tgt_audios):
        pred_stft_mag = src_stft_mag.unsqueeze(1) * pred_mask.permute(0, 2, 1, 3)
        pred_stft = pred_stft_mag * src_stft_phase.unsqueeze(1)
        bs, num_srcs = pred_stft.size(0), pred_stft.size(1)
        pred_audios = torch.istft(
                pred_stft.reshape(bs * num_srcs, pred_stft.size(2), pred_stft.size(3)).permute(0, 2, 1), 
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=torch.hann_window(self.win_length).to(pred_mask.device),
                center=True,
            )
        pred_audios = pred_audios.reshape(bs, num_srcs, pred_audios.size(1))
        # pad the predicted audios to the same length
        est_targets = torch.zeros_like(tgt_audios)
        est_targets[:, :, :pred_audios.size(2)] = pred_audios

        perm_list = [list(perm) for perm in list(permutations(range(self.num_srcs)))] 
        loss_list = []
        for perm in perm_list:
            targets = tgt_audios[:, perm, :]

             # Step 1. Zero-mean norm
            if self.zero_mean:
                mean_source = torch.mean(targets, dim=2, keepdim=True)
                mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
                targets = targets - mean_source
                est_targets = est_targets - mean_estimate
            # Step 2. Pair-wise SI-SDR.
            # [batch, n_src]
            pair_wise_dot = torch.sum(est_targets * targets, dim=2,
                                      keepdim=True)
            # [batch, n_src]
            s_target_energy = torch.sum(targets ** 2, dim=2,
                                        keepdim=True) + EPS
            # [batch, n_src, time]
            scaled_targets = pair_wise_dot * targets / s_target_energy
            e_noise = est_targets - scaled_targets

            # [batch, n_src]
            pair_wise_sdr = torch.sum(scaled_targets ** 2, dim=2) / (
                    torch.sum(e_noise ** 2, dim=2) + EPS)

            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + EPS)
            loss_perm = -torch.mean(pair_wise_sdr, dim=-1)
            loss_list.append(loss_perm)
        loss = torch.stack(loss_list, 1)
        min_loss, min_idx = torch.min(loss, 1)
        return torch.mean(min_loss), [perm_list[idx] for idx in min_idx], None

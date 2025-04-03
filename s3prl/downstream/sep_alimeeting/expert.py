# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ expert.py ]
#   Synopsis     [ the speech separation downstream wrapper ]
#   Source       [ Reference some code from https://github.com/funcwj/uPIT-for-speech-separation and https://github.com/asteroid-team/asteroid ]
#   Author       [ Zili Huang ]
#   Copyright    [ Copyright(c), Johns Hopkins University ]
"""*********************************************************************************************"""

###############
# IMPORTATION #
###############
import os
import math
import random
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict

# -------------#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence
import torch.nn.functional as F

# -------------#
from .model import SepRNN
from .dataset import SeparationDataset, OnlineSeparationDatasetAMI
from asteroid.metrics import get_metrics
from .loss import MaskLoss, SISDRLoss
import soundfile as sf

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

COMPUTE_METRICS = ["si_sdr"]
EPS = 1e-10

class DownstreamExpert(nn.Module):
    """
    Used to handle downstream-specific operations
    eg. downstream forward, metric computation, contents to log
    """

    def __init__(self, upstream_dim, upstream_rate, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.upstream_rate = upstream_rate
        self.datarc = downstream_expert["datarc"]
        self.loaderrc = downstream_expert["loaderrc"]
        self.modelrc = downstream_expert["modelrc"]
        self.expdir = expdir

        #self.train_dataset = OnlineSeparationDatasetAMI(
        #    data_dir=self.loaderrc["train_dir"],
        #    add_reverb=self.datarc["add_reverb"],
        #    RIR_dir=self.datarc["RIR_dir_train"],
        #    add_noise=self.datarc["add_noise"],
        #    noise_type=self.datarc["noise_type"] if "noise_type" in self.datarc else "none",
        #    noise_scp_file=self.datarc["noise_scp_file_train"],
        #    full_overlap=self.datarc["full_overlap"],
        #    s1_first=self.datarc["s1_first"],
        #    crop_dur=10.0,
        #    max_num_spk=self.datarc["max_num_spk"],
        #    min_num_spk=self.datarc["min_num_spk"],
        #    min_sir=self.datarc["min_sir"],
        #    max_sir=self.datarc["max_sir"],
        #    min_snr=self.datarc["min_snr"],
        #    max_snr=self.datarc["max_snr"],
        #    rate=self.datarc["rate"],
        #    channel=self.datarc["channel"],
        #    n_fft=self.datarc["n_fft"],
        #    hop_length=self.datarc["hop_length"],
        #    win_length=self.datarc["win_length"],
        #    chunk_size=self.datarc["chunk_size"],
        #    normalize=self.datarc["normalize"],
        #    target=self.datarc["target"] if "target" in self.datarc else "clean",
        #    ref_channel=self.datarc["ref_channel"] if "ref_channel" in self.datarc else 0,
        #)
        #self.dev_dataset = OnlineSeparationDatasetAMI(
        #    data_dir=self.loaderrc["dev_dir"],
        #    add_reverb=self.datarc["add_reverb"],
        #    RIR_dir=self.datarc["RIR_dir_dev"],
        #    add_noise=self.datarc["add_noise"],
        #    noise_type=self.datarc["noise_type"] if "noise_type" in self.datarc else "none",
        #    noise_scp_file=self.datarc["noise_scp_file_dev"],
        #    full_overlap=self.datarc["full_overlap"],
        #    s1_first=self.datarc["s1_first"],
        #    crop_dur=10.0,
        #    max_num_spk=self.datarc["max_num_spk"],
        #    min_num_spk=self.datarc["min_num_spk"],
        #    min_sir=self.datarc["min_sir"],
        #    max_sir=self.datarc["max_sir"],
        #    min_snr=self.datarc["min_snr"],
        #    max_snr=self.datarc["max_snr"],
        #    rate=self.datarc["rate"],
        #    channel=self.datarc["channel"],
        #    n_fft=self.datarc["n_fft"],
        #    hop_length=self.datarc["hop_length"],
        #    win_length=self.datarc["win_length"],
        #    chunk_size=-1,
        #    normalize=self.datarc["normalize"],
        #    target=self.datarc["target"] if "target" in self.datarc else "clean",
        #    ref_channel=self.datarc["ref_channel"] if "ref_channel" in self.datarc else 0,
        #)
        #self.test_dataset = OnlineSeparationDatasetAMI(
        #    data_dir=self.loaderrc["test_dir"],
        #    add_reverb=self.datarc["add_reverb"],
        #    RIR_dir=self.datarc["RIR_dir_test"],
        #    add_noise=self.datarc["add_noise"],
        #    noise_type=self.datarc["noise_type"] if "noise_type" in self.datarc else "none",
        #    noise_scp_file=self.datarc["noise_scp_file_test"],
        #    full_overlap=self.datarc["full_overlap"],
        #    s1_first=self.datarc["s1_first"],
        #    crop_dur=10.0,
        #    max_num_spk=self.datarc["max_num_spk"],
        #    min_num_spk=self.datarc["min_num_spk"],
        #    min_sir=self.datarc["min_sir"],
        #    max_sir=self.datarc["max_sir"],
        #    min_snr=self.datarc["min_snr"],
        #    max_snr=self.datarc["max_snr"],
        #    rate=self.datarc["rate"],
        #    channel=self.datarc["channel"],
        #    n_fft=self.datarc["n_fft"],
        #    hop_length=self.datarc["hop_length"],
        #    win_length=self.datarc["win_length"],
        #    chunk_size=-1,
        #    normalize=self.datarc["normalize"],
        #    target=self.datarc["target"] if "target" in self.datarc else "clean",
        #    ref_channel=self.datarc["ref_channel"] if "ref_channel" in self.datarc else 0,
        #)

        self.train_dataset = SeparationDataset(
                data_dir=self.loaderrc["train_dir"],
                rate=self.datarc['rate'],
                src_cond=self.datarc['src_cond'],
                tgt_conds=self.datarc['tgt_conds'],
                channel=self.datarc['channel'],
                n_fft=self.datarc['n_fft'],
                hop_length=self.datarc['hop_length'],
                win_length=self.datarc['win_length'],
                chunk_size=self.datarc['chunk_size'],
                ref_channel=self.datarc['ref_channel'],
            )
        self.dev_dataset = SeparationDataset(
                data_dir=self.loaderrc["dev_dir"],
                rate=self.datarc['rate'],
                src_cond=self.datarc['src_cond'],
                tgt_conds=self.datarc['tgt_conds'],
                channel=self.datarc['channel'],
                n_fft=self.datarc['n_fft'],
                hop_length=self.datarc['hop_length'],
                win_length=self.datarc['win_length'],
                chunk_size=-1,
                ref_channel=self.datarc['ref_channel'],
            )
        self.test_dataset = SeparationDataset(
                data_dir=self.loaderrc["test_dir"],
                rate=self.datarc['rate'],
                src_cond=self.datarc['src_cond'],
                tgt_conds=self.datarc['tgt_conds'],
                channel=self.datarc['channel'],
                n_fft=self.datarc['n_fft'],
                hop_length=self.datarc['hop_length'],
                win_length=self.datarc['win_length'],
                chunk_size=-1,
                ref_channel=self.datarc['ref_channel'],
            )

        if self.modelrc["model"] == "SepRNN":
            if self.datarc["concatenate"]:
                input_dim = self.upstream_dim + int(self.datarc['n_fft'] / 2 + 1)
            else:
                input_dim = self.upstream_dim
            self.model = SepRNN(
                input_dim=input_dim,
                num_bins=int(self.datarc["n_fft"] / 2 + 1),
                rnn=self.modelrc["rnn"],
                num_spks=self.datarc["num_speakers"],
                num_layers=self.modelrc["rnn_layers"],
                hidden_size=self.modelrc["hidden_size"],
                dropout=self.modelrc["dropout"],
                non_linear=self.modelrc["non_linear"],
                bidirectional=self.modelrc["bidirectional"]
            )
        else:
            raise ValueError("Model type not defined.")

        self.loss_type = self.modelrc["loss_type"]
        if self.loss_type == 'SISDR':
            self.objective = SISDRLoss(self.modelrc["loss_type"], self.datarc['num_speakers'], self.datarc['n_fft'], self.datarc['hop_length'], self.datarc['win_length'])
        elif self.loss_type in ["L1", "MSE"]:
            self.objective = MaskLoss(self.modelrc["mask_type"], self.modelrc["loss_type"], self.datarc['num_speakers'])
        
        self.register_buffer("best_score", torch.ones(1) * -10000)

    def _get_train_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.loaderrc["train_batchsize"],
            shuffle=True,
            num_workers=self.loaderrc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.loaderrc["eval_batchsize"],
            shuffle=False,
            num_workers=4,
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    def get_dataloader(self, mode):
        """
        Args:
            mode: string
                'train', 'dev' or 'test'
        Return:
            a torch.utils.data.DataLoader returning each batch in the format of:
            [wav1, wav2, ...], your_other_contents1, your_other_contents2, ...
            where wav1, wav2 ... are in variable length
            each wav is torch.FloatTensor in cpu with:
                1. dim() == 1
                2. sample_rate == 16000
                3. directly loaded by torchaudio
        """
        if mode == "train":
            return self._get_train_dataloader(self.train_dataset)
        elif mode == "dev":
            return self._get_eval_dataloader(self.dev_dataset)
        elif mode == "test":
            return self._get_eval_dataloader(self.test_dataset)

    def forward(self, mode, features, src_audio, tgt_audios, src_stft_mag, tgt_stft_mag, src_stft_phase, tgt_stft_phase, uttname_list, records, **kwargs):
        features = torch.stack(features)
        #print("features", features.shape)
        assert src_stft_mag.size(1) == src_stft_phase.size(1) == 1
        src_stft_mag = (src_stft_mag[:, 0, :, :]).to(features.device)
        tgt_stft_mag = tgt_stft_mag.to(features.device)
        src_stft_phase = (src_stft_phase[:, 0, :, :]).to(features.device)
        tgt_stft_phase = tgt_stft_phase.to(features.device)
        #print("src_stft_mag", src_stft_mag.shape, torch.min(src_stft_mag), torch.max(src_stft_mag))
        #print("tgt_stft_mag", tgt_stft_mag.shape, torch.min(tgt_stft_mag), torch.max(tgt_stft_mag))
        #print("src_stft_phase", src_stft_phase.shape, src_stft_phase.dtype)
        #print("tgt_stft_phase", tgt_stft_phase.shape, tgt_stft_phase.dtype)

        # repeat SSL representations to match the STFT rate
        assert self.upstream_rate == 160 or self.upstream_rate == 320
        assert self.datarc["hop_length"] == 160
        if self.upstream_rate == 320:
            features = torch.repeat_interleave(features, 2, dim=1)

        # match the length
        min_length = min(features.size(1), src_stft_mag.size(1))
        features = features[:, :min_length, :]
        src_stft_mag, tgt_stft_mag = src_stft_mag[:, :min_length, :], tgt_stft_mag[:, :, :min_length, :]
        src_stft_phase, tgt_stft_phase = src_stft_phase[:, :min_length, :], tgt_stft_phase[:, :, :min_length, :]

        if self.datarc["concatenate"]:
            if self.datarc["log1p"]:
                input_feature = torch.cat([features, torch.log1p(src_stft_mag)], 2)
            else:
                input_feature = torch.cat([features, src_stft_mag], 2)
        else:
            input_feature = features
        #print("input_feature", input_feature.shape)

        pred_mask = self.model(input_feature)
        #print("pred_mask", pred_mask.shape)

        loss, perm_list, ref_stft_mag = self.objective(pred_mask, src_stft_mag, tgt_stft_mag, src_stft_phase, tgt_stft_phase, src_audio, tgt_audios)
        records["loss"].append(loss.item())

        # evaluate the separation quality of predict sources
        if mode == 'dev' or mode == 'test':
            pred_stft_mag = src_stft_mag.unsqueeze(1) * pred_mask.permute(0, 2, 1, 3)
            pred_stft = pred_stft_mag * src_stft_phase.unsqueeze(1)
            bs, num_srcs = pred_stft.size(0), pred_stft.size(1)
            assert bs == 1
            pred_audios = torch.istft(
                    pred_stft.view(bs * num_srcs, pred_stft.size(2), pred_stft.size(3)).permute(0, 2, 1), 
                    n_fft=self.datarc["n_fft"],
                    hop_length=self.datarc["hop_length"],
                    win_length=self.datarc["win_length"],
                    window=torch.hann_window(self.datarc["win_length"]).to(features.device),
                    center=True,
                )
            pred_audios = pred_audios.view(bs, num_srcs, pred_audios.size(1))

            # pad the predicted audios to the same length
            pred_audios_pad = torch.zeros_like(tgt_audios)
            pred_audios_pad[:, :, :pred_audios.size(2)] = pred_audios
            assert len(tgt_audios) == len(pred_audios_pad) == 1

            tgt_audios_perm = tgt_audios[:, perm_list[0], :]
            #tgt_audios_perm = tgt_audios

            if len(src_audio.size()) == 2:
                mix_audio = src_audio.cpu().numpy()
            elif len(src_audio.size()) == 3:
                mix_audio = src_audio[:, self.datarc['ref_channel'], :].cpu().numpy()

            #if mode == 'test':
            #    sf.write('{}/predictions/{}_mix.wav'.format(self.expdir, uttname_list[0]), mix_audio[0], 16000)
            #    for i in range(2):
            #        sf.write('{}/predictions/{}_s{}_ref.wav'.format(self.expdir, uttname_list[0], i+1), tgt_audios_perm[0][i], 16000)
            #    for i in range(2):
            #        sf.write('{}/predictions/{}_s{}_pred.wav'.format(self.expdir, uttname_list[0], i+1), pred_audios_pad[0][i], 16000)

            utt_metrics = get_metrics(
                    mix_audio,
                    tgt_audios_perm[0].cpu().numpy(),
                    pred_audios_pad[0].cpu().numpy(),
                    sample_rate = self.datarc['rate'],
                    metrics_list = COMPUTE_METRICS,
                    compute_permutation=False,
                )

            for metric in COMPUTE_METRICS:
                input_metric = "input_" + metric
                assert metric in utt_metrics and input_metric in utt_metrics
                imp = utt_metrics[metric] - utt_metrics[input_metric]
                if metric not in records:
                    records[metric] = []
                if metric == "si_sdr":
                    records[metric].append(imp)
                elif metric == "stoi" or metric == "pesq":
                    records[metric].append(utt_metrics[metric])
                else:
                    raise ValueError("Metric type not defined.")

        return loss

    # interface
    def log_records(
        self, mode, records, logger, global_step, batch_ids, total_batch_num, **kwargs
    ):
        """
        Args:
            mode: string
                'train':
                    records and batchids contain contents for `log_step` batches
                    `log_step` is defined in your downstream config
                    eg. downstream/example/config.yaml
                'dev' or 'test' :
                    records and batchids contain contents for the entire evaluation dataset

            records:
                defaultdict(list), contents already appended

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            global_step:
                The global_step when training, which is helpful for Tensorboard logging

            batch_ids:
                The batches contained in records when enumerating over the dataloader

            total_batch_num:
                The total amount of batches in the dataloader

        Return:
            a list of string
                Each string is a filename we wish to use to save the current model
                according to the evaluation result, like the best.ckpt on the dev set
                You can return nothing or an empty list when no need to save the checkpoint
        """
        if mode == 'train':
            avg_loss = np.mean(records["loss"])
            logger.add_scalar(
                f"sep_ami/{mode}-loss", avg_loss, global_step=global_step
            )
            return []
        else:
            avg_loss = np.mean(records["loss"])
            logger.add_scalar(
                f"sep_ami/{mode}-loss", avg_loss, global_step=global_step
            )
            with (Path(self.expdir) / f"{mode}_metrics.txt").open("w") as output:
                for metric in COMPUTE_METRICS:
                    avg_metric = np.mean(records[metric])
                    if mode == "test" or mode == "dev":
                        print("Average {} of {} utts: {:.4f}".format(metric, len(records[metric]), avg_metric))
                        print(metric, avg_metric, file=output)

                    logger.add_scalar(
                        f'sep_ami/{mode}-'+metric,
                        avg_metric,
                        global_step=global_step
                    )

            save_ckpt = []
            assert 'si_sdr' in records
            if mode == "dev" and np.mean(records['si_sdr']) > self.best_score:
                self.best_score = torch.ones(1) * np.mean(records['si_sdr'])
                save_ckpt.append(f"best-states-{mode}.ckpt")
            return save_ckpt

# -*- coding: utf-8 -*- #

import os
import math
import h5py
import random
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed import is_initialized, get_rank
from torch.nn.utils.rnn import pad_sequence

from .model import Model
from .dataset import DiarizationDataset
from .utils import pit_loss, calc_diarization_error, get_label_perm

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

        config_frame_shift = self.datarc.get("frame_shift")
        if isinstance(config_frame_shift, int):
            logging.warning(
                f"Diarization label frame shift: {config_frame_shift}. "
                "It is set in the config field. You don't need to set this config field if "
                "you are training new downstream models. This module will then automatically "
                "use upstream's downsample rate as the training label frame shift. This "
                "'if condition' is designed only to inference the already trained downstream "
                "checkpoints with the command: python3 run_downstream.py -m evaluate -e [ckpt]. "
                "The checkpoint contains the frame_shift used for its training, and the same "
                "frame_shift should be prepared for the inference."
            )
            frame_shift = config_frame_shift
        else:
            logging.warning(
                f"Diarization label frame shift: {upstream_rate}. It is automatically set as "
                "upstream's downsample rate to best align the representation v.s. labels for training. "
                "This frame_shift information will be saved in the checkpoint for future inference."
            )
            frame_shift = upstream_rate

        self.datarc["frame_shift"] = frame_shift
        with (Path(expdir) / "frame_shift").open("w") as file:
            print(frame_shift, file=file)

        self.loaderrc = downstream_expert["loaderrc"]
        self.modelrc = downstream_expert["modelrc"]

        self.train_batch_size = self.loaderrc["train_batchsize"]
        self.eval_batch_size = self.loaderrc["eval_batchsize"]

        self.expdir = expdir

        self.model = Model(
            input_dim=self.upstream_dim,
            output_class_num=self.datarc["num_spks"] if "num_spks" in self.datarc else self.datarc["num_speakers"],
            **self.modelrc,
        )
        self.objective = pit_loss

        self.register_buffer("best_score", torch.zeros(1))

    # Interface
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
        if not hasattr(self, f"{mode}_dataset"):
            dataset = DiarizationDataset(
                self.loaderrc[f"{mode}_dir"],
                **self.datarc,
            )
            setattr(self, f"{mode}_dataset", dataset)

        if mode == "train":
            return self._get_train_dataloader(self.train_dataset)
        elif mode == "dev":
            return self._get_dev_dataloader(self.dev_dataset)
        elif mode == "test":
            return self._get_test_dataloader(self.test_dataset)

    """
    Datalaoder Specs:
        Each dataloader should output in the following format:

        [[wav1, wav2, ...], your_other_contents1, your_other_contents2, ...]

        where wav1, wav2 ... are in variable length
        each wav is torch.FloatTensor in cpu with dim()==1 and sample_rate==16000
    """

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.loaderrc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    def _get_dev_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.loaderrc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    def _get_test_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.loaderrc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    def _match_length(self, inputs, labels):
        """
        Since the upstream extraction process can sometimes cause a mismatch
        between the seq lenth of inputs and labels:
        - if len(inputs) > len(labels), we truncate the final few timestamp of inputs to match the length of labels
        - if len(inputs) < len(labels), we duplicate the last timestep of inputs to match the length of labels
        Note that the length of labels should never be changed.
        """
        input_len, label_len = inputs.size(1), labels.size(1)
        assert abs(input_len - label_len) <= 3

        if input_len > label_len:
            inputs = inputs[:, :label_len, :]
        elif input_len < label_len:
            pad_vec = inputs[:, -1, :].unsqueeze(1)  # (batch_size, 1, feature_dim)
            inputs = torch.cat(
                (inputs, pad_vec.repeat(1, label_len - input_len, 1)), dim=1
            )  # (batch_size, seq_len, feature_dim), where seq_len == labels.size(-1)
        return inputs, labels

    # Interface
    def forward(self, mode, features, labels, lengths, rec_id, records, **kwargs):
        """
        Args:
            mode: string
                'train', 'dev' or 'test' for this forward step

            features:
                list of unpadded features [feat1, feat2, ...]
                each feat is in torch.FloatTensor and already
                put in the device assigned by command-line args

            labels:
                the frame-wise speaker labels

            rec_id:
                related recording id, use for inference

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss:
                the loss to be optimized, should not be detached
        """

        features = (pad_sequence(features, batch_first=True)).float()
        labels = [torch.from_numpy(label).float() for label in labels]
        labels = pad_sequence(labels, batch_first=True).to(features.device)
        lengths = torch.tensor(lengths, dtype=torch.long)
        features, labels = self._match_length(features, labels)
        predicted = self.model(features)

        loss, perm_idx, perm_list = self.objective(predicted, labels, lengths)

        # get the best label permutation
        label_perm = get_label_perm(labels, perm_idx, perm_list)

        (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        ) = calc_diarization_error(predicted, label_perm, lengths)

        if speech_scored > 0 and speaker_scored > 0 and num_frames > 0:
            SAD_MR, SAD_FR, MI, FA, CF, ACC, DER = (
                speech_miss / speech_scored,
                speech_falarm / speech_scored,
                speaker_miss / speaker_scored,
                speaker_falarm / speaker_scored,
                speaker_error / speaker_scored,
                correct / num_frames,
                (speaker_miss + speaker_falarm + speaker_error) / speaker_scored,
            )
        else:
            SAD_MR, SAD_FR, MI, FA, CF, ACC, DER = 0, 0, 0, 0, 0, 0, 0

        # print("SAD_MR {}, SAD_FR {}, MI {}, FA {}, CF {}, ACC {}, DER {}".format(SAD_MR, SAD_FR, MI, FA, CF, ACC, DER))
        records["loss"].append(loss.item())
        records["acc"] += [ACC]
        records["err"] += [speaker_miss + speaker_falarm + speaker_error]
        records["total"] += [speaker_scored]
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
        average_loss = torch.FloatTensor(records['loss']).mean().item()
        average_acc = torch.FloatTensor(records["acc"]).mean().item()
        average_der = torch.FloatTensor(records["err"]).sum().item() / torch.FloatTensor(records["total"]).sum().item()

        logger.add_scalar(
            f"diar_ami/{mode}-loss", average_loss, global_step=global_step
        )
        logger.add_scalar(
            f"diar_ami/{mode}-acc", average_acc, global_step=global_step
        )
        logger.add_scalar(
            f"diar_ami/{mode}-der", average_der, global_step=global_step
        )
        print("mode {} acc {} der {} loss {}".format(mode, average_acc, average_der, average_loss))

        save_ckpt = []
        if mode == "dev" and average_acc > self.best_score:
            self.best_score = torch.ones(1) * average_acc
            save_ckpt.append(f"best-states-{mode}.ckpt")

        return save_ckpt

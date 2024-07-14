# -*- coding: utf-8 -*- #

import logging
import os
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from .dictionary import Dictionary
import soundfile as sf

class ASRDataset(Dataset):
    def __init__(self, split, data_dir, dictionary, channel="0", max_samples=480000, normalize=False, **kwargs):
        super(ASRDataset, self).__init__()
        self.dictionary = dictionary
        self.wav2path = self.load_mapping("{}/wav.scp".format(data_dir))
        self.utt2text = self.load_mapping("{}/text".format(data_dir))
        self.utt2numsamples = self.load_mapping("{}/utt2num_samples".format(data_dir))
        self.channel = [int(c) for c in channel.split(',')]
        if len(self.channel) == 1:
            self.channel = self.channel[0]
        self.normalize = normalize
        self.uttlist = list(self.wav2path.keys())
        self.uttlist.sort()
        len_uttlist_ori = len(self.uttlist)
        self.uttlist = [utt for utt in self.uttlist if int(self.utt2numsamples[utt]) <= max_samples]
        print("Split {} loading {}/{} utterances".format(split, len(self.uttlist), len_uttlist_ori))

    def load_mapping(self, fname):
        mapping = {}
        with open(fname, 'r') as fh:
            content = fh.readlines()
        for line in content:
            line = line.strip('\n')
            line_split = line.split(None, 1)
            mapping[line_split[0]] = line_split[1]
        return mapping

    def process_trans(self, transcript):
        #TODO: support character / bpe
        transcript = transcript.upper()
        return " ".join(list(transcript.replace(" ", "|"))) + " |"

    def _build_dictionary(self, transcripts, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        d = Dictionary()
        transcript_list = list(transcripts.values())
        Dictionary.add_transcripts_to_dictionary(
            transcript_list, d, workers
        )
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    def __len__(self):
        return len(self.uttlist)

    def __getitem__(self, index):
        utt = self.uttlist[index]
        path = self.wav2path[utt]
        audio = sf.read(path)[0]
        if self.normalize:
            audio = audio / (np.max(np.abs(audio)) + 1e-10)
        if len(audio.shape) == 2:
            audio = audio[:, self.channel]
        elif len(audio.shape) == 1:
            assert self.channel == 0
        else:
            raise ValueError("Invalid audio shape")
        audio = torch.from_numpy(audio).float()
        text = self.utt2text[utt]
        text = self.process_trans(text)
        text = self.dictionary.encode_line(text, line_tokenizer=lambda x: x.split()).long()
        return audio, text, utt 

    def collate_fn(self, samples):
        sorted_samples = sorted(samples, key=lambda x: -x[0].size(0))
        return zip(*sorted_samples)

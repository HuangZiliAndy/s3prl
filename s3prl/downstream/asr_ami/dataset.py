# -*- coding: utf-8 -*- #

"""
dataset.py — PyTorch Dataset for CTC-based ASR on segmented audio.

Loads pre-segmented utterances from a Kaldi-style data directory that contains:
  wav.scp          — utterance ID → wav file path
  text             — utterance ID → transcript
  utt2num_samples  — utterance ID → number of audio samples (used for length filtering)

Transcripts are converted to character-level token sequences using "|" as the
word boundary marker, matching the convention in token_to_word() in expert.py.
"""

import logging
import os
import random
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from .dictionary import Dictionary
import soundfile as sf


class ASRDataset(Dataset):
    """Dataset for pre-segmented ASR utterances stored in a Kaldi-style directory.

    Filters out utterances longer than max_samples at construction time.
    Audio channels are selected at load time: a single integer selects one channel;
    a comma-separated string (e.g. "0,1") returns all listed channels.

    Args:
        split:       Split name (used for logging only).
        data_dir:    Path to the Kaldi data directory.
        dictionary:  Dictionary used to encode transcripts.
        channel:     Comma-separated 0-indexed channel(s) to load (default: "0").
        max_samples: Maximum number of audio samples per utterance (default: 480000 = 30 s at 16 kHz).
        normalize:   If True, peak-normalize each waveform to [-1, 1] before returning.
        **kwargs:    Additional keyword arguments (ignored).
    """

    def __init__(self, split, data_dir, dictionary, channel="0", max_samples=480000, normalize=False, wav_preprocess=None, **kwargs):
        super(ASRDataset, self).__init__()
        self.dictionary = dictionary
        self.wav2path = self.load_mapping("{}/wav.scp".format(data_dir))
        self.utt2text = self.load_mapping("{}/text".format(data_dir))
        self.utt2numsamples = self.load_mapping("{}/utt2num_samples".format(data_dir))
        self.channel = [int(c) for c in channel.split(',')]
        if len(self.channel) == 1:
            self.channel = self.channel[0]
        self.normalize = normalize
        self.wav_preprocess = wav_preprocess
        self.uttlist = list(self.wav2path.keys())
        self.uttlist.sort()
        len_uttlist_ori = len(self.uttlist)
        self.uttlist = [utt for utt in self.uttlist if int(self.utt2numsamples[utt]) <= max_samples]
        print("Split {} loading {}/{} utterances".format(split, len(self.uttlist), len_uttlist_ori))

    def load_mapping(self, fname):
        """Load a two-column Kaldi file into a dict keyed by the first field.

        Lines are split on the first whitespace only, so values may contain spaces
        (e.g. paths or transcripts with multiple words).

        Args:
            fname: Path to the file (wav.scp, text, utt2num_samples, etc.).

        Returns:
            Dict mapping utterance ID → value string.
        """
        mapping = {}
        with open(fname, 'r') as fh:
            content = fh.readlines()
        for line in content:
            line = line.strip('\n')
            line_split = line.split(None, 1)
            mapping[line_split[0]] = line_split[1]
        return mapping

    def process_trans(self, transcript):
        """Convert a word-level transcript to a space-separated character token string.

        Words are uppercased, spaces are replaced with "|" (word boundary token),
        and a trailing "|" is appended to mark the end of the last word.

        Example: "hello world" -> "H E L L O | W O R L D |"

        TODO: support subword (BPE) tokenization.

        Args:
            transcript: Raw word-level transcript string.

        Returns:
            Space-separated character token string.
        """
        transcript = transcript.upper()
        return " ".join(list(transcript.replace(" ", "|"))) + " |"

    def _build_dictionary(self, transcripts, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        """Build a new Dictionary from a mapping of utterance IDs to transcripts.

        Intended as a utility for offline dictionary construction. Not called
        during normal dataset loading (the dictionary is passed in at init time).

        Args:
            transcripts:    Dict of utterance ID → transcript string.
            workers:        Number of parallel workers for token counting.
            threshold:      Minimum token count to include in the vocabulary.
            nwords:         Maximum vocabulary size (-1 for no limit).
            padding_factor: Pad vocabulary size to a multiple of this value.

        Returns:
            A finalized Dictionary instance.
        """
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
        """Load one utterance.

        Returns:
            audio: 1-D float32 Tensor of waveform samples (mono, 16 kHz assumed).
            text:  LongTensor of dictionary token IDs for the transcript.
            utt:   Utterance ID string.
        """
        utt = self.uttlist[index]
        path = self.wav2path[utt]
        audio, _ = sf.read(path, dtype='float32')
        if self.normalize:
            audio = audio / (np.max(np.abs(audio)) + 1e-10)
        if len(audio.shape) == 2:
            audio = np.ascontiguousarray(audio[:, self.channel])
        elif len(audio.shape) == 1:
            assert self.channel == 0, (
                f"Channel {self.channel} requested but audio is already mono: {path}"
            )
        else:
            raise ValueError("Invalid audio shape")
        audio = torch.from_numpy(audio)
        if self.wav_preprocess is not None:
            audio = self.wav_preprocess(audio)

        text = self.utt2text[utt]
        text = self.process_trans(text)
        text = self.dictionary.encode_line(text, line_tokenizer=lambda x: x.split()).long()
        return audio, text, utt

    def collate_fn(self, samples):
        """Collate a list of (audio, text, utt) samples into a batch.

        Sorts by descending audio length so that pack_padded_sequence in
        LSTMASRModel receives a properly ordered batch.

        Returns:
            A zip iterator yielding (audios, texts, utts) as tuples.
            The caller is expected to unpack these into separate sequences.
        """
        sorted_samples = sorted(samples, key=lambda x: -x[0].size(0))
        return tuple(zip(*sorted_samples))

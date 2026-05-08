import os
import torch
import torchaudio
import argparse
import random
import numpy as np
from s3prl.downstream.runner import Runner
from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote.audio.core.task import Specifications, Problem, Resolution
from pyannote.audio import Pipeline
from typing import Optional
from torch.nn.utils.rnn import pad_sequence
from s3prl.downstream.diar_ami.dataset import DiarizationDataset
from typing import Callable, Mapping, Optional, Text, Union
from pathlib import Path
from pyannote.audio.core.io import Audio
from tqdm import tqdm
from torch import Tensor

parser = argparse.ArgumentParser(description='Evaluate diarization model')
parser.add_argument('ckpt_path', type=str, help='checkpoint path')
parser.add_argument('test_dir', type=str, help='test directory')
parser.add_argument('output_dir', type=str, help='output directory')
parser.add_argument('--channel', type=str, default='0', help='audio channel')
parser.add_argument('--normalize', type=int, default=0, help='whether to normalize audio')
parser.add_argument('--segmentation_thres', type=float, default=0.5, help='segmentation threshold')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AudioCustom(Audio):
    def __init__(self, sample_rate=None, mono=None):
        super().__init__(sample_rate=sample_rate, mono=mono)

    def downmix_and_resample(self, waveform: Tensor, sample_rate: int) -> Tensor:
        """Downmix and resample

        Parameters
        ----------
        waveform : (channel, time) Tensor
            Waveform.
        sample_rate : int
            Sample rate.

        Returns
        -------
        waveform : (channel, time) Tensor
            Remixed and resampled waveform
        sample_rate : int
            New sample rate
        """

        # downmix to mono
        num_channels = waveform.shape[0]
        if num_channels > 1:
            if self.mono == "random":
                channel = random.randint(0, num_channels - 1)
                waveform = waveform[channel : channel + 1]
            elif self.mono == "downmix":
                waveform = waveform.mean(dim=0, keepdim=True)
            elif self.mono == "chan0":
                waveform = waveform[0 : 1]

        # resample
        if (self.sample_rate is not None) and (self.sample_rate != sample_rate):
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.sample_rate
            )
            sample_rate = self.sample_rate

        return waveform, sample_rate

class S3PRLModel(Model):
    def __init__(
        self,
        upstream,
        featurizer,
        downstream,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=1, task=task)

        self.upstream = upstream
        self.featurizer = featurizer
        self.downstream = downstream
        self.audio = AudioCustom(sample_rate=self.hparams.sample_rate, mono=None)

    def audio(self, file: str):
        waveform, sample_rate = torchaudio.load(file)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        return waveform, sample_rate

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Pass forward

        Parameters
        ----------
        waveforms : (batch, channel, sample)

        Returns
        -------
        scores : (batch, frame, classes)
        """
        if waveforms.size(1) == 1:
            wavs = [(wav[0, :]).to(device) for wav in waveforms]
        else:
            wavs = [(wav[:, :]).transpose(0, 1).to(device) for wav in waveforms]
        with torch.no_grad():
            features = self.upstream.model(wavs)
            features = self.featurizer.model(wavs, features)
            #print("features", [feat.size() for feat in features])
            features = pad_sequence(features, batch_first=True)
            #print("features", features.size())
            prediction = torch.sigmoid(self.downstream.model.model(features))
            #print("prediction", prediction.size())
            #print("prediction", torch.max(prediction), torch.min(prediction))
        return prediction

def main():
    print("Loading {}".format(args.ckpt_path))
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    ckpt_args, ckpt_config = ckpt['Args'], ckpt['Config']
    ckpt_args.init_ckpt = args.ckpt_path
    runner = Runner(ckpt_args, ckpt_config)
    upstream = runner.upstream
    featurizer = runner.featurizer
    downstream = runner.downstream
    upstream_rate = featurizer.model.downsample_rate

    segmentation_model = S3PRLModel(upstream, featurizer, downstream, num_channels=len(args.channel.split(',')))
    segmentation_model = segmentation_model.to(device)

    dataset = DiarizationDataset(args.test_dir, num_spks=5, frame_shift=upstream_rate, channel=args.channel, normalize=args.normalize)
    for i, v in enumerate(tqdm(dataset)):
        audio, label, length, uttname = v

        audio = (torch.from_numpy(audio).float()).to(device)
        if len(audio.size()) == 1:
            audio = audio.unsqueeze(0)
        elif len(audio.size()) == 2:
            audio = audio.transpose(0, 1)
        audio = audio.unsqueeze(0)

        seg_duration = 10.0
        seg_nsamples = int(seg_duration * 16000)
        if audio.shape[-1] <= seg_nsamples:
            prediction = segmentation_model(audio)
        else:
            nsegs = int(np.ceil(audio.shape[-1] / seg_nsamples)) 
            for j in range(nsegs):
                if j != nsegs - 1:
                    prediction = segmentation_model(audio[:, :, j * seg_nsamples:(j+1) * seg_nsamples])
                    print(j * seg_nsamples, (j+1) * seg_nsamples, "prediction", prediction.shape)
                else:
                    prediction = segmentation_model(audio[:, :, -seg_nsamples:])
                    print(audio.shape[-1] - seg_nsamples, audio.shape[-1], "prediction", prediction.shape)
        raise


        #with open('{}/{}.rttm'.format(args.output_dir, uttname), 'w') as rttm_f:
        #    diar_result.write_rttm(rttm_f)
    return 0

if __name__ == '__main__':
    main()
